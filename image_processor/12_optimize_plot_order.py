# 12_optimize_plot_order.py
# Joint travel optimization: interleave polylines and taps per layer to minimize pen-up travel.
# Inputs : <layer>/lines_cross.pkl, <layer>/taps_cross.pkl
# Outputs: <layer>/plot_ops.pkl  (interleaved ops: [{"type":"line","points":...}|{"type":"tap","x":...,"y":...}])
#          <output>/vector_manifest.json  (points to per-layer plot_ops.pkl)
#
# Algorithm:
#   1) Build a candidate set of ops: lines (with reversible orientation) + taps (points).
#   2) Greedy nearest-neighbor that chooses both next item and line orientation.
#   3) Limited 2-opt on the interleaved sequence (segment-reversal with orientation flip inside).
#      We accept a reversal only if it reduces the TOTAL route cost (safe but bounded by a pair limit).

from __future__ import annotations
import os, json, pickle, math
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2

from config import load_config, Config

# ---------------- size & color utils ----------------

def _target_size_px(cfg: Config) -> Tuple[int,int]:
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 and th > 0:
        return tw, th
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size.")
    h, w = base.shape[:2]
    return w, h

def _color_index(cfg: Config, name: str) -> int:
    try:
        return list(cfg.color_names).index(name)
    except ValueError:
        return 0

# ---------------- data model ----------------

class Op:
    __slots__ = ("kind", "pts", "xy", "flip")  # kind: "line"|"tap"
    def __init__(self, kind: str, pts: np.ndarray | None = None, xy: Tuple[int,int] | None = None):
        self.kind = kind
        self.pts  = None
        self.xy   = None
        self.flip = False
        if kind == "line":
            p = np.asarray(pts).reshape(-1,2).astype(np.int32)
            self.pts = p
        elif kind == "tap":
            x,y = xy
            self.xy = (int(x), int(y))
        else:
            raise ValueError("Unknown op kind")

    def start(self) -> Tuple[int,int]:
        if self.kind == "tap":
            return self.xy
        if not self.flip:
            a = self.pts[0]
        else:
            a = self.pts[-1]
        return (int(a[0]), int(a[1]))

    def end(self) -> Tuple[int,int]:
        if self.kind == "tap":
            return self.xy
        if not self.flip:
            b = self.pts[-1]
        else:
            b = self.pts[0]
        return (int(b[0]), int(b[1]))

    def oriented_points(self) -> np.ndarray:
        if self.kind == "tap":
            return np.empty((0,2), np.int32)
        return (self.pts if not self.flip else self.pts[::-1]).astype(np.int32, copy=False)

    def length(self) -> float:
        if self.kind == "tap":
            return 0.0
        d = np.diff(self.pts.astype(np.float32), axis=0)
        return float(np.hypot(d[:,0], d[:,1]).sum())

# ---------------- I/O ----------------

def _load_cross(layer_dir: str) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
    pL = os.path.join(layer_dir, "lines_cross.pkl")
    pT = os.path.join(layer_dir, "taps_cross.pkl")
    if not (os.path.exists(pL) and os.path.exists(pT)):
        raise RuntimeError(f"Missing cross-layer inputs in {layer_dir}")
    with open(pL, "rb") as f: lines = pickle.load(f)
    taps: List[Tuple[int,int]] = []
    with open(pT, "rb") as f:
        obj = pickle.load(f)
        for it in obj:
            a = np.asarray(it).reshape(-1)
            if a.size >= 2:
                taps.append((int(a[0]), int(a[1])))
    return lines, taps

def _save_ops(layer_dir: str, ops: List[Op], color_name: str, color_idx: int) -> str:
    packed_ops: List[Dict[str,Any]] = []
    for op in ops:
        if op.kind == "line":
            packed_ops.append({"type":"line","points": op.oriented_points().astype(np.int32)})
        else:
            x,y = op.xy
            packed_ops.append({"type":"tap","x":int(x),"y":int(y)})
    out_path = os.path.join(layer_dir, "plot_ops.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "color_name": color_name,
            "color_idx": int(color_idx),
            "ops": packed_ops
        }, f)
    return out_path

# ---------------- geometry ----------------

def _d2(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    dx = float(a[0]-b[0]); dy = float(a[1]-b[1])
    return dx*dx + dy*dy

def _next_choice(cur: Tuple[int,int], op: Op) -> Tuple[float, bool, Tuple[int,int]]:
    if op.kind == "tap":
        d = _d2(cur, op.xy)
        return d, False, op.xy
    s, e = op.start(), op.end()
    d_keep = _d2(cur, s)
    d_flip = _d2(cur, e)
    if d_flip < d_keep:
        return d_flip, True, s  # if we flip, new end will be original start
    else:
        return d_keep, False, e

def _route_cost(ops: List[Op]) -> float:
    if len(ops) <= 1: return 0.0
    cost = 0.0
    cur_end = ops[0].end()
    for i in range(1, len(ops)):
        nxt_start = ops[i].start()
        cost += math.hypot(cur_end[0]-nxt_start[0], cur_end[1]-nxt_start[1])
        cur_end = ops[i].end()
    return cost

# ---------------- optimization ----------------

def _build_ops(lines: List[np.ndarray], taps: List[Tuple[int,int]]) -> List[Op]:
    ops: List[Op] = [Op("line", pts=l) for l in lines] + [Op("tap", xy=t) for t in taps]
    # stable order: keep relative cross-layer order as a weak prior (lines then taps),
    # but greedy below will reorder anyway.
    return ops

def _greedy_interleaved(ops: List[Op]) -> List[Op]:
    if not ops: return []
    used = np.zeros(len(ops), dtype=bool)
    # start from the longest line if exists, else from the first tap
    start_idx = int(np.argmax([op.length() for op in ops]))
    if ops[start_idx].kind == "tap":
        # find any line; else keep tap
        idx_line = next((i for i,op in enumerate(ops) if op.kind=="line"), start_idx)
        start_idx = idx_line
    used[start_idx] = True
    seq: List[Op] = []
    # choose best orientation for the start line (flip if end farther from overall centroid)
    op0 = ops[start_idx]
    if op0.kind == "line":
        # heuristic: make the longer "tail" go forward by pointing to the farther endpoint from centroid
        pts = op0.pts
        cx, cy = float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))
        d0 = _d2((int(pts[0,0]),int(pts[0,1])), (int(cx),int(cy)))
        d1 = _d2((int(pts[-1,0]),int(pts[-1,1])), (int(cx),int(cy)))
        op0.flip = (d1 < d0)
    seq.append(op0)
    cur = op0.end()

    while not np.all(used):
        best_i = -1; best_flip = False; best_d = 1e30; best_exit = cur
        for i, op in enumerate(ops):
            if used[i]: continue
            d, flip, nxt_end = _next_choice(cur, op)
            if d < best_d:
                best_d = d; best_i = i; best_flip = flip; best_exit = nxt_end
        ops[best_i].flip = (ops[best_i].flip ^ best_flip) if ops[best_i].kind=="line" else False
        used[best_i] = True
        seq.append(ops[best_i])
        cur = best_exit
    return seq

def _two_opt_limited(seq: List[Op], max_pairs: int, max_passes: int, tol: float=1e-6) -> List[Op]:
    n = len(seq)
    if n < 4 or max_pairs <= 0 or max_passes <= 0:
        return seq
    total = _route_cost(seq)
    pairs_tried = 0
    improved_any = True
    passes = 0
    while improved_any and passes < max_passes and pairs_tried < max_pairs:
        improved_any = False
        passes += 1
        for i in range(0, n-3):
            for j in range(i+2, n-1):
                if pairs_tried >= max_pairs:
                    break
                pairs_tried += 1
                # try reversing [i+1 : j]
                # save state
                old_block = seq[i+1:j+1]
                # flip orientations inside the block because execution order flips
                for op in old_block:
                    if op.kind == "line":
                        op.flip = not op.flip
                seq[i+1:j+1] = old_block[::-1]
                new_total = _route_cost(seq)
                if new_total + tol < total:
                    total = new_total
                    improved_any = True
                else:
                    # revert
                    seq[i+1:j+1] = old_block[::-1]  # reverse back
                    for op in old_block:
                        if op.kind == "line":
                            op.flip = not op.flip
        # loop again if improved
    return seq

# ---------------- manifest ----------------

def _save_manifest(outdir: str, W: int, H: int, layers: List[Dict[str,Any]]) -> str:
    man = {"image_size":[int(W),int(H)], "layers": layers}
    dst = os.path.join(outdir, "vector_manifest.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    return dst

# ---------------- main ----------------

def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    W, H = _target_size_px(cfg)

    # tunables (safe defaults)
    two_opt_pairs   = int(getattr(cfg, "plotopt_two_opt_pairs", 2000))
    two_opt_passes  = int(getattr(cfg, "plotopt_two_opt_passes", 2))

    layers_meta: List[Dict[str,Any]] = []

    for name in cfg.color_names:
        layer_dir = os.path.join(outdir, name)
        if not os.path.isdir(layer_dir):
            raise RuntimeError(f"Missing layer dir: {layer_dir}")

        lines, taps = _load_cross(layer_dir)
        ops0 = _build_ops(lines, taps)
        seq  = _greedy_interleaved(ops0)
        seq  = _two_opt_limited(seq, max_pairs=two_opt_pairs, max_passes=two_opt_passes)

        out_pkl = _save_ops(layer_dir, seq, name, _color_index(cfg, name))
        layers_meta.append({
            "color_name": name,
            "color_index": _color_index(cfg, name),
            "file": os.path.relpath(out_pkl, outdir).replace("\\","/"),
            "stats": {
                "ops": len(seq),
                "lines": sum(1 for op in seq if op.kind=="line"),
                "taps":  sum(1 for op in seq if op.kind=="tap"),
            }
        })
        print(f"[plot-opt] {name}: ops={len(seq)} (lines={layers_meta[-1]['stats']['lines']}, taps={layers_meta[-1]['stats']['taps']})")

    man_path = _save_manifest(outdir, W, H, layers_meta)
    print(f"[plot-opt] manifest saved: {man_path}")

if __name__ == "__main__":
    main()
