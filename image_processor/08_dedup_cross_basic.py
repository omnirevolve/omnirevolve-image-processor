# 08_dedup_cross_basic.py
# Cross-layer dedup with 1-bit forbidden raster.
# Priority: darker → lighter. Inside a layer: LINES first, then sequential TAPS.
# After cutting lines against the forbidden mask:
#   • accepted lines are stamped into the mask with a THICK brush (D_lines, default 120px if pen=60px)
#   • taps are iterated SEQUENTIALLY; if a tap center hits a free pixel -> accept AND IMMEDIATELY stamp
#     the tap into the mask with a THICK brush (D_taps, default 120px if pen=60px)
#
# Strict I/O per layer dir:
#   INPUT : lines_intra.pkl , taps_intra.pkl           (from 07_dedup_layer_basic.py)
#   OUTPUT: lines_cross.pkl , taps_cross.pkl           (for previews / next steps)
# Debug:
#   output_dir/forbidden_after_XX_<layer>.png          (8-bit preview of mask after each layer)

from __future__ import annotations
import os, math, pickle
from typing import List, Tuple

import numpy as np
import cv2
from config import load_config, Config

# ───────────────── helpers ─────────────────

def _target_size_px(cfg: Config) -> tuple[int, int]:
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 and th > 0:
        return tw, th
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    img = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if img is None:
        raise RuntimeError("Cannot infer target size; set target_* in config.")
    h, w = img.shape[:2]
    return w, h

def _poly_len(pts: np.ndarray) -> float:
    p = pts.reshape(-1,1,2).astype(np.float32)
    return float(cv2.arcLength(p, False)) if len(p) >= 2 else 0.0

def _min_enclosing_diameter(pts: np.ndarray) -> float:
    (c, r) = cv2.minEnclosingCircle(pts.reshape(-1,1,2).astype(np.float32))
    return float(2.0 * r)

def _split_on_long_jumps(poly: np.ndarray, max_jump: float) -> List[np.ndarray]:
    pts = poly.reshape(-1,2).astype(np.float32)
    if len(pts) < 2: return []
    out: List[np.ndarray] = []
    cur = [tuple(pts[0])]
    for i in range(1, len(pts)):
        dx = float(pts[i,0]-pts[i-1,0]); dy = float(pts[i,1]-pts[i-1,1])
        if math.hypot(dx,dy) > max_jump:
            if len(cur) >= 2:
                out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
            cur = []
        cur.append((float(pts[i,0]), float(pts[i,1])))
    if len(cur) >= 2:
        out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
    return out

def _ends(poly: np.ndarray):
    p = poly.reshape(-1,2)
    return p[0], p[-1]

def _reorder_for_travel(contours: List[np.ndarray]) -> List[np.ndarray]:
    if not contours: return []
    used = np.zeros(len(contours), bool)
    starts = np.array([_ends(c)[0] for c in contours])
    ends   = np.array([_ends(c)[1] for c in contours])
    lengths = [_poly_len(c) for c in contours]

    cur = int(np.argmax(lengths))
    order=[cur]; flips=[False]; used[cur]=True
    cur_end = ends[cur]
    while not np.all(used):
        idxs = np.flatnonzero(~used)
        d2s = np.sum((starts[idxs].astype(np.float32)-cur_end.astype(np.float32))**2, axis=1)
        d2e = np.sum((ends[idxs].astype(np.float32)  -cur_end.astype(np.float32))**2, axis=1)
        best=-1; bf=False; bd=1e20
        for k,i in enumerate(idxs):
            if d2s[k]<=d2e[k]:
                if d2s[k]<bd: best=i; bf=False; bd=d2s[k]
            else:
                if d2e[k]<bd: best=i; bf=True;  bd=d2e[k]
        used[best]=True; order.append(best); flips.append(bf)
        cur_end = starts[best] if bf else ends[best]

    out=[]
    for i,f in zip(order,flips):
        pts = contours[i].reshape(-1,2)
        if f: pts = pts[::-1].copy()
        out.append(pts.reshape(-1,1,2).astype(np.int32))
    return out

def _tiny_and_taps(contours: List[np.ndarray],
                   tap_d: float,
                   min_keep_d: float,
                   tap_max_perim: float,
                   tap_max_vertices: int) -> tuple[List[np.ndarray], list[tuple[int,int]]]:
    kept: List[np.ndarray] = []
    taps_xy: list[tuple[int,int]] = []
    for c in contours:
        if c is None: continue
        d = _min_enclosing_diameter(c)
        if d <= tap_d:
            per  = _poly_len(c)
            verts= int(c.reshape(-1,2).shape[0])
            if per <= tap_max_perim and verts <= tap_max_vertices:
                (x,y),_ = cv2.minEnclosingCircle(c.reshape(-1,1,2).astype(np.float32))
                taps_xy.append((int(round(x)), int(round(y))))
                continue
        if d >= min_keep_d:
            kept.append(c)
    return kept, taps_xy

# ───────────── forbidden raster ─────────────

def _paint_thick(mask: np.ndarray,
                 lines: List[np.ndarray],
                 taps: List[tuple[int,int]],
                 brush_diam_px: float):
    """Paint into mask with given diameter. Lines as polylines, taps as filled discs."""
    thickness = int(max(1, round(brush_diam_px)))
    radius    = max(1, int(round(brush_diam_px/2.0)))

    if lines:
        arrs = []
        for p in lines:
            a = np.asarray(p).reshape(-1,1,2).astype(np.int32)
            if len(a) >= 2: arrs.append(a)
        if arrs:
            cv2.polylines(mask, arrs, isClosed=False, color=255,
                          thickness=thickness, lineType=cv2.LINE_8)

    for (x,y) in taps:
        cv2.circle(mask, (int(x),int(y)), radius, 255, thickness=-1, lineType=cv2.LINE_8)

def _cut_poly_against_mask(poly: np.ndarray,
                           forb: np.ndarray,
                           step_px: float) -> List[np.ndarray]:
    """Thin (≈1px) drawing of a polyline; split where mask is set."""
    pts = np.asarray(poly).reshape(-1,2).astype(np.float32)
    if len(pts) < 2: return []
    out: List[np.ndarray] = []
    cur: List[tuple[float,float]] = []

    def blocked(x: float, y: float) -> bool:
        xi = int(round(x)); yi = int(round(y))
        return (0 <= yi < forb.shape[0]) and (0 <= xi < forb.shape[1]) and (forb[yi, xi] != 0)

    if not blocked(float(pts[0,0]), float(pts[0,1])):
        cur.append((float(pts[0,0]), float(pts[0,1])))

    for i in range(1, len(pts)):
        p0 = pts[i-1]; p1 = pts[i]
        v  = p1 - p0
        L  = float(np.hypot(v[0], v[1]))
        if L <= 1e-6:
            continue
        n = max(1, int(math.ceil(L / max(1.0, step_px))))
        for k in range(1, n+1):
            t = k / n
            q = p0 + v * t
            if blocked(float(q[0]), float(q[1])):
                if len(cur) >= 2:
                    out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
                cur = []
            else:
                cur.append((float(q[0]), float(q[1])))

    if len(cur) >= 2:
        out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
    return out

# ───────────── I/O ─────────────

def _load_intra(layer_dir: str):
    pL = os.path.join(layer_dir, "lines_intra.pkl")
    pT = os.path.join(layer_dir, "taps_intra.pkl")
    lines: List[np.ndarray] = []
    taps:  List[tuple[int,int]] = []
    if os.path.exists(pL):
        with open(pL, "rb") as f: lines = pickle.load(f)
    else:
        print(f"[cross] WARNING: missing {pL}")
    if os.path.exists(pT):
        with open(pT, "rb") as f:
            obj = pickle.load(f)
            for it in obj:
                a = np.asarray(it).reshape(-1)
                if a.size >= 2: taps.append((int(a[0]), int(a[1])))
    else:
        print(f"[cross] WARNING: missing {pT}")
    return lines, taps

def _save_cross(layer_dir: str, lines: List[np.ndarray], taps: List[tuple[int,int]]):
    with open(os.path.join(layer_dir, "lines_cross.pkl"), "wb") as f:
        pickle.dump(lines, f)
    with open(os.path.join(layer_dir, "taps_cross.pkl"), "wb") as f:
        pickle.dump(taps, f)

def _darkness_rank(name: str) -> int:
    order = ["layer_dark", "layer_mid", "layer_skin", "layer_light"]
    return order.index(name) if name in order else 999

# ───────────── main ─────────────

def main():
    cfg: Config = load_config()
    W, H = _target_size_px(cfg)
    forbidden = np.zeros((H, W), np.uint8)

    pen_diam   = float(getattr(cfg, "pen_width_px", 60.0))
    tap_diam   = float(getattr(cfg, "tap_diameter_px", pen_diam))
    min_keep   = float(getattr(cfg, "min_keep_diameter_px", max(10.0, (pen_diam/2.0)*0.4)))
    tap_max_per= float(getattr(cfg, "tap_max_perimeter_px", 2.5*tap_diam))
    tap_max_v  = int(getattr(cfg, "tap_max_vertices", 50))
    max_jump   = float(getattr(cfg, "max_join_jump_px", 80.0))

    # **IMPORTANT**: thick brush for both lines and taps (D=120px if pen=60px)
    D_lines = float(getattr(cfg, "cross_lines_brush_diam_px", pen_diam * 2.0))
    D_taps  = float(getattr(cfg, "cross_taps_brush_diam_px", pen_diam * 2.0))

    step_px = float(getattr(cfg, "cross_cut_step_px", 1.0))
    dbg     = bool(getattr(cfg, "cross_debug_masks", False))

    names = list(cfg.color_names)
    names.sort(key=_darkness_rank)  # dark -> light

    print(f"[cross] forbidden: {W}x{H} | D_lines={D_lines:.1f}px | D_taps={D_taps:.1f}px | step={step_px:.1f}px")

    for idx, name in enumerate(names, 1):
        layer_dir = os.path.join(cfg.output_dir, name)
        os.makedirs(layer_dir, exist_ok=True)

        lines_in, taps_in = _load_intra(layer_dir)

        # 1) cut lines (thin), split jumps, small->taps, reorder
        cut: List[np.ndarray] = []
        for poly in lines_in:
            cut.extend(_cut_poly_against_mask(poly, forbidden, step_px))

        cut2: List[np.ndarray] = []
        for seg in cut:
            parts = _split_on_long_jumps(seg, max_jump)
            cut2.extend(parts if parts else [seg])

        lines_keep, taps_from_lines = _tiny_and_taps(cut2, tap_diam, min_keep, tap_max_per, tap_max_v)
        lines_out = _reorder_for_travel(lines_keep)

        # 2) LINES have priority → stamp thick
        _paint_thick(forbidden, lines_out, [], D_lines)

        # 3) TAPS: iterate SEQUENTIALLY, accept if center free, then IMMEDIATELY stamp thick
        def center_blocked(x: int, y: int) -> bool:
            return 0 <= y < forbidden.shape[0] and 0 <= x < forbidden.shape[1] and forbidden[y, x] != 0

        taps_seq = list(taps_in) + list(taps_from_lines)
        taps_out: List[tuple[int,int]] = []
        for (x, y) in taps_seq:
            if not center_blocked(x, y):
                taps_out.append((x, y))
                _paint_thick(forbidden, [], [(x, y)], D_taps)  # ← делает накладывание невозможным

        # 4) save results for this layer
        _save_cross(layer_dir, lines_out, taps_out)

        if dbg:
            cv2.imwrite(os.path.join(cfg.output_dir, f"forbidden_after_{idx:02d}_{name}.png"), forbidden)

        vin = sum(int(p.reshape(-1,2).shape[0]) for p in lines_in)
        vout= sum(int(p.reshape(-1,2).shape[0]) for p in lines_out)
        print(f"[cross] {name}: lines {len(lines_in)}→{len(lines_out)} (v {vin}→{vout}), "
              f"taps {len(taps_in)}+{len(taps_from_lines)}→{len(taps_out)}")

if __name__ == "__main__":
    main()
