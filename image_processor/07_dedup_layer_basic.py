# 07_dedup_layer_basic.py
# Intra-layer cleanup:
#  - detect taps (≤ tap_diameter_px, with guards)
#  - per-contour virtual draw: drop points that collide with "old" path (ignores recent tail)
#  - split on long jumps, drop tiny → taps, reorder for travel
# Outputs (per layer):
#   lines_intra.pkl  — polylines after intra pass (open)
#   taps_intra.pkl   — taps as [(x,y), ...]

from __future__ import annotations
import os, math, pickle
from collections import deque
from typing import List, Tuple, Dict

import numpy as np
import cv2
from config import load_config, Config


# ---------- geometry ----------
def _min_enclosing_diameter(pts: np.ndarray) -> float:
    (c, r) = cv2.minEnclosingCircle(pts.reshape(-1,1,2).astype(np.float32))
    return float(2.0 * r)

def _poly_perimeter(pts: np.ndarray) -> float:
    p = pts.reshape(-1,2).astype(np.float32)
    if len(p) < 2: return 0.0
    return float(cv2.arcLength(p.reshape(-1,1,2), False))

def _split_small_and_taps(
    contours: List[np.ndarray],
    tap_diam: float,
    min_keep_diam: float,
    tap_max_perimeter: float,
    tap_max_vertices: int,
) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
    kept: List[np.ndarray] = []
    taps_xy: List[Tuple[int,int]] = []
    for c in contours:
        if c is None: continue
        d = _min_enclosing_diameter(c)
        if d <= tap_diam:
            per = _poly_perimeter(c)
            verts = int(c.reshape(-1,2).shape[0])
            if per <= tap_max_perimeter and verts <= tap_max_vertices:
                (x,y), _ = cv2.minEnclosingCircle(c.reshape(-1,1,2).astype(np.float32))
                taps_xy.append((int(round(x)), int(round(y))))
                continue
        if d < min_keep_diam:
            # too small to keep as line (but didn't pass tap guards) → drop
            continue
        kept.append(c)
    return kept, taps_xy

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

def _reorder_only(contours: List[np.ndarray]) -> List[np.ndarray]:
    if not contours: return []
    used = np.zeros(len(contours), bool)
    starts = np.array([_ends(c)[0] for c in contours])
    ends   = np.array([_ends(c)[1] for c in contours])
    lengths = [float(cv2.arcLength(c.reshape(-1,1,2), False)) for c in contours]

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


# ---------- spatial hash over “old” points only ----------
class _PointHash:
    def __init__(self, radius: float):
        self.r = float(radius)
        self.cell = max(4.0, float(radius))
        self.inv  = 1.0/self.cell
        self.g: Dict[Tuple[int,int], List[Tuple[float,float]]] = {}
    def _key(self, x: float, y: float):
        return (int(math.floor(x*self.inv)), int(math.floor(y*self.inv)))
    def _nbrs(self, x: float, y: float):
        cx,cy = self._key(x,y)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                yield (cx+dx, cy+dy)
    def near(self, x: float, y: float) -> bool:
        R2 = self.r*self.r
        for k in self._nbrs(x,y):
            arr = self.g.get(k)
            if not arr: continue
            for (px,py) in arr:
                dx=px-x; dy=py-y
                if dx*dx+dy*dy <= R2: return True
        return False
    def add(self, x: float, y: float):
        k = self._key(x,y)
        a = self.g.get(k)
        if a is None: self.g[k]=[(x,y)]
        else: a.append((x,y))


def _virtual_draw_split_ignore_tail(
    pts: np.ndarray,
    pen_radius: float,
    recent_path_len_px: float
) -> List[np.ndarray]:
    """
    Walk points; drop a point if it is within pen_radius of *old* path.
    Points from the most recent arc-length window (recent_path_len_px) are NOT hashed,
    so the line can grow without self-cancelling. Split at drops.
    """
    H = _PointHash(pen_radius)
    out: List[np.ndarray] = []
    cur_xy: List[Tuple[float,float]] = []

    # distance-based 'recent' queue
    recent_q: deque[Tuple[float,float]] = deque()
    recent_s: deque[float] = deque()
    s = 0.0
    prev = None

    for p in pts.reshape(-1,2).astype(np.float32):
        x,y = float(p[0]), float(p[1])
        if prev is not None:
            dx=x-prev[0]; dy=y-prev[1]
            s += float(math.hypot(dx,dy))

        # expire recent into hash
        while recent_s and (s - recent_s[0]) > recent_path_len_px:
            px,py = recent_q.popleft()
            recent_s.popleft()
            H.add(px,py)

        # collide only with OLD points (hash)
        if H.near(x,y):
            if len(cur_xy)>=2:
                out.append(np.array(cur_xy, np.float32))
            cur_xy=[]; recent_q.clear(); recent_s.clear(); prev=None
            continue

        # accept point
        cur_xy.append((x,y))
        recent_q.append((x,y)); recent_s.append(s); prev=(x,y)

    if len(cur_xy)>=2: out.append(np.array(cur_xy, np.float32))
    return [c.reshape(-1,1,2).astype(np.int32) for c in out]


def _load_source(layer_dir: str) -> List[np.ndarray]:
    for fn in ("contours_sorted.pkl", "contours_scaled.pkl", "contours.pkl"):
        p = os.path.join(layer_dir, fn)
        if os.path.exists(p):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, list) else []
    return []


def process_layer(layer_dir: str, cfg: Config):
    # params
    pen_diam   = float(getattr(cfg, "pen_width_px", 60))
    pen_radius = float(getattr(cfg, "pen_radius_px", pen_diam/2.0))
    tap_diam   = float(getattr(cfg, "tap_diameter_px", pen_diam))
    min_keep   = float(getattr(cfg, "min_keep_diameter_px", max(10.0, pen_radius*0.4)))
    tap_max_per= float(getattr(cfg, "tap_max_perimeter_px", 2.5*tap_diam))
    tap_max_v  = int(getattr(cfg, "tap_max_vertices", 50))
    recent_len = float(getattr(cfg, "recent_path_len_px", max(2*pen_radius, pen_diam)))
    max_jump   = float(getattr(cfg, "max_join_jump_px", 80.0))

    polys = _load_source(layer_dir)
    if not polys:
        print(f"[intra] {os.path.basename(layer_dir)}: no input contours, skip.")
        return

    kept, taps = _split_small_and_taps(polys, tap_diam, min_keep, tap_max_per, tap_max_v)

    cleaned: List[np.ndarray] = []
    for c in kept:
        segs = _virtual_draw_split_ignore_tail(c, pen_radius, recent_len)
        if not segs:
            continue
        # extra split on long jumps
        for s in segs:
            parts = _split_on_long_jumps(s, max_jump)
            cleaned.extend(parts if parts else [s])

    # small→taps again
    lines2, taps2 = _split_small_and_taps(cleaned, tap_diam, min_keep, tap_max_per, tap_max_v)
    taps_all = taps2 if len(taps)==0 else (taps + taps2)

    # reorder for travel
    lines2 = _reorder_only(lines2)

    # save STRICT filenames
    out_lines = os.path.join(layer_dir, "lines_intra.pkl")
    out_taps  = os.path.join(layer_dir, "taps_intra.pkl")
    with open(out_lines, "wb") as f: pickle.dump(lines2, f)
    with open(out_taps,  "wb") as f: pickle.dump(taps_all, f)

    vin  = sum(int(p.reshape(-1,2).shape[0]) for p in polys)
    vout = sum(int(p.reshape(-1,2).shape[0]) for p in lines2)
    print(f"[intra] {os.path.basename(layer_dir)}: lines={len(lines2)}, taps={len(taps_all)}, "
          f"vertices_in={vin}, vertices_out={vout}")
    print(f"[intra]   saved → {os.path.relpath(out_lines, layer_dir)}, {os.path.relpath(out_taps, layer_dir)}")


def main():
    cfg: Config = load_config()
    for name in cfg.color_names:
        process_layer(os.path.join(cfg.output_dir, name), cfg)

if __name__ == "__main__":
    main()
