# 08_dedup_cross_basic.py
# Cross-layer deduplication using a 1-bit forbidden raster and a thick brush.
# Order: dark -> light. Each accepted layer updates the forbidden mask
# with "paint" of pen diameter; lighter layers are cut against it.
#
# Inputs per layer (STRICT):
#   lines_intra.pkl  - polylines after intra-dedup (list[np.ndarray Nx1x2 int32])
#   taps_intra.pkl   - taps after intra-dedup (list[(x,y)] or array Nx2)
#
# Outputs per layer (STRICT):
#   lines_cross.pkl  - polylines after cross-dedup
#   taps_cross.pkl   - taps after cross-dedup

from __future__ import annotations
import os, math, pickle
from typing import List, Tuple

import numpy as np
import cv2
from config import load_config, Config

# ─────────────────────────── utils / geometry ───────────────────────────

def _target_size_px(cfg: Config) -> Tuple[int,int]:
    tw_px = int(getattr(cfg,"target_width_px",0) or 0)
    th_px = int(getattr(cfg,"target_height_px",0) or 0)
    if tw_px>0 and th_px>0:
        return tw_px, th_px
    tw_mm = float(getattr(cfg,"target_width_mm",0) or 0)
    th_mm = float(getattr(cfg,"target_height_mm",0) or 0)
    ppm   = int(getattr(cfg,"pixels_per_mm",0) or 0)
    if tw_mm>0 and th_mm>0 and ppm>0:
        return int(round(tw_mm*ppm)), int(round(th_mm*ppm))
    # fallback: size of resized.png
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size; set target_* in config.")
    h, w = base.shape[:2]
    return w, h

def _poly_len(pts: np.ndarray) -> float:
    p = pts.reshape(-1,1,2).astype(np.float32)
    if len(p) < 2: return 0.0
    return float(cv2.arcLength(p, False))

def _min_enclosing_diameter(pts: np.ndarray) -> float:
    (c, r) = cv2.minEnclosingCircle(pts.reshape(-1,1,2).astype(np.float32))
    return float(2.0*r)

def _tiny_and_taps(
    contours: List[np.ndarray],
    tap_d: float,
    min_keep_d: float,
    tap_max_perim: float,
    tap_max_vertices: int,
) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
    kept: List[np.ndarray] = []
    taps_xy: List[Tuple[int,int]] = []
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

# ─────────────────────────── forbidden raster ───────────────────────────

def _rasterize_thick(mask: np.ndarray,
                     lines: List[np.ndarray],
                     taps: List[Tuple[int,int]],
                     radius_px: float):
    # Brush thickness ~= diameter. +2 чтобы закрывать дырки.
    thick = int(max(1, math.ceil(2.0*radius_px + 2.0)))
    rr    = int(max(1, round(radius_px)))

    if lines:
        arrs = []
        for p in lines:
            a = np.asarray(p).reshape(-1,1,2).astype(np.int32)
            if len(a) >= 2: arrs.append(a)
        if arrs:
            cv2.polylines(mask, arrs, isClosed=False, color=255, thickness=thick, lineType=cv2.LINE_8)

    for (x,y) in taps:
        cv2.circle(mask, (int(x),int(y)), rr, 255, thickness=-1, lineType=cv2.LINE_8)

def _cut_poly_against_mask(poly: np.ndarray,
                           forb: np.ndarray,
                           step_px: float) -> List[np.ndarray]:
    """
    Thin "drawing": iterate ~1px along the polyline and split on hits.
    We DO NOT modify the mask here.
    """
    pts = np.asarray(poly).reshape(-1,2).astype(np.float32)
    if len(pts) < 2: return []
    out: List[np.ndarray] = []

    cur: List[Tuple[float,float]] = []
    prev = pts[0].copy()

    def _blocked(qx: float, qy: float) -> bool:
        xi = int(round(qx)); yi = int(round(qy))
        if 0 <= yi < forb.shape[0] and 0 <= xi < forb.shape[1]:
            return forb[yi, xi] != 0
        return False

    # start point
    if not _blocked(float(prev[0]), float(prev[1])):
        cur.append((float(prev[0]), float(prev[1])))

    for i in range(1, len(pts)):
        p1    = pts[i]
        vec   = p1 - prev
        segln = float(np.hypot(vec[0], vec[1]))
        if segln <= 1e-6:
            prev = p1
            continue

        n_steps = max(1, int(math.ceil(segln / max(1.0, step_px))))
        for k in range(1, n_steps+1):
            t = k / n_steps
            q = prev + vec * t
            if _blocked(float(q[0]), float(q[1])):
                if len(cur) >= 2:
                    out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
                cur = []
            else:
                cur.append((float(q[0]), float(q[1])))

        prev = p1

    if len(cur) >= 2:
        out.append(np.array(cur, np.float32).reshape(-1,1,2).astype(np.int32))
    return out

# ─────────────────────────── I/O helpers ────────────────────────────────

def _load_intra(layer_dir: str):
    pL = os.path.join(layer_dir, "lines_intra.pkl")
    pT = os.path.join(layer_dir, "taps_intra.pkl")
    lines: List[np.ndarray] = []
    taps:  List[Tuple[int,int]] = []
    if os.path.exists(pL):
        with open(pL,"rb") as f: lines = pickle.load(f)
    else:
        print(f"[cross] WARNING: missing lines_intra.pkl in {os.path.basename(layer_dir)}")
    if os.path.exists(pT):
        with open(pT,"rb") as f:
            obj = pickle.load(f)
            for it in obj:
                a = np.asarray(it).reshape(-1)
                if a.size>=2: taps.append((int(a[0]), int(a[1])))
    else:
        print(f"[cross] WARNING: missing taps_intra.pkl in {os.path.basename(layer_dir)}")
    return lines, taps

def _save_cross(layer_dir: str, lines: List[np.ndarray], taps: List[Tuple[int,int]]):
    with open(os.path.join(layer_dir, "lines_cross.pkl"), "wb") as f:
        pickle.dump(lines, f)
    with open(os.path.join(layer_dir, "taps_cross.pkl"), "wb") as f:
        pickle.dump(taps, f)

# ─────────────────────────── order by darkness ──────────────────────────

def _darkness_rank(name: str) -> int:
    order = ["layer_dark", "layer_mid", "layer_skin", "layer_light"]
    return order.index(name) if name in order else 999

# ─────────────────────────── main ───────────────────────────────────────

def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir

    W, H = _target_size_px(cfg)
    forbidden = np.zeros((H, W), np.uint8)

    # params
    pen_diam   = float(getattr(cfg, "pen_width_px", 60.0))
    pen_radius = float(getattr(cfg, "pen_radius_px", pen_diam/2.0))
    tap_diam   = float(getattr(cfg, "tap_diameter_px", pen_diam))
    min_keep   = float(getattr(cfg, "min_keep_diameter_px", max(10.0, pen_radius*0.4)))
    tap_max_per= float(getattr(cfg, "tap_max_perimeter_px", 2.5*tap_diam))
    tap_max_v  = int(getattr(cfg, "tap_max_vertices", 50))
    max_jump   = float(getattr(cfg, "max_join_jump_px", 80.0))

    step_px    = float(getattr(cfg, "cross_cut_step_px", 1.0))  # ~1px thin drawing
    dbg_masks  = bool(getattr(cfg, "cross_debug_masks", False))

    names = list(cfg.color_names)
    names.sort(key=_darkness_rank)  # dark → light

    print(f"[cross] forbidden mask {W}x{H}, pen_d={pen_diam:.1f}px, step={step_px:.1f}px")

    # dark → light
    for idx, name in enumerate(names, 1):
        layer_dir = os.path.join(outdir, name)
        os.makedirs(layer_dir, exist_ok=True)

        lines_intra, taps_intra = _load_intra(layer_dir)

        # 1) cut lines against current forbidden mask (thin drawing)
        cut: List[np.ndarray] = []
        for poly in lines_intra:
            parts = _cut_poly_against_mask(poly, forbidden, step_px)
            cut.extend(parts)

        # 2) split on long jumps (safety for artifacts)
        cut2: List[np.ndarray] = []
        for seg in cut:
            parts = _split_on_long_jumps(seg, max_jump)
            cut2.extend(parts if parts else [seg])

        # 3) small → taps, keep sizable
        lines_keep, taps_new = _tiny_and_taps(cut2, tap_diam, min_keep, tap_max_per, tap_max_v)

        # 4) remove taps that collide with forbidden (via dilation)
        rr = int(max(1, round(pen_radius)))
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rr+1, 2*rr+1))
        forb_dil = cv2.dilate(forbidden, k, iterations=1)
        taps_out: List[Tuple[int,int]] = []
        for (x,y) in taps_intra + taps_new:
            if 0 <= y < forb_dil.shape[0] and 0 <= x < forb_dil.shape[1] and forb_dil[y, x]:
                continue
            taps_out.append((x,y))

        # 5) reorder for shorter travel
        lines_out = _reorder_for_travel(lines_keep)

        # 6) save STRICT outputs
        _save_cross(layer_dir, lines_out, taps_out)

        # 7) update forbidden: thick paint with pen diameter
        _rasterize_thick(forbidden, lines_out, taps_out, pen_radius)

        if dbg_masks:
            cv2.imwrite(os.path.join(outdir, f"forbidden_after_{idx:02d}_{name}.png"), forbidden)

        vin = sum(int(p.reshape(-1,2).shape[0]) for p in lines_intra)
        vout= sum(int(p.reshape(-1,2).shape[0]) for p in lines_out)
        print(f"[cross] {name}: lines {len(lines_intra)}→{len(lines_out)}, "
              f"taps {len(taps_intra)}→{len(taps_out)}, vertices {vin}→{vout}")

if __name__ == "__main__":
    main()
