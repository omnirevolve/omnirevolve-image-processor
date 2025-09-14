# 08_dedup_cross_basic.py
# Cross-layer pass (dark → light):
#  - split each lighter layer against cumulative hash of darker layers' lines
#  - drop tiny & convert taps, then remove taps near darker layers (lines/taps)
#  - update cumulative hash, save contours & taps

import os, pickle, math
from typing import List, Tuple, Dict
import numpy as np
import cv2
from config import load_config, Config

# ---- local helpers (no imports from "07_*.py") ----
def min_enclosing_diameter(pts: np.ndarray) -> float:
    (x, y), r = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2).astype(np.float32))
    return float(2.0 * r)

class CumHash:
    def __init__(self, radius: float):
        self.r = float(radius)
        self.cell = max(4.0, float(radius))
        self.inv = 1.0 / self.cell
        self.grid: Dict[Tuple[int,int], List[Tuple[float,float]]] = {}
    def _key(self, x: float, y: float): return (int(math.floor(x*self.inv)), int(math.floor(y*self.inv)))
    def _neighbors(self, x: float, y: float):
        cx, cy = self._key(x,y)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                yield (cx+dx, cy+dy)
    def near(self, x: float, y: float) -> bool:
        R2 = self.r*self.r
        for k in self._neighbors(x,y):
            arr = self.grid.get(k)
            if not arr: continue
            for (px,py) in arr:
                dx=px-x; dy=py-y
                if dx*dx+dy*dy<=R2: return True
        return False
    def add(self, x: float, y: float):
        k = self._key(x,y); a=self.grid.get(k)
        if a is None: self.grid[k]=[(x,y)]
        else: a.append((x,y))
    def add_poly(self, poly: np.ndarray):
        for (x,y) in poly.reshape(-1,2):
            self.add(float(x), float(y))

def taps_from_lines(lines: List[np.ndarray], tap_diam: float):
    taps = []; keep = []
    for c in lines:
        if min_enclosing_diameter(c) <= tap_diam:
            (x,y),_ = cv2.minEnclosingCircle(c.reshape(-1,1,2).astype(np.float32))
            taps.append((float(x), float(y)))
        else:
            keep.append(c)
    return keep, (np.array(taps, np.float32) if len(taps) else np.zeros((0,2), np.float32))

def filter_taps_against_hash(taps: np.ndarray, H: CumHash) -> np.ndarray:
    if taps is None or len(taps) == 0: return taps
    keep = []
    for (x,y) in taps:
        if not H.near(float(x), float(y)):
            keep.append((x,y))
    return np.array(keep, np.float32)

def darkness_rank(name: str) -> int:
    n = name.lower()
    if "dark" in n or "black" in n: return 0
    if "mid"  in n or "blue"  in n: return 1
    if "skin" in n or "green" in n: return 2
    if "light"in n or "red"   in n: return 3
    return 10

# ---- main ----
def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    pen_diam = float(getattr(cfg, "pen_width_px", 60))
    pen_radius = float(getattr(cfg, "pen_radius_px", pen_diam/2.0))
    tap_diam   = float(getattr(cfg, "tap_diameter_px", pen_diam))
    min_keep   = float(getattr(cfg, "min_keep_diameter_px", pen_radius))

    # stable sort: by darkness rank, then by original order
    names = list(cfg.color_names)
    orig_order = {n: i for i, n in enumerate(names)}
    names.sort(key=lambda n: (darkness_rank(n), orig_order.get(n, 1_000_000)))

    H_all = CumHash(pen_radius)  # darker geometry (lines)
    H_tap = CumHash(pen_radius)  # darker taps

    for name in names:
        cdir = os.path.join(outdir, name)
        os.makedirs(cdir, exist_ok=True)

        # prefer results of 07 → else sorted/scaled/raw
        src_lines = os.path.join(cdir, "contours_dedup_layer.pkl")
        if not os.path.exists(src_lines):
            for alt in ("contours_sorted.pkl", "contours_scaled.pkl", "contours.pkl"):
                p = os.path.join(cdir, alt)
                if os.path.exists(p):
                    src_lines = p; break

        with open(src_lines, "rb") as f:
            lines: List[np.ndarray] = pickle.load(f)

        taps = np.zeros((0,2), np.float32)
        p_taps = os.path.join(cdir, "taps.pkl")
        if os.path.exists(p_taps):
            with open(p_taps, "rb") as f:
                t = pickle.load(f)
                if isinstance(t, np.ndarray): taps = t.astype(np.float32)
                elif isinstance(t, list) and len(t): taps = np.array(t, np.float32)

        # split current lines against darker geometry
        split_lines: List[np.ndarray] = []
        for poly in lines:
            pts = poly.reshape(-1,2).astype(np.float32)
            cur=[]
            for (x,y) in pts:
                if H_all.near(float(x), float(y)):
                    if len(cur)>=2:
                        split_lines.append(np.array(cur,np.float32).reshape(-1,1,2).astype(np.int32))
                    cur=[]
                    continue
                cur.append((float(x), float(y)))
            if len(cur)>=2:
                split_lines.append(np.array(cur,np.float32).reshape(-1,1,2).astype(np.int32))

        # prune tiny + taps
        keep, extra_taps = taps_from_lines(split_lines, tap_diam)
        keep2=[]
        for c in keep:
            if min_enclosing_diameter(c) < min_keep:
                continue
            keep2.append(c)

        # taps: merge lists, then drop near darker (lines & taps)
        taps = extra_taps if len(taps)==0 else (np.vstack([taps, extra_taps]) if len(extra_taps) else taps)
        taps = filter_taps_against_hash(taps, H_all)
        taps = filter_taps_against_hash(taps, H_tap)

        # update cumulative hash for next (lighter) layers
        for poly in keep2: H_all.add_poly(poly)
        for (x,y) in taps: H_tap.add(float(x), float(y))

        with open(os.path.join(cdir, "contours_dedup_cross.pkl"), "wb") as f:
            pickle.dump(keep2, f)
        with open(os.path.join(cdir, "taps_final.pkl"), "wb") as f:
            pickle.dump(taps, f)

        print(f"[cross] {name}: lines={len(keep2)}, taps={len(taps)}")

if __name__ == "__main__":
    main()
