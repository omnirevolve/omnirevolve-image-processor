# 07_dedup_layer_basic.py
# Per-layer clean-up:
#  - detect taps (≤ tap_diameter_px) with extra guards (max perimeter/vertices)
#  - virtual draw per original contour; only “old” points (beyond recent_path_len_px) cause splits
#  - drop tiny shapes (< min_keep_diameter_px) and convert tiny to taps
# Outputs: <color>/contours_dedup_layer.pkl, <color>/taps.pkl

import os, pickle, math
from typing import List, Tuple, Dict
from collections import deque
import numpy as np
import cv2
from config import load_config, Config

# ---------------- helpers ----------------
def min_enclosing_diameter(pts: np.ndarray) -> float:
    (x, y), r = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2).astype(np.float32))
    return float(2.0 * r)

def poly_perimeter(pts: np.ndarray) -> float:
    p = pts.reshape(-1, 2).astype(np.float32)
    if len(p) < 2:
        return 0.0
    return float(cv2.arcLength(p.reshape(-1, 1, 2), False))

def split_small_and_taps(
    contours: List[np.ndarray],
    tap_diam: float,
    min_keep_diam: float,
    tap_max_perimeter: float,
    tap_max_vertices: int,
):
    kept: List[np.ndarray] = []
    taps_xy: List[Tuple[float, float]] = []
    for c in contours:
        d = min_enclosing_diameter(c)
        if d <= tap_diam:
            # guards to avoid turning long jaggy polylines into taps
            per = poly_perimeter(c)
            verts = int(c.reshape(-1, 2).shape[0])
            if per <= tap_max_perimeter and verts <= tap_max_vertices:
                (x, y), _ = cv2.minEnclosingCircle(
                    c.reshape(-1, 1, 2).astype(np.float32)
                )
                taps_xy.append((float(x), float(y)))
                continue
        if d < min_keep_diam:
            continue
        kept.append(c)
    return kept, (
        np.asarray(taps_xy, dtype=np.float32)
        if taps_xy
        else np.zeros((0, 2), np.float32)
    )

def _ends(poly: np.ndarray):
    pts = poly.reshape(-1, 2)
    closed = np.all(pts[0] == pts[-1])
    if closed and len(pts) > 1:
        pts = pts[:-1]
    return pts[0], pts[-1], closed

def reorder_only(contours: List[np.ndarray]) -> List[np.ndarray]:
    if not contours:
        return []
    used = np.zeros(len(contours), dtype=bool)
    starts, ends, closed = [], [], []
    for c in contours:
        s, e, cl = _ends(c)
        starts.append(s)
        ends.append(e)
        closed.append(cl)
    starts = np.array(starts)
    ends = np.array(ends)
    closed = np.array(closed, bool)
    order, flips = [], []
    lengths = [float(cv2.arcLength(c, True)) for c in contours]
    cur = int(np.argmax(lengths))
    order.append(cur)
    flips.append(False)
    used[cur] = True
    cur_end = ends[cur] if not closed[cur] else starts[cur]
    while not np.all(used):
        idxs = np.flatnonzero(~used)
        d2s = np.sum(
            (starts[idxs].astype(np.float32) - cur_end.astype(np.float32)) ** 2, axis=1
        )
        d2e = np.sum(
            (ends[idxs].astype(np.float32) - cur_end.astype(np.float32)) ** 2, axis=1
        )
        best = -1
        bf = False
        bd = 1e20
        for k, i in enumerate(idxs):
            if closed[i]:
                if d2s[k] < bd:
                    best = i
                    bf = False
                    bd = d2s[k]
            else:
                if d2s[k] <= d2e[k]:
                    if d2s[k] < bd:
                        best = i
                        bf = False
                        bd = d2s[k]
                else:
                    if d2e[k] < bd:
                        best = i
                        bf = True
                        bd = d2e[k]
        used[best] = True
        order.append(best)
        flips.append(bf)
        cur_end = starts[best] if (closed[best] or bf) else ends[best]
    out = []
    for i, f in zip(order, flips):
        pts = contours[i].reshape(-1, 2)
        if f:
            pts = pts[::-1].copy()
        if (
            np.all(contours[i].reshape(-1, 2)[0] == contours[i].reshape(-1, 2)[-1])
            and not np.all(pts[0] == pts[-1])
        ):
            pts = np.vstack([pts, pts[0]])
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out

# ---------- spatial hash over “old” points only ----------
class PointHash:
    def __init__(self, radius: float):
        self.r = float(radius)
        self.cell = max(4.0, float(radius))
        self.inv = 1.0 / self.cell
        self.grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def _key(self, x: float, y: float):
        return (int(math.floor(x * self.inv)), int(math.floor(y * self.inv)))

    def _neighbors(self, x: float, y: float):
        cx, cy = self._key(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield (cx + dx, cy + dy)

    def near(self, x: float, y: float) -> bool:
        R2 = self.r * self.r
        for k in self._neighbors(x, y):
            arr = self.grid.get(k)
            if not arr:
                continue
            for (px, py) in arr:
                dx = px - x
                dy = py - y
                if dx * dx + dy * dy <= R2:
                    return True
        return False

    def add(self, x: float, y: float):
        k = self._key(x, y)
        a = self.grid.get(k)
        if a is None:
            self.grid[k] = [(x, y)]
        else:
            a.append((x, y))

def virtual_draw_split(
    pts: np.ndarray, pen_radius: float, recent_path_len_px: float
) -> List[np.ndarray]:
    """
    Walk points; drop a point if it is within pen_radius of *old* path.
    Points from the most recent arc-length window (recent_path_len_px) are NOT hashed,
    so the line can grow normally without self-cancelling.
    Split when a point is dropped.
    """
    H = PointHash(pen_radius)
    out: List[np.ndarray] = []
    cur_xy: List[Tuple[float, float]] = []
    recent_q: deque[Tuple[float, float]] = deque()
    recent_s: deque[float] = deque()
    s = 0.0
    prev = None

    for p in pts.reshape(-1, 2).astype(np.float32):
        x, y = float(p[0]), float(p[1])
        if prev is not None:
            dx = x - prev[0]
            dy = y - prev[1]
            s += float(math.hypot(dx, dy))

        # expire recent points into the hash when they are far enough behind
        while recent_s and (s - recent_s[0]) > recent_path_len_px:
            px, py = recent_q.popleft()
            recent_s.popleft()
            H.add(px, py)

        # check collision only against OLD points (hash), not recent
        if H.near(x, y):
            if len(cur_xy) >= 2:
                out.append(np.array(cur_xy, dtype=np.float32))
            # start a new piece; reset recent (we start a fresh path)
            cur_xy = []
            recent_q.clear()
            recent_s.clear()
            prev = None
            continue

        # accept point
        cur_xy.append((x, y))
        recent_q.append((x, y))
        recent_s.append(s)
        prev = (x, y)

    if len(cur_xy) >= 2:
        out.append(np.array(cur_xy, dtype=np.float32))

    return [c.reshape(-1, 1, 2).astype(np.int32) for c in out]

# ---------------- main per-layer ----------------
def process_layer(
    color_dir: str,
    tap_diam: float,
    min_keep_diam: float,
    tap_max_perimeter: float,
    tap_max_vertices: int,
    pen_radius: float,
    recent_path_len_px: float,
):
    # prefer sorted → scaled → raw
    src_sorted = os.path.join(color_dir, "contours_sorted.pkl")
    src_scaled = os.path.join(color_dir, "contours_scaled.pkl")
    src_raw = os.path.join(color_dir, "contours.pkl")
    src = (
        src_sorted
        if os.path.exists(src_sorted)
        else (src_scaled if os.path.exists(src_scaled) else src_raw)
    )
    if not os.path.exists(src):
        print(f"[layer] skip (missing): {src}")
        return
    with open(src, "rb") as f:
        contours: List[np.ndarray] = pickle.load(f)

    kept, taps = split_small_and_taps(
        contours, tap_diam, min_keep_diam, tap_max_perimeter, tap_max_vertices
    )

    cleaned: List[np.ndarray] = []
    for c in kept:
        pieces = virtual_draw_split(c, pen_radius, recent_path_len_px)
        cleaned.extend(pieces)

    cleaned2, taps2 = split_small_and_taps(
        cleaned, tap_diam, min_keep_diam, tap_max_perimeter, tap_max_vertices
    )
    all_taps = (
        taps2
        if len(taps) == 0
        else (np.vstack([taps, taps2]) if len(taps2) else taps)
    )
    cleaned2 = reorder_only(cleaned2)

    with open(os.path.join(color_dir, "contours_dedup_layer.pkl"), "wb") as f:
        pickle.dump(cleaned2, f)
    with open(os.path.join(color_dir, "taps.pkl"), "wb") as f:
        pickle.dump(all_taps, f)
    print(
        f"[layer] {os.path.basename(color_dir)}: lines={len(cleaned2)}, taps={len(all_taps)}"
    )

def main():
    cfg: Config = load_config()
    pen_diam = float(getattr(cfg, "pen_width_px", 60))
    pen_radius = float(getattr(cfg, "pen_radius_px", pen_diam / 2.0))
    tap_diam = float(getattr(cfg, "tap_diameter_px", pen_diam))
    min_keep = float(getattr(cfg, "min_keep_diameter_px", pen_radius))
    recent_path_len = float(
        getattr(cfg, "recent_path_len_px", max(pen_diam, 2.0 * pen_radius))
    )
    # guards for taps
    tap_max_perimeter = float(getattr(cfg, "tap_max_perimeter_px", 3.0 * tap_diam))
    tap_max_vertices = int(getattr(cfg, "tap_max_vertices", 40))

    for name in cfg.color_names:
        os.makedirs(os.path.join(cfg.output_dir, name), exist_ok=True)
        process_layer(
            os.path.join(cfg.output_dir, name),
            tap_diam,
            min_keep,
            tap_max_perimeter,
            tap_max_vertices,
            pen_radius,
            recent_path_len,
        )

if __name__ == "__main__":
    main()
