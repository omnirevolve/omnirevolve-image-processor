# 06_sort_contours.py
# Reorder polylines to reduce travel. Reads scaled contours if present.
# Writes: <color>/contours_sorted.pkl
import os
import time
import pickle
import numpy as np
import cv2
from typing import List, Tuple
from config import load_config, Config

def _ends(poly: np.ndarray):
    pts = poly.reshape(-1, 2)
    closed = np.all(pts[0] == pts[-1])
    if closed and len(pts) > 1:
        pts = pts[:-1]
    return pts[0], pts[-1], closed

def reorder_one_color(color_dir: str) -> None:
    src_scaled = os.path.join(color_dir, "contours_scaled.pkl")
    src = src_scaled if os.path.exists(src_scaled) else os.path.join(color_dir, "contours.pkl")
    if not os.path.exists(src):
        print(f"[sort] skip (missing): {src}")
        return

    with open(src, "rb") as f:
        contours: List[np.ndarray] = pickle.load(f)

    before_pts = int(sum(c.reshape(-1,2).shape[0] for c in contours))
    if not contours:
        with open(os.path.join(color_dir, "contours_sorted.pkl"), "wb") as f:
            pickle.dump([], f)
        print(f"[sort] {os.path.basename(color_dir)}: empty")
        return

    t0 = time.perf_counter()
    used = np.zeros(len(contours), dtype=bool)
    starts = []
    ends = []
    closed = []
    for c in contours:
        s, e, cl = _ends(c)
        starts.append(s); ends.append(e); closed.append(cl)
    starts = np.array(starts); ends = np.array(ends)
    closed = np.array(closed, dtype=bool)

    order = []
    flips = []

    lengths = [float(cv2.arcLength(c, True)) for c in contours]
    cur = int(np.argmax(lengths))
    order.append(cur); flips.append(False); used[cur] = True
    cur_end = ends[cur] if not closed[cur] else starts[cur]

    while not np.all(used):
        best_i = -1; best_flip = False; best_d2 = 1e20
        idxs = np.flatnonzero(~used)
        if len(idxs) == 0: break
        d2_start = np.sum((starts[idxs].astype(np.float32) - cur_end.astype(np.float32))**2, axis=1)
        d2_end   = np.sum((ends[idxs].astype(np.float32)   - cur_end.astype(np.float32))**2, axis=1)
        for k, i in enumerate(idxs):
            if closed[i]:
                d2 = d2_start[k]
                if d2 < best_d2:
                    best_d2 = d2; best_i = i; best_flip = False
            else:
                if d2_start[k] <= d2_end[k]:
                    if d2_start[k] < best_d2:
                        best_d2 = d2_start[k]; best_i = i; best_flip = False
                else:
                    if d2_end[k] < best_d2:
                        best_d2 = d2_end[k]; best_i = i; best_flip = True
        used[best_i] = True
        order.append(best_i)
        flips.append(best_flip)
        if closed[best_i]:
            cur_end = starts[best_i]
        else:
            cur_end = (ends[best_i] if not best_flip else starts[best_i])

    sorted_contours: List[np.ndarray] = []
    for idx, flip in zip(order, flips):
        c = contours[idx]
        pts = c.reshape(-1, 2)
        if flip: pts = pts[::-1].copy()
        if np.all(c.reshape(-1,2)[0] == c.reshape(-1,2)[-1]) and not np.all(pts[0] == pts[-1]):
            pts = np.vstack([pts, pts[0]])
        sorted_contours.append(pts.reshape(-1,1,2).astype(np.int32))

    after_pts = int(sum(c.reshape(-1,2).shape[0] for c in sorted_contours))
    with open(os.path.join(color_dir, "contours_sorted.pkl"), "wb") as f:
        pickle.dump(sorted_contours, f)

    print(f"[sort] {os.path.basename(color_dir)}: contours={len(sorted_contours)}, "
          f"vertices before={before_pts}, after={after_pts}, time={time.perf_counter()-t0:.2f}s")

def main():
    cfg: Config = load_config()
    for name in cfg.color_names:
        color_dir = os.path.join(cfg.output_dir, name)
        os.makedirs(color_dir, exist_ok=True)
        reorder_one_color(color_dir)

if __name__ == "__main__":
    main()
