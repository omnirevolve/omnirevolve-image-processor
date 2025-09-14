# 08_simplify.py
import os
import pickle
import cv2
import numpy as np
from typing import List
from config import load_config, Config

def chaikin_smooth(contour: np.ndarray, iters: int = 2, closed: bool = True) -> np.ndarray:
    """Chaikin corner-cutting. Returns smoothed contour."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2 or iters <= 0:
        return contour
    for _ in range(iters):
        new_pts = []
        n = len(pts)
        last = n if closed else n - 1
        for i in range(last):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_pts.extend([Q, R])
        pts = np.asarray(new_pts, dtype=np.float32)
    return pts.reshape(-1, 1, 2).astype(np.int32)

def simplify_contour(contour, epsilon_factor: float):
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def simplify_layer(color_name: str, cfg: Config) -> List[np.ndarray]:
    base = os.path.join(cfg.output_dir, color_name)
    src_cross = os.path.join(base, "contours_dedup_cross.pkl")
    src_layer = os.path.join(base, "contours_dedup_layer.pkl")
    src_raw   = os.path.join(base, "contours.pkl")
    src = src_cross if os.path.exists(src_cross) else (src_layer if os.path.exists(src_layer) else src_raw)

    with open(src, "rb") as f:
        contours = pickle.load(f)

    simplified, pts_before, pts_after = [], 0, 0
    for c in contours:
        pts_before += len(c)
        s = chaikin_smooth(c, iters=max(0, int(cfg.smoothing_iterations)))
        s = simplify_contour(s, cfg.epsilon_factor)  # no extra *2
        simplified.append(s)
        pts_after += len(s)

    out_path = os.path.join(base, "contours_simplified.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(simplified, f)

    print(f"Simplify {color_name}: {pts_before} -> {pts_after} points")
    return simplified

def simplify_all_layers(cfg: Config):
    for name in cfg.color_names:
        simplify_layer(name, cfg)

if __name__ == "__main__":
    config = load_config()
    simplify_all_layers(config)
