# 05_sort_contours.py
import os
import cv2  # required for cv2.moments
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from typing import List
from config import load_config, Config

def contour_center(c) -> tuple[int, int]:
    """Centroid of a contour; falls back to first point if area is zero."""
    m = cv2.moments(c)
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    else:
        cx, cy = c[0][0]
    return (cx, cy)

def nearest_neighbor_order(contours: List[np.ndarray]) -> List[int]:
    """Greedy nearest-neighbor ordering for drawing path."""
    n = len(contours)
    if n <= 1:
        return list(range(n))
    centers = np.array([contour_center(c) for c in contours])
    D = cdist(centers, centers)
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = order[-1]
        nxt = np.argmin([D[cur, j] if not visited[j] else np.inf for j in range(n)])
        order.append(int(nxt))
        visited[int(nxt)] = True
    return order

def sort_layer(color_name: str, cfg: Config):
    src = os.path.join(cfg.output_dir, color_name, "contours.pkl")
    with open(src, "rb") as f:
        contours = pickle.load(f)

    order = nearest_neighbor_order(contours)
    sorted_contours = [contours[i] for i in order]

    dst = os.path.join(cfg.output_dir, color_name, "contours_sorted.pkl")
    with open(dst, "wb") as f:
        pickle.dump(sorted_contours, f)

    draw_order = os.path.join(cfg.output_dir, color_name, "draw_order.txt")
    with open(draw_order, "w") as f:
        for i, idx in enumerate(order):
            f.write(f"{i}: original_index={idx}\n")

    print(f"Contours sorted for {color_name}")
    return sorted_contours

def sort_all(cfg: Config):
    for name in cfg.color_names:
        sort_layer(name, cfg)

if __name__ == "__main__":
    config = load_config()
    sort_all(config)
