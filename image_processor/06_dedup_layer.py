# 06_dedup_layer.py
import os
import pickle
import numpy as np
import cv2
from typing import List
from config import load_config, Config

def contours_are_similar(c1, c2, dist_thresh: float) -> bool:
    """Similarity: close centroids AND similar area."""
    m1, m2 = cv2.moments(c1), cv2.moments(c2)
    if m1["m00"] == 0 or m2["m00"] == 0:
        return False
    cx1, cy1 = m1["m10"]/m1["m00"], m1["m01"]/m1["m00"]
    cx2, cy2 = m2["m10"]/m2["m00"], m2["m01"]/m2["m00"]
    dist = np.hypot(cx1 - cx2, cy1 - cy2)
    a1, a2 = cv2.contourArea(c1), cv2.contourArea(c2)
    area_ratio = (min(a1, a2) / max(a1, a2)) if max(a1, a2) > 0 else 0.0
    return dist < dist_thresh and area_ratio > 0.7

def dedup_layer(color_name: str, cfg: Config) -> List:
    src = os.path.join(cfg.output_dir, color_name, "contours_sorted.pkl")
    with open(src, "rb") as f:
        contours = pickle.load(f)

    if len(contours) <= 1:
        unique = contours
    else:
        unique, used = [], [False] * len(contours)
        for i in range(len(contours)):
            if used[i]:
                continue
            group = [contours[i]]
            used[i] = True
            for j in range(i + 1, len(contours)):
                if not used[j] and contours_are_similar(contours[i], contours[j], cfg.dedup_distance_threshold):
                    group.append(contours[j])
                    used[j] = True
            unique.append(group[0])

    dst = os.path.join(cfg.output_dir, color_name, "contours_dedup_layer.pkl")
    with open(dst, "wb") as f:
        pickle.dump(unique, f)

    print(f"Dedup {color_name}: {len(contours)} -> {len(unique)}")
    return unique

def dedup_all(cfg: Config):
    for name in cfg.color_names:
        dedup_layer(name, cfg)

if __name__ == "__main__":
    config = load_config()
    dedup_all(config)
