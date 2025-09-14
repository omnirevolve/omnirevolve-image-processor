# 04_find_contours.py
import os
import cv2
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
from config import load_config, Config

def find_contours_for_layer(color_name: str, cfg: Config) -> Tuple[str, List]:
    edges_path = os.path.join(cfg.output_dir, color_name, "edges.png")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edges image not found: {edges_path}")

    edges = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise ValueError(f"Failed to load edges image: {edges_path}")

    # keep all points; approximate later (step 8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered = [c for c in contours if cv2.contourArea(c) >= cfg.min_contour_area]

    out_path = os.path.join(cfg.output_dir, color_name, "contours.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(filtered, f)

    print(f"Found contours for {color_name}: {len(filtered)}")
    return color_name, filtered

def find_all_contours(cfg: Config):
    results = []
    with ProcessPoolExecutor(max_workers=min(4, cfg.n_cores)) as ex:
        futures = {ex.submit(find_contours_for_layer, name, cfg): name for name in cfg.color_names}
        for fut in as_completed(futures):
            results.append(fut.result())
    return dict(results)

if __name__ == "__main__":
    config = load_config()
    find_all_contours(config)
