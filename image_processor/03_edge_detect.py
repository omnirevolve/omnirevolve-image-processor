# 03_edge_detect.py
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from config import load_config, Config

def _ensure_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1

def process_color(color_name: str, cfg: Config) -> Tuple[str, str]:
    mask_path = os.path.join(cfg.output_dir, color_name, "mask.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    # --- smooth masks to avoid zig-zags ---
    k_m = max(1, int(cfg.edge_morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_m, k_m))
    if cfg.edge_morph_open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(cfg.edge_morph_open_iters))
    if cfg.edge_morph_close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(cfg.edge_morph_close_iters))

    k = _ensure_odd(cfg.edge_kernel_size)
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    edges = cv2.Canny(blurred, cfg.edge_low_threshold, cfg.edge_high_threshold)

    out_path = os.path.join(cfg.output_dir, color_name, "edges.png")
    cv2.imwrite(out_path, edges)
    print(f"Edges extracted: {color_name}")
    return color_name, out_path

def detect_all_edges(cfg: Config):
    results = []
    with ProcessPoolExecutor(max_workers=cfg.n_cores) as ex:
        futures = {ex.submit(process_color, name, cfg): name for name in cfg.color_names}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

if __name__ == "__main__":
    config = load_config()
    detect_all_edges(config)
