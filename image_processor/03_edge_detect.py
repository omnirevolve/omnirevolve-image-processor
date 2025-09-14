import os
import json
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from config import load_config, Config  # :contentReference[oaicite:2]{index=2}

def _ensure_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1

def process_color(color_name: str, cfg: Config) -> Tuple[str, str]:
    """Load layer mask, smooth, Canny, save edges.png."""
    mask_path = os.path.join(cfg.output_dir, color_name, "mask.png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image: {mask_path}")

    k_m = max(1, int(getattr(cfg, "edge_morph_kernel", 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_m, k_m))
    if getattr(cfg, "edge_morph_open_iters", 1) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=int(cfg.edge_morph_open_iters))
    if getattr(cfg, "edge_morph_close_iters", 1) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=int(cfg.edge_morph_close_iters))

    k = _ensure_odd(cfg.edge_kernel_size)
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    edges = cv2.Canny(blurred, cfg.edge_low_threshold, cfg.edge_high_threshold)

    out_path = os.path.join(cfg.output_dir, color_name, "edges.png")
    cv2.imwrite(out_path, edges)
    nz = int(np.count_nonzero(edges))
    print(f"Edges extracted: {color_name} | nz={nz}")
    return color_name, out_path

def detect_all_edges(cfg: Config):
    results = []
    with ProcessPoolExecutor(max_workers=cfg.n_cores) as ex:
        futures = {ex.submit(process_color, name, cfg): name for name in cfg.color_names}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

def _load_palette_by_name(outdir: str) -> dict | None:
    p = os.path.join(outdir, "palette_by_name.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_edges_composite(cfg: Config) -> str:
    """
    Overlay all <output_dir>/<color>/edges.png onto a white canvas.
    Prefer per-layer colors from palette_by_name.json; fallback to cfg.colors[i].
    """
    resized = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if resized is None:
        # infer from any edges.png
        h, w = None, None
        for name in cfg.color_names:
            ep = os.path.join(cfg.output_dir, name, "edges.png")
            if os.path.exists(ep):
                e = cv2.imread(ep, cv2.IMREAD_GRAYSCALE)
                if e is not None:
                    h, w = e.shape[:2]; break
        if h is None:
            raise FileNotFoundError("No edges found to build edges_composite.png")
        canvas = np.full((h, w, 3), 255, np.uint8)
    else:
        h, w = resized.shape[:2]
        canvas = np.full((h, w, 3), 255, np.uint8)

    # try palette_by_name first
    palette = _load_palette_by_name(cfg.output_dir) or {}
    name_to_bgr = {}
    for i, name in enumerate(cfg.color_names):
        if name in palette and "bgr" in palette[name]:
            b,g,r = palette[name]["bgr"]
        else:
            # fallback to indexed cfg.colors
            b,g,r = cfg.colors[i]
        name_to_bgr[name] = (b,g,r)

    for name in cfg.color_names:
        edge_path = os.path.join(cfg.output_dir, name, "edges.png")
        if not os.path.exists(edge_path):
            continue
        edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edges is None:
            continue
        mask = edges > 0
        if not np.any(mask):
            continue
        b,g,r = name_to_bgr[name]
        layer = np.zeros_like(canvas)
        layer[mask] = (b, g, r)
        canvas[mask] = layer[mask]

    out_path = os.path.join(cfg.output_dir, "edges_composite.png")
    cv2.imwrite(out_path, canvas)
    print(f"Edges composite saved: {out_path}")
    return out_path

if __name__ == "__main__":
    config = load_config()
    detect_all_edges(config)
    save_edges_composite(config)
