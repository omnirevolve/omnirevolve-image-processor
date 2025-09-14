# 09_scale_vector.py
import os
import json
import pickle
import numpy as np
import cv2
from typing import List, Tuple
from config import load_config, Config

def scale_contours_to_target(contours, original_size: Tuple[int, int],
                             target_size_mm: Tuple[int, int], px_per_mm: int):
    """Scale contours to fit a target physical size in mm (converted to pixels)."""
    target_w_px = target_size_mm[0] * px_per_mm
    target_h_px = target_size_mm[1] * px_per_mm

    sx = target_w_px / original_size[0]
    sy = target_h_px / original_size[1]
    scale = min(sx, sy)

    scaled = []
    for c in contours:
        c_scaled = c.astype(np.float32) * scale
        scaled.append(c_scaled.astype(np.int32))
    return scaled, scale

def export_to_svg(contours, color_name: str, color_rgb: Tuple[int, int, int],
                  size_px: Tuple[int, int], cfg: Config):
    w, h = size_px
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
        f'<g id="{color_name}" fill="none" stroke="rgb{color_rgb}" stroke-width="1">'
    ]
    for cnt in contours:
        path = "M " + " ".join(f"{pt[0][0]},{pt[0][1]}" for pt in cnt) + " Z"
        lines.append(f'  <path d="{path}"/>')
    lines.append('</g>')
    lines.append('</svg>')

    out_svg = os.path.join(cfg.output_dir, color_name, "output.svg")
    with open(out_svg, "w") as f:
        f.write("\n".join(lines))

def scale_all_vectors(cfg: Config):
    resized_path = os.path.join(cfg.output_dir, "resized.png")
    img = cv2.imread(resized_path)
    if img is None:
        raise ValueError(f"Failed to load resized image: {resized_path}")
    original_size = (img.shape[1], img.shape[0])  # (w, h) in pixels

    target_mm = (cfg.target_width_mm, cfg.target_height_mm)
    target_px = (int(target_mm[0] * cfg.pixels_per_mm),
                 int(target_mm[1] * cfg.pixels_per_mm))

    for i, color_name in enumerate(cfg.color_names):
        with open(os.path.join(cfg.output_dir, color_name, "contours_simplified.pkl"), "rb") as f:
            contours = pickle.load(f)

        scaled_contours, scale = scale_contours_to_target(
            contours, original_size, target_mm, cfg.pixels_per_mm
        )

        with open(os.path.join(cfg.output_dir, color_name, "contours_final.pkl"), "wb") as f:
            pickle.dump(scaled_contours, f)

        # BGR -> RGB for SVG stroke color
        bgr = cfg.colors[i]
        rgb = (bgr[2], bgr[1], bgr[0])
        export_to_svg(scaled_contours, color_name, rgb, target_px, cfg)

        meta = {
            "color_name": color_name,
            "original_size": original_size,
            "target_size_mm": target_mm,
            "target_size_px": target_px,
            "scale": scale,
            "num_contours": len(scaled_contours),
        }
        with open(os.path.join(cfg.output_dir, color_name, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Scale {color_name}: scale={scale:.3f}")

if __name__ == "__main__":
    config = load_config()
    scale_all_vectors(config)
