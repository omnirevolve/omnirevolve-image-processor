# 01_resize.py
import os
import cv2
import numpy as np
from config import load_config, Config

def resize_if_needed(image_path: str, cfg: Config) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]
    max_dim = max(h, w)

    if max_dim > cfg.max_dimension:
        scale = cfg.max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"Resizing: {w}x{h} -> {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        print(f"No resize required: {w}x{h}")
    return img

if __name__ == "__main__":
    config = load_config()          # <- reads OUTPUT_DIR/config.json via PIPELINE_CONFIG
    config.ensure_output_dirs()
    img = resize_if_needed(config.input_image, config)
    out_path = os.path.join(config.output_dir, "resized.png")
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
