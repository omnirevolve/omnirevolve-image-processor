# 10_preview.py
import os
import pickle
import cv2
import numpy as np
from config import load_config

def main():
    cfg = load_config()

    # Canvas size = target physical size converted to pixels
    w_px = int(cfg.target_width_mm  * cfg.pixels_per_mm)
    h_px = int(cfg.target_height_mm * cfg.pixels_per_mm)

    canvas = np.full((h_px, w_px, 3), 255, dtype=np.uint8)  # white background

    for i, name in enumerate(cfg.color_names):
        pkl = os.path.join(cfg.output_dir, name, "contours_final.pkl")
        if not os.path.exists(pkl):
            # fallback: allow preview right after step 8
            pkl = os.path.join(cfg.output_dir, name, "contours_simplified.pkl")
            if not os.path.exists(pkl):
                continue

        with open(pkl, "rb") as f:
            contours = pickle.load(f)

        # OpenCV expects BGR; config.colors is already BGR
        bgr = tuple(int(c) for c in cfg.colors[i])
        for c in contours:
            cv2.polylines(canvas, [c], isClosed=True, color=bgr, thickness=2, lineType=cv2.LINE_AA)

    out_png = os.path.join(cfg.output_dir, "preview.png")
    cv2.imwrite(out_png, canvas)
    print(f"Preview saved: {out_png}")

if __name__ == "__main__":
    main()
