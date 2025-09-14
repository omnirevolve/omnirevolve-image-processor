# 10_preview.py
import os
import pickle
import cv2
import numpy as np
from config import load_config

def main():
    cfg = load_config()

    target_px = (int(cfg.target_width_mm * cfg.pixels_per_mm),
                 int(cfg.target_height_mm * cfg.pixels_per_mm))
    canvas = np.full((target_px[1], target_px[0], 3), 255, dtype=np.uint8)

    for i, name in enumerate(cfg.color_names):
        base = os.path.join(cfg.output_dir, name)

        # prefer final (scaled) contours; fallback to simplified (unscaled) is not used here
        pkl = os.path.join(base, "contours_final.pkl")
        if not os.path.exists(pkl):
            # nothing to draw in target coords
            continue
        with open(pkl, "rb") as f:
            contours = pickle.load(f)

        color = tuple(int(c) for c in cfg.colors[i])  # BGR
        for c in contours:
            cv2.polylines(canvas, [c], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        taps_path = os.path.join(base, "taps_final.pkl")
        if os.path.exists(taps_path):
            with open(taps_path, "rb") as f:
                taps = pickle.load(f)  # Nx2 float32 in target coords
            R = max(1, int(cfg.pen_radius_px // 2))
            for (x, y) in taps:
                cv2.circle(canvas, (int(x), int(y)), R, color, thickness=-1, lineType=cv2.LINE_AA)

    out_png = os.path.join(cfg.output_dir, "preview.png")
    cv2.imwrite(out_png, canvas)
    print(f"Preview saved: {out_png}")

if __name__ == "__main__":
    main()
