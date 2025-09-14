# 07_dedup_cross.py
import os
import pickle
from config import load_config

def save_cross_dedup() -> None:
    """
    Placeholder cross-layer deduplication:
    copies per-layer dedup results into `contours_dedup_cross.pkl`
    so the next stage can proceed.
    """
    cfg = load_config()
    for color_name in cfg.color_names:
        base = os.path.join(cfg.output_dir, color_name)
        src = os.path.join(base, "contours_dedup_layer.pkl")
        dst = os.path.join(base, "contours_dedup_cross.pkl")
        if not os.path.exists(src):
            print(f"[cross-dedup] Missing layer file: {src}")
            continue
        with open(src, "rb") as f:
            contours = pickle.load(f)
        with open(dst, "wb") as f:
            pickle.dump(contours, f)
        print(f"[cross-dedup] Wrote {dst} ({len(contours)} contours)")

if __name__ == "__main__":
    save_cross_dedup()
