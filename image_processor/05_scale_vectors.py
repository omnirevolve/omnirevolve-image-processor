# 05_scale_vectors.py
# Scale vector contours (from find-contours step) to the final drawing size.
# Writes: <color>/contours_scaled.pkl
import os
import pickle
import cv2
import numpy as np
from typing import List, Tuple
from config import load_config, Config

def _get_target_scale(cfg: Config, w_src: int, h_src: int) -> Tuple[float, float]:
    """
    Priority:
      1) cfg.vector_scale (single float) -> isotropic
      2) cfg.target_width_px / cfg.target_height_px -> fit keeping aspect
      3) cfg.canvas_width_px / cfg.canvas_height_px -> fit keeping aspect
      4) default 1.0
    """
    # explicit scale
    s = float(getattr(cfg, "vector_scale", 0.0) or 0.0)
    if s > 0:
        return s, s

    # explicit target box
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 or th > 0:
        if tw <= 0:  # scale by height
            s = th / float(h_src)
            return s, s
        if th <= 0:  # scale by width
            s = tw / float(w_src)
            return s, s
        # both given -> fit
        sx = tw / float(w_src)
        sy = th / float(h_src)
        s = min(sx, sy)
        return s, s

    # canvas box
    cw = int(getattr(cfg, "canvas_width_px", 0) or 0)
    ch = int(getattr(cfg, "canvas_height_px", 0) or 0)
    if cw > 0 and ch > 0:
        sx = cw / float(w_src)
        sy = ch / float(h_src)
        s = min(sx, sy)
        return s, s

    return 1.0, 1.0

def _scale_one(contours: List[np.ndarray], sx: float, sy: float) -> List[np.ndarray]:
    out = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out

def main():
    cfg: Config = load_config()
    resized_path = os.path.join(cfg.output_dir, "resized.png")
    img = cv2.imread(resized_path)
    if img is None:
        raise RuntimeError(f"Cannot read resized image: {resized_path}")
    h_src, w_src = img.shape[:2]

    sx, sy = _get_target_scale(cfg, w_src, h_src)
    print(f"[scale] source={w_src}x{h_src}, scale=({sx:.4f},{sy:.4f})")

    for name in cfg.color_names:
        cdir = os.path.join(cfg.output_dir, name)
        os.makedirs(cdir, exist_ok=True)
        src = os.path.join(cdir, "contours.pkl")
        if not os.path.exists(src):
            print(f"[scale] skip (missing): {src}")
            continue
        with open(src, "rb") as f:
            contours: List[np.ndarray] = pickle.load(f)

        scaled = _scale_one(contours, sx, sy)
        dst = os.path.join(cdir, "contours_scaled.pkl")
        with open(dst, "wb") as f:
            pickle.dump(scaled, f)
        pts_before = sum(c.reshape(-1,2).shape[0] for c in contours)
        pts_after  = sum(c.reshape(-1,2).shape[0] for c in scaled)
        print(f"[scale] {name}: contours={len(contours)}, vertices={pts_before} → {pts_after} → {dst}")

if __name__ == "__main__":
    main()
