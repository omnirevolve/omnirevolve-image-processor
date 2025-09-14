# 05_scale_vectors.py
# Scale vector contours (from find-contours step) to the final drawing size.
# Inputs : <output>/<layer>/contours.pkl
# Outputs: <output>/<layer>/contours_scaled.pkl
from __future__ import annotations
import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np
from config import load_config, Config


def _target_size_px(cfg: Config) -> Tuple[int, int]:
    """
    Determine target canvas size in pixels (W,H).
    Priority:
      1) target_width_px / target_height_px (if present in config.json)
      2) target_width_mm / target_height_mm with pixels_per_mm
      3) fallback to resized.png size (keeps old behavior)
    """
    tw_px = int(getattr(cfg, "target_width_px", 0) or 0)
    th_px = int(getattr(cfg, "target_height_px", 0) or 0)

    if tw_px > 0 and th_px > 0:
        return tw_px, th_px

    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))

    # fallback: use resized.png shape
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size: no target_* set and no resized.png found.")
    h, w = base.shape[:2]
    return w, h


def _source_size_px(cfg: Config) -> Tuple[int, int]:
    """Source reference size (W,H) — we use resized.png dimensions."""
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Missing resized.png (run step 1 first).")
    h, w = base.shape[:2]
    return w, h


def _get_scale_factors(cfg: Config, w_src: int, h_src: int, w_tgt: int, h_tgt: int) -> Tuple[float, float]:
    """
    Compute scale factors (sx, sy).
    If cfg.keep_aspect (default True), use isotropic scale = min(sx, sy).
    Otherwise, allow anisotropic scaling.
    """
    keep_aspect = bool(getattr(cfg, "keep_aspect", True))
    sx = w_tgt / max(1e-6, w_src)
    sy = h_tgt / max(1e-6, h_src)
    if keep_aspect:
        s = min(sx, sy)
        return s, s
    return sx, sy


def _scale_one(polys: List[np.ndarray], sx: float, sy: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if not polys:
        return out
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float32)
    for p in polys:
        pts = p.reshape(-1, 2).astype(np.float32)
        pts = (pts @ S.T)
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out


def main():
    cfg: Config = load_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    w_src, h_src = _source_size_px(cfg)
    w_tgt, h_tgt = _target_size_px(cfg)
    sx, sy = _get_scale_factors(cfg, w_src, h_src, w_tgt, h_tgt)

    print(f"[scale] source={w_src}x{h_src}, target={w_tgt}x{h_tgt}, scale=({sx:.4f},{sy:.4f})")

    for name in cfg.color_names:
        cdir = os.path.join(cfg.output_dir, name)
        os.makedirs(cdir, exist_ok=True)
        src = os.path.join(cdir, "contours.pkl")
        if not os.path.exists(src):
            print(f"[scale] {name}: missing {src}, skipping")
            continue
        with open(src, "rb") as f:
            contours: List[np.ndarray] = pickle.load(f)

        scaled = _scale_one(contours, sx, sy)
        dst = os.path.join(cdir, "contours_scaled.pkl")
        with open(dst, "wb") as f:
            pickle.dump(scaled, f)
        pts_before = sum(c.reshape(-1, 2).shape[0] for c in contours)
        pts_after  = sum(c.reshape(-1, 2).shape[0] for c in scaled)
        print(f"[scale] {name}: contours={len(contours)}, vertices={pts_before} → {pts_after} → {dst}")


if __name__ == "__main__":
    main()
