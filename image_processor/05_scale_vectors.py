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


def _margins_px(cfg: Config) -> Tuple[int, int, int, int]:
    ppm = int(getattr(cfg, "pixels_per_mm", 40) or 40)
    ml = int(round(float(getattr(cfg, "margin_left_mm", 10.0))   * ppm))
    mr = int(round(float(getattr(cfg, "margin_right_mm", 10.0))  * ppm))
    mt = int(round(float(getattr(cfg, "margin_top_mm", 10.0))    * ppm))
    mb = int(round(float(getattr(cfg, "margin_bottom_mm", 10.0)) * ppm))
    # clamp negative
    ml = max(0, ml); mr = max(0, mr); mt = max(0, mt); mb = max(0, mb)
    return ml, mr, mt, mb


def _get_scale_factors_into_inner(cfg: Config, w_src: int, h_src: int,
                                  w_full: int, h_full: int,
                                  ml: int, mr: int, mt: int, mb: int) -> Tuple[float, float, int, int]:
    """
    Compute scale factors to fit into inner rect (full minus margins).
    Return (sx, sy, inner_w, inner_h).
    If cfg.keep_aspect (default True), use isotropic scale = min(sx, sy).
    """
    inner_w = max(1, w_full - ml - mr)
    inner_h = max(1, h_full - mt - mb)
    keep_aspect = bool(getattr(cfg, "keep_aspect", True))
    sx = inner_w / max(1e-6, w_src)
    sy = inner_h / max(1e-6, h_src)
    if keep_aspect:
        s = min(sx, sy)
        return s, s, inner_w, inner_h
    return sx, sy, inner_w, inner_h


def _scale_one(polys: List[np.ndarray], sx: float, sy: float, dx: float, dy: float) -> List[np.ndarray]:
    """
    Scale each poly by (sx, sy) and then translate by (dx, dy).
    Returns int32 polys ready for downstream steps.
    """
    out: List[np.ndarray] = []
    if not polys:
        return out
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float32)
    T = np.array([dx, dy], dtype=np.float32)
    for p in polys:
        pts = p.reshape(-1, 2).astype(np.float32)
        pts = (pts @ S.T) + T
        out.append(pts.reshape(-1, 1, 2).astype(np.int32))
    return out


def main():
    cfg: Config = load_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    w_src, h_src = _source_size_px(cfg)
    w_full, h_full = _target_size_px(cfg)
    ml, mr, mt, mb = _margins_px(cfg)
    sx, sy, inner_w, inner_h = _get_scale_factors_into_inner(cfg, w_src, h_src, w_full, h_full, ml, mr, mt, mb)

    # offset = top-left corner of the inner rect (in pixels)
    dx, dy = ml, mt

    print(f"[scale] source={w_src}x{h_src}, target(full)={w_full}x{h_full}, inner={inner_w}x{inner_h}, "
          f"margins(l,r,t,b)=({ml},{mr},{mt},{mb}), scale=({sx:.4f},{sy:.4f}), offset=({dx},{dy})")

    for name in cfg.color_names:
        cdir = os.path.join(cfg.output_dir, name)
        os.makedirs(cdir, exist_ok=True)
        src = os.path.join(cdir, "contours.pkl")
        if not os.path.exists(src):
            print(f"[scale] {name}: missing {src}, skipping")
            continue
        with open(src, "rb") as f:
            contours: List[np.ndarray] = pickle.load(f)

        scaled = _scale_one(contours, sx, sy, dx, dy)
        dst = os.path.join(cdir, "contours_scaled.pkl")
        with open(dst, "wb") as f:
            pickle.dump(scaled, f)
        pts_before = sum(c.reshape(-1, 2).shape[0] for c in contours)
        pts_after  = sum(c.reshape(-1, 2).shape[0] for c in scaled)
        print(f"[scale] {name}: contours={len(contours)}, vertices={pts_before} → {pts_after} → {dst}")


if __name__ == "__main__":
    main()
