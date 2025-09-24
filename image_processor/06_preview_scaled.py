# 05_1_scaled_preview.py — rasterize scaled vectors per layer + composite preview.
# Inputs : <output>/<layer>/contours_scaled.pkl  (fallback: contours_sorted.pkl → contours.pkl)
# Canvas : EXACT target canvas in px (from target_*_px or target_*_mm * pixels_per_mm)
# Outputs: <output>/<layer>/scaled_preview.png, <output>/scaled_preview_composite.png
from __future__ import annotations
import os
import json
import pickle
from typing import List, Dict, Tuple

import cv2
import numpy as np
from config import load_config, Config


def _target_size_px(cfg: Config) -> Tuple[int, int]:
    tw_px = int(getattr(cfg, "target_width_px", 0) or 0)
    th_px = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw_px > 0 and th_px > 0:
        return tw_px, th_px
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    # fallback to resized.png to avoid hard crash
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size (no target_* and no resized.png).")
    h, w = base.shape[:2]
    return w, h


def _margins_px(cfg: Config) -> Tuple[int, int, int, int]:
    ppm = int(getattr(cfg, "pixels_per_mm", 40) or 40)
    ml = int(round(float(getattr(cfg, "margin_left_mm", 10.0))   * ppm))
    mr = int(round(float(getattr(cfg, "margin_right_mm", 10.0))  * ppm))
    mt = int(round(float(getattr(cfg, "margin_top_mm", 10.0))    * ppm))
    mb = int(round(float(getattr(cfg, "margin_bottom_mm", 10.0)) * ppm))
    # clamp non-negative
    ml = max(0, ml); mr = max(0, mr); mt = max(0, mt); mb = max(0, mb)
    return ml, mr, mt, mb


def _load_palette_by_name(outdir: str, color_names: List[str], fallback_colors: List[Tuple[int,int,int]]) -> Dict[str, Tuple[int,int,int]]:
    path = os.path.join(outdir, "palette_by_name.json")
    mapping: Dict[str, Tuple[int,int,int]] = {}
    data = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    for i, name in enumerate(color_names):
        if data and name in data and "approx_bgr" in data[name]:
            bgr = data[name]["approx_bgr"]
            mapping[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        else:
            b,g,r = fallback_colors[i]
            mapping[name] = (int(b), int(g), int(r))
    return mapping


def _load_scaled_contours(layer_dir: str) -> List[np.ndarray]:
    for fname in ("contours_scaled.pkl", "contours_sorted.pkl", "contours.pkl"):
        p = os.path.join(layer_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                return obj
    return []


def _draw_layer(polys: List[np.ndarray], size: Tuple[int,int], color: Tuple[int,int,int], thickness: int, aa: bool) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    if not polys:
        return img
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    for p in polys:
        if p is None:
            continue
        arr = p.reshape(-1, 1, 2).astype(np.int32)
        if len(arr) >= 2:
            cv2.polylines(img, [arr], isClosed=False, color=color, thickness=thickness, lineType=lt)
    return img


def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)

    # render params
    thickness = int(getattr(cfg, "scaled_preview_thickness_px", 1))
    aa = bool(getattr(cfg, "scaled_preview_antialiased", True))

    size = _target_size_px(cfg)  # (W,H) exact full canvas
    ml, mr, mt, mb = _margins_px(cfg)
    inner_w = max(1, size[0] - ml - mr)
    inner_h = max(1, size[1] - mt - mb)
    print(f"[scaled_preview] canvas(full)={size[0]}x{size[1]}, margins(l,r,t,b)=({ml},{mr},{mt},{mb}), inner={inner_w}x{inner_h}, offset=({ml},{mt})")

    palette = _load_palette_by_name(outdir, cfg.color_names, cfg.colors)

    composite = np.full((size[1], size[0], 3), 255, np.uint8)
    total_polys = 0
    total_vertices = 0

    for name in cfg.color_names:
        layer_dir = os.path.join(outdir, name)
        os.makedirs(layer_dir, exist_ok=True)
        polys = _load_scaled_contours(layer_dir)

        vcount = sum(int(p.reshape(-1,2).shape[0]) for p in polys) if polys else 0
        total_polys += len(polys)
        total_vertices += vcount

        # per-layer preview (black)
        layer_img = _draw_layer(polys, size, (0,0,0), thickness, aa)
        out_layer = os.path.join(layer_dir, "scaled_preview.png")
        cv2.imwrite(out_layer, layer_img)

        # add to composite with per-layer color
        bgr = palette.get(name, (0,0,0))
        color_img = _draw_layer(polys, size, bgr, thickness, aa)
        mask = (color_img != 255).any(axis=2)
        composite[mask] = color_img[mask]

        print(f"[scaled_preview] {name}: contours={len(polys)}, vertices={vcount} → {out_layer}")

    out_comp = os.path.join(outdir, "scaled_preview_composite.png")
    cv2.imwrite(out_comp, composite)
    print(f"[scaled_preview] composite saved: {out_comp}")
    print(f"[scaled_preview] totals: contours={total_polys}, vertices={total_vertices}")


if __name__ == "__main__":
    main()
