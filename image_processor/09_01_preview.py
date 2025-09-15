# 09_01_preview.py — preview after simplify (STRICT)
# Input per layer: contours_simplified.pkl
# Outputs: <layer>/preview_simplified.png, preview_simplified_composite.png

from __future__ import annotations
import os, json, pickle
from typing import List, Tuple, Dict
import numpy as np, cv2
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
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size: resized.png is missing.")
    h, w = base.shape[:2]
    return w, h


def _palette(outdir: str, names: List[str], fallback) -> Dict[str, Tuple[int, int, int]]:
    path = os.path.join(outdir, "palette_by_name.json")
    mapping: Dict[str, Tuple[int, int, int]] = {}
    data = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
    for i, n in enumerate(names):
        if data and n in data and "approx_bgr" in data[n]:
            b, g, r = data[n]["approx_bgr"]
        else:
            b, g, r = fallback[i]
        mapping[n] = (int(b), int(g), int(r))
    return mapping


def _load_polys_strict(layer_dir: str) -> List[np.ndarray]:
    p = os.path.join(layer_dir, "contours_simplified.pkl")
    if not os.path.exists(p):
        raise RuntimeError(f"Missing required input: {p}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Invalid pickle format: {p}")
    return obj


def _draw(polys: List[np.ndarray], size: Tuple[int, int], color: Tuple[int, int, int], th: int, aa: bool) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    for p in polys:
        if p is None:
            continue
        arr = np.asarray(p).reshape(-1, 1, 2).astype(np.int32)
        if len(arr) >= 2:
            cv2.polylines(img, [arr], False, color, th, lt)
    return img


def main():
    cfg = load_config()
    outdir = cfg.output_dir
    size = _target_size_px(cfg)
    th = int(getattr(cfg, "preview_line_thickness_px", 1))
    aa = bool(getattr(cfg, "preview_antialiased", True))

    palette = _palette(outdir, cfg.color_names, cfg.colors)
    composite = np.full((size[1], size[0], 3), 255, np.uint8)

    for name in cfg.color_names:
        layer_dir = os.path.join(outdir, name)
        polys = _load_polys_strict(layer_dir)

        lay = _draw(polys, size, (0, 0, 0), th, aa)
        out_path = os.path.join(layer_dir, "preview_simplified.png")
        cv2.imwrite(out_path, lay)

        col = palette.get(name, (0, 0, 0))
        col_img = _draw(polys, size, col, th, aa)
        mask = (col_img != 255).any(axis=2)
        composite[mask] = col_img[mask]

        print(f"[preview_simplified] {name}: polylines={len(polys)} → {out_path}")

    comp = os.path.join(outdir, "preview_simplified_composite.png")
    cv2.imwrite(comp, composite)
    print(f"[preview_simplified] composite saved: {comp}")


if __name__ == "__main__":
    main()
