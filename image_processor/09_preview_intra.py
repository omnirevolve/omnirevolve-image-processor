# 07_1_preview.py — preview after intra-layer dedup (STRICT)
# Inputs per layer: lines_intra.pkl, taps_intra.pkl
# Outputs: <layer>/preview_intra.png, preview_intra_composite.png

from __future__ import annotations
import os, json, pickle
from typing import List, Tuple, Dict

import numpy as np
import cv2
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


def _load_lines_strict(layer_dir: str) -> List[np.ndarray]:
    p = os.path.join(layer_dir, "lines_intra.pkl")
    if not os.path.exists(p):
        raise RuntimeError(f"Missing required input: {p}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Invalid pickle format: {p}")
    return obj


def _load_taps_strict(layer_dir: str) -> List[Tuple[int, int]]:
    p = os.path.join(layer_dir, "taps_intra.pkl")
    if not os.path.exists(p):
        raise RuntimeError(f"Missing required input: {p}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    taps: List[Tuple[int, int]] = []
    for it in obj:
        a = np.asarray(it).reshape(-1)
        if a.size >= 2:
            taps.append((int(a[0]), int(a[1])))
    return taps


def _draw_lines(polys: List[np.ndarray], size: Tuple[int, int], color: Tuple[int, int, int], th: int, aa: bool) -> np.ndarray:
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


def _draw_taps(img: np.ndarray, taps: List[Tuple[int, int]], color: Tuple[int, int, int], r: int) -> None:
    for (x, y) in taps:
        cv2.circle(img, (int(x), int(y)), int(r), color, thickness=-1, lineType=cv2.LINE_AA)


def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    size = _target_size_px(cfg)

    pen_r = int(getattr(cfg, "pen_radius_px", max(1, int(round(getattr(cfg, "pixels_per_mm", 40) * 0.75)))))
    th    = int(getattr(cfg, "preview_line_thickness_px", 1))
    aa    = bool(getattr(cfg, "preview_antialiased", True))

    palette = _palette(outdir, cfg.color_names, cfg.colors)
    composite = np.full((size[1], size[0], 3), 255, np.uint8)

    for name in cfg.color_names:
        layer_dir = os.path.join(outdir, name)
        os.makedirs(layer_dir, exist_ok=True)

        lines = _load_lines_strict(layer_dir)
        taps  = _load_taps_strict(layer_dir)

        lay = _draw_lines(lines, size, (0, 0, 0), th, aa)
        _draw_taps(lay, taps, (0, 0, 255), pen_r)
        out_l = os.path.join(layer_dir, "preview_intra.png")
        cv2.imwrite(out_l, lay)

        col = palette.get(name, (0, 0, 0))
        layc = _draw_lines(lines, size, col, th, aa)
        _draw_taps(layc, taps, col, pen_r)
        mask = (layc != 255).any(axis=2)
        composite[mask] = layc[mask]

        print(f"[preview_intra] {name}: lines={len(lines)}, taps={len(taps)} → {out_l}")

    out_c = os.path.join(outdir, "preview_intra_composite.png")
    cv2.imwrite(out_c, composite)
    print(f"[preview_intra] composite saved: {out_c}")


if __name__ == "__main__":
    main()
