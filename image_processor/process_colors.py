#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_colors.py  — v1.1.1 (fix)
Разделение изображения на строго one-hot слои + сохранение индексной карты меток.

Выход:
  <out>/labels.png        — карта индексов (uint8, 0..K-1)
  <out>/labels.npy        — та же карта (numpy)
  <out>/palette.json      — фактически использованная палитра (name, rgb, index)
  <out>/layer_<idx>_<name>.png — one-hot маски (0/255)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def kmeans_palette(img_rgb: np.ndarray, k: int, samples: int = 200000, seed: int = 1) -> np.ndarray:
    h, w, _ = img_rgb.shape
    N = h * w
    rs = np.random.RandomState(seed)
    if N > samples:
        idx = rs.choice(N, size=samples, replace=False)
        sample = img_rgb.reshape(-1, 3)[idx].astype(np.float32)
    else:
        sample = img_rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(sample, K=k, bestLabels=None, criteria=criteria, attempts=3, flags=flags)
    return centers.astype(np.uint8)  # RGB


def palette_from_json(path: str) -> Tuple[np.ndarray, List[str]]:
    with open(path, 'r') as f:
        data = json.load(f)

    if 'recommended_colors' in data:
        items = sorted(data['recommended_colors'], key=lambda x: x.get('position', 1e9))
        rgb, names = [], []
        for it in items:
            names.append(str(it.get('name', f"color_{len(names)}")))
            rgb.append(it['rgb'])
        return np.array(rgb, dtype=np.uint8), names
    elif 'palette' in data:
        rgb = [c['rgb'] for c in data['palette']]
        names = [str(c.get('name', f"color_{i}")) for i, _ in enumerate(rgb)]
        return np.array(rgb, dtype=np.uint8), names
    else:
        raise ValueError(f"Unsupported palette JSON structure: {path}")


def assign_labels(img_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = img_rgb.shape
    img_flat = img_rgb.reshape(-1, 3).astype(np.int16)
    pal = palette_rgb.astype(np.int16)  # [K,3]
    diff = img_flat[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist2, axis=1).astype(np.uint8)
    return labels.reshape(h, w)


def default_color_names(k: int) -> List[str]:
    base = ["red", "green", "blue", "black"]
    return [base[i] if i < len(base) else f"color_{i}" for i in range(k)]


def save_labels_png(path: Path, labels: np.ndarray) -> None:
    img = Image.fromarray(labels.astype(np.uint8), mode='L')
    img.save(str(path))


def main():
    ap = argparse.ArgumentParser(description="One-hot color layer generator with labels output")
    ap.add_argument("input", help="Input image")
    ap.add_argument("-o", "--output", default="layers", help="Output directory")
    ap.add_argument("-m", "--mode", choices=["adaptive", "palette"], default="adaptive",
                    help="adaptive: k-means; palette: palette JSON")
    ap.add_argument("-n", "--colors", type=int, default=4, help="Number of colors for adaptive")
    ap.add_argument("--palette", help="Palette JSON (from analyze_colors.py) for mode=palette")
    ap.add_argument("--edges-only", action="store_true", help="Kept for pipeline compatibility (ignored)")
    args = ap.parse_args()

    out_dir = Path(args.output).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[process_colors] v1.1.1")
    print(f"[process_colors] Input: {args.input}")
    print(f"[process_colors] Output dir: {out_dir}")

    img_rgb = load_image_rgb(args.input)
    h, w, _ = img_rgb.shape
    print(f"[process_colors] Size: {w}x{h}")

    # палитра
    if args.mode == "palette":
        if not args.palette:
            raise ValueError("Mode 'palette' requires --palette JSON")
        palette_rgb, names = palette_from_json(args.palette)
        K = len(palette_rgb)
        if args.colors and args.colors != K:
            print(f"[WARN] --colors={args.colors} ignored; palette has {K} entries.")
    else:
        K = int(args.colors) if args.colors else 4
        # FIX: параметр называется k, а не K
        palette_rgb = kmeans_palette(img_rgb, k=K)
        names = default_color_names(K)

    # назначаем метки
    labels = assign_labels(img_rgb, palette_rgb)

    # сводка
    total = labels.size
    print("[process_colors] Class distribution:")
    for i in range(K):
        cnt = int((labels == i).sum())
        p = 100.0 * cnt / total if total else 0.0
        rgb = tuple(int(v) for v in palette_rgb[i])
        nm = names[i] if i < len(names) else f"color_{i}"
        print(f"  [{i}] {nm:12s}  {rgb}  pixels={cnt:8d}  {p:5.1f}%")

    # сохраняем labels
    labels_png = out_dir / "labels.png"
    labels_npy = out_dir / "labels.npy"
    save_labels_png(labels_png, labels)
    np.save(str(labels_npy), labels.astype(np.uint8))
    print(f"[process_colors] Saved labels PNG: {labels_png}")
    print(f"[process_colors] Saved labels NPY: {labels_npy}")

    # сохраняем палитру
    pal_json = out_dir / "palette.json"
    palette_dump = {
        "colors": [
            {"index": int(i),
             "name": (names[i] if i < len(names) else f"color_{i}"),
             "rgb": [int(c) for c in palette_rgb[i].tolist()]}
            for i in range(K)
        ]
    }
    with open(pal_json, "w") as f:
        json.dump(palette_dump, f, indent=2)
    print(f"[process_colors] Saved palette JSON: {pal_json}")

    # one-hot маски
    print("[process_colors] Saving one-hot masks...")
    for i in range(K):
        nm = names[i] if i < len(names) else f"color_{i}"
        mask = (labels == i).astype(np.uint8) * 255
        out_path = out_dir / f"layer_{i+1}_{nm}.png"
        ok = cv2.imwrite(str(out_path), mask)
        if not ok:
            raise RuntimeError(f"Failed to write mask: {out_path}")
    print(f"[process_colors] Done. {K} layer files written to: {out_dir}")

    if args.edges_only:
        print("[process_colors] NOTE: --edges-only is ignored here (kept for pipeline compatibility).")

if __name__ == "__main__":
    main()
