#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-based multi-layer edge extractor with cross-layer deduplication + fallback.

- Граница назначается классу с меньшей яркостью (темнее) по palette.json.
- Фильтры шума: min-area, min-length, close-gaps, smooth.
- --thicken: утолщение (dilate) тонких краёв перед фильтрацией.
- Подробная статистика по слоям.
- Авто-фолбэк: если слишком мало пикселей после фильтрации, ослабляем фильтры.

Совместимость: принимает -c/--config (игнорируется).
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


def load_labels(layers_dir: Path) -> np.ndarray:
    png = layers_dir / "labels.png"
    npy = layers_dir / "labels.npy"
    if png.exists():
        lab = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if lab is None:
            raise ValueError(f"Cannot read labels.png at {png}")
        return lab.astype(np.int32)
    if npy.exists():
        return np.load(str(npy)).astype(np.int32)
    raise FileNotFoundError(f"labels.png / labels.npy not found in {layers_dir}")


def load_palette(layers_dir: Path, k: int) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    names = [f"color_{i}" for i in range(k)]
    rgbs  = [(0,0,0)] * k
    pal = layers_dir / "palette.json"
    if pal.exists():
        with open(pal, "r") as f:
            data = json.load(f)
        if "colors" in data:
            for item in data["colors"]:
                idx = int(item.get("index", 0))
                if 0 <= idx < k:
                    names[idx] = str(item.get("name", names[idx]))
                    rgb = item.get("rgb", [0,0,0])
                    rgbs[idx] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    else:
        base_names = ["red", "green", "blue", "black"]
        base_rgbs  = [(255,0,0), (0,255,0), (0,0,255), (0,0,0)]
        for i in range(min(k,4)):
            names[i] = base_names[i]
            rgbs[i]  = base_rgbs[i]
    return names, rgbs


def luminance(rgb: Tuple[int,int,int]) -> float:
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b


def assign_edges_to_darker(labels: np.ndarray,
                           lumas: np.ndarray,
                           close_gaps: bool,
                           gap_kernel: int,
                           thicken: int) -> List[np.ndarray]:
    """Строит границы между классами и назначает их более тёмному классу."""
    h, w = labels.shape
    k = int(labels.max()) + 1 if labels.size else 0
    edges = [np.zeros((h, w), np.uint8) for _ in range(k)]

    # горизонтальные пары
    L = labels[:, :-1]
    R = labels[:, 1:]
    diff = (L != R)
    if np.any(diff):
        take_L = (lumas[L] <= lumas[R]) & diff
        take_R = (~(lumas[L] <= lumas[R])) & diff
        if np.any(take_L):
            ys, xs = np.where(take_L)
            cls = L[take_L]
            for c in range(k):
                m = (cls == c)
                if np.any(m):
                    edges[c][ys[m], xs[m]] = 255
        if np.any(take_R):
            ys, xs = np.where(take_R)
            cls = R[take_R]
            for c in range(k):
                m = (cls == c)
                if np.any(m):
                    edges[c][ys[m], xs[m] + 1] = 255

    # вертикальные пары
    T = labels[:-1, :]
    B = labels[1:, :]
    diff = (T != B)
    if np.any(diff):
        take_T = (lumas[T] <= lumas[B]) & diff
        take_B = (~(lumas[T] <= lumas[B])) & diff
        if np.any(take_T):
            ys, xs = np.where(take_T)
            cls = T[take_T]
            for c in range(k):
                m = (cls == c)
                if np.any(m):
                    edges[c][ys[m], xs[m]] = 255
        if np.any(take_B):
            ys, xs = np.where(take_B)
            cls = B[take_B]
            for c in range(k):
                m = (cls == c)
                if np.any(m):
                    edges[c][ys[m] + 1, xs[m]] = 255

    if close_gaps and gap_kernel > 1:
        ker = np.ones((gap_kernel, gap_kernel), np.uint8)
        for i in range(k):
            if edges[i].any():
                edges[i] = cv2.morphologyEx(edges[i], cv2.MORPH_CLOSE, ker)

    if thicken > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thicken, thicken))
        for i in range(k):
            if edges[i].any():
                edges[i] = cv2.dilate(edges[i], ker, iterations=1)

    return edges


def filter_edges(edges: List[np.ndarray],
                 min_area: int,
                 min_length: int,
                 smooth: int) -> List[np.ndarray]:
    out = []
    for e in edges:
        if not e.any():
            out.append(e)
            continue
        work = e.copy()

        if smooth > 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth, smooth))
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, ker)

        if min_area > 0:
            n, lab, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), connectivity=8)
            keep = np.zeros_like(work)
            for i in range(1, n):  # 0 — фон
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == i] = 255
            work = keep

        if min_length > 0 and work.any():
            cnts, _ = cv2.findContours(work, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            canvas = np.zeros_like(work)
            for c in cnts:
                if cv2.arcLength(c, False) >= float(min_length):
                    cv2.drawContours(canvas, [c], -1, 255, 1)
            work = canvas

        out.append(work)
    return out


def save_edges(out_dir: Path, edges: List[np.ndarray], names: List[str]) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, e in enumerate(edges, start=1):
        name = names[i-1] if i-1 < len(names) and names[i-1] else f"color_{i-1}"
        path = out_dir / f"edges_layer_{i}_{name}.png"
        cv2.imwrite(str(path), e)
        paths.append(str(path))
    return paths


def summarize(edges: List[np.ndarray], names: List[str]):
    total = 0
    for i, e in enumerate(edges):
        cnt = int((e > 0).sum())
        total += cnt
        print(f"[edges] Layer {i}: {names[i] if i < len(names) else i} → pixels={cnt}")
    print(f"[edges] Total edge pixels: {total}")
    return total


def main():
    ap = argparse.ArgumentParser(description="Label-based edge extraction with dedup to the darker class (with fallback)")
    ap.add_argument("input", help="Directory with labels.png / labels.npy and palette.json")
    ap.add_argument("-o", "--output", default="edges", help="Output directory")

    # шумоподавление (помягче дефолты)
    ap.add_argument("--min-area", type=int, default=80, help="Drop tiny components (< pixels)")
    ap.add_argument("--min-length", type=int, default=40, help="Drop short contours (< perimeter in px)")
    ap.add_argument("--close-gaps", type=int, choices=[0,1], default=1, help="Morph CLOSE after building edges")
    ap.add_argument("--gap", type=int, default=3, help="Kernel size for gap closing")
    ap.add_argument("--smooth", type=int, default=0, help="Morph OPEN kernel (0 disables)")
    ap.add_argument("--thicken", type=int, default=0, help="Dilate kernel size before filtering (0 disables)")

    ap.add_argument("--composite", action="store_true", help="Also write edges_composite.png")

    # совместимость с оркестратором
    ap.add_argument("-c", "--config", default=None, help="(ignored) config file")
    args = ap.parse_args()

    layers_dir = Path(args.input)
    out_dir = Path(args.output)

    if args.config:
        print(f"[edges] Note: config '{args.config}' is ignored here.")

    print("[edges] Loading labels...")
    labels = load_labels(layers_dir)
    h, w = labels.shape
    k = int(labels.max()) + 1 if labels.size else 0
    print(f"[edges] Size: {w}x{h}, classes: {k}")

    names, rgbs = load_palette(layers_dir, k)
    lumas = np.array([luminance(rgb) for rgb in rgbs], dtype=np.float32)
    print("[edges] Classes:", ", ".join(f"{i}:{names[i]}" for i in range(k)))
    print("[edges] Luminance:", ", ".join(f"{i}:{lumas[i]:.1f}" for i in range(k)))

    print(f"[edges] Build → darker | close_gaps={args.close_gaps}, gap={args.gap}, thicken={args.thicken}")
    edges = assign_edges_to_darker(labels, lumas, bool(args.close_gaps), max(1, int(args.gap)), max(0, int(args.thicken)))
    print("[edges] Raw summary:")
    raw_total = summarize(edges, names)

    print(f"[edges] Filter: min_area={args.min_area}, min_length={args.min_length}, smooth={args.smooth}")
    filtered = filter_edges(edges, max(0, args.min_area), max(0, args.min_length), max(0, args.smooth))
    print("[edges] After filter summary:")
    filt_total = summarize(filtered, names)

    # авто-фолбэк, если «всё пропало»
    min_keep = max(200, int(0.0005 * w * h))  # ~0.05% пикселей или хотя бы 200
    if filt_total < min_keep and raw_total > 0:
        print(f"[edges] WARNING: too few edge pixels after filtering ({filt_total} < {min_keep}). "
              f"Applying fallback: no close_gaps, no smooth, min-area=0, min-length=0, thicken=max(1,thicken).")
        edges_fb = assign_edges_to_darker(labels, lumas, False, 1, max(1, int(args.thicken)))
        filtered = edges_fb  # без фильтрации вовсе
        print("[edges] Fallback summary:")
        summarize(filtered, names)

    print("[edges] Saving layers...")
    paths = save_edges(out_dir, filtered, names)
    print(f"[edges] Done: {len(paths)} files → {out_dir}")

    if args.composite and paths:
        comp = np.full((h, w, 3), 255, np.uint8)
        for p, col in zip(paths, rgbs):
            e = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if e is None:
                continue
            mask = e > 127
            comp[mask] = col
        comp_path = out_dir / "edges_composite.png"
        cv2.imwrite(str(comp_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"[edges] Composite: {comp_path}")

    print("[edges] OK.")


if __name__ == "__main__":
    main()