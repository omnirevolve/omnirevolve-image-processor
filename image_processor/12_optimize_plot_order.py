#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_optimize_plot_order.py — build drawing order (ops) for each layer.

Input (strict):
  <out>/<layer>/lines_cross.pkl
  <out>/<layer>/taps_cross.pkl

Output:
  <out>/<layer>/ops.pkl  (list of operations with interleaved lines and taps)
  <out>/vector_manifest.json

Routing strategy (per layer):
  • Polylines may be reversed (choose the endpoint closer to the current position).
  • Start with the longest polyline to get a good seed.
  • After each chosen operation, "drain" nearby taps within radius R_insert so we don't come back later.
  • Then pick the next nearest operation (line/tap) and repeat.

Assumes inputs are already cross-deduplicated.
"""

from __future__ import annotations
import os
import json
import pickle
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2

from config import load_config, Config


# ------------------------------ utils ------------------------------

def _target_size_px(cfg: Config) -> Tuple[int, int]:
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 and th > 0:
        return tw, th
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise SystemExit("Cannot infer target size; run step 1.")
    h, w = base.shape[:2]
    return w, h


def _load_cross(layer_dir: str) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    pL = os.path.join(layer_dir, "lines_cross.pkl")
    pT = os.path.join(layer_dir, "taps_cross.pkl")
    if not os.path.exists(pL) or not os.path.exists(pT):
        raise SystemExit(f"Missing cross artifacts in {layer_dir}")
    with open(pL, "rb") as f:
        lines = pickle.load(f)
    with open(pT, "rb") as f:
        taps = []
        for it in pickle.load(f):
            a = np.asarray(it).reshape(-1)
            if a.size >= 2:
                taps.append((int(a[0]), int(a[1])))
    return lines, taps


def _poly_len(pts: np.ndarray) -> float:
    a = np.asarray(pts).reshape(-1, 2).astype(np.float32)
    if a.shape[0] < 2:
        return 0.0
    d = a[1:] - a[:-1]
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


# ------------------------------ optimizer ------------------------------

def _build_ops_for_layer(
    lines: List[np.ndarray],
    taps: List[Tuple[int, int]],
    R_insert: float,
) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []

    line_candidates = []
    for i, c in enumerate(lines):
        p = np.asarray(c).reshape(-1, 2).astype(np.float32)
        if p.shape[0] < 2:
            continue
        line_candidates.append({
            "i": i,
            "points": p,
            "start": (float(p[0, 0]), float(p[0, 1])),
            "end": (float(p[-1, 0]), float(p[-1, 1])),
            "len": _poly_len(p),
        })
    tap_candidates = [{"i": i, "pt": (float(x), float(y))} for i, (x, y) in enumerate(taps)]

    if not line_candidates and not tap_candidates:
        return ops

    pos = (0.0, 0.0)

    if line_candidates:
        s = max(range(len(line_candidates)), key=lambda k: line_candidates[k]["len"])
        first = line_candidates.pop(s)
        if _dist(pos, first["end"]) < _dist(pos, first["start"]):
            first["points"] = first["points"][::-1].copy()
            first["start"], first["end"] = first["end"], first["start"]
        ops.append({"type": "line", "points": first["points"]})
        pos = first["end"]

        kept = []
        for t in tap_candidates:
            if _dist(pos, t["pt"]) <= R_insert:
                ops.append({"type": "tap", "x": int(round(t["pt"][0])), "y": int(round(t["pt"][1]))})
                pos = t["pt"]
            else:
                kept.append(t)
        tap_candidates = kept
    else:
        s = min(range(len(tap_candidates)), key=lambda k: _dist(pos, tap_candidates[k]["pt"]))
        first = tap_candidates.pop(s)
        ops.append({"type": "tap", "x": int(round(first["pt"][0])), "y": int(round(first["pt"][1]))})
        pos = first["pt"]

    while line_candidates or tap_candidates:
        best_kind = None
        best_idx = -1
        best_cost = 1e20
        best_flip = False

        for k in range(len(line_candidates)):
            st = line_candidates[k]["start"]
            en = line_candidates[k]["end"]
            d1 = _dist(pos, st)
            d2 = _dist(pos, en)
            if d1 < best_cost:
                best_cost = d1
                best_kind = "L"
                best_idx = k
                best_flip = False
            if d2 < best_cost:
                best_cost = d2
                best_kind = "L"
                best_idx = k
                best_flip = True
        for k in range(len(tap_candidates)):
            d = _dist(pos, tap_candidates[k]["pt"])
            if d < best_cost:
                best_cost = d
                best_kind = "T"
                best_idx = k
                best_flip = False

        if best_kind == "L":
            cur = line_candidates.pop(best_idx)
            pts = cur["points"]
            if best_flip:
                pts = pts[::-1].copy()
                cur_en = cur["start"]
            else:
                cur_en = cur["end"]
            ops.append({"type": "line", "points": pts})
            pos = cur_en

            kept = []
            for t in tap_candidates:
                if _dist(pos, t["pt"]) <= R_insert:
                    ops.append({"type": "tap", "x": int(round(t["pt"][0])), "y": int(round(t["pt"][1]))})
                    pos = t["pt"]
                else:
                    kept.append(t)
            tap_candidates = kept
        else:
            cur = tap_candidates.pop(best_idx)
            ops.append({"type": "tap", "x": int(round(cur["pt"][0])), "y": int(round(cur["pt"][1]))})
            pos = cur["pt"]

    return ops


# ------------------------------ I/O & main ------------------------------

def main():
    cfg: Config = load_config()
    out = cfg.output_dir
    W, H = _target_size_px(cfg)

    R_insert = float(getattr(cfg, "plotopt_tap_insert_radius_px", max(80.0, getattr(cfg, "pen_width_px", 60))))

    layers = []
    for name in cfg.color_names:
        layer_dir = os.path.join(out, name)
        os.makedirs(layer_dir, exist_ok=True)
        lines, taps = _load_cross(layer_dir)
        ops = _build_ops_for_layer(lines, taps, R_insert)

        p_ops = os.path.join(layer_dir, "ops.pkl")
        with open(p_ops, "wb") as f:
            pickle.dump(ops, f)

        if "dark" in name:
            color_idx = 3
        elif "skin" in name:
            color_idx = 0
        elif "mid" in name:
            color_idx = 1
        elif "light" in name:
            color_idx = 2
        else:
            color_idx = 0

        layers.append({
            "name": name,
            "color_name": name,
            "color_index": color_idx,
            "file": os.path.relpath(p_ops, out),
            "count_ops": len(ops),
        })
        nL = sum(1 for o in ops if o["type"] == "line")
        nT = sum(1 for o in ops if o["type"] == "tap")
        print(f"[plot-opt] {name}: ops={len(ops)} (lines={nL}, taps={nT})")

    manifest = {"image_size": [W, H], "layers": layers, "coords": "pixel_top_left"}
    with open(os.path.join(out, "vector_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[plot-opt] manifest saved: {os.path.join(out, 'vector_manifest.json')}")

if __name__ == "__main__":
    main()
