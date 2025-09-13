#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xyplotter_stream_creator.py — build a single NEW-protocol stream from vector layers.

Reads a vector manifest (JSON) with per-layer pickles, converts them into
plotter steps, and emits a binary stream using the shared motion/encoding
engine from xyplotter_stream_creator_helper.py.

Key points:
- Pen-up travels use fixed-window accel/decel (configurable).
- Pen-down polylines are corner-aware (slow-in/out at sharp angles).
- Service bytes (speed/pen/color/EOF) and step packing are handled by the helper.
"""

from __future__ import annotations
import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from xyplotter_stream_creator_helper import (
    Config,
    StreamWriter,
    emit_polyline,
    travel_ramped,
)

# ------------------------------- Data models -------------------------------

@dataclass
class LayerInfo:
    color_name: str
    color_index: int
    contours_steps: List[np.ndarray]          # list of (N,2) int arrays
    taps_steps: List[Tuple[int, int]]         # list of (x,y) in steps
    total_length_steps: float


# ------------------------------- Utilities ---------------------------------

def _ensure_xy(contour) -> np.ndarray:
    """Normalize OpenCV-like contours to (N,2) float64."""
    pts = np.asarray(contour)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)
    return pts.astype(np.float64, copy=False)


def _finalize_point(x: float, y: float, invert_y: bool, tgt_w: int, tgt_h: int) -> Tuple[int, int]:
    """To integer steps inside canvas; optional Y inversion for top-left previews."""
    xi = int(round(x))
    yi = int(round(y))
    if invert_y:
        yi = tgt_h - 1 - yi
    xi = max(0, min(tgt_w - 1, xi))
    yi = max(0, min(tgt_h - 1, yi))
    return xi, yi


def _contour_to_steps(contour: np.ndarray, invert_y: bool, tgt_w: int, tgt_h: int) -> np.ndarray:
    pts = _ensure_xy(contour)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    out = np.empty((len(pts), 2), dtype=np.int32)
    for i, (x, y) in enumerate(pts):
        out[i, 0], out[i, 1] = _finalize_point(x, y, invert_y, tgt_w, tgt_h)
    return out


def _load_vector_layers(manifest_path: Path, invert_y: bool, target_w_steps: int, target_h_steps: int) -> List[LayerInfo]:
    """Load layers from manifest; each layer file is a pickle with 'contours' and 'taps'."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mw, mh = map(int, manifest.get("image_size", [target_w_steps, target_h_steps]))
    if (mw, mh) != (target_w_steps, target_h_steps):
        raise ValueError(f"Canvas mismatch: manifest {mw}x{mh} != target {target_w_steps}x{target_h_steps}")

    base_dir = manifest_path.parent
    layers: List[LayerInfo] = []

    for entry in manifest.get("layers", []):
        layer_file = base_dir / entry["file"]
        if not layer_file.exists():
            print(f"Warning: missing layer file: {layer_file}")
            continue

        layer_data = pickle.loads(layer_file.read_bytes())

        contours_steps: List[np.ndarray] = []
        total_len = 0.0
        for item in layer_data.get("contours", []):
            cs = _contour_to_steps(_ensure_xy(item["points"]), invert_y, target_w_steps, target_h_steps)
            if cs.shape[0] >= 2:
                contours_steps.append(cs)
                d = np.diff(cs.astype(np.float64), axis=0)
                total_len += float(np.sum(np.hypot(d[:, 0], d[:, 1])))

        taps_steps: List[Tuple[int, int]] = []
        for t in layer_data.get("taps", []):
            tx, ty = _finalize_point(float(t["x"]), float(t["y"]), invert_y, target_w_steps, target_h_steps)
            taps_steps.append((tx, ty))

        layers.append(LayerInfo(
            color_name=str(layer_data.get("color_name", entry.get("color_name", "unknown"))),
            color_index=int(layer_data.get("color_idx", entry.get("color_index", 0))),
            contours_steps=contours_steps,
            taps_steps=taps_steps,
            total_length_steps=total_len,
        ))

    # Draw layers in deterministic color order
    layers.sort(key=lambda L: L.color_index)
    return layers


# ------------------------------- Core logic --------------------------------

def _generate_stream(manifest_path: Path, output_file: Path,
                     target_w_steps: int, target_h_steps: int, cfg: Config) -> None:
    w = StreamWriter()
    w.pen_up()

    layers = _load_vector_layers(manifest_path, cfg.invert_y, target_w_steps, target_h_steps)

    cur_x, cur_y = 0, 0
    total_contours = sum(len(L.contours_steps) for L in layers)
    total_taps = sum(len(L.taps_steps) for L in layers)

    for L in layers:
        # Pre-travel to the first element of the layer (optional), then select color
        first_xy: Optional[Tuple[int, int]] = None
        if L.contours_steps:
            first_xy = (int(L.contours_steps[0][0, 0]), int(L.contours_steps[0][0, 1]))
        elif L.taps_steps:
            first_xy = (int(L.taps_steps[0][0]), int(L.taps_steps[0][1]))
        if first_xy and (cur_x, cur_y) != first_xy:
            travel_ramped(w, cur_x, cur_y, first_xy[0], first_xy[1], cfg)
            cur_x, cur_y = first_xy

        w.select_color(L.color_index)

        # Contours (corner-aware)
        for cs in L.contours_steps:
            start = (int(cs[0, 0]), int(cs[0, 1]))
            if (cur_x, cur_y) != start:
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, start[0], start[1], cfg)
                cur_x, cur_y = start
            w.pen_down()
            pts = [(int(x), int(y)) for x, y in cs]
            emit_polyline(w, cfg, pts, color_index=None)
            w.pen_up()
            cur_x, cur_y = pts[-1]

        # Taps
        for (tx, ty) in L.taps_steps:
            if (cur_x, cur_y) != (tx, ty):
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, tx, ty, cfg)
                cur_x, cur_y = tx, ty
            w.tap()

    data = w.finalize()
    output_file.write_bytes(data)

    meta = {
        "target_steps": {"width": target_w_steps, "height": target_h_steps},
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "stats": {"layers": len(layers), "contours": total_contours, "taps": total_taps},
        "bytes": len(data),
        "manifest": str(manifest_path),
    }
    output_file.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✓ Stream saved:", str(output_file))
    print("  Size:", len(data), "bytes")
    print("  Layers:", len(layers), "Contours:", total_contours, "Taps:", total_taps)


# ---------------------------------- CLI ------------------------------------

def _locate_manifest(arg: str) -> Path:
    p = Path(arg)
    if p.is_file() and p.name == "vector_manifest.json":
        return p
    if p.is_dir():
        cand = p / "vector_manifest.json"
        if cand.exists():
            return cand
        cand = p / "vector_data" / "vector_manifest.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Cannot find vector_manifest.json in {arg}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a NEW-protocol binary stream from vector layers (color-batched)."
    )
    ap.add_argument("input", help="Path to vector_manifest.json or its parent directory")
    ap.add_argument("-o", "--output", default="stream.bin")
    ap.add_argument("--target-width-steps", type=int, required=True)
    ap.add_argument("--target-height-steps", type=int, required=True)

    # Motion/kinematics (aligned with xyplotter_stream_creator_helper.Config)
    ap.add_argument("--steps-per-mm", type=float, default=40.0)
    ap.add_argument("--invert-y", type=int, default=1)

    # Drawing (pen-down) profile
    ap.add_argument("--div-start", type=int, default=64,
                    help="Start/stop divider (slow end) for drawing/travel")
    ap.add_argument("--div-fast", type=int, default=20,
                    help="Drawing cruise divider")
    ap.add_argument("--profile", choices=["triangle", "scurve"], default="triangle")
    ap.add_argument("--corner-deg", type=float, default=85.0)
    ap.add_argument("--corner-div", type=int, default=64,
                    help="Divider near sharp corners and as accel/decel endpoint")
    ap.add_argument("--corner-window-steps", type=int, default=800,
                    help="Fixed accel/decel window length in steps (~2 cm @ 40 steps/mm)")

    # Pen-up travel cruise
    ap.add_argument("--travel-div-fast", type=int, default=10,
                    help="Pen-up cruise divider (must be <= div-start)")

    args = ap.parse_args()

    mp = _locate_manifest(args.input)

    # Validate dividers for travel engine
    if args.div_start < args.travel_div_fast:
        raise SystemExit("Error: --div-start must be >= --travel-div-fast")

    cfg = Config(
        steps_per_mm=args.steps_per_mm,
        invert_y=bool(args.invert_y),
        div_start=args.div_start,
        div_fast=args.div_fast,
        profile=args.profile,
        travel_div_fast=args.travel_div_fast,
        corner_deg=args.corner_deg,
        corner_div=args.corner_div,
        corner_window_steps=args.corner_window_steps,
    )

    _generate_stream(mp, Path(args.output), args.target_width_steps, args.target_height_steps, cfg)


if __name__ == "__main__":
    main()
