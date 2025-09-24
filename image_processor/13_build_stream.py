#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_build_stream.py — build a binary plotter stream from vector_manifest.json.

Manifest uses pixel coordinates with top-left origin.
Stream is emitted in plotter step space with bottom-left origin: stepY = H-1 - pixelY.
Before selecting a color, we always approach the first operation of the layer.

Color remap sources (from the already-loaded pipeline cfg, i.e. config.json):
  • cfg.stream_force_color_index : int 0..7 — force the same color for all layers
  • cfg.stream_color_by_name     : {"layer_dark":3, "layer_mid":1, ...}
  • cfg.stream_color_by_order    : [3,0,1,2] — by layer order
ENV overrides:
  • STREAM_FORCE_COLOR_INDEX
  • STREAM_COLOR_ORDER="3,0,1,2"
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2

from config import load_config, Config

# helpers from ../shared (project: omnirevolve_plotter)
_SHARED_DIR = (Path(__file__).resolve().parent / "../shared").resolve()
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from omnirevolve_plotter_stream_creator_helper import (  # type: ignore
    Config as StreamCfg,
    StreamWriter,
    emit_polyline,
    travel_ramped,
)

# ───────────────────────── helpers ─────────────────────────

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


def _stream_cfg_from_pipeline(cfg: Config) -> StreamCfg:
    # invert_y=True matches plotter step space (bottom-left origin)
    return StreamCfg(
        steps_per_mm=float(getattr(cfg, "pixels_per_mm", 40.0)),
        invert_y=True,
        div_start=int(getattr(cfg, "draw_div_start", 25)),
        div_fast=int(getattr(cfg, "draw_div_fast", 15)),
        profile=str(getattr(cfg, "draw_profile", "triangle")),
        corner_deg=float(getattr(cfg, "corner_deg", 85.0)),
        corner_div=int(getattr(cfg, "corner_div", 30)),
        corner_window_steps=int(getattr(cfg, "corner_window_steps", 800)),
        travel_div_fast=int(getattr(cfg, "travel_div_fast", 10)),
    )


def _ensure_xy(arr) -> np.ndarray:
    pts = np.asarray(arr)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts.reshape(-1, 2)
    return pts.astype(np.float64, copy=False)


def _to_steps(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    xi = max(0, min(W - 1, int(round(x))))
    yi = max(0, min(H - 1, int(round(y))))
    yi = H - 1 - yi  # Y inversion
    return xi, yi

# ───────────────────── color remap (from cfg + ENV) ─────────────────────

def _sanitize_color_idx(x) -> int:
    try:
        return int(x) & 7
    except Exception:
        return 0


def _as_dict_or_none(x):
    return x if isinstance(x, dict) else None


def _as_list_or_none(x):
    return x if isinstance(x, (list, tuple)) and len(x) > 0 else None


def _load_color_maps(cfg: Config):
    """
    Read color mapping rules from the pipeline config and apply ENV overrides.
    """
    force_idx = getattr(cfg, "stream_force_color_index", None)
    if force_idx is not None:
        force_idx = _sanitize_color_idx(force_idx)

    by_name = _as_dict_or_none(getattr(cfg, "stream_color_by_name", None))
    if by_name is not None:
        by_name = {str(k): _sanitize_color_idx(v) for k, v in by_name.items()}

    by_order = _as_list_or_none(getattr(cfg, "stream_color_by_order", None))
    if by_order is not None:
        by_order = [_sanitize_color_idx(v) for v in by_order]

    # ENV overrides
    env_force = os.environ.get("STREAM_FORCE_COLOR_INDEX")
    if env_force is not None:
        try:
            force_idx = _sanitize_color_idx(env_force)
        except Exception:
            pass

    env_order = os.environ.get("STREAM_COLOR_ORDER")
    if env_order:
        try:
            by_order = [_sanitize_color_idx(v) for v in env_order.split(",")]
        except Exception:
            pass

    print(f"[stream] color maps: force={force_idx} by_name={by_name} by_order={by_order}")
    return force_idx, by_name, by_order


def _resolve_color_index(
    layer_name: str,
    orig_idx: int,
    ordinal: int,
    force_idx: int | None,
    map_by_name: dict | None,
    map_by_order: list | None,
) -> int:
    if force_idx is not None:
        return force_idx
    if map_by_name and layer_name in map_by_name:
        return map_by_name[layer_name]
    if map_by_order:
        return map_by_order[ordinal % len(map_by_order)]
    return _sanitize_color_idx(orig_idx)

# ───────────────────── manifest I/O ─────────────────────

def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("coords") not in (None, "pixel_top_left"):
        raise SystemExit("Unsupported coordinates in manifest; expected pixel_top_left")
    return obj


def _load_ops(base: Path, entry: Dict[str, Any]) -> Tuple[str, int, List[Dict[str, Any]]]:
    color_name = str(entry.get("color_name", entry.get("name", "unknown")))
    color_idx = int(entry.get("color_index", 0))
    pkl = base / entry["file"]
    if not pkl.exists():
        raise SystemExit(f"Missing layer file: {pkl}")
    import pickle
    ops = pickle.loads(pkl.read_bytes())
    return color_name, color_idx, ops

# ───────────────────── emission ─────────────────────

def _emit_layer(
    w: StreamWriter,
    ops: List[Dict[str, Any]],
    color_idx: int,
    W: int,
    H: int,
    scfg: StreamCfg,
    cur_x: int,
    cur_y: int,
) -> Tuple[int, int]:
    # Approach the first operation before color select
    if ops:
        first = ops[0]
        if first["type"] == "tap":
            sx, sy = _to_steps(first["x"], first["y"], W, H)
        else:
            q = _ensure_xy(first["points"])
            sx, sy = _to_steps(q[0, 0], q[0, 1], W, H)
        if (cur_x, cur_y) != (sx, sy):
            travel_ramped(w, cur_x, cur_y, sx, sy, scfg)
            cur_x, cur_y = sx, sy

    w.select_color(color_idx)

    for op in ops:
        if op["type"] == "tap":
            tx, ty = _to_steps(op["x"], op["y"], W, H)
            if (cur_x, cur_y) != (tx, ty):
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, tx, ty, scfg)
                cur_x, cur_y = tx, ty
            w.tap()
            continue

        pts = _ensure_xy(op["points"])
        if len(pts) < 2:
            continue
        start = _to_steps(pts[0, 0], pts[0, 1], W, H)
        if (cur_x, cur_y) != start:
            w.pen_up()
            travel_ramped(w, cur_x, cur_y, start[0], start[1], scfg)
            cur_x, cur_y = start
        w.pen_down()
        plist: List[Tuple[int, int]] = [_to_steps(x, y, W, H) for x, y in pts]
        emit_polyline(w, scfg, plist)
        w.pen_up()
        cur_x, cur_y = plist[-1]

    return cur_x, cur_y

# ───────────────────── main ─────────────────────

def main():
    cfg: Config = load_config()
    out = Path(cfg.output_dir)
    W, H = _target_size_px(cfg)
    scfg = _stream_cfg_from_pipeline(cfg)

    man_path = out / "vector_manifest.json"
    if not man_path.exists():
        raise SystemExit(f"Missing manifest: {man_path}")
    man = _load_manifest(man_path)
    ms = man.get("image_size")
    if not (isinstance(ms, (list, tuple)) and len(ms) == 2 and int(ms[0]) == W and int(ms[1]) == H):
        print(f"[stream] WARN: manifest size {ms} != target {W}x{H}")

    force_idx, map_by_name, map_by_order = _load_color_maps(cfg)

    w = StreamWriter()
    w.pen_up()
    cur_x = 0
    cur_y = 0
    total_lines = total_taps = 0

    for ordinal, entry in enumerate(man.get("layers", [])):
        cname, cidx_orig, ops = _load_ops(out, entry)
        cidx = _resolve_color_index(cname, cidx_orig, ordinal, force_idx, map_by_name, map_by_order)

        print(f"[stream] layer#{ordinal+1} '{cname}': color {cidx_orig} → {cidx} | ops={len(ops)}")

        total_lines += sum(1 for o in ops if o["type"] == "line")
        total_taps += sum(1 for o in ops if o["type"] == "tap")
        cur_x, cur_y = _emit_layer(w, ops, cidx, W, H, scfg, cur_x, cur_y)

    data = w.finalize()
    dst = out / "plot_stream.bin"
    dst.write_bytes(data)
    (out / "plot_stream.json").write_text(
        json.dumps(
            {
                "target_steps": {"width": W, "height": H},
                "bytes": len(data),
                "lines": total_lines,
                "taps": total_taps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("✓ Stream saved:", str(dst))
    print("  Size:", len(data), "bytes")
    print("  Lines:", total_lines, "Taps:", total_taps)

if __name__ == "__main__":
    main()
