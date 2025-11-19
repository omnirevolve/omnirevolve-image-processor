#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_plotter_gcode_stream_creator.py — G-code → OmniRevolve NEW-protocol stream.

Features:
  - Parses a simple G-code subset (G0/G1, G90/G91, G20/G21, M3/M4/M5, X/Y).
  - Extracts all pen-down polylines in millimeters.
  - Converts them to step coordinates (A4 @ 40 steps/mm by default).
  - Optionally reorders polylines with a nearest-neighbor heuristic.
  - Draws contours using the same emit_polyline() corner-aware engine
    as the original omnirevolve_plotter_stream_creator.py.
  - Supports a global speed scale factor (similar to 3D-printer feed rate):
      * --speed-scale > 1.0 → faster (smaller dividers)
      * --speed-scale < 1.0 → slower (larger dividers)

Defaults:
  - Target page: A4 210x297 mm
  - steps_per_mm: 40
  - canvas: 8400 x 11880 steps
  - single color: index 3 (black)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# A4 @ 40 steps/mm defaults
DEFAULT_A4_W_MM = 210.0
DEFAULT_A4_H_MM = 297.0
DEFAULT_STEPS_PER_MM = 40.0
INCH_TO_MM = 25.4

# Shared helper path (same style as in your existing scripts)
_SHARED_DIR = (Path(__file__).resolve().parent / "../shared").resolve()
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from omnirevolve_plotter_stream_creator_helper import (  # type: ignore
    Config,
    StreamWriter,
    travel_ramped,
    emit_polyline,
)

Point = Tuple[int, int]


# ------------------------- G-code state -------------------------

@dataclass
class GCodeState:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    absolute: bool = True     # G90 / G91
    units_in_mm: bool = True  # G21 / G20
    pen_down: bool = False


# ------------------------- Utilities -------------------------

def clamp_xy(x: int, y: int, wmax: int, hmax: int) -> Point:
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > wmax:
        x = wmax
    if y > hmax:
        y = hmax
    return x, y


def mm_to_steps(
    x_mm: float,
    y_mm: float,
    steps_per_mm: float,
    target_w_steps: int,
    target_h_steps: int,
    invert_y: bool,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Point:
    """
    Map physical coordinates (mm) into step-space.

    Note:
      - invert_y applies a simple flip around the vertical center of the canvas.
      - offsets and scales are applied in mm-space before conversion to steps.
    """
    x_mm_adj = x_mm * scale_x + offset_x_mm
    y_mm_adj = y_mm * scale_y + offset_y_mm

    xs_f = x_mm_adj * steps_per_mm
    ys_f = y_mm_adj * steps_per_mm

    if invert_y:
        ys_f = (target_h_steps - 1) - ys_f

    xs = int(round(xs_f))
    ys = int(round(ys_f))
    xs, ys = clamp_xy(xs, ys, target_w_steps - 1, target_h_steps - 1)
    return xs, ys


def strip_comments(line: str) -> str:
    """
    Strip simple G-code comments:
      - everything after ';'
      - bracket comments: ( ... ) without nesting
    """
    if ';' in line:
        line = line.split(';', 1)[0]

    out = []
    in_paren = False
    for ch in line:
        if ch == '(':
            in_paren = True
            continue
        if ch == ')':
            in_paren = False
            continue
        if not in_paren:
            out.append(ch)
    return ''.join(out).strip()


def parse_gcode_file(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = strip_comments(raw)
        if line:
            lines.append(line)
    return lines


# ---------------------- Nearest-neighbor ordering ----------------------

def _dist_l1(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def order_paths_nearest(paths: List[List[Point]], start_xy: Point) -> List[List[Point]]:
    """
    Reorder polylines to minimize travel distance using a simple
    nearest-neighbor heuristic.
    """
    remaining = [p for p in paths if len(p) >= 2]
    ordered: List[List[Point]] = []
    cur = start_xy

    while remaining:
        best_i = 0
        best_d = 10**18
        for i, p in enumerate(remaining):
            d = _dist_l1(cur, p[0])
            if d < best_d:
                best_d = d
                best_i = i
        chosen = remaining.pop(best_i)
        ordered.append(chosen)
        cur = chosen[-1]

    return ordered


# ---------------------- Core: G-code → polylines (mm) ----------------------

def extract_polylines_mm(
    gcode_path: Path,
) -> Tuple[List[List[Tuple[float, float]]], int]:
    """
    Parse G-code and extract all pen-down polylines in millimeters.

    Returns:
        (paths_mm, pen_down_moves)
    where paths_mm is a list of polylines (each a list of (x_mm, y_mm)).
    """
    lines = parse_gcode_file(gcode_path)
    st = GCodeState()

    paths_mm: List[List[Tuple[float, float]]] = []
    current_path: List[Tuple[float, float]] = []

    def close_current_path():
        nonlocal current_path
        if len(current_path) >= 2:
            paths_mm.append(current_path)
        current_path = []

    pen_down_moves = 0

    for lineno, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            continue

        new_pen_down: Optional[bool] = None
        new_x_val: Optional[float] = None
        new_y_val: Optional[float] = None
        new_z_val: Optional[float] = None

        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            cmd = tok[0].upper()
            val_str = tok[1:]
            if not val_str:
                continue

            # G-codes
            if cmd == 'G':
                try:
                    gcode_num = int(float(val_str))
                except ValueError:
                    continue
                if gcode_num == 90:
                    st.absolute = True
                elif gcode_num == 91:
                    st.absolute = False
                elif gcode_num == 21:
                    st.units_in_mm = True
                elif gcode_num == 20:
                    st.units_in_mm = False

            # M-codes
            elif cmd == 'M':
                try:
                    mnum = int(float(val_str))
                except ValueError:
                    continue
                # Simple mapping:
                #   M3/M4 → pen down
                #   M5    → pen up
                if mnum in (3, 4):
                    new_pen_down = True
                elif mnum == 5:
                    new_pen_down = False

            # Coordinates
            elif cmd in ('X', 'Y', 'Z'):
                try:
                    v = float(val_str)
                except ValueError:
                    continue
                if not st.units_in_mm:
                    v *= INCH_TO_MM
                if cmd == 'X':
                    new_x_val = v
                elif cmd == 'Y':
                    new_y_val = v
                else:
                    new_z_val = v

        # Z is unused here (no Z in your papapishu file), but left for completeness
        if new_z_val is not None:
            st.z_mm = new_z_val
            # If M-code did not explicitly override, infer from Z:
            if new_pen_down is None:
                new_pen_down = (st.z_mm <= 0.0)

        # Apply pen up/down if it changed
        if new_pen_down is not None and new_pen_down != st.pen_down:
            if st.pen_down and not new_pen_down:
                close_current_path()
            st.pen_down = new_pen_down

        # X/Y motion
        move_involved = (new_x_val is not None) or (new_y_val is not None)
        if move_involved:
            old_x, old_y = st.x_mm, st.y_mm

            if st.absolute:
                if new_x_val is not None:
                    st.x_mm = new_x_val
                if new_y_val is not None:
                    st.y_mm = new_y_val
            else:
                if new_x_val is not None:
                    st.x_mm += new_x_val
                if new_y_val is not None:
                    st.y_mm += new_y_val

            if st.pen_down:
                if not current_path:
                    current_path = [(old_x, old_y)]
                current_path.append((st.x_mm, st.y_mm))
                pen_down_moves += 1

    close_current_path()
    return paths_mm, pen_down_moves


# ---------------------- mm polylines → step polylines ----------------------

def convert_polylines_to_steps(
    paths_mm: List[List[Tuple[float, float]]],
    cfg: Config,
    target_w_steps: int,
    target_h_steps: int,
    offset_x_mm: float,
    offset_y_mm: float,
    scale_x: float,
    scale_y: float,
) -> List[List[Point]]:
    out: List[List[Point]] = []

    for poly_mm in paths_mm:
        if len(poly_mm) < 2:
            continue
        step_poly: List[Point] = []
        last: Optional[Point] = None
        for x_mm, y_mm in poly_mm:
            xs, ys = mm_to_steps(
                x_mm,
                y_mm,
                cfg.steps_per_mm,
                target_w_steps,
                target_h_steps,
                cfg.invert_y,
                offset_x_mm,
                offset_y_mm,
                scale_x,
                scale_y,
            )
            if last is None or last != (xs, ys):
                step_poly.append((xs, ys))
                last = (xs, ys)
        if len(step_poly) >= 2:
            out.append(step_poly)

    return out


# ---------------------- Stream generator ----------------------

def generate_stream_from_gcode(
    gcode_path: Path,
    output_file: Path,
    cfg: Config,
    target_w_steps: int,
    target_h_steps: int,
    color_index: int,
    offset_x_mm: float,
    offset_y_mm: float,
    scale_x: float,
    scale_y: float,
    reorder: bool,
) -> None:
    # 1) Extract polylines in millimeters
    paths_mm, pen_moves = extract_polylines_mm(gcode_path)
    print(f"[gcode] File: {gcode_path}")
    print(f"[gcode] Pen-down polylines (mm): {len(paths_mm)}, pen-down moves: {pen_moves}")

    if not paths_mm:
        print("[gcode] No pen-down paths found — emitting empty stream.")
        w_empty = StreamWriter()
        data_empty = w_empty.finalize()
        output_file.write_bytes(data_empty)
        print("✓ Stream saved (empty).")
        return

    # 2) Convert to step-space
    paths_steps = convert_polylines_to_steps(
        paths_mm=paths_mm,
        cfg=cfg,
        target_w_steps=target_w_steps,
        target_h_steps=target_h_steps,
        offset_x_mm=offset_x_mm,
        offset_y_mm=offset_y_mm,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    print(f"[gcode] Step-space polylines: {len(paths_steps)}")

    if not paths_steps:
        print("[gcode] All paths collapsed or out of canvas — empty stream.")
        w_empty = StreamWriter()
        data_empty = w_empty.finalize()
        output_file.write_bytes(data_empty)
        print("✓ Stream saved (empty).")
        return

    # 3) Optional nearest-neighbor reordering
    if reorder:
        ordered_paths = order_paths_nearest(paths_steps, start_xy=(0, 0))
    else:
        ordered_paths = paths_steps

    # 4) Build stream using emit_polyline for drawing
    w = StreamWriter()
    w.pen_up()
    w.set_speed(cfg.div_start)
    w.select_color(color_index)

    cur_x, cur_y = 0, 0

    for path in ordered_paths:
        if len(path) < 2:
            continue

        start = path[0]
        end = path[-1]

        # Travel (pen up) to the start of this polyline
        if (cur_x, cur_y) != start:
            travel_ramped(w, cur_x, cur_y, start[0], start[1], cfg)
            cur_x, cur_y = start

        # Draw the polyline with the corner-aware engine
        w.pen_down()
        emit_polyline(w, cfg, path, color_index=None)
        w.pen_up()
        cur_x, cur_y = end

    data = w.finalize()
    output_file.write_bytes(data)

    print("✓ Stream saved:", str(output_file))
    print("  Size:", len(data), "bytes")
    print("  Paths:", len(ordered_paths))
    print("  Target steps: %d x %d, steps/mm=%.3f" % (target_w_steps, target_h_steps, cfg.steps_per_mm))


# ------------------------------- CLI ---------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert G-code to OmniRevolve NEW-protocol stream (A4 defaults)."
    )
    ap.add_argument("input", help="Input G-code file")
    ap.add_argument(
        "-o",
        "--output",
        default="stream_from_gcode.bin",
        help="Output binary stream file (default: stream_from_gcode.bin)",
    )

    # If not provided, computed as A4 @ steps-per-mm
    ap.add_argument(
        "--target-width-steps",
        type=int,
        default=None,
        help="Canvas width in steps; default: A4_width_mm * steps_per_mm",
    )
    ap.add_argument(
        "--target-height-steps",
        type=int,
        default=None,
        help="Canvas height in steps; default: A4_height_mm * steps_per_mm",
    )

    ap.add_argument(
        "--steps-per-mm",
        type=float,
        default=DEFAULT_STEPS_PER_MM,
        help=f"Steps per millimeter (default: {DEFAULT_STEPS_PER_MM})",
    )
    ap.add_argument(
        "--invert-y",
        type=int,
        default=0,
        help="Invert Y axis in generator (1/0). Default: 0 (no flip).",
    )

    # Affine transform (shift/scale G-code geometry in mm)
    ap.add_argument(
        "--offset-x-mm",
        type=float,
        default=0.0,
        help="X offset in mm (before converting to steps)",
    )
    ap.add_argument(
        "--offset-y-mm",
        type=float,
        default=0.0,
        help="Y offset in mm (before converting to steps)",
    )
    ap.add_argument(
        "--scale-x",
        type=float,
        default=1.0,
        help="Scale factor along X (default: 1.0)",
    )
    ap.add_argument(
        "--scale-y",
        type=float,
        default=1.0,
        help="Scale factor along Y (default: 1.0)",
    )

    # Single pen / color index for entire G-code
    ap.add_argument(
        "--color-index",
        type=int,
        default=3,
        help="Pen/color index (0..7), default: 3 (black)",
    )

    # Motion profile — same fields as Config in helper
    ap.add_argument("--div-start", type=int, default=28)
    ap.add_argument("--div-fast", type=int, default=15)
    ap.add_argument("--profile", choices=["triangle", "scurve"], default="triangle")
    ap.add_argument("--corner-deg", type=float, default=85.0)
    ap.add_argument("--corner-div", type=int, default=28)
    ap.add_argument("--corner-window-steps", type=int, default=300)

    ap.add_argument("--travel-div-fast", type=int, default=10)
    ap.add_argument("--travel-start-div", type=int, default=28)
    ap.add_argument("--travel-window-steps", type=int, default=240)
    ap.add_argument("--travel-quant-step", type=int, default=4)

    ap.add_argument("--short-len-steps", type=int, default=120)
    ap.add_argument("--short-div", type=int, default=16)

    # Global speed scale
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help=(
            "Global speed multiplier for all motion dividers. "
            ">1.0 = faster (smaller dividers), <1.0 = slower. Default: 1.0"
        ),
    )

    # Path reordering
    ap.add_argument(
        "--no-reorder",
        action="store_true",
        help="Disable nearest-neighbor reordering of polylines.",
    )

    return ap


def apply_speed_scale(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply global speed scaling to all divider-related arguments.

    Divider is an inverse speed: smaller divider → faster.
    For speed_scale > 1.0, we reduce divisors; for <1.0, increase.
    """
    scale = float(args.speed_scale)
    if scale <= 0.0:
        raise SystemExit("Error: --speed-scale must be > 0")

    if abs(scale - 1.0) < 1e-6:
        # No scaling needed
        return args

    def scale_div(v: int) -> int:
        new_v = int(round(v / scale))
        if new_v < 1:
            new_v = 1
        return new_v

    args.div_start = scale_div(args.div_start)
    args.div_fast = scale_div(args.div_fast)
    args.corner_div = scale_div(args.corner_div)
    args.short_div = scale_div(args.short_div)

    args.travel_div_fast = scale_div(args.travel_div_fast)
    args.travel_start_div = scale_div(args.travel_start_div)

    # Constraints
    if args.div_start < args.div_fast:
        args.div_start = args.div_fast
    if args.corner_div < args.div_fast:
        args.corner_div = args.div_fast
    if args.short_div < args.div_fast:
        args.short_div = args.div_fast
    if args.travel_start_div < args.travel_div_fast:
        args.travel_start_div = args.travel_div_fast
    if args.div_start < args.travel_div_fast:
        args.div_start = args.travel_div_fast

    return args


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    # Apply global speed scaling to divider-related args
    args = apply_speed_scale(args)

    # A4 defaults if target size not provided
    if args.target_width_steps is None or args.target_height_steps is None:
        target_w_steps = int(round(DEFAULT_A4_W_MM * args.steps_per_mm))
        target_h_steps = int(round(DEFAULT_A4_H_MM * args.steps_per_mm))
    else:
        target_w_steps = args.target_width_steps
        target_h_steps = args.target_height_steps

    # Build Config for travel/drawing
    cfg = Config(
        steps_per_mm=args.steps_per_mm,
        invert_y=bool(args.invert_y),
        div_start=args.div_start,
        div_fast=args.div_fast,
        profile=args.profile,
        corner_deg=args.corner_deg,
        corner_div=args.corner_div,
        corner_window_steps=args.corner_window_steps,
        short_len_steps=args.short_len_steps,
        short_div=args.short_div,
        travel_div_fast=args.travel_div_fast,
        travel_start_div=args.travel_start_div,
        travel_window_steps=args.travel_window_steps,
        travel_quant_step=args.travel_quant_step,
    )

    gcode_path = Path(args.input)
    out_path = Path(args.output)

    generate_stream_from_gcode(
        gcode_path=gcode_path,
        output_file=out_path,
        cfg=cfg,
        target_w_steps=target_w_steps,
        target_h_steps=target_h_steps,
        color_index=args.color_index,
        offset_x_mm=args.offset_x_mm,
        offset_y_mm=args.offset_y_mm,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        reorder=not args.no_reorder,
    )


if __name__ == "__main__":
    main()
