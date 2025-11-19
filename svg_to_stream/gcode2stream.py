#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_plotter_gcode_stream_creator.py — G-code → OmniRevolve NEW-protocol stream.

Uses your shared helper module:
    omnirevolve_plotter_stream_creator_helper.py

Supported G-code subset:
  - G0  : rapid move (pen-up travel_ramped)
  - G1  : linear move
  - G90 : absolute coordinates (default)
  - G91 : relative coordinates
  - G21 : millimeters (default)
  - G20 : inches (converted to mm)
  - M3/M4 : pen down
  - M5    : pen up
  - Z     : Z > 0 → pen up, Z <= 0 → pen down
            (unless explicitly overridden by M-codes)

Defaults:
  - A4 (210x297 mm)
  - 40 steps/mm
  - target canvas: 8400 x 11880 steps
  - single color (color_index = 3, black)
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
    emit_polyline,
    travel_ramped,
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
    current_g: int = 0        # modal: 0 (rapid) or 1 (linear)


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


# ---------------------- Core: G-code → stream ----------------------

def generate_stream_from_gcode(
    gcode_path: Path,
    output_file: Path,
    cfg: Config,
    target_w_steps: int,
    target_h_steps: int,
    color_index: int = 3,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    lines = parse_gcode_file(gcode_path)
    print(f"[gcode] File: {gcode_path}, lines: {len(lines)}")

    st = GCodeState()
    w = StreamWriter()
    w.pen_up()
    w.set_speed(cfg.div_start)
    w.select_color(color_index)

    cur_x_steps, cur_y_steps = mm_to_steps(
        st.x_mm,
        st.y_mm,
        cfg.steps_per_mm,
        target_w_steps,
        target_h_steps,
        cfg.invert_y,
        offset_x_mm,
        offset_y_mm,
        scale_x,
        scale_y,
    )

    current_poly: List[Point] = []

    def flush_poly():
        nonlocal current_poly
        if len(current_poly) >= 2:
            w.pen_down()
            emit_polyline(w, cfg, current_poly, color_index=None)
            w.pen_up()
        current_poly = []

    moves_count = 0

    for lineno, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            continue

        new_g: Optional[int] = None
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
                if gcode_num in (0, 1):
                    new_g = gcode_num
                elif gcode_num == 90:
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
                if cmd == 'X':
                    new_x_val = v
                elif cmd == 'Y':
                    new_y_val = v
                elif cmd == 'Z':
                    new_z_val = v

        # Convert coordinates to mm if units = inches (G20)
        def to_mm(v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            return v if st.units_in_mm else v * INCH_TO_MM

        new_x_val = to_mm(new_x_val)
        new_y_val = to_mm(new_y_val)
        new_z_val = to_mm(new_z_val)

        # Apply modal G0/G1
        if new_g is not None:
            st.current_g = new_g

        # Update Z and optionally pen state from Z
        if new_z_val is not None:
            st.z_mm = new_z_val
            # If M-code did not explicitly override, infer from Z:
            if new_pen_down is None:
                new_pen_down = (st.z_mm <= 0.0)

        # Apply pen up/down if it changed
        if new_pen_down is not None and new_pen_down != st.pen_down:
            if st.pen_down and not new_pen_down:
                flush_poly()
            st.pen_down = new_pen_down

        # X/Y motion
        move_involved = (new_x_val is not None) or (new_y_val is not None)
        if move_involved:
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

            new_x_steps, new_y_steps = mm_to_steps(
                st.x_mm,
                st.y_mm,
                cfg.steps_per_mm,
                target_w_steps,
                target_h_steps,
                cfg.invert_y,
                offset_x_mm,
                offset_y_mm,
                scale_x,
                scale_y,
            )

            if (new_x_steps, new_y_steps) == (cur_x_steps, cur_y_steps):
                continue

            moves_count += 1

            # G0 or pen-up → travel_ramped
            if st.current_g == 0 or not st.pen_down:
                flush_poly()
                w.pen_up()
                travel_ramped(w, cur_x_steps, cur_y_steps, new_x_steps, new_y_steps, cfg)
                cur_x_steps, cur_y_steps = new_x_steps, new_y_steps
            else:
                # G1 with pen-down → accumulate polyline
                if not current_poly:
                    current_poly = [(cur_x_steps, cur_y_steps)]
                current_poly.append((new_x_steps, new_y_steps))
                cur_x_steps, cur_y_steps = new_x_steps, new_y_steps

    # Flush last polyline
    flush_poly()

    data = w.finalize()
    output_file.write_bytes(data)

    print("✓ Stream saved:", str(output_file))
    print("  Size:", len(data), "bytes")
    print("  Moves:", moves_count)
    print(f"  Target steps: {target_w_steps} x {target_h_steps}, steps/mm={cfg.steps_per_mm}")


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
        help="Canvas width in steps; default: A4_w_mm * steps_per_mm",
    )
    ap.add_argument(
        "--target-height-steps",
        type=int,
        default=None,
        help="Canvas height in steps; default: A4_h_mm * steps_per_mm",
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
        help="Invert Y axis (1/0), default: 0",
    )

    # Simple affine transform (if you want to shift/scale the G-code)
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

    # Motion profile — same defaults as in your other scripts
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

    return ap


def main(argv: Optional[list[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.div_start < args.travel_div_fast:
        raise SystemExit("Error: --div-start must be >= --travel-div-fast")

    # A4 defaults if target size not provided
    if args.target_width_steps is None or args.target_height_steps is None:
        target_w_steps = int(round(DEFAULT_A4_W_MM * args.steps_per_mm))
        target_h_steps = int(round(DEFAULT_A4_H_MM * args.steps_per_mm))
    else:
        target_w_steps = args.target_width_steps
        target_h_steps = args.target_height_steps

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
    )


if __name__ == "__main__":
    main()
