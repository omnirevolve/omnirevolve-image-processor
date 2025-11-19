#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_svg_to_stream_pipeline.py — full pipeline:
    SVG -> G-code -> OmniRevolve plotter stream -> preview.

Assumptions:
  - This script lives in the same directory as:
        svg2gcode.py
        gcode2stream.py
  - Previewer is at:
        ../shared/omnirevolve_plotter_stream_previewer.py

Steps:
  1) Call svg2gcode.py to convert SVG into G-code with automatic A4 scaling.
  2) Call gcode2stream.py to convert G-code into a NEW-protocol stream.
  3) Optionally call the previewer to visualize the result.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_STEPS_PER_MM = 40.0
DEFAULT_PAGE_W_MM = 210.0
DEFAULT_PAGE_H_MM = 297.0
DEFAULT_MARGIN_MM = 10.0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert SVG directly to OmniRevolve plotter stream and preview it."
    )
    ap.add_argument("input", help="Input SVG file")

    ap.add_argument(
        "-o",
        "--output-stream",
        default=None,
        help="Output stream file (.bin). Default: <svg_stem>_stream.bin",
    )
    ap.add_argument(
        "--gcode-output",
        default=None,
        help="Optional G-code output file. "
             "If omitted, <svg_stem>.gcode is used.",
    )

    # Page / SVG scaling parameters (forwarded to svg2gcode.py)
    ap.add_argument(
        "--page-width-mm",
        type=float,
        default=DEFAULT_PAGE_W_MM,
        help=f"Target page width in mm (default: {DEFAULT_PAGE_W_MM}, A4 width).",
    )
    ap.add_argument(
        "--page-height-mm",
        type=float,
        default=DEFAULT_PAGE_H_MM,
        help=f"Target page height in mm (default: {DEFAULT_PAGE_H_MM}, A4 height).",
    )
    ap.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help=f"Margin from page border in mm (default: {DEFAULT_MARGIN_MM}).",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Uniform scale factor SVG-units -> mm (overrides auto-fit).",
    )
    ap.add_argument(
        "--scale-x",
        type=float,
        default=None,
        help="Explicit X scale factor (SVG-units -> mm). Overrides --scale for X.",
    )
    ap.add_argument(
        "--scale-y",
        type=float,
        default=None,
        help="Explicit Y scale factor (SVG-units -> mm). Overrides --scale for Y.",
    )

    # Plotter / stream parameters (forwarded to gcode2stream.py)
    ap.add_argument(
        "--steps-per-mm",
        type=float,
        default=DEFAULT_STEPS_PER_MM,
        help=f"Steps per millimeter for the plotter (default: {DEFAULT_STEPS_PER_MM}).",
    )
    ap.add_argument(
        "--target-width-steps",
        type=int,
        default=None,
        help="Canvas width in steps. "
             "Default: page_width_mm * steps_per_mm (A4 if not changed).",
    )
    ap.add_argument(
        "--target-height-steps",
        type=int,
        default=None,
        help="Canvas height in steps. "
             "Default: page_height_mm * steps_per_mm (A4 if not changed).",
    )
    ap.add_argument(
        "--invert-y",
        type=int,
        default=0,
        help="Y inversion flag for the generator (1/0). Default: 0 (no flip).",
    )
    ap.add_argument(
        "--color-index",
        type=int,
        default=3,
        help="Pen/color index (0..7) for the entire drawing. Default: 3 (black).",
    )
    ap.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Global speed multiplier for all motion dividers. "
             ">1.0 = faster, <1.0 = slower. Default: 1.0.",
    )
    ap.add_argument(
        "--no-reorder",
        action="store_true",
        help="Disable nearest-neighbor reordering of polylines in gcode2stream.",
    )

    # Preview control
    ap.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not run the previewer at the end.",
    )
    ap.add_argument(
        "--preview-render-width",
        type=int,
        default=1200,
        help="Preview render surface width in pixels (default: 1200).",
    )
    ap.add_argument(
        "--preview-render-height",
        type=int,
        default=900,
        help="Preview render surface height in pixels (default: 900).",
    )

    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    svg2gcode = script_dir / "svg2gcode.py"
    gcode2stream = script_dir / "gcode2stream.py"
    previewer = (script_dir / "../shared/omnirevolve_plotter_stream_previewer.py").resolve()

    if not svg2gcode.is_file():
        raise SystemExit(f"svg2gcode.py not found at {svg2gcode}")
    if not gcode2stream.is_file():
        raise SystemExit(f"gcode2stream.py not found at {gcode2stream}")
    if not previewer.is_file() and not args.no_preview:
        raise SystemExit(f"Previewer not found at {previewer}")

    svg_path = Path(args.input)
    if not svg_path.is_file():
        raise SystemExit(f"Input SVG not found: {svg_path}")

    # Determine G-code and stream paths
    if args.gcode_output:
        gcode_path = Path(args.gcode_output)
    else:
        # Put G-code next to SVG: <stem>.gcode
        gcode_path = svg_path.with_suffix(".gcode")

    if args.output_stream:
        stream_path = Path(args.output_stream)
    else:
        # <stem>_stream.bin (cannot use with_suffix for this)
        stream_path = svg_path.with_name(svg_path.stem + "_stream.bin")

    # Compute canvas size in steps (for both generator and previewer)
    if args.target_width_steps is not None and args.target_height_steps is not None:
        canvas_w_steps = args.target_width_steps
        canvas_h_steps = args.target_height_steps
    else:
        canvas_w_steps = int(round(args.page_width_mm * args.steps_per_mm))
        canvas_h_steps = int(round(args.page_height_mm * args.steps_per_mm))

    # ------------------------------------------------------------------
    # 1) SVG -> G-code via svg2gcode.py
    # ------------------------------------------------------------------
    svg_cmd = [
        sys.executable,
        str(svg2gcode),
        str(svg_path),
        "-o",
        str(gcode_path),
        "--page-width-mm",
        str(args.page_width_mm),
        "--page-height-mm",
        str(args.page_height_mm),
        "--margin-mm",
        str(args.margin_mm),
    ]
    if args.scale is not None:
        svg_cmd += ["--scale", str(args.scale)]
    if args.scale_x is not None:
        svg_cmd += ["--scale-x", str(args.scale_x)]
    if args.scale_y is not None:
        svg_cmd += ["--scale-y", str(args.scale_y)]

    print("=== [1/3] SVG -> G-code ===")
    print("Running:", " ".join(svg_cmd))
    subprocess.run(svg_cmd, check=True)

    # ------------------------------------------------------------------
    # 2) G-code -> stream via gcode2stream.py
    # ------------------------------------------------------------------
    stream_cmd = [
        sys.executable,
        str(gcode2stream),
        str(gcode_path),
        "-o",
        str(stream_path),
        "--steps-per-mm",
        str(args.steps_per_mm),
        "--invert-y",
        str(args.invert-y) if False else str(args.invert_y),  # keep normal: args.invert_y
        "--color-index",
        str(args.color_index),
        "--speed-scale",
        str(args.speed_scale),
        "--scale-x",
        "1.0",
        "--scale-y",
        "1.0",
        "--offset-x-mm",
        "0.0",
        "--offset-y-mm",
        "0.0",
        "--target-width-steps",
        str(canvas_w_steps),
        "--target-height-steps",
        str(canvas_h_steps),
    ]
    if args.no_reorder:
        stream_cmd.append("--no-reorder")

    # fix small typo from the inline comment above:
    stream_cmd[stream_cmd.index(str(args.invert-y)) if False else 0] = "dummy"  # this line won't be used
    # Actually we will just rebuild stream_cmd correctly below:

    stream_cmd = [
        sys.executable,
        str(gcode2stream),
        str(gcode_path),
        "-o",
        str(stream_path),
        "--steps-per-mm",
        str(args.steps_per_mm),
        "--invert-y",
        str(args.invert_y),
        "--color-index",
        str(args.color_index),
        "--speed-scale",
        str(args.speed_scale),
        "--scale-x",
        "1.0",
        "--scale-y",
        "1.0",
        "--offset-x-mm",
        "0.0",
        "--offset-y-mm",
        "0.0",
        "--target-width-steps",
        str(canvas_w_steps),
        "--target-height-steps",
        str(canvas_h_steps),
    ]
    if args.no_reorder:
        stream_cmd.append("--no-reorder")

    print("=== [2/3] G-code -> stream ===")
    print("Running:", " ".join(stream_cmd))
    subprocess.run(stream_cmd, check=True)

    print(f"✓ Stream written to {stream_path}")

    # ------------------------------------------------------------------
    # 3) Preview using omnirevolve_plotter_stream_previewer.py
    # ------------------------------------------------------------------
    if not args.no_preview:
        preview_cmd = [
            sys.executable,
            str(previewer),
            str(stream_path),
            "--canvas-w-steps",
            str(canvas_w_steps),
            "--canvas-h-steps",
            str(canvas_h_steps),
            "--invert-y",
            "1",
            "--render-width",
            str(args.preview_render_width),
            "--render-height",
            str(args.preview_render_height),
        ]
        print("=== [3/3] Preview ===")
        print("Running:", " ".join(preview_cmd))
        subprocess.run(preview_cmd, check=True)
    else:
        print("Preview disabled (--no-preview).")


if __name__ == "__main__":
    main()
