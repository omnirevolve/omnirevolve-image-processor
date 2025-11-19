#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_svg_to_gcode.py — SVG → G-code converter for pen plotter / laser.

Pipeline:
- Read SVG, inspect its viewBox / width / height.
- Compute a default scale so that the SVG fits into an A4 page (210×297 mm)
  with a configurable margin, keeping aspect ratio.
- Optionally override that scale via --scale / --scale-x / --scale-y.
- Ensure the <svg> root element has width/height attributes so that
  svg-to-gcode does not crash on "None.isnumeric()".
- Use svg-to-gcode to generate "raw" G-code (in SVG units).
- Post-process the G-code: scale and offset all X/Y coordinates according
  to the chosen scale and margin.

Dependencies:
    pip install svg-to-gcode
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import xml.etree.ElementTree as ET

from svg_to_gcode.svg_parser import parse_file
from svg_to_gcode.compiler import Compiler, interfaces


# ---------------- SVG helpers ----------------


def _parse_float(s: str) -> Optional[float]:
    """Parse a float from a CSS length like '123', '123px', '210mm'. Unit is ignored."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    m = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([a-zA-Z%]*)$", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def read_svg_geometry(svg_path: Path) -> Tuple[float, float, float, float]:
    """
    Read svg width/height and viewBox.

    Returns:
        (min_x, min_y, width_units, height_units)
    where all values are in "SVG units". Units are *not* converted to mm;
    we only care about relative scaling.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespace handling (strip it if present)
    tag = root.tag
    if "}" in tag:
        _, bare = tag.split("}", 1)
    else:
        bare = tag
    if bare.lower() != "svg":
        raise ValueError(f"Root element is not <svg>: {tag}")

    view_box_raw = root.get("viewBox") or root.get("viewbox")
    if view_box_raw:
        parts = view_box_raw.replace(",", " ").split()
        if len(parts) == 4:
            min_x = float(parts[0])
            min_y = float(parts[1])
            width_units = float(parts[2])
            height_units = float(parts[3])
        else:
            # Fallback to width/height
            min_x = 0.0
            min_y = 0.0
            width_units = _parse_float(root.get("width") or "100") or 100.0
            height_units = _parse_float(root.get("height") or "100") or 100.0
    else:
        min_x = 0.0
        min_y = 0.0
        width_units = _parse_float(root.get("width") or "100") or 100.0
        height_units = _parse_float(root.get("height") or "100") or 100.0

    if width_units <= 0 or height_units <= 0:
        raise ValueError("Bad SVG dimensions: width/height must be > 0")

    return min_x, min_y, width_units, height_units


def ensure_svg_has_size(
    svg_path: Path,
    width_units: float,
    height_units: float,
) -> Path:
    """
    Make sure the <svg> root element has width/height attributes,
    because svg-to-gcode's parse_file() expects them and crashes
    if height is missing.

    If width/height are present, return the original svg_path.
    Otherwise, create a temporary SVG file with integer width/height
    and return its path.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    has_width = bool(root.get("width"))
    has_height = bool(root.get("height"))

    if has_width and has_height:
        return svg_path  # nothing to fix

    if not has_width:
        root.set("width", str(int(round(width_units))))
    if not has_height:
        root.set("height", str(int(round(height_units))))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
    tmp_path = Path(tmp.name)
    tmp.close()

    tree.write(tmp_path, encoding="utf-8")
    return tmp_path


# ---------------- G-code helpers ----------------

_COORD_RE = re.compile(r"([XY])([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")


def scale_and_offset_gcode(
    text: str,
    sx: float,
    sy: float,
    offset_x: float,
    offset_y: float,
) -> str:
    """
    Multiply all X/Y coordinates by sx/sy and add offsets.
    Works for absolute G0/G1 style G-code.
    """

    def repl(m: re.Match) -> str:
        axis = m.group(1)
        val = float(m.group(2))
        if axis == "X":
            new_val = val * sx + offset_x
        else:
            new_val = val * sy + offset_y
        # 4 decimal places is usually enough for plotters/lasers
        return f"{axis}{new_val:.4f}"

    out_lines = []
    for line in text.splitlines():
        # Fast path: only touch lines that mention X or Y
        if "X" in line or "Y" in line:
            new_line = _COORD_RE.sub(repl, line)
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


# ---------------- CLI ----------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert SVG to G-code with automatic A4 scaling (via svg-to-gcode)."
    )
    ap.add_argument("input", help="Input SVG file")
    ap.add_argument(
        "-o",
        "--output",
        default="from_svg.gcode",
        help="Output G-code file (default: from_svg.gcode)",
    )

    # Machine / motion parameters
    ap.add_argument(
        "--movement-speed",
        type=float,
        default=8000.0,
        help="Rapid move speed (mm/min), default: 8000",
    )
    ap.add_argument(
        "--cutting-speed",
        type=float,
        default=2000.0,
        help="Drawing / cutting speed (mm/min), default: 2000",
    )
    ap.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Number of passes over the same paths, default: 1",
    )
    ap.add_argument(
        "--pass-depth",
        type=float,
        default=0.0,
        help="Depth per pass along Z (mm). For a pen plotter this is usually 0.0.",
    )

    # Page geometry (defaults: A4 portrait with small margin)
    ap.add_argument(
        "--page-width-mm",
        type=float,
        default=210.0,
        help="Target page width in mm (default: 210, i.e. A4 width).",
    )
    ap.add_argument(
        "--page-height-mm",
        type=float,
        default=297.0,
        help="Target page height in mm (default: 297, i.e. A4 height).",
    )
    ap.add_argument(
        "--margin-mm",
        type=float,
        default=10.0,
        help="Margin from page border in mm (default: 10). "
             "SVG content is placed inside [margin, page - margin].",
    )

    # Scaling overrides
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Uniform scale factor (overrides automatic A4 fit). "
             "If set, --scale-x/--scale-y are ignored unless specified explicitly.",
    )
    ap.add_argument(
        "--scale-x",
        type=float,
        default=None,
        help="Explicit X scale factor (SVG units → mm). Overrides --scale for X.",
    )
    ap.add_argument(
        "--scale-y",
        type=float,
        default=None,
        help="Explicit Y scale factor (SVG units → mm). Overrides --scale for Y.",
    )

    return ap


# ---------------- Main ----------------


def main(argv: list[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    svg_path = Path(args.input)
    out_path = Path(args.output)

    if not svg_path.is_file():
        raise SystemExit(f"Input SVG not found: {svg_path}")

    # 1) Read SVG geometry (in SVG units)
    min_x, min_y, svg_w_units, svg_h_units = read_svg_geometry(svg_path)

    # 1a) Ensure svg-to-gcode sees width/height in the root element
    fixed_svg_path = ensure_svg_has_size(svg_path, svg_w_units, svg_h_units)

    # 2) Compute automatic scale to fit page (A4 by default)
    avail_w_mm = max(1e-6, args.page_width_mm - 2.0 * args.margin_mm)
    avail_h_mm = max(1e-6, args.page_height_mm - 2.0 * args.margin_mm)

    auto_scale_x = avail_w_mm / svg_w_units
    auto_scale_y = avail_h_mm / svg_h_units
    auto_uniform = min(auto_scale_x, auto_scale_y)

    # Start from automatic uniform scale
    sx = auto_uniform
    sy = auto_uniform

    # Apply user overrides
    if args.scale is not None:
        sx = sy = args.scale
    if args.scale_x is not None:
        sx = args.scale_x
    if args.scale_y is not None:
        sy = args.scale_y

    # Offsets: map SVG (min_x, min_y) → (margin, margin)
    # new_x = (x * sx) + offset_x; we want x = min_x → margin
    # so offset_x = margin - min_x * sx
    offset_x = args.margin_mm - min_x * sx
    offset_y = args.margin_mm - min_y * sy

    # 3) Use svg-to-gcode to compile raw (unscaled) G-code into a temp file
    #    Note: svg-to-gcode interprets coordinates in SVG units; we rescale later.
    curves = parse_file(str(fixed_svg_path))

    # If we created a temporary fixed SVG, we can safely remove it now
    if fixed_svg_path != svg_path:
        try:
            fixed_svg_path.unlink(missing_ok=True)
        except Exception:
            pass

    gcode_compiler = Compiler(
        interfaces.Gcode,
        movement_speed=args.movement_speed,
        cutting_speed=args.cutting_speed,
        pass_depth=args.pass_depth,
    )

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".gcode", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Compile SVG → raw G-code (SVG units)
    gcode_compiler.append_curves(curves)
    gcode_compiler.compile_to_file(str(tmp_path), passes=args.passes)

    raw_text = tmp_path.read_text(encoding="utf-8")
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 4) Post-process G-code: scale and offset X/Y
    scaled_text = scale_and_offset_gcode(
        raw_text,
        sx=sx,
        sy=sy,
        offset_x=offset_x,
        offset_y=offset_y,
    )

    out_path.write_text(scaled_text, encoding="utf-8")

    # 5) Report
    print(f"✓ G-code saved to {out_path}")
    print(f"  Source SVG: {svg_path}")
    print(f"  SVG units: width={svg_w_units:.3f}, height={svg_h_units:.3f}, "
          f"min_x={min_x:.3f}, min_y={min_y:.3f}")
    print(f"  Page: {args.page_width_mm} × {args.page_height_mm} mm, "
          f"margin={args.margin_mm} mm")
    print(f"  Auto scale (fit): sx={auto_scale_x:.5f}, sy={auto_scale_y:.5f}, "
          f"uniform={auto_uniform:.5f}")
    print(f"  Final scale: sx={sx:.5f}, sy={sy:.5f}")
    print(f"  Offsets: offset_x={offset_x:.3f} mm, offset_y={offset_y:.3f} mm")
    print(f"  movement_speed={args.movement_speed} mm/min, "
          f"cutting_speed={args.cutting_speed} mm/min, "
          f"passes={args.passes}, pass_depth={args.pass_depth} mm")


if __name__ == "__main__":
    main()
