#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

INCH_TO_MM = 25.4


def strip_comments(line: str) -> str:
    """Remove ';...' and '(...)' comments from a G-code line."""
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


def parse_gcode_paths(gcode_path: Path):
    """
    Parse G-code and return a list of pen-down paths in mm:
        [ [(x1,y1), (x2,y2), ...], ... ]
    """
    lines = gcode_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    x = y = 0.0
    absolute = True
    units_mm = True
    pen_down = False

    paths = []
    cur = []

    def close_path():
        nonlocal cur
        if len(cur) >= 2:
            paths.append(cur)
        cur = []

    for raw in lines:
        line = strip_comments(raw)
        if not line:
            continue

        toks = line.split()
        new_x = None
        new_y = None
        new_pen = None

        for t in toks:
            if len(t) < 2:
                continue
            c = t[0].upper()
            vstr = t[1:]

            # G-codes
            if c == 'G':
                try:
                    g = int(float(vstr))
                except ValueError:
                    continue
                if g == 90:
                    absolute = True
                elif g == 91:
                    absolute = False
                elif g == 21:
                    units_mm = True
                elif g == 20:
                    units_mm = False

            # M-codes -> pen state
            elif c == 'M':
                try:
                    m = int(float(vstr))
                except ValueError:
                    continue
                if m in (3, 4):
                    new_pen = True
                elif m == 5:
                    new_pen = False

            # Coordinates
            elif c in ('X', 'Y'):
                try:
                    v = float(vstr)
                except ValueError:
                    continue
                if not units_mm:
                    v *= INCH_TO_MM
                if c == 'X':
                    new_x = v
                else:
                    new_y = v

        if new_pen is not None and new_pen != pen_down:
            if pen_down and not new_pen:
                close_path()
            pen_down = new_pen

        if new_x is not None or new_y is not None:
            old_x, old_y = x, y

            if absolute:
                if new_x is not None:
                    x = new_x
                if new_y is not None:
                    y = new_y
            else:
                if new_x is not None:
                    x += new_x
                if new_y is not None:
                    y += new_y

            if pen_down:
                if not cur:
                    cur = [(old_x, old_y)]
                cur.append((x, y))

    close_path()
    return paths


def paths_bbox(paths):
    xs = []
    ys = []
    for p in paths:
        for x, y in p:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def write_svg(paths, out_path: Path, scale=3.0, margin_px=10):
    """
    Write paths to a simple SVG file. Y axis goes down (like screen coordinates).
    """
    if not paths:
        out_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'></svg>",
            encoding="utf-8",
        )
        return

    min_x, min_y, max_x, max_y = paths_bbox(paths)
    width_mm = max_x - min_x
    height_mm = max_y - min_y

    width_px = int(width_mm * scale) + 2 * margin_px
    height_px = int(height_mm * scale) + 2 * margin_px

    lines = []
    lines.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{width_px}' height='{height_px}' "
        f"viewBox='0 0 {width_px} {height_px}'>"
    )
    lines.append("<g fill='none' stroke='black' stroke-width='1'>")

    for path in paths:
        pts = []
        for x_mm, y_mm in path:
            x_px = (x_mm - min_x) * scale + margin_px
            y_px = (y_mm - min_y) * scale + margin_px  # Y down
            pts.append(f"{x_px:.2f},{y_px:.2f}")
        d = " ".join(pts)
        lines.append(f"  <polyline points='{d}' />")

    lines.append("</g>")
    lines.append("</svg>")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print("Usage: view_gcode_svg.py file.gcode [out.svg]")
        sys.exit(1)

    gcode_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_svg = Path(sys.argv[2])
    else:
        out_svg = gcode_file.with_suffix(".gcode_view.svg")

    paths = parse_gcode_paths(gcode_file)
    print(f"Pen-down paths: {len(paths)}")

    write_svg(paths, out_svg)
    print(f"SVG written to: {out_svg}")


if __name__ == "__main__":
    main()
