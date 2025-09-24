#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo stream generator (A4 @ 40 steps/mm).

Idle-travel optimizations:
- Parts of shapes are batched by color (quarter of circle/polygon, sine periods).
- Inside a color group, paths are ordered by nearest neighbor with possible
  stroke reversal (if the end is closer to the current position).
- Color is selected ONCE at the start of a group: first approach, then select_color.

Palette: 0=R, 1=G, 2=B, 3=K.
"""

from dataclasses import dataclass, replace
from math import sin, cos, tau
from pathlib import Path
from typing import List, Tuple, Iterable
import argparse
import json

from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

# helpers (protocol/profiles)
from xyplotter_stream_creator_helper import (
    Config as HWConfig,
    StreamWriter,
    travel_ramped,
)

# ---------- Canvas constants ----------
STEPS_PER_MM = 40.0
A4_W_MM, A4_H_MM = 210.0, 297.0
CANVAS_W = int(round(A4_W_MM * STEPS_PER_MM))   # 8400
CANVAS_H = int(round(A4_H_MM * STEPS_PER_MM))   # 11880
INVERT_Y = True

Point = Tuple[int, int]

FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
font_prop = FontProperties(fname=FONT_PATH)


@dataclass
class Drawer:
    w: StreamWriter
    cfg: HWConfig
    x: int = 0
    y: int = 0
    pen_is_down: bool = False

    def _pen_up(self):
        if self.pen_is_down:
            self.w.pen_up()
            self.pen_is_down = False

    def _pen_down(self):
        if not self.pen_is_down:
            self.w.pen_down()
            self.pen_is_down = True

    def travel_to(self, tx: int, ty: int):
        """Pen-up travel with accel/decel profile."""
        self._pen_up()
        if (tx, ty) != (self.x, self.y):
            travel_ramped(self.w, self.x, self.y, tx, ty, self.cfg)
            self.x, self.y = tx, ty

    def line_to(self, tx: int, ty: int):
        """
        Pen-down move with the same profile as travel_ramped.
        For short segments — use a softer local profile (12..10).
        """
        if (tx, ty) == (self.x, self.y):
            return
        self._pen_down()

        dx = abs(tx - self.x)
        dy = abs(ty - self.y)
        est_steps = max(dx, dy)

        short_thresh = 2 * int(self.cfg.corner_window_steps)
        if est_steps < short_thresh:
            local_start = max(self.cfg.div_start, 12)
            local_fast  = min(10, local_start)
            local_cfg = replace(self.cfg, div_start=local_start, div_fast=local_fast)
            travel_ramped(self.w, self.x, self.y, tx, ty, local_cfg)
        else:
            travel_ramped(self.w, self.x, self.y, tx, ty, self.cfg)

        self.x, self.y = tx, ty

    def draw_path(self, pts: List[Point]):
        """Pen-down polyline."""
        if len(pts) < 2:
            return
        self.travel_to(*pts[0])
        self._pen_down()
        for px, py in pts[1:]:
            self.line_to(px, py)
        self._pen_up()

    def tap(self):
        self._pen_up()
        self.w.tap()


# ---------- Geometry helpers ----------
def poly_length(poly: List[Point]) -> float:
    L = 0.0
    for i in range(len(poly) - 1):
        dx = poly[i + 1][0] - poly[i][0]
        dy = poly[i + 1][1] - poly[i][1]
        L += (dx * dx + dy * dy) ** 0.5
    return L

def split_poly_by_quarters(poly: List[Point]) -> List[List[Point]]:
    """Split polyline into four parts by cumulative length (approximate quarters)."""
    if len(poly) < 2:
        return [poly]

    segL = []
    for i in range(len(poly) - 1):
        dx = poly[i + 1][0] - poly[i][0]
        dy = poly[i + 1][1] - poly[i][1]
        segL.append((dx * dx + dy * dy) ** 0.5)

    total = sum(segL)
    if total == 0:
        return [poly]

    cuts = [total * 0.25, total * 0.50, total * 0.75, total]
    parts: List[List[Point]] = []
    acc = 0.0
    cur = [poly[0]]
    cut_idx = 0
    target = cuts[cut_idx]

    for i in range(1, len(poly)):
        cur.append(poly[i])
        if i - 1 < len(segL):
            acc += segL[i - 1]
        while acc >= target and cut_idx < 4:
            parts.append(cur[:])
            cur = [poly[i]]
            cut_idx += 1
            if cut_idx < 4:
                target = cuts[cut_idx]

    if cur:
        parts.append(cur)
    while len(parts) < 4:
        parts.append([parts[-1][-1]] if parts else [poly[-1]])
    return parts[:4]

def circle(cx: int, cy: int, r: int, n: int = 480) -> List[Point]:
    return [(int(cx + r * cos(t)), int(cy + r * sin(t))) for t in [i * tau / (n - 1) for i in range(n)]]

def rectangle(x: int, y: int, w: int, h: int, n_per_side: int = 160) -> List[Point]:
    pts: List[Point] = []
    for i in range(n_per_side):
        t = i / (n_per_side - 1); pts.append((x + int(w * t), y))
    for i in range(1, n_per_side):
        t = i / (n_per_side - 1); pts.append((x + w, y + int(h * t)))
    for i in range(1, n_per_side):
        t = i / (n_per_side - 1); pts.append((x + w - int(w * t), y + h))
    for i in range(1, n_per_side - 1):
        t = i / (n_per_side - 1); pts.append((x, y + h - int(h * t)))
    return pts

def triangle(ax: int, ay: int, bx: int, by: int, cx: int, cy: int, n_per_side: int = 150) -> List[Point]:
    pts: List[Point] = []
    for i in range(n_per_side):
        t = i / (n_per_side - 1); pts.append((int(ax + (bx - ax) * t), int(ay + (by - ay) * t)))
    for i in range(1, n_per_side):
        t = i / (n_per_side - 1); pts.append((int(bx + (cx - bx) * t), int(by + (cy - by) * t)))
    for i in range(1, n_per_side):
        t = i / (n_per_side - 1); pts.append((int(cx + (ax - cx) * t), int(cy + (ay - cy) * t)))
    return pts

def sine_wave(x0: int, x1: int, y0: int, amp: int, periods: int, pts_per_period: int) -> List[Point]:
    n = int(periods * pts_per_period)
    dx = (x1 - x0) / (n - 1)
    return [(int(x0 + i * dx), int(y0 + amp * sin(tau * periods * i / (n - 1)))) for i in range(n)]

def dot_matrix_full(x: int, y: int, w: int, h: int, cols: int, rows: int, margin: int) -> List[Point]:
    dx = (w - 2 * margin) / max(1, cols - 1)
    dy = (h - 2 * margin) / max(1, rows - 1)
    return [(int(x + margin + c * dx), int(y + margin + r * dy)) for r in range(rows) for c in range(cols)]


# ---------- Ordering helpers (nearest-neighbor w/ reversal) ----------
def _dist(a: Point, b: Point) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])  # L1 distance; fast and sufficient for ordering

def order_paths_nearest(paths: List[List[Point]], start_xy: Point) -> List[List[Point]]:
    """Reorder (and possibly reverse) paths to minimize approaches."""
    remain = [p for p in paths if len(p) >= 2]
    ordered: List[List[Point]] = []
    cur = start_xy
    while remain:
        best_i = 0; best_rev = False; best_d = 10**12
        for i, p in enumerate(remain):
            d_fwd = _dist(cur, p[0])
            d_rev = _dist(cur, p[-1])
            if d_fwd < best_d:
                best_d, best_i, best_rev = d_fwd, i, False
            if d_rev < best_d:
                best_d, best_i, best_rev = d_rev, i, True
        chosen = remain.pop(best_i)
        if best_rev:
            chosen = list(reversed(chosen))
        ordered.append(chosen)
        cur = chosen[-1]
    return ordered


# ---------- Hatching (text fill) ----------
def hatch_fill(
    D: Drawer,
    polys: List[List[Point]],
    spacing: int,
    color_idx: int,
    inset: int = 27,
    serpentine: bool = True,
):
    if not polys:
        return
    min_y = min(min(py for _, py in poly) for poly in polys)
    max_y = max(max(py for _, py in poly) for poly in polys)

    y0 = ((min_y + spacing // 2) // spacing) * spacing
    left_to_right = True

    for y_line in range(y0, max_y + 1, spacing):
        xs = []
        for poly in polys:
            n = len(poly)
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if y1 == y2: continue
                if y1 > y2:  x1, y1, x2, y2 = x2, y2, x1, y1
                if y1 < y_line <= y2:
                    t = (y_line - y1) / (y2 - y1)
                    xs.append(x1 + t * (x2 - x1))
        xs.sort()

        for i in range(0, len(xs), 2):
            if i + 1 < len(xs):
                sx = int(xs[i] + inset)
                ex = int(xs[i + 1] - inset)
                if ex <= sx: continue
                if serpentine and not left_to_right:
                    D.travel_to(ex, y_line); D._pen_down(); D.line_to(sx, y_line)
                else:
                    D.travel_to(sx, y_line); D._pen_down(); D.line_to(ex, y_line)
        if serpentine:
            left_to_right = not left_to_right


# ---------- Text drawing ----------
def draw_text_contour(D: Drawer, text: str, x: int, y_baseline: int,
                      height_mm: float, spacing_mm: float,
                      color_idx: int, fill_letters: str = ""):
    scale = (height_mm * STEPS_PER_MM) / 100.0
    spacing = int(spacing_mm * STEPS_PER_MM)

    cur_x = x
    current_color = color_idx

    for char in text:
        if char == " ":
            cur_x += int(height_mm * STEPS_PER_MM * 0.5) + spacing
            continue

        tp = TextPath((0, 0), char, prop=font_prop, size=100)
        polys = tp.to_polygons()

        transformed_polys: List[List[Point]] = []
        for poly in polys:
            pts = [(int(cur_x + px * scale), int(y_baseline + py * scale)) for px, py in poly]
            if pts:
                transformed_polys.append(pts)

        # Outline (switch color on the first subpath of the letter; no intra-letter ordering optimization)
        color_selected = False
        for pts in transformed_polys:
            D.travel_to(*pts[0])
            if not color_selected:
                D.w.select_color(current_color)
                color_selected = True
            D._pen_down()
            for px, py in pts[1:]:
                D.line_to(px, py)
            D.line_to(*pts[0])
            D._pen_up()

        # Hatching if requested
        if char.upper() in fill_letters.upper() and transformed_polys:
            hatch_fill(D, polys=transformed_polys, spacing=int(40), color_idx=current_color)

        # Next letter — next color (palette demo)
        current_color = (current_color + 1) % 4

        # Advance X
        if transformed_polys:
            char_w = max(p[0] for pts in transformed_polys for p in pts) - \
                     min(p[0] for pts in transformed_polys for p in pts)
        else:
            char_w = int(height_mm * STEPS_PER_MM * 0.6)
        cur_x += char_w + spacing


# ---------- Color-batched drawers with ordering ----------
def draw_color_group(D: Drawer, paths: List[List[Point]], color: int):
    """Order paths from the current position and draw with a single color selection."""
    paths = [p for p in paths if len(p) >= 2]
    if not paths:
        return

    ordered = order_paths_nearest(paths, (D.x, D.y))

    D.travel_to(*ordered[0][0])
    D.w.select_color(color)

    for pts in ordered:
        D.travel_to(*pts[0])
        D._pen_down()
        for px, py in pts[1:]:
            D.line_to(px, py)
        D._pen_up()

def draw_sine_by_periods_batched_optimized(D: Drawer, poly: List[Point], periods: int, pts_per_period: int):
    n_per = pts_per_period
    per_parts: List[List[Point]] = []
    per_colors: List[int] = []
    for p in range(periods):
        start = p * n_per
        end = min((p + 1) * n_per, len(poly))
        part = poly[start:end]
        if len(part) >= 2:
            per_parts.append(part)
            per_colors.append(p % 4)
    # batch by color and optimize order
    for c in (0, 1, 2, 3):
        group = [part for part, col in zip(per_parts, per_colors) if col == c]
        draw_color_group(D, group, color=c)

def draw_poly_quarters_batched_optimized(D: Drawer, poly: List[Point], start_color_idx: int = 0):
    parts = split_poly_by_quarters(poly)
    colored = [(part, (start_color_idx + i) % 4) for i, part in enumerate(parts) if len(part) >= 2]
    for c in (0, 1, 2, 3):
        group = [p for (p, col) in colored if col == c]
        draw_color_group(D, group, color=c)


def draw_tap_shape(D: Drawer, pts: List[Point], color_idx: int, serpentine: bool = True):
    """Serpentine by rows, starting from the nearest edge to the current position."""
    if not pts:
        return
    xs = sorted({x for x, _ in pts}); ys = sorted({y for _, y in pts})
    cols, rows = len(xs), len(ys)
    if cols == 0 or rows == 0:
        return

    start_from_top = abs(D.y - ys[0]) <= abs(D.y - ys[-1])
    row_indices = list(range(rows)) if start_from_top else list(reversed(range(rows)))
    ltr_first = abs(D.x - xs[0]) <= abs(D.x - xs[-1])

    r0 = row_indices[0]
    c0 = 0 if ltr_first else cols - 1
    D.travel_to(xs[c0], ys[r0])
    D.w.select_color(color_idx)

    for idx, r in enumerate(row_indices):
        ltr = (ltr_first if (idx % 2 == 0) else (not ltr_first)) if serpentine else ltr_first
        col_iter = range(cols) if ltr else reversed(range(cols))
        for c in col_iter:
            px, py = xs[c], ys[r]
            D.travel_to(px, py)
            D.tap()


# ---------- Stream assembly ----------
def generate_demo_stream(output_file: str = "demo_stream.bin"):
    cfg = HWConfig(
        steps_per_mm=STEPS_PER_MM,
        invert_y=INVERT_Y,
        # pen-down profile (fast & quiet)
        div_start=25,
        div_fast=12,
        profile="triangle",
        # travel (pen-up)
        travel_div_fast=10,
        # corner accel/decel windows
        corner_deg=85.0,
        corner_div=25,
        corner_window_steps=300,
    )

    w = StreamWriter()
    D = Drawer(w, cfg, x=0, y=0)

    margin = int(10 * STEPS_PER_MM)
    left, top = margin, margin
    right, bottom = CANVAS_W - margin, CANVAS_H - margin

    # ---- TEXT (top) ----
    text_left_x = left + int(10 * STEPS_PER_MM)
    text_baseline_y = top + int(240 * STEPS_PER_MM)

    draw_text_contour(D, "OmniRevolve", x=text_left_x, y_baseline=text_baseline_y,
                      height_mm=20.0, spacing_mm=5.0, color_idx=0, fill_letters="OmniRevolve")

    draw_text_contour(D, "Plotter Demo", x=text_left_x,
                      y_baseline=text_baseline_y - int(40 * STEPS_PER_MM),
                      height_mm=20.0, spacing_mm=5.0, color_idx=3, fill_letters="Plotter Demo")

    # ---- SINE (8 periods; batched by color and NN-ordered) ----
    periods = 8
    pts_per_period = 300
    sin_y = top + int(45 * STEPS_PER_MM)
    sin_poly = sine_wave(
        x0=left, x1=right,
        y0=sin_y,
        amp=int(12 * STEPS_PER_MM),
        periods=periods, pts_per_period=pts_per_period
    )
    draw_sine_by_periods_batched_optimized(D, sin_poly, periods=periods, pts_per_period=pts_per_period)

    # ---- CIRCLE / TRIANGLE / RECTANGLE — quarters, color-batched with NN order ----
    circ = circle(cx=left + int(45 * STEPS_PER_MM),
                  cy=top + int(95 * STEPS_PER_MM),
                  r=int(30 * STEPS_PER_MM), n=480)
    draw_poly_quarters_batched_optimized(D, circ, start_color_idx=0)

    tri = triangle(ax=left + int(60 * STEPS_PER_MM), ay=top + int(160 * STEPS_PER_MM),
                   bx=left + int(110 * STEPS_PER_MM), by=top + int(160 * STEPS_PER_MM),
                   cx=left + int(85 * STEPS_PER_MM),  cy=top + int(120 * STEPS_PER_MM),
                   n_per_side=150)
    draw_poly_quarters_batched_optimized(D, tri, start_color_idx=2)

    rect_x = left + int(110 * STEPS_PER_MM)
    rect_y = top + int(70 * STEPS_PER_MM)
    rect_w = int(80 * STEPS_PER_MM)
    rect_h = int(50 * STEPS_PER_MM)
    rect_poly = rectangle(x=rect_x, y=rect_y, w=rect_w, h=rect_h, n_per_side=160)
    draw_poly_quarters_batched_optimized(D, rect_poly, start_color_idx=1)

    # ---- TAP matrix (whole grid with a single color selection; approach nearest edge) ----
    taps_cols, taps_rows = 9, 7
    inner_margin = int(8 * STEPS_PER_MM)
    taps_pts = dot_matrix_full(rect_x, rect_y, rect_w, rect_h, taps_cols, taps_rows, inner_margin)
    draw_tap_shape(D, taps_pts, color_idx=1)

    # ---- Finalize ----
    stream = w.finalize()
    Path(output_file).write_bytes(stream)

    meta = {
        "canvas_steps": {"width": CANVAS_W, "height": CANVAS_H},
        "steps_per_mm": STEPS_PER_MM,
        "invert_y": INVERT_Y,
        "bytes": len(stream),
        "notes": [
            "Color-batched NN ordering inside groups",
            "One color-select per group (pre-travel then select)",
            "Short segments use soft local profile",
        ],
    }
    Path(output_file).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("✓ Stream saved:", output_file)
    print("  Size:", len(stream), "bytes")
    print(f"\nPreview:\n  python3 shared/xyplotter_stream_previewer.py --canvas-w-steps {CANVAS_W} --canvas-h-steps {CANVAS_H} --invert-y {1 if INVERT_Y else 0} {output_file}")


def main():
    ap = argparse.ArgumentParser(description="Demo stream generator (optimized travels).")
    ap.add_argument("-o", "--output", default="demo_stream.bin", help="Output stream file (.bin)")
    args = ap.parse_args()
    generate_demo_stream(args.output)

if __name__ == "__main__":
    main()
