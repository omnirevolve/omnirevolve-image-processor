#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_plotter_svg_stream_creator.py — SVG → NEW-protocol stream.

Использует тот же движок, что omnirevolve_plotter_stream_creator.py:
Config, StreamWriter, emit_polyline, travel_ramped из
omnirevolve_plotter_stream_creator_helper.py

Пример:
    python omnirevolve_plotter_svg_stream_creator.py input.svg stream.bin \
        --target-width-steps 8400 \
        --target-height-steps 11880 \
        --steps-per-mm 40 \
        --margin-mm 5 \
        --pen-map "#000000=3;#ff0000=0;#00ff00=1;#0000ff=2"
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys
import math
import re

try:
    from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc
except ImportError as e:
    raise SystemExit(
        "Требуется svgpathtools.\n"
        "Установи: pip install svgpathtools\n"
    ) from e

# Путь к shared, как в omnirevolve_plotter_stream_creator.py
_SHARED_DIR = (Path(__file__).resolve().parent / "../../shared").resolve()
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from omnirevolve_plotter_stream_creator_helper import (  # type: ignore
    Config,
    StreamWriter,
    emit_polyline,
    travel_ramped,
)

Point = Tuple[int, int]

# ------------------------- Цвета и pen-map -------------------------

HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")

def normalize_hex(color: str) -> str:
    if not color:
        return ""
    m = HEX_COLOR_RE.search(color.strip())
    if not m:
        return ""
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.lower()

def parse_pen_map(spec: str) -> Dict[str, int]:
    """
    "#000000=3;#ff0000=0" -> {"#000000":3, "#ff0000":0}
    """
    result: Dict[str, int] = {}
    if not spec:
        return result
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Неверный pen-map элемент: '{part}' (ожидается '#rrggbb=index')")
        color_s, idx_s = part.split("=", 1)
        color_n = normalize_hex(color_s)
        if not color_n:
            raise ValueError(f"Неверный цвет в pen-map: '{color_s}'")
        result[color_n] = int(idx_s)
    return result

# ------------------------- SVG → полилинии -------------------------

def _sample_segment(seg, tol: float) -> List[complex]:
    """Дискретизация сегмента в точки complex(x + i*y)."""
    if isinstance(seg, Line):
        return [seg.start, seg.end]

    length = seg.length(error=1e-3)
    if length == 0:
        return [seg.start, seg.end]

    n = max(2, int(math.ceil(length / tol)))
    ts = [i / (n - 1) for i in range(n)]
    return [seg.point(t) for t in ts]

@dataclass
class ColoredPolyline:
    pts: List[Tuple[float, float]]  # SVG coords
    color: str                      # "#rrggbb"

def svg_to_colored_polylines(svg_path: Path, curve_tol_px: float) -> List[ColoredPolyline]:
    paths, attrs, svg_attribs = svg2paths2(str(svg_path))

    result: List[ColoredPolyline] = []

    for path, a in zip(paths, attrs):
        stroke = normalize_hex(a.get("stroke") or "")
        fill   = normalize_hex(a.get("fill") or "")
        color  = stroke or fill
        if not color:
            continue  # ничего рисовать

        current: List[Tuple[float, float]] = []
        last_end: complex | None = None

        for seg in path:
            pts_complex = _sample_segment(seg, curve_tol_px)
            if not pts_complex:
                continue

            # разрыв path -> новый polyline
            if last_end is not None and pts_complex[0] != last_end and current:
                result.append(ColoredPolyline(current, color))
                current = []

            for p in pts_complex:
                current.append((float(p.real), float(p.imag)))
            last_end = pts_complex[-1]

        if current:
            result.append(ColoredPolyline(current, color))

    return result

# ----------------------- Масштабирование в шаги -----------------------

def compute_bounds(polys: Iterable[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for poly in polys:
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        raise ValueError("SVG не содержит видимых путей с цветом (stroke/fill).")
    return min(xs), min(ys), max(xs), max(ys)

def transform_svg_to_steps(
    polys: List[ColoredPolyline],
    target_w_steps: int,
    target_h_steps: int,
    margin_steps: int,
    invert_y: bool,
) -> List[ColoredPolyline]:
    raw_polys = [p.pts for p in polys]
    xmin, ymin, xmax, ymax = compute_bounds(raw_polys)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        raise ValueError("Degenerate SVG bounds (width/height <= 0)")

    inner_w = target_w_steps - 2 * margin_steps
    inner_h = target_h_steps - 2 * margin_steps
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Слишком большой margin относительно рабочей области")

    sx = inner_w / w
    sy = inner_h / h
    s = min(sx, sy)

    offset_x = margin_steps + (inner_w - w * s) / 2.0
    offset_y = margin_steps + (inner_h - h * s) / 2.0

    out: List[ColoredPolyline] = []

    for cp in polys:
        pts_steps: List[Point] = []
        for x, y in cp.pts:
            xs = int(round((x - xmin) * s + offset_x))
            ys_float = (y - ymin) * s + offset_y
            if invert_y:
                ys = int(round(target_h_steps - 1 - ys_float))
            else:
                ys = int(round(ys_float))
            # clamp (как _finalize_point)
            xs = max(0, min(target_w_steps - 1, xs))
            ys = max(0, min(target_h_steps - 1, ys))
            pts_steps.append((xs, ys))
        out.append(ColoredPolyline(pts_steps, cp.color))

    return out

# ---------------------- Генерация потока протокола ----------------------

def _group_by_color(polys: List[ColoredPolyline], pen_map: Dict[str, int], default_pen: int) -> Dict[int, List[List[Point]]]:
    groups: Dict[int, List[List[Point]]] = {}
    for cp in polys:
        if len(cp.pts) < 2:
            continue
        color_idx = pen_map.get(cp.color, default_pen)
        groups.setdefault(color_idx, []).append([(int(x), int(y)) for x, y in cp.pts])
    return groups

def generate_stream_from_svg(
    svg_path: Path,
    output_file: Path,
    target_w_steps: int,
    target_h_steps: int,
    cfg: Config,
    pen_map: Dict[str, int],
    default_pen: int = 3,  # по умолчанию чёрный (K)
    curve_tol_px: float = 0.25,
    margin_mm: float = 5.0,
) -> None:
    margin_steps = int(round(margin_mm * cfg.steps_per_mm))

    print(f"[svg] Читаю {svg_path}")
    polys = svg_to_colored_polylines(svg_path, curve_tol_px=curve_tol_px)
    if not polys:
        raise SystemExit("В SVG не найдено путей со stroke/fill.")

    print(f"[svg] Найдено контуров: {len(polys)}")

    polys_steps = transform_svg_to_steps(
        polys,
        target_w_steps=target_w_steps,
        target_h_steps=target_h_steps,
        margin_steps=margin_steps,
        invert_y=cfg.invert_y,
    )

    groups = _group_by_color(polys_steps, pen_map, default_pen)
    if not groups:
        raise SystemExit("После преобразования не осталось контуров для рисования.")

    w = StreamWriter()
    w.pen_up()
    w.set_speed(cfg.div_start)

    cur_x, cur_y = 0, 0

    total_contours = sum(len(v) for v in groups.values())

    print(f"[stream] Цветов: {len(groups)}, контуров: {total_contours}")

    for color_idx in sorted(groups.keys()):
        paths = groups[color_idx]
        if not paths:
            continue

        print(f"[stream] Цвет {color_idx}: контуров {len(paths)}")

        # Пред-подъезд к первому контуру этого цвета
        first = paths[0][0]
        if (cur_x, cur_y) != first:
            travel_ramped(w, cur_x, cur_y, first[0], first[1], cfg)
            cur_x, cur_y = first

        w.select_color(color_idx)

        for poly in paths:
            if len(poly) < 2:
                continue
            start = poly[0]
            if (cur_x, cur_y) != start:
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, start[0], start[1], cfg)
                cur_x, cur_y = start
            w.pen_down()
            emit_polyline(w, cfg, poly, color_index=None)
            w.pen_up()
            cur_x, cur_y = poly[-1]

    data = w.finalize()
    output_file.write_bytes(data)

    print("✓ Stream saved:", str(output_file))
    print("  Size:", len(data), "bytes")
    print("  Target steps:", target_w_steps, "x", target_h_steps)

# ------------------------------- CLI ---------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="SVG → OmniRevolve NEW-protocol stream (color-batched)."
    )
    ap.add_argument("input", help="SVG-файл")
    ap.add_argument("-o", "--output", default="stream_from_svg.bin")

    ap.add_argument("--target-width-steps", type=int, required=True)
    ap.add_argument("--target-height-steps", type=int, required=True)

    ap.add_argument("--steps-per-mm", type=float, default=40.0)
    ap.add_argument("--invert-y", type=int, default=1)

    ap.add_argument("--margin-mm", type=float, default=5.0)
    ap.add_argument("--curve-tol-px", type=float, default=0.25)

    ap.add_argument(
        "--pen-map",
        type=str,
        default="#000000=3",
        help="Соответствие SVG-цветов перам, формат '#rrggbb=index;#ff0000=0;...'",
    )
    ap.add_argument(
        "--default-pen",
        type=int,
        default=3,
        help="Индекс пера по умолчанию для цветов, не попавших в pen-map",
    )

    # Профиль движения — те же параметры, что в omnirevolve_plotter_stream_creator.py
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

def main(argv: list[str] | None = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.div_start < args.travel_div_fast:
        raise SystemExit("Error: --div-start must be >= --travel-div-fast")

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

    pen_map = parse_pen_map(args.pen_map)
    svg_path = Path(args.input)
    out_path = Path(args.output)

    generate_stream_from_svg(
        svg_path,
        out_path,
        target_w_steps=args.target_width_steps,
        target_h_steps=args.target_height_steps,
        cfg=cfg,
        pen_map=pen_map,
        default_pen=args.default_pen,
        curve_tol_px=args.curve_tol_px,
        margin_mm=args.margin_mm,
    )

if __name__ == "__main__":
    main()
