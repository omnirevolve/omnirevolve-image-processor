#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omnirevolve_plotter_stream_creator_helper.py — universal helpers for building XY-plotter streams.

Protocol:
- Step byte (MSB=1):
  * Two steps: 11 FFF SSS   (FFF=first step dir 0..7, SSS=second step dir 0..7)
  * Single  : 10 SSS 000
- Service byte (MSB=0):
  * Set speed: 0x40 | (divider & 0x3F)   (divider 0..63) — applies immediately
  * Pen: 0x01=pen up, 0x02=pen down, 0x03=tap
  * Color select: 0x08..0x0F  (index 0..7)
  * EOF: 0x3F

This version:
- Explicit initial speed set at stream start (to avoid accidentally starting at 63).
- Separate travel profile: travel_start_div + travel_window_steps.
- Quantized ramps (coarse levels) to reduce speed flips (less motor "singing").
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Any
import math

SPI_CHUNK_SIZE = 1024
WORK_MAX_X = 13210
WORK_MAX_Y = 13019

# Direction codes (0..7): 0=+Y, 1=NE, 2=+X, 3=SE, 4=-Y, 5=SW, 6=-X, 7=NW
DIR_POSY, DIR_NE, DIR_POSX, DIR_SE, DIR_NEGY, DIR_SW, DIR_NEGX, DIR_NW = 0, 1, 2, 3, 4, 5, 6, 7

__all__ = [
    "SPI_CHUNK_SIZE", "WORK_MAX_X", "WORK_MAX_Y",
    "DIR_POSY", "DIR_NE", "DIR_POSX", "DIR_SE", "DIR_NEGY", "DIR_SW", "DIR_NEGX", "DIR_NW",
    "Config", "StreamWriter",
    "make_speed_byte", "pack_steps",
    "build_counts_triangle", "build_counts_scurve",
    "bresenham_dir_codes",
    "emit_steps_accel", "emit_steps_decel",
    "emit_segment_with_corner_profile", "emit_polyline",
    "travel_ramped",
]

# ------------------------------- Encoding core -------------------------------

def make_speed_byte(divider: int) -> int:
    if (int(divider) > 63):
        divider = 63
    if (int(divider) < 0):
        divider = 0
    return 0x40 | (int(divider) & 0x3F)

def pack_steps(step_codes: Iterable[int]) -> bytearray:
    out = bytearray()
    codes = [int(c) & 0x07 for c in step_codes]
    i = 0
    while i < len(codes):
        s1 = codes[i]
        if i + 1 < len(codes):
            s2 = codes[i + 1]
            out.append(0x80 | 0x40 | (s1 << 3) | s2)
            i += 2
        else:
            out.append(0x80 | (s1 << 3))
            i += 1
    return out

# ------------------------------- Ramps & helpers -------------------------------

def _distribute_even(total_steps: int, levels: int) -> List[int]:
    if levels <= 0: return []
    base = total_steps // levels
    rem  = total_steps % levels
    return [base + (1 if i < rem else 0) for i in range(levels)]

def build_counts_triangle(length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    if length <= 0: return {}
    if div_slow < div_fast: raise ValueError("div_slow must be >= div_fast")
    levels = div_slow - div_fast + 1
    per = _distribute_even(length, levels)
    out: Dict[int,int] = {}
    for i, cnt in enumerate(per):
        div = div_slow - i
        if cnt > 0: out[div] = out.get(div, 0) + cnt
    return out

def build_counts_scurve(length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    if length <= 0: return {}
    if div_slow < div_fast: raise ValueError("div_slow must be >= div_fast")
    span = div_slow - div_fast
    out: Dict[int,int] = {}
    for i in range(length):
        t = (i + 0.5) / length
        s = (3 * t * t) - (2 * t * t * t)
        div = round(div_slow - s * span)
        div = max(div_fast, min(div_slow, div))
        out[div] = out.get(div, 0) + 1
    return out

def _quantized_levels(div_slow: int, div_fast: int, step: int = 4) -> List[int]:
    """Coarse levels from slow to fast inclusive, e.g. 28,24,20,16,12,10 for step=4."""
    if div_slow < div_fast: div_slow, div_fast = div_fast, div_slow
    levels = list(range(div_slow, div_fast - 1, -step))
    if levels[-1] != div_fast:
        levels.append(div_fast)
    return levels

# --------------------------------- Writer -----------------------------------

@dataclass
class Config:
    steps_per_mm: float = 40.0
    invert_y: bool = True

    # Drawing profile (pen-down)
    div_start: int = 28
    div_fast: int  = 15
    profile: str = "triangle"

    # Corner handling
    corner_deg: float = 85.0
    corner_div: int = 28
    corner_window_steps: int = 300

    # Short edges (no corners)
    short_len_steps: int = 120
    short_div: int = 16

    # Travel (pen-up) profile — separate
    travel_div_fast: int = 10
    travel_start_div: int = 28
    travel_window_steps: int = 240
    travel_quant_step: int = 4   # quantization of speed levels for travel ramps

    # Optional soft tail (unused)
    soft_tail_steps: int = 0
    soft_tail_div: int = 20

    def to_steps(self, mm: float) -> int:
        return int(round(mm * self.steps_per_mm))

class StreamWriter:
    def __init__(self, *_: Any, **__: Any):
        self.out = bytearray()
        self._cur_speed: Optional[int] = None
        self._started = False

    # service bytes
    def set_speed(self, divider: int):
        b = make_speed_byte(divider)
        if self._cur_speed != divider:
            self.out.append(b)
            self._cur_speed = divider

    def pen_up(self):   self.out.append(0x01)
    def pen_down(self): self.out.append(0x02)
    def tap(self):      self.out.append(0x03)

    def select_color(self, color_index: int):
        if not (0 <= color_index <= 7): raise ValueError("color index 0..7")
        self.out.append(0x08 | (color_index & 0x07))

    # steps
    def add_steps(self, step_codes: Iterable[int]):
        self.out += pack_steps(step_codes)

    def finalize(self) -> bytes:
        self.out.append(0x3F)  # EOF
        pad = (-len(self.out)) % SPI_CHUNK_SIZE
        if pad:
            self.out += b"\x00" * pad
        return bytes(self.out)

# --------------------------- Geometry / direction ---------------------------

def clamp_xy(x: int, y: int, wmax: int = WORK_MAX_X, hmax: int = WORK_MAX_Y) -> Tuple[int, int]:
    x = 0 if x < 0 else (wmax if x > wmax else x)
    y = 0 if y < 0 else (hmax if y > hmax else y)
    return x, y

def bresenham_dir_codes(x0: int, y0: int, x1: int, y1: int) -> List[int]:
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    codes: List[int] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = err * 2
        moved_x = moved_y = 0
        if e2 > -dy:
            err -= dy; x += sx; moved_x = 1
        if e2 < dx:
            err += dx; y += sy; moved_y = 1
        if moved_x and moved_y:
            if sx > 0 and sy > 0: codes.append(DIR_NE)
            elif sx > 0 and sy < 0: codes.append(DIR_SE)
            elif sx < 0 and sy < 0: codes.append(DIR_SW)
            else: codes.append(DIR_NW)
        elif moved_x:
            codes.append(DIR_POSX if sx > 0 else DIR_NEGX)
        elif moved_y:
            codes.append(DIR_POSY if sy > 0 else DIR_NEGY)
    return codes

# --------------------------- Ramp helpers ---------------------------

def _build_counts(profile: str, length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    if profile == "scurve":
        return build_counts_scurve(length, div_fast, div_slow)
    elif profile == "triangle":
        return build_counts_triangle(length, div_fast, div_slow)
    raise ValueError("profile must be 'triangle' or 'scurve'")

def emit_steps_accel(w: StreamWriter, step_codes: List[int], profile: str, div_fast: int, start_div: int):
    if not step_codes: return
    if start_div <= div_fast:
        w.set_speed(div_fast); w.add_steps(step_codes); return
    counts = _build_counts(profile, len(step_codes), div_fast, start_div)
    idx = 0
    for div in range(start_div, div_fast - 1, -1):
        cnt = counts.get(div, 0)
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(step_codes[idx:idx+cnt]); idx += cnt

def emit_steps_decel(w: StreamWriter, step_codes: List[int], profile: str, div_fast: int, end_div: int):
    if not step_codes: return
    if end_div <= div_fast:
        w.set_speed(div_fast); w.add_steps(step_codes); return
    counts = _build_counts(profile, len(step_codes), div_fast, end_div)
    idx = 0
    for div in range(div_fast, end_div + 1):
        cnt = counts.get(div, 0)
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(step_codes[idx:idx+cnt]); idx += cnt

# --------------------------- Corner-aware polylines --------------------------

def angle_degrees(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0: return 180.0
    dot = (v1x*v2x + v1y*v2y) / (n1*n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def emit_segment_with_corner_profile(
    w: StreamWriter,
    step_codes: List[int],
    profile: str,
    div_fast: int,
    div_start: int,
    corner_div: int,
    corner_window_steps: int,
    slow_in: bool,
    slow_out: bool,
    short_len_steps: int = 120,
    short_div: int = 16,
):
    N = len(step_codes)
    if N == 0: return

    if not slow_in and not slow_out:
        w.set_speed(short_div if N <= short_len_steps else div_fast)
        w.add_steps(step_codes)
        return

    entry_len = min(corner_window_steps if slow_in else 0, N)
    exit_len  = min(corner_window_steps if slow_out else 0, max(0, N - entry_len))
    mid_len   = max(0, N - entry_len - exit_len)

    if entry_len + exit_len >= N:
        half = N // 2
        if half > 0:
            emit_steps_accel(w, step_codes[:half], profile, div_fast, corner_div if slow_in else div_start)
        if N % 2 == 1:
            w.set_speed(div_fast); w.add_steps(step_codes[half:half+1]); half += 1
        rest = step_codes[half:]
        if rest:
            emit_steps_decel(w, rest, profile, div_fast, corner_div if slow_out else div_start)
        return

    if entry_len > 0:
        emit_steps_accel(w, step_codes[:entry_len], profile, div_fast, corner_div)
    if mid_len > 0:
        w.set_speed(div_fast); w.add_steps(step_codes[entry_len:entry_len+mid_len])
    if exit_len > 0:
        emit_steps_decel(w, step_codes[-exit_len:], profile, div_fast, corner_div)

def emit_polyline(w: StreamWriter, cfg: Config, *args, color_index: Optional[int] = None):
    if len(args) == 1:
        pts = args[0]
    else:
        raise TypeError("emit_polyline(w, cfg, pts, ...) expected")
    if not pts or len(pts) < 2: return
    if color_index is not None: w.select_color(color_index)

    for i in range(len(pts)-1):
        a = pts[i-1] if i-1 >= 0 else pts[i]
        b = pts[i]; c = pts[i+1]
        ang_in = angle_degrees(a[0],a[1], b[0],b[1], c[0],c[1]) if i>0 else 180.0
        slow_in = (i>0 and ang_in < cfg.corner_deg)
        if (i+2) < len(pts):
            ang_out = angle_degrees(b[0],b[1], c[0],c[1], pts[i+2][0],pts[i+2][1])
            slow_out = (ang_out < cfg.corner_deg)
        else:
            slow_out = False
        codes = bresenham_dir_codes(b[0], b[1], c[0], c[1])
        emit_segment_with_corner_profile(
            w, codes, cfg.profile, cfg.div_fast, cfg.div_start,
            cfg.corner_div, cfg.corner_window_steps,
            slow_in=slow_in, slow_out=slow_out,
            short_len_steps=cfg.short_len_steps, short_div=cfg.short_div
        )

# ----------------------------- Pen-up travel -----------------------------

def _emit_quantized_accel(w: StreamWriter, codes: List[int], levels: List[int], profile: str):
    """Split codes evenly across given speed levels (slow→fast)."""
    if not codes or not levels: return
    parts = _distribute_even(len(codes), len(levels))
    idx = 0
    for div, cnt in zip(levels, parts):
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(codes[idx:idx+cnt]); idx += cnt

def _emit_quantized_decel(w: StreamWriter, codes: List[int], levels: List[int], profile: str):
    """Split codes evenly across given speed levels (fast→slow)."""
    if not codes or not levels: return
    parts = _distribute_even(len(codes), len(levels))
    idx = 0
    for div, cnt in zip(levels, parts):
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(codes[idx:idx+cnt]); idx += cnt

def travel_ramped(w: StreamWriter, *args):
    """
    Pen-up travel with short, quantized accel/decel ramps.
    Signature: travel_ramped(w, x0, y0, x1, y1, cfg)
    """
    if len(args) != 5:
        raise TypeError("travel_ramped(w, x0, y0, x1, y1, cfg) expected")
    x0, y0, x1, y1, cfg = args

    codes = bresenham_dir_codes(x0, y0, x1, y1)
    N = len(codes)
    if N == 0: return

    win       = int(cfg.travel_window_steps)
    div_fast  = int(cfg.travel_div_fast)
    div_start = int(cfg.travel_start_div)

    if div_start < div_fast:
        div_start = div_fast

    # If the segment is very short — use a simple triangular profile
    if N <= 2 * win:
        half = max(1, N // 2)
        emit_steps_accel(w, codes[:half], cfg.profile, div_fast, div_start)
        if N % 2 == 1:
            w.set_speed(div_fast); w.add_steps(codes[half:half+1]); half += 1
        emit_steps_decel(w, codes[half:], cfg.profile, div_fast, div_start)
        return

    entry = codes[:win]
    cruise = codes[win:N - win]
    exitc  = codes[N - win:]

    # Quantize speed levels for accel/decel
    levels_down = _quantized_levels(div_start, div_fast, step=max(1, int(cfg.travel_quant_step)))
    levels_up   = list(reversed(levels_down))

    _emit_quantized_accel(w, entry, levels_down, cfg.profile)
    if cruise:
        w.set_speed(div_fast); w.add_steps(cruise)
    _emit_quantized_decel(w, exitc, levels_up, cfg.profile)
