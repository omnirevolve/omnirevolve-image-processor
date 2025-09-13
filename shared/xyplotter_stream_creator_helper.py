#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xyplotter_stream_creator_helper.py — universal helpers for building XY-plotter streams.

Protocol:
- Step byte (MSB=1):
  * Two steps: 11 FFF SSS   (FFF=first step dir 0..7, SSS=second step dir 0..7)
  * Single  : 10 SSS 000
- Service byte (MSB=0):
  * Set speed: 0x40 | (divider & 0x3F)   (divider 0..63) — applies immediately
  * Pen: 0x01=pen up, 0x02=pen down, 0x03=tap
  * Color select: 0x08..0x0F  (index 0..7)
  * EOF: 0x3F
- Streams are padded to SPI_CHUNK_SIZE (1024 bytes).

This module provides:
- Config: kinematics & motion defaults (mm/step, dividers, ramp lengths, corner handling).
- StreamWriter: protocol-aware writer (service bytes, packed step bytes, finalize).
- bresenham_dir_codes: direction codes (0..7) between points.
- Ramp helpers: S-curve / triangle distribution, fixed-length accel/decel windows.
- Corner-aware polylines: slow-in/slow-out around sharp angles.
- travel_ramped: pen-up travel with fixed 2 cm accel/decel windows by default.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional, Any
import math

# --------------------------- Workspace & directions ---------------------------

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
    "emit_steps_accel", "emit_steps_decel", "emit_steps_with_symmetric_ramp",
    "emit_segment_with_corner_profile", "emit_polyline",
    "travel_ramped",
]

# ------------------------------- Encoding core -------------------------------

def make_speed_byte(divider: int) -> int:
    """Service byte for speed setting: MSB=0, 0x40 | (divider & 0x3F)."""
    if (int(divider) > 63):
        divider = 63
    if (int(divider) < 0):
        divider = 0
    return 0x40 | (int(divider) & 0x3F)

def pack_steps(step_codes: Iterable[int]) -> bytearray:
    """
    Pack 0..7 step codes using:
      - two-step bytes: 11 FFF SSS
      - single-step bytes: 10 SSS 000
    Preference: emit as many two-step bytes as possible; last odd one as single-step.
    """
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

# ------------------------------- S-curve / triangle ramps -------------------------------

def _distribute_even(total_steps: int, levels: int) -> List[int]:
    if levels <= 0:
        return []
    base = total_steps // levels
    rem = total_steps % levels
    return [base + (1 if i < rem else 0) for i in range(levels)]

def build_counts_triangle(length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    """
    Linear ramp distribution over `length` steps across dividers from div_slow→div_fast (inclusive).
    Returns: dict {divider -> steps_at_that_divider}.
    """
    if length <= 0:
        return {}
    if div_slow < div_fast:
        raise ValueError("div_slow must be >= div_fast")
    levels = div_slow - div_fast + 1
    per = _distribute_even(length, levels)
    out: Dict[int, int] = {}
    for i, cnt in enumerate(per):
        div = div_slow - i
        out[div] = out.get(div, 0) + cnt
    return out

def build_counts_scurve(length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    """
    Smoothstep (3t^2 - 2t^3) ramp distribution over `length` steps, div_slow→div_fast.
    Returns: dict {divider -> steps_at_that_divider}.
    """
    if length <= 0:
        return {}
    if div_slow < div_fast:
        raise ValueError("div_slow must be >= div_fast")
    span = div_slow - div_fast
    out: Dict[int, int] = {}
    for i in range(length):
        t = (i + 0.5) / length
        s = (3 * t * t) - (2 * t * t * t)
        div = round(div_slow - s * span)
        div = max(div_fast, min(div_slow, div))
        out[div] = out.get(div, 0) + 1
    return out

# --------------------------------- Writer -----------------------------------

@dataclass
class Config:
    steps_per_mm: float = 40.0
    invert_y: bool = True
    # base ramp profile (for drawing/pen-down)
    div_start: int = 30        # start/stop divider (slow end)
    div_fast: int  = 15        # drawing cruise divider
    profile: str = "triangle"    # "scurve" or "triangle"
    # pen-up travel cruise
    travel_div_fast: int = 10  # travel (pen-up) cruise divider
    # corner handling (also used as default accel/decel window length)
    corner_deg: float = 85.0
    corner_div: int = 30       # target slow divider at sharp corners
    corner_window_steps: int = 400  # default accel/decel window ≈ 1 cm @ 40 steps/mm
    # optional soft tail (not used here by default)
    soft_tail_steps: int = 0
    soft_tail_div: int = 20

    def to_steps(self, mm: float) -> int:
        return int(round(mm * self.steps_per_mm))

class StreamWriter:
    """
    Protocol-aware byte stream builder.
    """
    def __init__(self, *_ignored: Any, **__ignored: Any):
        self.out = bytearray()
        self._cur_speed: Optional[int] = None

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

    # step bytes
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
    """Generate direction codes (0..7) to move from (x0,y0) to (x1,y1)."""
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
            err -= dy
            x += sx
            moved_x = 1
        if e2 < dx:
            err += dx
            y += sy
            moved_y = 1
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

# ------------------------------- Ramp helpers -------------------------------

def _build_counts(profile: str, length: int, div_fast: int, div_slow: int) -> Dict[int, int]:
    if profile == "scurve":
        return build_counts_scurve(length, div_fast, div_slow)
    elif profile == "triangle":
        return build_counts_triangle(length, div_fast, div_slow)
    raise ValueError("profile must be 'triangle' or 'scurve'")

def emit_steps_accel(w: StreamWriter, step_codes: List[int], profile: str, div_fast: int, start_div: int):
    """Accelerate from start_div → div_fast across provided step_codes (forward order)."""
    counts = _build_counts(profile, len(step_codes), div_fast, start_div)
    idx = 0
    for div in range(start_div, div_fast - 1, -1):
        cnt = counts.get(div, 0)
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(step_codes[idx:idx+cnt]); idx += cnt

def emit_steps_decel(w: StreamWriter, step_codes: List[int], profile: str, div_fast: int, end_div: int):
    """Decelerate from div_fast → end_div across provided step_codes (forward order)."""
    counts = _build_counts(profile, len(step_codes), div_fast, end_div)
    idx = 0
    for div in range(div_fast, end_div + 1):
        cnt = counts.get(div, 0)
        if cnt <= 0: continue
        w.set_speed(div); w.add_steps(step_codes[idx:idx+cnt]); idx += cnt

def emit_steps_with_symmetric_ramp(w: StreamWriter, step_codes: List[int], profile: str, div_fast: int, div_start: int):
    """(Legacy) symmetric accel/decel over the whole segment (no fixed ramp length)."""
    N = len(step_codes)
    if N == 0: return
    half = N // 2
    emit_steps_accel(w, step_codes[:half], profile, div_fast, div_start)
    if N % 2 == 1:
        w.set_speed(div_fast); w.add_steps(step_codes[half:half+1]); half += 1
    emit_steps_decel(w, step_codes[half:], profile, div_fast, div_start)

# --------------------------- Corner-aware polylines --------------------------

def angle_degrees(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    """Return interior angle at B formed by A-B-C in degrees."""
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
):
    """
    Emit one segment with fixed-length entry/exit windows.
    - Entry: accel from (corner_div if slow_in else div_start) → div_fast over `corner_window_steps`.
    - Cruise: div_fast mid part (if any).
    - Exit : decel from div_fast → (corner_div if slow_out else div_start) over `corner_window_steps`.
    If segment is shorter than entry+exit windows, a triangular profile is used.
    """
    N = len(step_codes)
    if N == 0: return
    start_div = corner_div if slow_in else div_start
    end_div   = corner_div if slow_out else div_start

    if N <= 2 * corner_window_steps:
        half = max(1, N // 2)
        emit_steps_accel(w, step_codes[:half], profile, div_fast, start_div)
        if N % 2 == 1:
            w.set_speed(div_fast); w.add_steps(step_codes[half:half+1]); half += 1
        emit_steps_decel(w, step_codes[half:], profile, div_fast, end_div)
        return

    entry_len = corner_window_steps
    exit_len  = corner_window_steps
    cruise_len = N - entry_len - exit_len

    entry_codes  = step_codes[:entry_len]
    cruise_codes = step_codes[entry_len:entry_len+cruise_len]
    exit_codes   = step_codes[-exit_len:]

    emit_steps_accel(w, entry_codes, profile, div_fast, start_div)
    if cruise_codes:
        w.set_speed(div_fast); w.add_steps(cruise_codes)
    emit_steps_decel(w, exit_codes, profile, div_fast, end_div)

def emit_polyline(
    w: StreamWriter,
    cfg: Config,
    *args,
    color_index: Optional[int] = None,
):
    """
    Emit a polyline with corner-aware entry/exit using fixed-length accel/decel windows.
    Signature: emit_polyline(w, cfg, pts, color_index=None)
    """
    if len(args) == 1:
        pts = args[0]
    else:
        raise TypeError("emit_polyline(w, cfg, pts, ...) expected")

    if not pts or len(pts) < 2:
        return
    if color_index is not None:
        w.select_color(color_index)

    for i in range(len(pts)-1):
        a = pts[i-1] if i-1 >= 0 else pts[i]
        b = pts[i]
        c = pts[i+1]
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
            slow_in=slow_in, slow_out=slow_out
        )

# ----------------------------- Pen-up travel -----------------------------

def travel_ramped(w: StreamWriter, *args):
    """
    Pen-up travel with fixed-length accel/decel windows (default 2 cm each side).
    Signature: travel_ramped(w, x0, y0, x1, y1, cfg)
    - Accelerate from cfg.div_start → cfg.travel_div_fast over cfg.corner_window_steps.
    - Cruise at cfg.travel_div_fast (if distance allows).
    - Decelerate from cfg.travel_div_fast → cfg.div_start over cfg.corner_window_steps.
    If the move is too short, a triangular profile is used automatically.
    """
    if len(args) != 5:
        raise TypeError("travel_ramped(w, x0, y0, x1, y1, cfg) expected")
    x0, y0, x1, y1, cfg = args

    codes = bresenham_dir_codes(x0, y0, x1, y1)
    N = len(codes)
    if N == 0:
        return

    # Use the same fixed window length as for corners; equals ≈2 cm by default.
    win = int(cfg.corner_window_steps)
    div_fast = int(cfg.travel_div_fast)
    div_start = int(cfg.div_start)

    if div_start < div_fast:
        raise ValueError("Config error: div_start must be >= travel_div_fast")

    if N <= 2 * win:
        # Triangular profile
        half = max(1, N // 2)
        emit_steps_accel(w, codes[:half], cfg.profile, div_fast, div_start)
        if N % 2 == 1:
            w.set_speed(div_fast); w.add_steps(codes[half:half+1]); half += 1
        emit_steps_decel(w, codes[half:], cfg.profile, div_fast, div_start)
        return

    # Entry → cruise → exit
    entry = codes[:win]
    cruise = codes[win:N - win]
    exitc  = codes[N - win:]

    emit_steps_accel(w, entry, cfg.profile, div_fast, div_start)
    if cruise:
        w.set_speed(div_fast); w.add_steps(cruise)
    emit_steps_decel(w, exitc, cfg.profile, div_fast, div_start)
