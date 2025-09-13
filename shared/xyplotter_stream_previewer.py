#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotter Preview — step/service stream visualizer.

Stream format (bytes):
- Step byte (MSB=1):
    bit7=1
    bit6=1 → two steps in one byte: 11 FFF SSS   (FFF=first step 0..7, SSS=second step 0..7)
    bit6=0 → single step:           10 SSS 000   (SSS=step 0..7)
- Service byte (MSB=0):
    0x01 = Pen Up
    0x02 = Pen Down
    0x03 = Tap (draw filled dot; pen returns Up)
    0x08..0x0F = Select Color (index 0..7; preview clamps to palette length)
    0x40..0x7F = Set Speed divider (0..63)  [affects stats; preview playback ignores timing]
    0x3F = EOF (end of stream)
All other values are treated as unknown service bytes and ignored with a warning.

UI:
- SPACE play/pause, R reset
- → step +100 commands, ← step -100 commands
- +/- zoom; mouse wheel zoom
- Slider to seek
- Buttons: Play, Pause, Reset, Step, 0.5x/2x playback multipliers
"""

from __future__ import annotations
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pygame
from PIL import Image

# ---------------------------- Color Helpers ----------------------------

def parse_color(spec: str) -> Tuple[int,int,int]:
    s = spec.strip().lower()
    named = {
        'r': (255, 0, 0), 'red': (255, 0, 0),
        'g': (0, 255, 0), 'green': (0, 255, 0),
        'b': (0, 0, 255), 'blue': (0, 0, 255),
        'k': (0, 0, 0), 'black': (0, 0, 0),
        'w': (255, 255, 255), 'white': (255, 255, 255),
        'y': (255, 255, 0), 'yellow': (255, 255, 0),
        'c': (0, 255, 255), 'cyan': (0, 255, 255),
        'm': (255, 0, 255), 'magenta': (255, 0, 255),
    }
    if s in named:
        return named[s]
    if s.startswith('#') and len(s) == 7:
        r = int(s[1:3], 16); g = int(s[3:5], 16); b = int(s[5:7], 16)
        return (r, g, b)
    if ',' in s:
        r, g, b = (int(p) for p in s.split(','))
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    raise ValueError(f"Bad color spec: {spec}")

# ---------------------------- Config & State ----------------------------

@dataclass
class Config:
    render_width_px: int = 1200
    render_height_px: int = 900
    canvas_steps_w: int = 13210
    canvas_steps_h: int = 13019
    invert_y: bool = True
    render_taps: bool = True
    tick_frequency: int = 10000   # base “tick” for playback; UI multiplies this
    colors: tuple = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0))  # palette indices 0..3
    background_white: bool = True

# diameter in screen pixels for tap visualization per color index (0..3)
PEN_DIAM_PX = (10, 10, 10, 10)

@dataclass
class PlotterState:
    x: int = 0       # step space X
    y: int = 0       # step space Y
    pen_down: bool = False
    color_idx: int = 0    # palette index

@dataclass
class Statistics:
    total_bytes: int = 0
    service_bytes: int = 0
    step_bytes: int = 0
    single_steps: int = 0
    double_steps: int = 0
    steps_total: int = 0
    pen_down_segments: int = 0
    color_changes: int = 0
    speed_changes: int = 0
    eof_seen: bool = False
    tail_after_eof: int = 0
    off_canvas_draws: int = 0
    final_x: int = 0
    final_y: int = 0

# ---------------------------- Decoder ----------------------------

# Direction codes: 0=+Y, 1=NE, 2=+X, 3=SE, 4=-Y, 5=SW, 6=-X, 7=NW
STEP_DIRS = {
    0: (0, +1), 1: (+1, +1), 2: (+1, 0), 3: (+1, -1),
    4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, +1),
}

class StreamDecoder:
    """Parse a binary stream into ('service', code|('color',idx)|('speed',div)|('step',dir))."""

    def __init__(self, data: bytes):
        self.data = data
        self.stats = Statistics(total_bytes=len(data))
        self.commands: List[Tuple[str, int]] = []
        self._decode()

    @staticmethod
    def _is_step_byte(b: int) -> bool:
        return (b & 0x80) != 0

    def _decode(self):
        n = len(self.data)
        i = 0
        eof_index: Optional[int] = None

        while i < n:
            b = self.data[i]

            # Step byte
            if self._is_step_byte(b):
                self.stats.step_bytes += 1
                if (b & 0x40):  # two steps
                    first = (b >> 3) & 0x07
                    second = b & 0x07
                    self.commands.append(('step', first))
                    self.commands.append(('step', second))
                    self.stats.double_steps += 1
                    self.stats.steps_total += 2
                else:            # single step
                    code = (b >> 3) & 0x07
                    self.commands.append(('step', code))
                    self.stats.single_steps += 1
                    self.stats.steps_total += 1
                i += 1
                continue

            # Service byte
            self.stats.service_bytes += 1

            # EOF
            if b == 0x3F:
                self.stats.eof_seen = True
                eof_index = i
                i += 1
                break

            # Pen control
            if b in (0x01, 0x02, 0x03):
                self.commands.append(('service', b))
                i += 1
                continue

            # Color select 0..7
            if 0x08 <= b <= 0x0F:
                idx = b & 0x07
                self.commands.append(('color', idx))
                i += 1
                continue

            # Speed set (divider)
            if (b & 0xC0) == 0x40:
                self.commands.append(('speed', b & 0x3F))
                self.stats.speed_changes += 1
                i += 1
                continue

            # Unknown service byte
            sys.stderr.write(f"WARNING: Unknown service byte 0x{b:02X} at offset {i}, ignored.\n")
            i += 1

        if eof_index is not None and eof_index < n:
            self.stats.tail_after_eof = n - (eof_index + 1)

# ---------------------------- Simulator & Renderer ----------------------------

class PlotterPreview:
    def __init__(self, decoder: StreamDecoder, cfg: Config):
        self.decoder = decoder
        self.cfg = cfg
        self.stats = decoder.stats
        self.state = PlotterState()
        self.palette = list(cfg.colors)
        self.cur_color = self._color_for_index(self.state.color_idx)

        # Playback
        self.playing = False
        self.current_command = 0
        self.tick_hz = cfg.tick_frequency
        self.speed_mult = 1.0
        self._tick_accum = 0.0

        # Pygame init
        pygame.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.controls_h = 150

        # Window & surfaces
        info = pygame.display.Info()
        init_w = max(800, int(info.current_w * 0.8))
        init_h = max(600, int(info.current_h * 0.8))
        self.win_w, self.win_h = init_w, init_h
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Plotter Preview")
        self.clock = pygame.time.Clock()

        # Render surface (where drawing accumulates)
        self._rebuild_render_surface()

        # UI
        self._rebuild_ui()
        print(f"Canvas steps: {self.cfg.canvas_steps_w}x{self.cfg.canvas_steps_h}")
        print(f"Render surface: {self.render_w}x{self.render_h} px; window: {self.win_w}x{self.win_h}")

    # ------------- geometry & transforms -------------
    def _rebuild_render_surface(self):
        # Available drawing area inside window (exclude controls)
        avail_w = self.win_w - 40
        avail_h = self.win_h - self.controls_h - 40
        self.render_w = max(400, min(self.cfg.render_width_px, avail_w))
        self.render_h = max(300, min(self.cfg.render_height_px, avail_h))

        self.render_surface = pygame.Surface((self.render_w, self.render_h))
        bg = (255, 255, 255) if self.cfg.background_white else (0, 0, 0)
        self.render_surface.fill(bg)

        # Uniform scale to preserve aspect ratio
        sx = self.render_w / max(1, self.cfg.canvas_steps_w)
        sy = self.render_h / max(1, self.cfg.canvas_steps_h)
        self.step_scale = min(sx, sy)
        # Center the drawing area with small margins
        used_w = int(self.cfg.canvas_steps_w * self.step_scale)
        used_h = int(self.cfg.canvas_steps_h * self.step_scale)
        self.offset_x = (self.render_w - used_w) // 2
        self.offset_y = (self.render_h - used_h) // 2

    def _rebuild_ui(self):
        base_y = self.win_h - self.controls_h + 10
        self.buttons = {
            'play': pygame.Rect(10, base_y, 80, 40),
            'pause': pygame.Rect(100, base_y, 80, 40),
            'reset': pygame.Rect(190, base_y, 80, 40),
            'step': pygame.Rect(280, base_y, 80, 40),
            'speed_down': pygame.Rect(370, base_y, 40, 40),
            'speed_up': pygame.Rect(420, base_y, 40, 40),
        }
        slider_w = max(200, min(self.win_w - 40, 1200))
        self.slider_rect = pygame.Rect(10, base_y + 50, slider_w, 20)
        self.slider_handle = pygame.Rect(10, base_y + 45, 10, 30)
        self.dragging_slider = False

    def _on_resize(self, w, h):
        if (w, h) != (self.win_w, self.win_h):
            self.win_w, self.win_h = w, h
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self._rebuild_render_surface()
        self._rebuild_ui()
        # Repaint up to current command after resize
        self._replay_to(self.current_command)

    # ------------- transforms -------------
    def _steps_to_px(self, x: int, y: int) -> Tuple[int, int]:
        # Apply uniform scale and center offsets
        px = int(self.offset_x + x * self.step_scale)
        py = int(self.offset_y + (self.cfg.canvas_steps_h - 1 - y) * self.step_scale) if self.cfg.invert_y \
             else int(self.offset_y + y * self.step_scale)
        return px, py

    # ------------- drawing helpers -------------
    def _draw_line_steps(self, x1: int, y1: int, x2: int, y2: int):
        x1p, y1p = self._steps_to_px(x1, y1)
        x2p, y2p = self._steps_to_px(x2, y2)
        # Clip count (optional; we skip strict check, lines outside will be clipped by pygame)
        pygame.draw.line(self.render_surface, self.cur_color, (x1p, y1p), (x2p, y2p), 1)

    def _draw_tap_steps(self, x: int, y: int, diam_px: int):
        px, py = self._steps_to_px(x, y)
        r = max(1, diam_px // 2)
        pygame.draw.circle(self.render_surface, self.cur_color, (px, py), r, 0)

    # ------------- simulation -------------
    def _color_for_index(self, idx: int):
        # clamp to palette length
        return self.palette[min(idx, len(self.palette) - 1)]

    def _process_one(self):
        if self.current_command >= len(self.decoder.commands):
            self.playing = False
            return
        t, v = self.decoder.commands[self.current_command]

        if t == 'service':
            if v == 0x01:       # pen up
                self.state.pen_down = False
            elif v == 0x02:     # pen down
                # Count start of a new stroke
                if not self.state.pen_down:
                    self.stats.pen_down_segments += 1
                self.state.pen_down = True
            elif v == 0x03:     # tap
                if self.cfg.render_taps:
                    diam = PEN_DIAM_PX[min(self.state.color_idx, len(PEN_DIAM_PX)-1)]
                    self._draw_tap_steps(self.state.x, self.state.y, diam)
                self.state.pen_down = False

        elif t == 'color':
            self.state.color_idx = int(v)
            self.cur_color = self._color_for_index(self.state.color_idx)
            self.stats.color_changes += 1
            # No implicit pen state change here (pen up/down handled explicitly)

        elif t == 'speed':
            # Playback ignores divider for timing; we only count it
            pass

        elif t == 'step':
            dx, dy = STEP_DIRS.get(int(v), (0, 0))
            x0, y0 = self.state.x, self.state.y
            self.state.x += dx
            self.state.y += dy
            if self.state.pen_down and (x0 != self.state.x or y0 != self.state.y):
                self._draw_line_steps(x0, y0, self.state.x, self.state.y)

        self.current_command += 1
        self.stats.final_x, self.stats.final_y = self.state.x, self.state.y

    def _reset(self):
        self.state = PlotterState()
        self.cur_color = self._color_for_index(self.state.color_idx)
        self.current_command = 0
        self.playing = False
        self._tick_accum = 0.0
        bg = (255, 255, 255) if self.cfg.background_white else (0, 0, 0)
        self.render_surface.fill(bg)

    def _replay_to(self, target_idx: int):
        target_idx = max(0, min(len(self.decoder.commands), target_idx))
        self._reset()
        while self.current_command < target_idx:
            self._process_one()

    # ------------- UI drawing -------------
    def _btn(self, name, label):
        rect = self.buttons[name]
        mouse = pygame.mouse.get_pos()
        hot = rect.collidepoint(mouse)
        color = (120, 120, 220) if hot else (100, 100, 200)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (50, 50, 50), rect, 2)
        txt = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_slider(self):
        pygame.draw.rect(self.screen, (200, 200, 200), self.slider_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), self.slider_rect, 2)
        total = len(self.decoder.commands)
        if total:
            progress = self.current_command / total
            filled = int(self.slider_rect.width * progress)
            filled_rect = pygame.Rect(self.slider_rect.x, self.slider_rect.y, filled, self.slider_rect.height)
            pygame.draw.rect(self.screen, (100, 200, 100), filled_rect)
            self.slider_handle.x = self.slider_rect.x + filled - 5
            pygame.draw.rect(self.screen, (50, 50, 150), self.slider_handle)

    def _draw_info(self):
        base_y = self.win_h - self.controls_h + 95
        info = [
            f"Cmd: {self.current_command}/{len(self.decoder.commands)}",
            f"Pos: ({self.state.x}, {self.state.y})  Pen: {'DOWN' if self.state.pen_down else 'UP'}",
            f"Playback: {self.speed_mult:.1f}x  Tick: {self.tick_hz} Hz",
            f"Bytes: {self.stats.total_bytes}  StepBytes: {self.stats.step_bytes}  ServiceBytes: {self.stats.service_bytes}",
            f"Steps: {self.stats.steps_total}  Singles: {self.stats.single_steps}  Doubles: {self.stats.double_steps}",
            f"Color changes: {self.stats.color_changes}  Speed changes: {self.stats.speed_changes}",
            f"Final: ({self.stats.final_x}, {self.stats.final_y})",
        ]
        for i, s in enumerate(info):
            surf = self.small_font.render(s, True, (35, 35, 35))
            self.screen.blit(surf, (10 + (i % 3) * 340, base_y + (i // 3) * 22))

        # Palette legend
        lx = self.win_w - 220
        ly = base_y - 10
        self.screen.blit(self.small_font.render("Palette:", True, (40, 40, 40)), (lx, ly))
        for i, c in enumerate(self.palette):
            pygame.draw.rect(self.screen, c, (lx + 80 + i * 22, ly, 18, 18))
            pygame.draw.rect(self.screen, (20, 20, 20), (lx + 80 + i * 22, ly, 18, 18), 1)

    # ------------- main loop -------------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

                elif e.type == pygame.VIDEORESIZE:
                    self._on_resize(e.w, e.h)

                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    pos = e.pos
                    if self.buttons['play'].collidepoint(pos): self.playing = True
                    elif self.buttons['pause'].collidepoint(pos): self.playing = False
                    elif self.buttons['reset'].collidepoint(pos): self._reset()
                    elif self.buttons['step'].collidepoint(pos):
                        self.playing = False; self._process_one()
                    elif self.buttons['speed_down'].collidepoint(pos): self.speed_mult = max(0.1, self.speed_mult / 2)
                    elif self.buttons['speed_up'].collidepoint(pos): self.speed_mult = min(100.0, self.speed_mult * 2)
                    if self.slider_rect.collidepoint(pos): self.dragging_slider = True

                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    self.dragging_slider = False

                elif e.type == pygame.MOUSEMOTION and self.dragging_slider:
                    rel_x = e.pos[0] - self.slider_rect.x
                    p = max(0.0, min(1.0, rel_x / self.slider_rect.width))
                    self._replay_to(int(p * len(self.decoder.commands)))

                elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (4, 5):  # wheel
                    factor = 1.1 if e.button == 4 else (1/1.1)
                    # zoom by rebuilding render surface size target
                    self.cfg.render_width_px = int(self.cfg.render_width_px * factor)
                    self.cfg.render_height_px = int(self.cfg.render_height_px * factor)
                    self._rebuild_render_surface()
                    self._replay_to(self.current_command)

                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE: self.playing = not self.playing
                    elif e.key == pygame.K_r: self._reset()
                    elif e.key == pygame.K_RIGHT:
                        self.playing = False
                        for _ in range(100): self._process_one()
                    elif e.key == pygame.K_LEFT:
                        self._replay_to(max(0, self.current_command - 100))
                    elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.cfg.render_width_px = int(self.cfg.render_width_px * 1.2)
                        self.cfg.render_height_px = int(self.cfg.render_height_px * 1.2)
                        self._rebuild_render_surface(); self._replay_to(self.current_command)
                    elif e.key == pygame.K_MINUS:
                        self.cfg.render_width_px = int(self.cfg.render_width_px / 1.2)
                        self.cfg.render_height_px = int(self.cfg.render_height_px / 1.2)
                        self._rebuild_render_surface(); self._replay_to(self.current_command)

            # playback
            if self.playing and self.current_command < len(self.decoder.commands):
                self._tick_accum += dt * self.tick_hz * self.speed_mult
                steps = int(min(self._tick_accum, 5000))
                if steps > 0:
                    for _ in range(steps):
                        if self.current_command >= len(self.decoder.commands): break
                        self._process_one()
                    self._tick_accum -= steps

            # draw frame
            self.screen.fill((240, 240, 240))
            # Place render surface centered
            cx = (self.win_w - self.render_w) // 2
            cy = (self.win_h - self.controls_h - self.render_h) // 2
            self.screen.blit(self.render_surface, (cx, cy))
            pygame.draw.rect(self.screen, (100, 100, 100), (cx, cy, self.render_w, self.render_h), 2)

            # cursor
            cur_px, cur_py = self._steps_to_px(self.state.x, self.state.y)
            pygame.draw.circle(self.screen, self.cur_color if self.state.pen_down else (0, 200, 0),
                               (cx + cur_px, cy + cur_py), 5)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx + cur_px, cy + cur_py), 5, 1)

            # UI
            for k, label in [('play','Play'),('pause','Pause'),('reset','Reset'),
                             ('step','Step'),('speed_down','0.5x'),('speed_up','2x')]:
                self._btn(k, label)
            self._draw_slider()
            self._draw_info()
            pygame.display.flip()

        pygame.quit()
        self._print_stats()

    def _print_stats(self):
        e = self.stats
        print("\n=== Statistics ===", file=sys.stderr)
        print(f"Total bytes: {e.total_bytes}", file=sys.stderr)
        print(f"Step bytes: {e.step_bytes}  Service bytes: {e.service_bytes}", file=sys.stderr)
        print(f"Steps: {e.steps_total}  Singles: {e.single_steps}  Doubles: {e.double_steps}", file=sys.stderr)
        print(f"Color changes: {e.color_changes}  Speed changes: {e.speed_changes}", file=sys.stderr)
        print(f"EOF seen: {e.eof_seen}  Tail after EOF: {e.tail_after_eof}", file=sys.stderr)
        print(f"Final position: ({e.final_x}, {e.final_y})", file=sys.stderr)

    def save_image(self, path: str):
        # Render the whole stream, then save render surface
        self._replay_to(len(self.decoder.commands))
        w, h = self.render_surface.get_size()
        buf = pygame.image.tostring(self.render_surface, 'RGB')
        Image.frombytes('RGB', (w, h), buf).save(path)
        print(f"Image saved: {path}")

# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Plotter Preview (step/service stream visualizer)")
    ap.add_argument('input', help='Input binary stream file')
    ap.add_argument('-o', '--output', help='Save rendered PNG to file')

    # Render surface (pixels)
    ap.add_argument('--render-width', type=int, default=1200, help='Render surface width in pixels')
    ap.add_argument('--render-height', type=int, default=900, help='Render surface height in pixels')

    # Canvas in steps (mechanical workspace)
    ap.add_argument('--canvas-w-steps', type=int, default=13210)
    ap.add_argument('--canvas-h-steps', type=int, default=13019)
    ap.add_argument('--invert-y', type=int, choices=[0,1], default=1)
    ap.add_argument('--background-white', type=int, choices=[0,1], default=1)
    ap.add_argument('--render-taps', type=int, choices=[0,1], default=1)
    ap.add_argument('--tick-freq', type=int, default=10000)

    # Palette (color indices)
    ap.add_argument('--c0', default='R'); ap.add_argument('--c1', default='G')
    ap.add_argument('--c2', default='B'); ap.add_argument('--c3', default='K')

    args = ap.parse_args()
    palette = (parse_color(args.c0), parse_color(args.c1),
               parse_color(args.c2), parse_color(args.c3))

    cfg = Config(
        render_width_px=args.render_width,
        render_height_px=args.render_height,
        canvas_steps_w=args.canvas_w_steps,
        canvas_steps_h=args.canvas_h_steps,
        invert_y=bool(args.invert_y),
        background_white=bool(args.background_white),
        render_taps=bool(args.render_taps),
        tick_frequency=args.tick_freq,
        colors=palette,
    )

    data = open(args.input, 'rb').read()
    dec = StreamDecoder(data)
    sim = PlotterPreview(dec, cfg)

    if args.output:
        sim.save_image(args.output)
        sim._print_stats()
    else:
        print("Controls: SPACE play/pause, R reset, arrows seek, +/- zoom, wheel zoom, slider seek.")
        sim.run()

if __name__ == '__main__':
    main()
