# 14_preview_stream.py
# Wrapper: launch shared/xyplotter_stream_previewer.py on plot_stream.bin with proper canvas & palette.

from __future__ import annotations
import os
import sys
import json
import subprocess
from typing import Tuple, List

import cv2  # only for target size fallback if needed (kept consistent with other steps)

from config import load_config, Config

def _target_size_px(cfg: Config) -> Tuple[int, int]:
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 and th > 0:
        return tw, th
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("Cannot infer target size; set target_* or run earlier steps.")
    h, w = base.shape[:2]
    return w, h

def _palette_args(bgr_list: List[Tuple[int,int,int]]) -> List[str]:
    # Previewer expects RGB; cfg.colors are BGR. Clamp to 4 entries.
    args: List[str] = []
    for i in range(min(4, len(bgr_list))):
        b, g, r = bgr_list[i]
        args += [f"--c{i}", f"{int(r)},{int(g)},{int(b)}"]
    # Fill missing with defaults if fewer than 4 colors present
    defaults = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
    for i in range(len(bgr_list), 4):
        r,g,b = defaults[i]
        args += [f"--c{i}", f"{r},{g},{b}"]
    return args

def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    stream_path = os.path.join(outdir, "plot_stream.bin")
    if not os.path.exists(stream_path):
        raise SystemExit(f"[preview] Missing stream: {stream_path}. Run step 13 first.")

    W, H = _target_size_px(cfg)

    here = os.path.dirname(os.path.abspath(__file__))
    previewer = os.path.normpath(os.path.join(here, "..", "shared", "xyplotter_stream_previewer.py"))
    if not os.path.exists(previewer):
        raise SystemExit(f"[preview] Missing previewer script: {previewer}")

    # Build CLI for previewer
    argv = [
        sys.executable, previewer,
        stream_path,
        "--canvas-w-steps", str(W),
        "--canvas-h-steps", str(H),
        "--invert-y", "1",
        "--background-white", "1",
        "--render-taps", "1",
        "--tick-freq", "10000",
        "--render-width", str(int(getattr(cfg, "preview_render_width_px", 1200))),
        "--render-height", str(int(getattr(cfg, "preview_render_height_px", 900))),
    ] + _palette_args(getattr(cfg, "colors", []))

    print("[preview] launching previewer…")
    print("[preview] cmd:", " ".join(argv))

    # Forward stdout/stderr; interactive UI will open
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(argv, env=env)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"[preview] previewer exited with code {proc.returncode}")

if __name__ == "__main__":
    main()
