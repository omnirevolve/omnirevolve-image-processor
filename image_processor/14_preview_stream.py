#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[14/14] Preview stream — launches the visualizer with canvas parameters from config/stream_meta.
"""

from __future__ import annotations
import os
import json
import subprocess
import sys
from pathlib import Path
from config import load_config

def main():
    cfg = load_config()
    outdir = Path(cfg.output_dir)

    stream = outdir / "plot_stream.bin"
    if not stream.exists():
        raise SystemExit(f"[preview] ERROR: stream file not found: {stream}")

    # Read canvas steps/inversion so the preview matches what goes to the plotter
    meta_path = outdir / "stream_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        W, H = meta.get("canvas_steps", [8400, 11880])
        invert_y = 1 if meta.get("invert_y", True) else 0
    else:
        W = int(getattr(cfg, "target_width_mm", 210) * getattr(cfg, "steps_per_mm", 40))
        H = int(getattr(cfg, "target_height_mm", 297) * getattr(cfg, "steps_per_mm", 40))
        invert_y = 1  # default to mirrored Y like on the plotter

    # RGBK palette as a conventional mapping
    palette = ["255,0,0", "0,255,0", "0,0,255", "0,0,0"]

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "../shared" / "omnirevolve_plotter_stream_previewer.py").resolve()),
        str(stream),
        "--canvas-w-steps", str(W),
        "--canvas-h-steps", str(H),
        "--invert-y", str(invert_y),
        "--background-white", "1",
        "--render-taps", "1",
        "--tick-freq", "10000",
        "--render-width", "1200",
        "--render-height", "900",
        "--c0", palette[0], "--c1", palette[1], "--c2", palette[2], "--c3", palette[3],
    ]

    print("[preview] launching previewer…")
    print("[preview] cmd:", " ".join(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=False)

if __name__ == "__main__":
    main()
