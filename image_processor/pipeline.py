#!/usr/bin/env python3
"""
Raster → Vector pipeline launcher.

Key behavior:
- Never overwrites the root config.json.
- Loads base settings from root config.json if present.
- Applies only CLI overrides that were explicitly provided.
- Saves the effective config to <output_dir>/config.json.
- Passes PIPELINE_CONFIG=<output_dir>/config.json to every module.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List
from config import Config  # dataclass with defaults

LINE = "=" * 50

def load_base_config(path: str = "config.json") -> dict:
    """Load base JSON if it exists; otherwise return empty dict."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def build_from_base(base: dict) -> Config:
    """Create Config and apply keys from base dict."""
    cfg = Config()
    for k, v in base.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def apply_overrides(cfg: Config, args) -> None:
    """Apply only CLI options that were explicitly provided."""
    # required positional
    cfg.input_image = args.input

    # optional overrides
    if args.output is not None:
        cfg.output_dir = args.output
    if args.max_dim is not None:
        cfg.max_dimension = args.max_dim
    if args.tolerance is not None:
        cfg.color_tolerance = args.tolerance
    if args.edge_low is not None:
        cfg.edge_low_threshold = args.edge_low
    if args.edge_high is not None:
        cfg.edge_high_threshold = args.edge_high
    if args.kernel is not None:
        cfg.edge_kernel_size = args.kernel
    if args.cores is not None:
        cfg.n_cores = args.cores
    if args.stop_edges:
        cfg.stop_after_edges = True  # only override when explicitly requested

def to_json_dict(cfg: Config) -> dict:
    """Serialize Config to a plain dict (for JSON)."""
    return {
        "input_image": cfg.input_image,
        "output_dir": cfg.output_dir,
        "n_cores": cfg.n_cores,
        "max_dimension": cfg.max_dimension,
        "colors": cfg.colors,
        "color_names": cfg.color_names,
        "color_tolerance": cfg.color_tolerance,
        "edge_low_threshold": cfg.edge_low_threshold,
        "edge_high_threshold": cfg.edge_high_threshold,
        "edge_kernel_size": cfg.edge_kernel_size,
        "min_contour_area": cfg.min_contour_area,
        "epsilon_factor": cfg.epsilon_factor,
        "dedup_distance_threshold": cfg.dedup_distance_threshold,
        "target_width_mm": cfg.target_width_mm,
        "target_height_mm": cfg.target_height_mm,
        "pixels_per_mm": cfg.pixels_per_mm,
        "stop_after_edges": cfg.stop_after_edges,
    }

def run_step(title: str, module: str, env: dict) -> None:
    """Run a single module and stop on error."""
    print(f"\n[{title}]")
    try:
        result = subprocess.run(
            [sys.executable, module],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        print(f"Error: module not found: {module}")
        sys.exit(1)

    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"Error in {module}:")
        if result.stderr:
            print(result.stderr.strip())
        sys.exit(1)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Raster → Vector processing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to input image")
    # All optional args default to None so we only override when explicitly set
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--max-dim", type=int, default=None, help="Max side after resize (px)")
    p.add_argument("--tolerance", type=int, default=None, help="RGB tolerance")
    p.add_argument("--edge-low", type=int, default=None, help="Canny low threshold")
    p.add_argument("--edge-high", type=int, default=None, help="Canny high threshold")
    p.add_argument("--kernel", type=int, default=None, help="Gaussian kernel size (odd)")
    p.add_argument("--cores", type=int, default=None, help="CPU cores to use")
    p.add_argument("--stop-edges", action="store_true", help="Stop after edge detection")
    return p

def main() -> None:
    args = build_argparser().parse_args()

    # 1) base config from root (if any), without writing it back
    base = load_base_config("config.json")
    cfg = build_from_base(base)

    # 2) apply only explicit CLI overrides
    apply_overrides(cfg, args)

    # 3) persist effective config into <output_dir>/config.json
    os.makedirs(cfg.output_dir, exist_ok=True)
    effective_cfg_path = os.path.join(cfg.output_dir, "config.json")
    with open(effective_cfg_path, "w") as f:
        json.dump(to_json_dict(cfg), f, indent=2)

    # 4) env for subprocesses: tell modules where to load the config from
    env = os.environ.copy()
    env["PIPELINE_CONFIG"] = effective_cfg_path

    # Header
    print(LINE)
    print("RASTER → VECTOR PIPELINE")
    print(LINE)
    print(f"Input image: {cfg.input_image}")
    print(f"Output dir:  {cfg.output_dir}")
    print(f"Config saved to {effective_cfg_path}")

    steps: List[str] = [
        "01_resize.py",
        "02_color_extract.py",
        "03_edge_detect.py",
        "04_find_contours.py",
        "05_sort_contours.py",
        "06_dedup_layer.py",
        "07_dedup_cross.py",
        "08_simplify.py",
        "09_scale_vector.py",
        "10_preview.py"
    ]

    titles: List[str] = [
        "1/10 Image resize…",
        "2/10 RGBK color extraction…",
        "3/10 Edge detection…",
        "4/10 Find contours…",
        "5/10 Sort contours…",
        "6/10 Intra-layer dedup…",
        "7/10 Cross-layer dedup…",
        "8/10 Simplify contours…",
        "9/10 Scale & export vectors…",
        "10/10 Vector preview (PNG)…"
    ]

    for t, m in zip(titles, steps):
        run_step(t, m, env)
        if cfg.stop_after_edges and m == "03_edge_detect.py":
            print("\nStopping after edge detection as requested.")
            break

if __name__ == "__main__":
    main()
