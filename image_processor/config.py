from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import os
import json

BGR = Tuple[int, int, int]


@dataclass
class Config:
    # IO
    input_image: str = "input.png"
    output_dir: str = "output"
    n_cores: int = 12

    # Resize cap for the input raster (longest side in px)
    max_dimension: int = 2000

    # Color layer names (order matters for dark→light logic elsewhere)
    color_names: List[str] = field(
        default_factory=lambda: ["layer_dark", "layer_mid", "layer_skin", "layer_light"]
    )

    # Optional BGR swatches; used by color extraction if needed
    colors: List[BGR] = field(
        default_factory=lambda: [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    )
    color_tolerance: int = 30

    # Edge detection
    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    edge_kernel_size: int = 3
    edge_morph_kernel: int = 3
    edge_morph_open_iters: int = 1
    edge_morph_close_iters: int = 1
    smoothing_iterations: int = 2

    # Contours / vectorization
    min_contour_area: float = 10.0
    epsilon_factor: float = 0.002  # used by optional simplify
    dedup_max_passes: int = 10

    # Plotter geometry (A4 @ 40 px/mm by default)
    target_width_mm: int = 210
    target_height_mm: int = 297
    pixels_per_mm: int = 40

    # Pen geometry
    pen_width_px: int = 60                 # 1.5 mm with 40 px/mm
    pen_radius_px: int = 30                # convenience

    # Tap (dot) detection thresholds
    tap_max_area: float = 1200.0           # px^2
    tap_max_perimeter: float = 160.0       # px
    tap_max_dim: int = 25                  # px bbox max dimension
    tap_merge_radius_px: int = 30          # px

    # Thinning / centerline (if used by any step)
    thinning_min_segment_len: int = 5
    thinning_dt_margin: float = 0.0

    # In-layer dedup parameters (conservative defaults)
    dedup_sample_step: int = 8
    dedup_overlap_threshold: float = 0.60
    dedup_draw_antialiased: bool = False

    # Hash-based collision guards
    ignore_tail_points_intra: int = 120
    collision_radius_intra_px: float = 18.0
    collision_radius_global_px: float = 21.0
    hash_stride_px: float = 18.0
    max_join_jump_px: float = 80.0

    # Optional simplifier
    simplify_enabled: bool = False

    # Debug helper: stop early after edge maps
    stop_after_edges: bool = False

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for name in self.color_names:
            os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)


def _merge_defaults(base: dict, override: dict | None) -> dict:
    out = dict(base)
    if override:
        out.update(override)
    return out


def load_config(path: str | None = None) -> Config:
    """
    Load config JSON with this priority:
      1) explicit path argument,
      2) env CONFIG_PATH,
      3) ./config.json.
    Falls back to dataclass defaults if missing or invalid.
    """
    cfg_path = path or os.environ.get("CONFIG_PATH") or "config.json"
    cfg_path = os.path.abspath(cfg_path)
    print(f"[config] Loading config: {cfg_path} (exists={os.path.exists(cfg_path)})")

    defaults = asdict(Config())

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = _merge_defaults(defaults, data)
            return Config(**merged)
        except Exception as e:
            print(f"[config] WARNING: failed to read JSON ({e}); using defaults.")

    return Config(**defaults)
