from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os

BGR = Tuple[int, int, int]


@dataclass
class Config:
    # I/O
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

    # Internal page margins (default 10 mm each side).
    # All downstream geometry (starting from scaling) is fitted into A4 minus these margins.
    margin_left_mm: float = 10.0
    margin_right_mm: float = 10.0
    margin_top_mm: float = 10.0
    margin_bottom_mm: float = 10.0

    # Pen geometry
    pen_width_px: int = 60                 # ~1.5 mm with 40 px/mm
    pen_radius_px: int = 30                # convenience

    # Tap (dot) detection thresholds
    tap_max_area: float = 1200.0           # px^2
    tap_max_perimeter: float = 160.0       # px
    tap_max_dim: int = 25                  # px (max bbox dimension)
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

    # Stream color remap (optional)
    stream_force_color_index: Optional[int] = None
    stream_color_by_name: Optional[Dict[str, int]] = None
    stream_color_by_order: Optional[List[int]] = None

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for name in self.color_names:
            os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)


def _merge_defaults(base: dict, override: dict | None) -> dict:
    """Shallow-merge helper: returns {**base, **override} (override may be None)."""
    out = dict(base)
    if override:
        out.update(override)
    return out


def load_config(path: str | None = None) -> Config:
    """
    Load configuration from JSON (path or CONFIG_PATH env var). Unknown keys are ignored.
    On failure, returns default Config().
    """
    import json
    p = path or os.environ.get("CONFIG_PATH")
    if not p:
        return Config()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[config] WARNING: failed to read JSON ({e}); using defaults.")
        return Config()

    # Keep only dataclass fields; ignore extras
    fields = set(Config.__dataclass_fields__.keys())
    known = {k: v for k, v in data.items() if k in fields}

    cfg = Config(**known)
    # Convenience: stash raw JSON and source path
    setattr(cfg, "_raw", data)
    setattr(cfg, "_path", p)
    print(f"[config] Loading config: {p} (exists=True)")
    return cfg
