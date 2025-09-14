# config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import os
import json

BGR = Tuple[int, int, int]

@dataclass
class Config:
    input_image: str = "input.png"
    output_dir: str = "output"
    n_cores: int = 12

    max_dimension: int = 2000

    colors: List[BGR] = field(default_factory=lambda: [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)
    ])
    color_names: List[str] = field(default_factory=lambda: ["black", "red", "green", "blue"])
    color_tolerance: int = 30

    edge_low_threshold: int = 50
    edge_high_threshold: int = 150
    edge_kernel_size: int = 3

    min_contour_area: float = 10.0
    epsilon_factor: float = 0.002
    dedup_distance_threshold: float = 5.0

    target_width_mm: int = 210
    target_height_mm: int = 297
    pixels_per_mm: int = 40

    edge_morph_kernel: int = 3
    edge_morph_open_iters: int = 1
    edge_morph_close_iters: int = 1
    smoothing_iterations: int = 2

    # Iterations for in-layer conflict resolution
    dedup_max_passes: int = 10

    pen_width_px: int = 60                  # 1.5 mm @ 40 px/mm
    pen_radius_px: int = 30                 # convenience = pen_width_px // 2

    # taps (point strokes)
    tap_max_area: float = 1200.0            # px^2 → convert tiny contours to taps
    tap_max_perimeter: float = 160.0        # px   → or short perimeters to taps
    tap_max_dim: int = 25                   # px   → or small bbox turns into a tap
    tap_merge_radius_px: int = 30           # px   → collapse nearby taps until stable

    # thinning
    thinning_min_segment_len: int = 5       # points; discard ultra short fragments
    thinning_dt_margin: float = 0.0         # extra safety distance over pen_radius    

    pen_width_px: int = 60           # pen width in pixels (no double passes within this)
    dedup_sample_step: int = 8       # subsampling step for distance check
    
    dedup_overlap_threshold: float = 0.60  # drop if ≥60% of stroke overlaps kept strokes
    dedup_draw_antialiased: bool = False   # AA off → crisp binary overlap

    stop_after_edges: bool = False

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for name in self.color_names:
            os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)

def load_config() -> Config:
    """
    Load config from JSON pointed to by PIPELINE_CONFIG, then ./config.json, otherwise defaults.
    """
    cfg = Config()
    path = os.environ.get("PIPELINE_CONFIG", "config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg
