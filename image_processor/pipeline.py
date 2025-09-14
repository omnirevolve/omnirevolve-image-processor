# Raster → Vector pipeline runner with step range, streamed logs, and CONFIG_PATH propagation.
import argparse
import json
import os
import sys
import subprocess
from dataclasses import asdict
from typing import List, Tuple


# ----------------- config I/O -----------------
def load_default_config():
    # Lazy import to avoid importing project code at CLI parse time
    from config import Config
    return Config()


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def config_path(outdir: str) -> str:
    return os.path.join(outdir, "config.json")


def write_config(cfg_obj, outdir: str, overrides: dict):
    """
    Persist config to outdir/config.json.
    If a file already exists, we merge it with CLI overrides (CLI wins).
    """
    dst = config_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Start from current on-disk config if present
    if os.path.exists(dst):
        try:
            with open(dst, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    else:
        merged = asdict(cfg_obj)

    # Apply CLI overrides (do not write None values)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return dst


def read_existing_config(outdir: str):
    p = config_path(outdir)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------- steps -----------------
def module_path(preferred: str, fallback: str | None = None) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.join(here, preferred)
    if os.path.exists(cand1):
        return cand1
    if fallback:
        cand2 = os.path.join(here, fallback)
        if os.path.exists(cand2):
            return cand2
    return cand1  # let subprocess fail clearly if truly missing


def build_steps() -> List[Tuple[str, str]]:
    return [
        ("[1/10] Image resize…",            module_path("01_resize.py")),
        ("[2/10] RGBK color extraction…",   module_path("02_color_extract.py")),
        ("[3/10] Edge detection…",          module_path("03_edge_detect.py")),
        ("[4/10] Find contours…",           module_path("04_find_contours.py")),
        ("[5/10] Scale vectors…",           module_path("05_scale_vectors.py")),
        ("[6/10] Sort contours…",           module_path("06_sort_contours.py")),
        ("[7/10] Intra-layer dedup…",       module_path("07_dedup_layer_basic.py")),
        ("[8/10] Cross-layer dedup…",       module_path("08_dedup_cross_basic.py")),
        ("[9/10] Simplify contours…",       module_path("09_simplify.py")),
        ("[10/10] Preview…",                module_path("10_preview.py")),
    ]


# ----------------- stdout streaming -----------------
def run_step(title: str, module: str, env: dict) -> None:
    print(f"\n{title}")
    try:
        proc = subprocess.Popen(
            [sys.executable, module],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,  # line-buffered
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        raise
    if proc.returncode != 0:
        print(f"\nError in {module} (exit={proc.returncode})")
        sys.exit(1)


# ----------------- prereq checks (lightweight) -----------------
def missing_for_step(step_idx: int, outdir: str, color_names: List[str]) -> List[str]:
    """Return a list of expected files that are missing for a given start step."""
    need: List[str] = []
    # step_idx is 1-based
    if step_idx >= 2:
        need.append(os.path.join(outdir, "resized.png"))
    if step_idx >= 3:
        need += [os.path.join(outdir, c, "mask.png") for c in color_names]
    if step_idx >= 4:
        need += [os.path.join(outdir, c, "edges.png") for c in color_names]
    if step_idx >= 5:
        need += [os.path.join(outdir, c, "contours.pkl") for c in color_names]
    if step_idx >= 6:
        # sorter typically consumes contours_scaled.pkl but can fall back;
        # we don't hard-fail here
        pass
    if step_idx >= 7:
        # dedup-layer will produce these; not a prereq
        pass
    if step_idx >= 8:
        # cross-dedup may consume dedup_layer; keep soft
        pass
    if step_idx >= 9:
        # simplify consumes dedup_cross or dedup_layer
        pass
    if step_idx >= 10:
        # preview can run off any stage; no strict prereq here
        pass
    return [p for p in need if not os.path.exists(p)]


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Raster → Vector pipeline")
    ap.add_argument("input_image", help="Input raster image (e.g., portrait0.jpg)")
    ap.add_argument("--output", required=True, dest="output_dir", help="Output directory")
    ap.add_argument("--start-step", type=int, default=1, help="1..10 (default: 1)")
    ap.add_argument("--end-step", type=int, default=10, help="1..10 (default: 10)")

    # Optional overrides to inject into config.json
    ap.add_argument("--pixels-per-mm", type=int, dest="pixels_per_mm")
    ap.add_argument("--target-width-mm", type=int, dest="target_width_mm")
    ap.add_argument("--target-height-mm", type=int, dest="target_height_mm")
    ap.add_argument(
        "--colors",
        dest="colors_json",
        help="Override colors as JSON, e.g. [[0,0,0],[255,0,0],[0,255,0],[0,0,255]] (BGR)",
    )
    return ap.parse_args()


# ----------------- main -----------------
def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    # Base defaults
    cfg_defaults = load_default_config()

    # Merge strategy:
    #  - if output_dir/config.json exists → load & apply CLI overrides
    #  - else → start from defaults & apply CLI overrides
    overrides = {
        "input_image": args.input_image,
        "output_dir": args.output_dir,
        "pixels_per_mm": args.pixels_per_mm,
        "target_width_mm": args.target_width_mm,
        "target_height_mm": args.target_height_mm,
    }
    if args.colors_json:
        try:
            overrides["colors"] = json.loads(args.colors_json)
        except Exception as e:
            print(f"Failed to parse --colors JSON: {e}", file=sys.stderr)

    cfg_file = write_config(cfg_defaults, args.output_dir, overrides)
    print("Config saved to", cfg_file)

    # Read color_names for preflight checks / logs
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_colors = cfg_dict.get("color_names", ["layer_dark", "layer_mid", "layer_skin", "layer_light"])
    except Exception:
        cfg_colors = ["layer_dark", "layer_mid", "layer_skin", "layer_light"]

    # Child process environment: propagate CONFIG_PATH so every module reads the right config
    env = os.environ.copy()
    env["CONFIG_PATH"] = cfg_file
    env["PYTHONUNBUFFERED"] = "1"  # stream child stdout in real time

    # Header
    print("=" * 50)
    print("RASTER → VECTOR PIPELINE")
    print("=" * 50)
    print("Input image:", args.input_image)
    print("Output dir: ", args.output_dir)

    steps = build_steps()
    total_steps = len(steps)

    # Clamp / validate range
    s0 = max(1, min(args.start_step, total_steps))
    s1 = max(1, min(args.end_step,   total_steps))
    if s0 > s1:
        s0, s1 = s1, s0

    # Preflight for chosen start step
    missing = missing_for_step(s0, args.output_dir, cfg_colors)
    if missing:
        print("\n[Preflight] Warning: missing inputs for the chosen start step:")
        for p in missing:
            print(" -", p)
        print("The step may fail; consider starting earlier or generating the missing artifacts.\n")

    # Run selected steps
    for i in range(s0 - 1, s1):
        title, module = steps[i]
        run_step(title, module, env)

    print("\nDone.")


if __name__ == "__main__":
    main()
