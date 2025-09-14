# pipeline.py
# Raster → Vector pipeline runner with step range and streamed logs.
import argparse
import json
import os
import sys
import subprocess
from dataclasses import asdict
from typing import List, Tuple

# ----------------- config I/O -----------------
def load_default_config():
    # Import lazily to avoid side effects if config has heavy deps
    from config import Config
    return Config()

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def config_path(outdir: str) -> str:
    return os.path.join(outdir, "config.json")

def write_config(cfg_obj, outdir: str, overrides: dict):
    """Merge overrides into cfg_obj, then persist to outdir/config.json."""
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(cfg_obj, k):
            setattr(cfg_obj, k, v)
    cfg_dict = asdict(cfg_obj)
    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})
    with open(config_path(outdir), "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

def read_existing_config(outdir: str):
    p = config_path(outdir)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------- steps -----------------
def module_path(preferred: str, fallback: str | None = None) -> str:
    """Return existing path for a step module; prefer preferred, fallback if needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.join(here, preferred)
    if os.path.exists(cand1):
        return cand1
    if fallback:
        cand2 = os.path.join(here, fallback)
        if os.path.exists(cand2):
            return cand2
    return cand1  # return preferred anyway (subprocess will error clearly)

def build_steps() -> List[Tuple[str, str]]:
    return [
        ("1/10 Image resize…",        module_path("01_resize.py")),
        ("2/10 RGBK color extraction…", module_path("02_color_extract.py")),
        ("3/10 Edge detection…",      module_path("03_edge_detect.py")),
        ("4/10 Find contours…",       module_path("04_find_contours.py")),
        ("5/10 Sort contours…",       module_path("05_sort_contours.py")),
        ("6/10 Intra-layer dedup…",   module_path("06_dedup_layer.py")),
        ("7/10 Cross-layer dedup…",   module_path("07_dedup_cross.py")),
        ("8/10 Simplify contours…",   module_path("08_simplify.py")),
        ("9/10 Scale vectors…",       module_path("09_scale_vector.py", "09_scale vector.py")),
        ("10/10 Preview…",            module_path("10_preview.py")),
    ]

# ----------------- stdout streaming -----------------
def run_step(title: str, module: str, env: dict) -> None:
    print(f"\n[{title}]")
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
def require_files(files: List[str]) -> List[str]:
    missing = []
    for p in files:
        if not os.path.exists(p):
            missing.append(p)
    return missing

def prereqs_for_step(step_idx: int, outdir: str, color_names: List[str]) -> List[str]:
    # step_idx is 1-based
    if step_idx <= 1:
        return []
    need = []
    if step_idx >= 2:
        need += [os.path.join(outdir, "resized.png")]
    if step_idx >= 3:
        need += [os.path.join(outdir, c, "mask.png") for c in color_names]
    if step_idx >= 4:
        need += [os.path.join(outdir, c, "edges.png") for c in color_names]
    if step_idx >= 5:
        need += [os.path.join(outdir, c, "contours.pkl") for c in color_names]
    if step_idx >= 6:
        # sort step may be skipped; dedup can fall back to contours.pkl
        # but we warn if sorted is missing for transparency
        pass
    if step_idx >= 7:
        need += [os.path.join(outdir, c, "contours_dedup_layer.pkl") for c in color_names]
    if step_idx >= 8:
        # depending on your 08 implementation, it may read dedup_cross or dedup_layer
        # we don't hard-fail here
        pass
    if step_idx >= 9:
        need += [os.path.join(outdir, c, "contours_simplified.pkl") for c in color_names]
    if step_idx >= 10:
        need += [os.path.join(outdir, c, "contours_final.pkl") for c in color_names]
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
    ap.add_argument("--colors", dest="colors_json", help="Override colors as JSON, e.g. [[0,0,0],[0,0,255],[0,255,0],[255,0,0]] (BGR)")
    return ap.parse_args()

# ----------------- main -----------------
def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    # Load or create config, then persist to output_dir/config.json
    cfg_existing = read_existing_config(args.output_dir)
    cfg = load_default_config()

    # Minimal overrides we always apply
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

    # If config exists, merge; else write a fresh one
    if cfg_existing:
        # merge existing with overrides and rewrite (idempotent)
        for k, v in overrides.items():
            if v is not None:
                cfg_existing[k] = v
        with open(config_path(args.output_dir), "w", encoding="utf-8") as f:
            json.dump(cfg_existing, f, indent=2, ensure_ascii=False)
        cfg_colors = cfg_existing.get("color_names", ["black","red","green","blue"])
        print("Config saved to", config_path(args.output_dir))
    else:
        write_config(cfg, args.output_dir, overrides)
        cfg_colors = getattr(cfg, "color_names", ["black","red","green","blue"])
        print("Config saved to", config_path(args.output_dir))

    # Environment for child steps
    env = os.environ.copy()
    env["PIPELINE_CONFIG"] = config_path(args.output_dir)
    env["PYTHONUNBUFFERED"] = "1"  # force unbuffered prints in children

    # Header
    print("=" * 50)
    print("RASTER → VECTOR PIPELINE")
    print("=" * 50)
    print("Input image:", args.input_image)
    print("Output dir: ", args.output_dir)

    steps = build_steps()
    total_steps = len(steps)

    # Clamp / validate step range
    s0 = max(1, min(args.start_step, total_steps))
    s1 = max(1, min(args.end_step,   total_steps))
    if s0 > s1:
        s0, s1 = s1, s0

    # Preflight: warn about missing prereqs for chosen start
    missing = prereqs_for_step(s0, args.output_dir, cfg_colors)
    if missing:
        print("\n[Preflight] Warning: some expected inputs for this start step are missing:")
        for p in missing:
            print(" -", p)
        print("The step may fail; consider starting earlier or generating the missing artifacts.\n")

    # Run selected range
    for i in range(s0-1, s1):
        title, module = steps[i]
        run_step(title, module, env)

    print("\nDone.")

if __name__ == "__main__":
    main()
