# 10_preview.py
# Render final preview of vectors per layer (no travel lines). Draw taps as small dots.
# Picks the newest available stage by default; can be forced via PREVIEW_STAGE env.

import os, pickle
from typing import List, Tuple
import numpy as np
import cv2
from config import load_config, Config

# File candidates: (filename, label)
SRC_CANDIDATES: List[Tuple[str, str]] = [
    ("contours_simplified.pkl", "simplified"),
    ("contours_dedup_cross.pkl","dedup_cross"),
    ("contours_dedup_layer.pkl","dedup_layer"),
    ("contours_sorted.pkl",     "sorted"),
    ("contours_scaled.pkl",     "scaled"),
    ("contours.pkl",            "raw"),
]

COLOR_MAP = {
    "layer_dark":  (30, 30, 30),     # BGR
    "layer_mid":   (180, 120, 80),
    "layer_skin":  (180, 170, 140),
    "layer_light": (200, 210, 240),
    # legacy names
    "black": (0,0,0), "blue": (180,120,80), "green": (140,170,140), "red": (130,170,240)
}

def pick_source(dirpath: str) -> Tuple[str, str]:
    """Return (path, label) for the best available source in dirpath.
       If PREVIEW_STAGE is set, try to use that; otherwise pick the newest by mtime."""
    force = os.environ.get("PREVIEW_STAGE", "").strip().lower()
    # map label -> filename
    label2file = {lbl: fn for fn, lbl in SRC_CANDIDATES}
    if force:
        fn = label2file.get(force)
        if fn:
            p = os.path.join(dirpath, fn)
            if os.path.exists(p):
                return p, force
    # else pick the newest existing file
    best_path, best_label, best_mtime = "", "", -1.0
    for fn, lbl in SRC_CANDIDATES:
        p = os.path.join(dirpath, fn)
        if os.path.exists(p):
            mt = os.path.getmtime(p)
            if mt > best_mtime:
                best_path, best_label, best_mtime = p, lbl, mt
    return best_path, best_label

def load_polylines(dirpath: str) -> Tuple[List[np.ndarray], str, str]:
    p, label = pick_source(dirpath)
    if p:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return (data if isinstance(data, list) else []), p, label
    return [], "", ""

def load_taps(dirpath: str) -> np.ndarray:
    # Prefer taps_final if exists
    for name in ("taps_final.pkl", "taps.pkl"):
        p = os.path.join(dirpath, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                t = pickle.load(f)
            if isinstance(t, list):
                t = np.array(t, np.float32)
            return t.astype(np.float32)
    return np.zeros((0,2), np.float32)

def main():
    cfg: Config = load_config()
    base = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if base is None:
        raise RuntimeError("resized.png not found in output dir")
    h, w = base.shape[:2]
    canvas = np.full_like(base, 255)

    for name in cfg.color_names:
        layer_dir = os.path.join(cfg.output_dir, name)
        lines, src_path, src_label = load_polylines(layer_dir)
        taps  = load_taps(layer_dir)

        color = COLOR_MAP.get(name, (80, 80, 80))

        # draw polylines (no implicit connections)
        n_lines = 0
        for poly in lines:
            pts = poly.reshape(-1,1,2).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, color, 1, lineType=cv2.LINE_AA)
                n_lines += 1

        # draw taps (small dots)
        if len(taps):
            for (x,y) in taps:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 2, color, -1, lineType=cv2.LINE_AA)

        # log what's used
        used = f"{src_label} ({os.path.basename(src_path)})" if src_path else "—"
        print(f"[preview] {name}: lines={n_lines}, taps={len(taps)}, src={used}")

    out_path = os.path.join(cfg.output_dir, "preview.png")
    cv2.imwrite(out_path, canvas)
    print(f"Preview saved: {out_path}")

if __name__ == "__main__":
    main()
