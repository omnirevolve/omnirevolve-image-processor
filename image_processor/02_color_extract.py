# 02_color_extract.py — robust 4-layer extraction
# Default mode: KMeans (Lab) → map clusters to layer names by darkness (dark→…→light).
# Legacy mode (optional): set "extraction_mode": "swatch" in config.json to threshold by swatches.

from __future__ import annotations
import os
import json
from typing import List, Tuple, Dict
import numpy as np
import cv2

from config import load_config, Config


# ---------------- helpers ----------------

def _darkness_rank(name: str) -> int:
    s = name.lower()
    if "dark" in s: return 0
    if "mid"  in s: return 1
    if "skin" in s: return 2
    if "light" in s: return 3
    return 2

def _ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _kmeans_lab(img_bgr: np.ndarray, k: int, sample_limit: int = 200_000, attempts: int = 3):
    """KMeans clustering in Lab. Returns (centers_lab[K,3], labels_full[H,W])."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)

    # random uniform subsample for speed
    n = data.shape[0]
    if n > sample_limit:
        idx = np.random.default_rng(42).choice(n, size=sample_limit, replace=False)
        sample = data[idx]
    else:
        sample = data

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compactness, labels_s, centers = cv2.kmeans(
        sample, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)  # [k,3] Lab

    # assign all pixels to nearest center (vectorized)
    diffs = data[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    labels_full = np.argmin(d2, axis=1).astype(np.int32).reshape(h, w)
    return centers, labels_full

def _lab_to_bgr(lab: np.ndarray) -> Tuple[int,int,int]:
    lab_img = np.array([[lab]], dtype=np.uint8)  # 1x1x3
    bgr = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------- main ----------------

def main():
    cfg: Config = load_config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    src_path = os.path.join(cfg.output_dir, "resized.png")
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read resized image: {src_path}")
    img = _ensure_bgr(img)

    names: List[str] = list(cfg.color_names)
    K_cfg = int(getattr(cfg, "cluster_k", len(names)))
    K = max(2, min(len(names), K_cfg))

    mode = str(getattr(cfg, "extraction_mode", "kmeans")).lower()

    if mode == "swatch":
        # --- legacy swatch-based extraction (kept for compatibility) ---
        tol = int(getattr(cfg, "color_tolerance", 30))
        colors = list(getattr(cfg, "colors", []))
        if not colors or len(colors) < len(names):
            raise RuntimeError("swatch mode: 'colors' must have ≥ len(color_names) entries.")
        # Try both RGB→BGR and BGR-as-is per swatch; pick better coverage.
        for i, name in enumerate(names, 1):
            layer_dir = os.path.join(cfg.output_dir, name); os.makedirs(layer_dir, exist_ok=True)
            rgb = tuple(int(v) for v in colors[i-1])
            bgr1 = (rgb[2], rgb[1], rgb[0])        # RGB→BGR
            bgr2 = rgb                              # as-is
            lower1 = np.array([max(0,bgr1[0]-tol), max(0,bgr1[1]-tol), max(0,bgr1[2]-tol)], np.uint8)
            upper1 = np.array([min(255,bgr1[0]+tol), min(255,bgr1[1]+tol), min(255,bgr1[2]+tol)], np.uint8)
            lower2 = np.array([max(0,bgr2[0]-tol), max(0,bgr2[1]-tol), max(0,bgr2[2]-tol)], np.uint8)
            upper2 = np.array([min(255,bgr2[0]+tol), min(255,bgr2[1]+tol), min(255,bgr2[2]+tol)], np.uint8)
            m1 = cv2.inRange(img, lower1, upper1)
            m2 = cv2.inRange(img, lower2, upper2)
            nz1, nz2 = int(np.count_nonzero(m1)), int(np.count_nonzero(m2))
            mask = m1 if nz1 >= nz2 else m2
            # small clean up
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            cv2.imwrite(os.path.join(layer_dir, "mask.png"), mask)
            print(f"Extracted (swatch): {name} | nz={int(np.count_nonzero(mask))}")
        print("Color extraction: done.")
        return

    # --- default: KMeans-based adaptive extraction ---
    centers_lab, labels = _kmeans_lab(
        img,
        k=K,
        sample_limit=int(getattr(cfg, "kmeans_sample_limit", 200_000)),
        attempts=int(getattr(cfg, "kmeans_attempts", 3)),
    )
    h, w = labels.shape

    # sort cluster ids by darkness (Lab L channel, ascending)
    Ls = centers_lab[:, 0]
    order = np.argsort(Ls)  # dark→light
    centers_lab = centers_lab[order]
    # relabel to 0..K-1 in dark→light order
    lut = np.zeros_like(order)
    lut[order] = np.arange(len(order))
    labels = lut[labels]

    # sort names by intended darkness
    names_sorted = sorted(names, key=_darkness_rank)

    # finalize mapping: darkest cluster -> darkest name, etc.
    mapping = list(zip(names_sorted, range(K)))

    # small morphological refine per cluster mask
    open_iters = int(getattr(cfg, "extract_open_iters", 1))
    close_iters = int(getattr(cfg, "extract_close_iters", 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # summary for palette_by_name.json
    palette: Dict[str, Dict] = {}

    # cluster sizes
    counts = [(labels == k).sum() for k in range(K)]

    for name, k_idx in mapping:
        layer_dir = os.path.join(cfg.output_dir, name)
        os.makedirs(layer_dir, exist_ok=True)

        mask = (labels == k_idx).astype(np.uint8) * 255
        if open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        if close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

        cv2.imwrite(os.path.join(layer_dir, "mask.png"), mask)
        nz = int(np.count_nonzero(mask))

        bgr = _lab_to_bgr(centers_lab[k_idx].astype(np.uint8))
        palette[name] = {
            "mode": "kmeans",
            "cluster_index": int(k_idx),
            "cluster_lab": [int(v) for v in centers_lab[k_idx]],
            "approx_bgr": list(bgr),
            "pixels": int(counts[k_idx]),
            "mask_nonzero": nz
        }
        print(f"Extracted (kmeans): {name} | cluster={k_idx} | L*={centers_lab[k_idx,0]:.1f} | "
              f"pixels={counts[k_idx]} | nz={nz}")

    pal_path = os.path.join(cfg.output_dir, "palette_by_name.json")
    with open(pal_path, "w", encoding="utf-8") as f:
        json.dump(palette, f, ensure_ascii=False, indent=2)
    print(f"Palette saved: {pal_path}")
    print("Color extraction: done.")


if __name__ == "__main__":
    main()
