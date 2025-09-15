# 09_simplify.py
from __future__ import annotations
import os
import pickle
from typing import List, Tuple, Dict, Any
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# prevent thread oversubscription inside workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

from config import load_config, Config


# ---------------------------- geometry helpers ----------------------------

def _is_closed(pts: np.ndarray) -> bool:
    p = pts.reshape(-1, 2)
    return len(p) > 2 and np.all(p[0] == p[-1])

def _ensure_closed(pts: np.ndarray) -> np.ndarray:
    p = pts.reshape(-1, 2)
    if len(p) > 0 and not np.all(p[0] == p[-1]):
        p = np.vstack([p, p[0]])
    return p.reshape(-1, 1, 2).astype(np.int32)

def _arc_lengths(p: np.ndarray) -> np.ndarray:
    d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1))
    return np.concatenate([[0.0], np.cumsum(d)])

def _resample_by_arclen(pts: np.ndarray, step: float, closed: bool) -> np.ndarray:
    p = pts.reshape(-1, 2).astype(np.float32)
    if len(p) < 2:
        return pts.astype(np.int32)
    if closed:
        p = np.vstack([p, p[0]])
    s = _arc_lengths(p)
    if s[-1] <= step:
        q = p[:-1] if closed else p
        return q.reshape(-1, 1, 2).astype(np.int32)
    new_s = np.arange(0.0, s[-1], step, dtype=np.float32)
    if closed and (len(new_s) == 0 or new_s[-1] != s[-1]):
        new_s = np.append(new_s, s[-1])
    seg = np.searchsorted(s, new_s, side="right") - 1
    seg = np.clip(seg, 0, len(p) - 2)
    t = (new_s - s[seg]) / np.maximum(1e-6, s[seg + 1] - s[seg])
    q = p[seg] * (1.0 - t[:, None]) + p[seg + 1] * t[:, None]
    if closed and len(q) > 0:
        q = q[:-1]
    return q.reshape(-1, 1, 2).astype(np.int32)

def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    p = points.reshape(-1, 2)
    n = len(p)
    if n <= 2:
        return p
    stack = [(0, n - 1)]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = p[s], p[e]
        seg = b - a
        seg_n = np.array([-seg[1], seg[0]], dtype=np.float32)
        seg_len = np.linalg.norm(seg) + 1e-12
        idx = slice(s + 1, e)
        d = np.abs((p[idx] - a) @ seg_n) / seg_len
        i = int(np.argmax(d))
        if d[i] > eps:
            k = s + 1 + i
            keep[k] = True
            stack.append((s, k))
            stack.append((k, e))
    return p[keep]


# --------------------------- rasterization (per ROI) -----------------------

def _polyline_bbox(pts: np.ndarray, pad: int) -> Tuple[int, int, int, int]:
    p = pts.reshape(-1, 2)
    x0 = int(np.floor(p[:, 0].min())) - pad
    y0 = int(np.floor(p[:, 1].min())) - pad
    x1 = int(np.ceil(p[:, 0].max())) + pad
    y1 = int(np.ceil(p[:, 1].max())) + pad
    return x0, y0, x1, y1

def _rasterize_polyline_roi(pts: np.ndarray, brush: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    p = pts.reshape(-1, 2).astype(np.int32)
    pad = max(brush + 2, 6)
    x0, y0, x1, y1 = _polyline_bbox(p, pad)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    roi = np.zeros((h, w), np.uint8)
    q = p - np.array([x0, y0], dtype=np.int32)
    if len(q) >= 2:
        cv2.polylines(roi, [q.reshape(-1, 1, 2)], False, 255, thickness=brush, lineType=cv2.LINE_8)
    else:
        cv2.circle(roi, tuple(q[0]), max(1, brush // 2), 255, -1, lineType=cv2.LINE_8)
    return roi, (x0, y0)


# ---------------------- Zhang–Suen thinning (vectorized) -------------------

# img is padded inside the function; all ops are NumPy boolean arithmetic
def _zhang_suen_thinning_fast(bin_img: np.ndarray) -> np.ndarray:
    img = (bin_img > 0).astype(np.uint8)
    if img.size == 0:
        return bin_img
    img = np.pad(img, 1, mode='constant', constant_values=0)
    changed = True
    while changed:
        changed = False
        P2 = img[:-2, 1:-1];  P3 = img[:-2, 2:];   P4 = img[1:-1, 2:]
        P5 = img[2:,  2:];    P6 = img[2:,  1:-1]; P7 = img[2:,  0:-2]
        P8 = img[1:-1, 0:-2]; P9 = img[0:-2,0:-2]; C = img[1:-1,1:-1]
        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m1 = (C==1)&(B>=2)&(B<=6)&(A==1)&((P2*P4*P6)==0)&((P4*P6*P8)==0)
        if m1.any():
            C[m1] = 0; changed = True

        P2 = img[:-2, 1:-1];  P3 = img[:-2, 2:];   P4 = img[1:-1, 2:]
        P5 = img[2:,  2:];    P6 = img[2:,  1:-1]; P7 = img[2:,  0:-2]
        P8 = img[1:-1, 0:-2]; P9 = img[0:-2,0:-2]; C = img[1:-1,1:-1]
        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m2 = (C==1)&(B>=2)&(B<=6)&(A==1)&((P2*P4*P8)==0)&((P2*P6*P8)==0)
        if m2.any():
            C[m2] = 0; changed = True

    skel = img[1:-1, 1:-1].astype(np.uint8)*255
    return skel


# ------------------------ skeleton → ordered centerline --------------------

_OFFS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def _degree(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, np.uint8)
    for dy, dx in _OFFS:
        out[1:-1,1:-1] += img[1+dy:img.shape[0]-1+dy, 1+dx:img.shape[1]-1+dx]
    return out

def _trace_from(img: np.ndarray, y0: int, x0: int) -> List[Tuple[int,int]]:
    path: List[Tuple[int,int]] = []
    h, w = img.shape
    y, x = y0, x0
    py, px = -1, -1
    while True:
        path.append((y, x))
        img[y, x] = 0  # mark visited
        found = False
        for dy, dx in _OFFS:
            ny, nx = y + dy, x + dx
            if 0 < ny < h-1 and 0 < nx < w-1 and img[ny, nx] > 0:
                if ny == py and nx == px:
                    continue
                py, px = y, x
                y, x = ny, nx
                found = True
                break
        if not found:
            break
    return path

def _skeleton_longest_path(skel: np.ndarray, origin_xy: Tuple[int,int]) -> np.ndarray | None:
    img = (skel > 0).astype(np.uint8)
    if img.sum() == 0:
        return None
    deg = _degree(img)
    h, w = img.shape

    # endpoints first
    ys, xs = np.where((img == 1) & (deg == 1))
    paths: List[List[Tuple[int,int]]] = []
    for y, x in zip(ys, xs):
        if img[y, x] == 0:
            continue
        p = _trace_from(img, y, x)
        if len(p) >= 2:
            paths.append(p)

    # leftover cycles
    ys, xs = np.where(img == 1)
    for y, x in zip(ys, xs):
        if img[y, x] == 0:
            continue
        p = _trace_from(img, y, x)
        if len(p) >= 2:
            if p[0] != p[-1]:
                p.append(p[0])
            paths.append(p)

    if not paths:
        return None
    paths.sort(key=len, reverse=True)

    ox, oy = origin_xy  # origin is (x, y)
    pts = [(x + ox, y + oy) for (y, x) in paths[0]]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


# ----------------------- one-polyline centerline pipeline ------------------

def _simplify_one_centerline(contour: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    brush = int(params["brush_d_px"])
    step  = float(params["resample_step_px"])
    rdp_eps = float(params["rdp_epsilon_px"])

    closed = _is_closed(contour)

    roi, org = _rasterize_polyline_roi(contour, brush=brush)
    skel = _zhang_suen_thinning_fast(roi)
    path = _skeleton_longest_path(skel, origin_xy=org)

    if path is None or path.reshape(-1, 2).shape[0] < 2:
        # keep original polyline if skeletonization failed
        q = contour.reshape(-1, 2)
    else:
        q = path.reshape(-1, 2)

    q = _resample_by_arclen(q, step=step, closed=closed).reshape(-1, 2)
    q = _rdp(q.reshape(-1, 1, 2), eps=rdp_eps).reshape(-1, 2)
    if closed:
        q = _ensure_closed(q.reshape(-1, 1, 2))
    return q.reshape(-1, 1, 2).astype(np.int32)


# ------------------------------- layer I/O ---------------------------------

def _load_lines_strict(layer_dir: str) -> List[np.ndarray]:
    path = os.path.join(layer_dir, "lines_cross.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required input: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Invalid pickle format: {path}")
    return obj


# ------------------------------ layer process ------------------------------

def _pack_params(cfg: Config) -> Dict[str, Any]:
    pen_w = float(getattr(cfg, "pen_width_px", 60))
    brush = int(getattr(cfg, "centerline_brush_d_px", round(pen_w)))
    step  = float(getattr(cfg, "simplify_step_px", max(1, int(round(pen_w * 0.25)))))
    rdp_e = float(getattr(cfg, "rdp_base_epsilon_px", max(1.0, 0.33 * pen_w)))
    return dict(
        brush_d_px=max(3, brush),
        resample_step_px=max(1.0, step),
        rdp_epsilon_px=max(0.5, rdp_e),
    )

def simplify_layer(color_name: str, cfg: Config) -> List[np.ndarray]:
    layer_dir = os.path.join(cfg.output_dir, color_name)
    polylines = _load_lines_strict(layer_dir)  # STRICT
    n = len(polylines)
    verts_in = int(sum(p.reshape(-1, 2).shape[0] for p in polylines))
    print(f"[simplify] {color_name}: start — polylines={n}, vertices_in={verts_in}", flush=True)

    params = _pack_params(cfg)
    t0 = perf_counter()
    results: List[np.ndarray] = [None] * n
    done = 0
    tick_each = max(1, n // 20)
    last_tick = t0

    max_workers = max(1, int(getattr(cfg, "n_cores", os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2idx = {ex.submit(_simplify_one_centerline, c, params): i for i, c in enumerate(polylines)}
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                # preserve geometry on failure
                results[i] = polylines[i]
                print(f"[simplify] {color_name}: poly {i} FAILED — {e}", flush=True)
            done += 1
            now = perf_counter()
            if (done % tick_each == 0) or (now - last_tick >= 2.0):
                pct = (done * 100) // max(1, n)
                print(f"[simplify] {color_name}: {done}/{n} ({pct}%)", flush=True)
                last_tick = now

    verts_out = int(sum(r.reshape(-1, 2).shape[0] for r in results))
    out_path = os.path.join(layer_dir, "contours_simplified.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[simplify] {color_name}: done — vertices {verts_in} → {verts_out}, time={perf_counter()-t0:.1f}s → {out_path}", flush=True)
    return results


def simplify_all_layers(cfg: Config) -> None:
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    print("[simplify] ===== Simplify contours (start) =====", flush=True)
    t_all = perf_counter()
    max_workers = min(int(getattr(cfg, "n_cores", os.cpu_count() or 1)), len(cfg.color_names))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(simplify_layer, name, cfg): name for name in cfg.color_names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[simplify] {name}: ERROR — {e}", flush=True)
    print(f"[simplify] ===== Simplify contours (end) — total {perf_counter()-t_all:.1f}s =====", flush=True)


# ---------------------------------- main -----------------------------------

if __name__ == "__main__":
    cfg = load_config()
    simplify_all_layers(cfg)
