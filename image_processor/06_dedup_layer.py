# 06_dedup_layer.py
# In-layer deduplication with progress logs.
import os
import time
import pickle
import cv2
import numpy as np
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict
from config import load_config, Config

# ---------- helpers ----------
def contour_area(c: np.ndarray) -> float:
    return float(abs(cv2.contourArea(c)))

def contour_perimeter(c: np.ndarray) -> float:
    return float(cv2.arcLength(c, True))

def contour_bbox(c: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def contour_center(c: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(c)
    if m["m00"] != 0:
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    pts = c.reshape(-1, 2)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

def load_resized_shape(cfg: Config) -> Tuple[int, int]:
    img = cv2.imread(os.path.join(cfg.output_dir, "resized.png"))
    if img is None:
        raise ValueError("Failed to load resized.png for canvas size")
    h, w = img.shape[:2]
    return h, w

# ---------- taps ----------
def is_tap_candidate(c: np.ndarray, cfg: Config) -> bool:
    if contour_area(c) <= cfg.tap_max_area:
        return True
    if contour_perimeter(c) <= cfg.tap_max_perimeter:
        return True
    x, y, w, h = contour_bbox(c)
    if max(w, h) <= cfg.tap_max_dim:
        return True
    return False

def cluster_merge_points(points: np.ndarray, radius: float, layer_name: str) -> np.ndarray:
    """Iteratively merge points (<= radius) into cluster centroids; prints progress."""
    if len(points) == 0:
        print(f"[{layer_name}] Taps: none", flush=True)
        return points
    pts = points.astype(np.float32).copy()
    r2 = radius * radius
    it = 0
    while True:
        it += 1
        n0 = len(pts)
        used = np.zeros(n0, dtype=bool)
        new_pts = []
        merges = 0
        for i in range(n0):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, n0):
                if used[j]:
                    continue
                d = pts[j] - pts[i]
                if d[0]*d[0] + d[1]*d[1] <= r2:
                    group.append(j)
                    used[j] = True
            if len(group) > 1:
                merges += len(group) - 1
            new_pts.append(np.mean(pts[group], axis=0))
        pts = np.asarray(new_pts, dtype=np.float32)
        print(f"[{layer_name}] Taps merge iter {it}: merged={merges}, taps={len(pts)}", flush=True)
        if merges == 0:
            break
    return pts

# ---------- spatial grid for neighbor search ----------
class GridIndex:
    """Uniform grid index for 2D points; stores (contour_id, point_id, x, y) for alive points."""
    def __init__(self, cell: float):
        self.cell = float(max(1.0, cell))
        self.cells: DefaultDict[Tuple[int,int], List[Tuple[int,int,float,float]]] = defaultdict(list)

    def _key(self, x: float, y: float) -> Tuple[int,int]:
        return (int(np.floor(x / self.cell)), int(np.floor(y / self.cell)))

    def rebuild(self, contours_pts: List[np.ndarray], alive_mask: List[np.ndarray]):
        self.cells.clear()
        for ci, pts in enumerate(contours_pts):
            alive = alive_mask[ci]
            if len(pts) == 0:
                continue
            for pi, (x, y) in enumerate(pts):
                if not alive[pi]:
                    continue
                self.cells[self._key(x, y)].append((ci, pi, float(x), float(y)))

    def query(self, x: float, y: float, radius: float) -> List[Tuple[int,int,float,float]]:
        r = int(np.ceil(radius / self.cell))
        kx, ky = self._key(x, y)
        out = []
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                out.extend(self.cells.get((kx+dx, ky+dy), []))
        return out

# ---------- main per-layer routine with progress ----------
def dedup_layer(color_name: str, cfg: Config):
    LOG_INTERVAL_SEC = 1.5  # time-based progress during passes

    base = os.path.join(cfg.output_dir, color_name)
    src_sorted = os.path.join(base, "contours_sorted.pkl")
    src_raw = os.path.join(base, "contours.pkl")
    src = src_sorted if os.path.exists(src_sorted) else src_raw
    with open(src, "rb") as f:
        contours: List[np.ndarray] = pickle.load(f)

    print(f"\n[{color_name}] In-layer dedup started…", flush=True)
    print(f"[{color_name}] Input contours: {len(contours)}", flush=True)

    # Split to taps vs lines
    tap_pts = []
    line_contours = []
    for c in contours:
        if contour_area(c) < cfg.min_contour_area:
            continue
        if is_tap_candidate(c, cfg):
            tap_pts.append(contour_center(c))
        else:
            line_contours.append(c)
    print(f"[{color_name}] Classified: lines={len(line_contours)}, taps={len(tap_pts)} (pre-merge)", flush=True)

    tap_pts = np.asarray(tap_pts, dtype=np.float32)
    tap_pts = cluster_merge_points(tap_pts, radius=float(cfg.tap_merge_radius_px), layer_name=color_name)

    # Work copy of polylines as float Nx2; remember if closed
    polys: List[np.ndarray] = []
    closed_flags: List[bool] = []
    for c in line_contours:
        pts = c.reshape(-1, 2).astype(np.float32)
        closed = np.all(pts[0] == pts[-1])
        if closed:
            pts = pts[:-1]
        polys.append(pts)
        closed_flags.append(closed)

    n = len(polys)
    if n == 0:
        with open(os.path.join(base, "contours_dedup_layer.pkl"), "wb") as f:
            pickle.dump([], f)
        with open(os.path.join(base, "taps.pkl"), "wb") as f:
            pickle.dump(tap_pts, f)
        print(f"[{color_name}] No line contours; taps={len(tap_pts)}. Saved.", flush=True)
        return [], tap_pts

    # Per-point state
    alive: List[np.ndarray] = [np.ones(len(p), dtype=bool) for p in polys]
    dead:  List[np.ndarray] = [np.zeros(len(p), dtype=bool) for p in polys]

    # Grid
    R = float(cfg.pen_radius_px)
    grid = GridIndex(cell=R)
    grid.rebuild(polys, alive)

    # Passes
    max_passes = max(1, int(getattr(cfg, "dedup_max_passes", 4)))
    changed_any = False
    for p in range(1, max_passes + 1):
        # count alive points
        total_alive = int(sum(int(a.sum()) for a in alive))
        print(f"[{color_name}] Pass {p}/{max_passes}: alive points={total_alive}, contours={n}", flush=True)

        pass_deletes = 0
        pass_moves = 0
        t_pass = time.perf_counter()
        last_log_t = t_pass
        processed_pts = 0

        try:
            for ci in range(n):
                pts = polys[ci]
                if len(pts) == 0:
                    continue

                to_del = []
                planned_moves: Dict[Tuple[int,int], Tuple[float,float]] = {}

                for pi, (x, y) in enumerate(pts):
                    if not alive[ci][pi]:
                        continue

                    processed_pts += 1
                    now = time.perf_counter()
                    if now - last_log_t > LOG_INTERVAL_SEC:
                        print(f"[{color_name}]  • progress: processed={processed_pts}/{total_alive} "
                              f"(~{100.0*processed_pts/max(1,total_alive):.1f}%), "
                              f"del={pass_deletes}, move={pass_moves}",
                              flush=True)
                        last_log_t = now

                    neigh = grid.query(x, y, R)
                    if not neigh:
                        continue

                    alive_targets = []
                    any_neighbor = False
                    for cj, pj, qx, qy in neigh:
                        if cj == ci:
                            continue
                        if not alive[cj][pj]:
                            continue
                        any_neighbor = True
                        if not dead[cj][pj]:
                            alive_targets.append((cj, pj, qx, qy))

                    if not any_neighbor:
                        continue

                    to_del.append(pi)

                    if alive_targets:
                        acc = np.array([x, y], dtype=np.float32)
                        for _, _, qx, qy in alive_targets:
                            acc += np.array([qx, qy], dtype=np.float32)
                        acc /= (1 + len(alive_targets))
                        for cj, pj, _, _ in alive_targets:
                            planned_moves[(cj, pj)] = (float(acc[0]), float(acc[1]))

                if to_del or planned_moves:
                    # apply deletions
                    if to_del:
                        alive[ci][to_del] = False
                        pass_deletes += len(to_del)

                    # apply moves
                    for (cj, pj), (nx, ny) in planned_moves.items():
                        polys[cj][pj, 0] = nx
                        polys[cj][pj, 1] = ny
                        if not dead[cj][pj]:
                            dead[cj][pj] = True
                            pass_moves += 1

                    # rebuild grid after modifications
                    grid.rebuild(polys, alive)

            print(f"[{color_name}] Pass {p} done in {time.perf_counter()-t_pass:.2f}s: "
                  f"deleted={pass_deletes}, moved={pass_moves}", flush=True)

            if pass_deletes == 0 and pass_moves == 0:
                print(f"[{color_name}] No changes in pass {p}; stopping passes.", flush=True)
                break
            else:
                changed_any = True

        except KeyboardInterrupt:
            print(f"[{color_name}] Interrupted during pass {p}. Saving current state.", flush=True)
            break

    # Rebuild final polylines from alive points
    merged_lines: List[np.ndarray] = []
    new_taps: List[Tuple[float, float]] = []
    for ci in range(n):
        pts = polys[ci]
        mask = alive[ci]
        if not np.any(mask):
            continue
        out = pts[mask]
        if len(out) < cfg.thinning_min_segment_len:
            for (x, y) in out:
                new_taps.append((float(x), float(y)))
            continue
        if closed_flags[ci]:
            out = np.vstack([out, out[0]])
        merged_lines.append(out.reshape(-1, 1, 2).astype(np.int32))

    if new_taps:
        if len(tap_pts) == 0:
            tap_pts = np.array(new_taps, dtype=np.float32)
        else:
            tap_pts = np.vstack([tap_pts, np.array(new_taps, dtype=np.float32)])

    # Second tap merge and tap vs line cleanup
    if len(tap_pts) > 0:
        tap_pts = cluster_merge_points(tap_pts, radius=float(cfg.tap_merge_radius_px), layer_name=color_name)

    if merged_lines and len(tap_pts) > 0:
        h, w = load_resized_shape(cfg)
        occ = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(occ, [c.astype(np.int32) for c in merged_lines], True,
                      255, thickness=int(cfg.pen_width_px), lineType=cv2.LINE_8)
        dt = cv2.distanceTransform(255 - occ, cv2.DIST_L2, 3)
        keep = []
        for (x, y) in tap_pts:
            xi = int(np.clip(x, 0, w - 1)); yi = int(np.clip(y, 0, h - 1))
            keep.append(dt[yi, xi] >= R)
        tap_pts = tap_pts[np.array(keep, dtype=bool)]

    # Persist
    with open(os.path.join(base, "contours_dedup_layer.pkl"), "wb") as f:
        pickle.dump(merged_lines, f)
    with open(os.path.join(base, "taps.pkl"), "wb") as f:
        pickle.dump(tap_pts, f)

    print(f"[{color_name}] Saved: lines={len(merged_lines)}, taps={len(tap_pts)} → "
          f"{os.path.join(base, 'contours_dedup_layer.pkl')}", flush=True)

    return merged_lines, tap_pts

def dedup_all(cfg: Config):
    for name in cfg.color_names:
        dedup_layer(name, cfg)

if __name__ == "__main__":
    cfg = load_config()
    dedup_all(cfg)
