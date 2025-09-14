# 04_find_contours.py
# Edges -> 1px skeleton (Zhang–Suen, ROI) -> centerline polylines
# With detailed progress that always flushes to stdout.
import os, time
import cv2
import numpy as np
import pickle
from typing import List, Tuple
from config import load_config, Config

# ---------- small utils ----------
NEIGH8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]

def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros_like(img)
    ys = slice(max(0, dy), min(h, h + dy))
    xs = slice(max(0, dx), min(w, w + dx))
    ysrc = slice(max(0, -dy), min(h, h - dy))
    xsrc = slice(max(0, -dx), min(w, w - dx))
    out[ys, xs] = img[ysrc, xsrc]
    return out

def _bbox_of_nonzero(img: np.ndarray, pad: int = 2) -> Tuple[int,int,int,int] | None:
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(img.shape[1]-1, x1 + pad); y1 = min(img.shape[0]-1, y1 + pad)
    return x0, y0, x1, y1

# ---------- thinning (Zhang–Suen) with ROI + progress ----------
def thinning_zhangsuen(bin_0_255: np.ndarray, layer: str) -> np.ndarray:
    bbox = _bbox_of_nonzero(bin_0_255)
    if bbox is None:
        print(f"[{layer}] Thinning: empty edges.", flush=True)
        return np.zeros_like(bin_0_255)

    x0, y0, x1, y1 = bbox
    roi = (bin_0_255[y0:y1+1, x0:x1+1] > 0).astype(np.uint8)
    h, w = roi.shape
    total0 = int(roi.sum())
    print(f"[{layer}] Thinning ROI {w}x{h}, fg={total0} px", flush=True)

    changed, it, stall = True, 0, 0
    max_iter = 120
    t0 = time.perf_counter()
    while changed and it < max_iter and stall < 3:
        it += 1; t_it = time.perf_counter(); changed = False

        P2 = _shift(roi, -1,  0); P3 = _shift(roi, -1,  1); P4 = _shift(roi,  0,  1)
        P5 = _shift(roi,  1,  1); P6 = _shift(roi,  1,  0); P7 = _shift(roi,  1, -1)
        P8 = _shift(roi,  0, -1); P9 = _shift(roi, -1, -1)

        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)

        del1 = (roi == 1) & (A == 1) & (B >= 2) & (B <= 6) & ((P2*P4*P6) == 0) & ((P4*P6*P8) == 0)
        n1 = int(del1.sum())
        if n1: roi[del1] = 0; changed = True

        P2 = _shift(roi, -1,  0); P3 = _shift(roi, -1,  1); P4 = _shift(roi,  0,  1)
        P5 = _shift(roi,  1,  1); P6 = _shift(roi,  1,  0); P7 = _shift(roi,  1, -1)
        P8 = _shift(roi,  0, -1); P9 = _shift(roi, -1, -1)

        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2 == 0) & (P3 == 1)).astype(np.uint8) + \
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) + \
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) + \
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) + \
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) + \
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) + \
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) + \
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)

        del2 = (roi == 1) & (A == 1) & (B >= 2) & (B <= 6) & ((P2*P4*P8) == 0) & ((P2*P6*P8) == 0)
        n2 = int(del2.sum())
        if n2: roi[del2] = 0; changed = True

        removed = n1 + n2
        total_removed = total0 - int(roi.sum())
        pct = (total_removed / max(1, total0)) * 100.0
        print(f"[{layer}] Thin {it:02d}: removed={removed:6d} | total={total_removed:7d} ({pct:5.1f}%) | {time.perf_counter()-t_it:.2f}s", flush=True)
        stall = 0 if removed > 0 else stall + 1

    print(f"[{layer}] Thinning done in {time.perf_counter()-t0:.2f}s, fg_now={int(roi.sum())} px", flush=True)

    out = np.zeros_like(bin_0_255)
    out[y0:y1+1, x0:x1+1] = (roi * 255).astype(np.uint8)
    return out

# ---------- centerline tracing with robust progress ----------
def trace_centerlines(skel_0_255: np.ndarray, layer: str) -> List[np.ndarray]:
    t0 = time.perf_counter()
    S = (skel_0_255 > 0).astype(np.uint8)
    total_fg = int(S.sum())
    if total_fg == 0:
        print(f"[{layer}] Trace: empty skeleton.", flush=True)
        return []

    # connected components -> process one-by-one (gives natural progress)
    num, labels = cv2.connectedComponents(S, connectivity=8)
    print(f"[{layer}] Trace: fg={total_fg} px | components={num-1}", flush=True)

    # fast degrees via convolution (neighbors in 8-connectivity)
    kernel = np.ones((3,3), np.uint8); kernel[1,1] = 0

    paths: List[np.ndarray] = []
    visited_global = np.zeros_like(S, dtype=np.uint8)

    comp_fg_done = 0
    for comp_id in range(1, num):
        comp_mask = (labels == comp_id).astype(np.uint8)

        fg_comp = int(comp_mask.sum())
        comp_fg_done += fg_comp
        print(f"[{layer}]  • Component {comp_id}/{num-1}: pixels={fg_comp}", flush=True)

        deg = cv2.filter2D(comp_mask, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (comp_mask == 1) & (deg == 1)
        junctions = (comp_mask == 1) & (deg >= 3)

        visited = np.zeros_like(comp_mask, dtype=np.uint8)
        visited_count = 0
        t_comp = time.perf_counter()
        last_print_t = t_comp

        def neighbors(x:int,y:int):
            for dx,dy in NEIGH8:
                nx,ny = x+dx,y+dy
                if 0<=nx<S.shape[1] and 0<=ny<S.shape[0] and comp_mask[ny,nx]:
                    yield nx,ny

        # 1) paths from endpoints
        ys, xs = np.nonzero(endpoints)
        for y0, x0 in zip(ys, xs):
            if visited[y0, x0]: continue
            path=[(x0,y0)]
            visited[y0,x0]=1; visited_global[y0,x0]=1; visited_count+=1
            px,py=x0,y0; prev=None
            # walk until endpoint/junction
            step_guard = 0
            while True:
                nb = [(nx,ny) for nx,ny in neighbors(px,py) if (nx,ny)!=prev and not visited[ny,nx]]
                if not nb: break
                nx,ny = nb[0]
                path.append((nx,ny))
                visited[ny,nx]=1; visited_global[ny,nx]=1; visited_count+=1
                prev=(px,py); px,py=nx,ny
                # stop at junction or next endpoint
                if junctions[py,px] or endpoints[py,px]:
                    break
                step_guard+=1
                if step_guard>total_fg*2: break
                # progress every ~1.5s
                if time.perf_counter()-last_print_t>1.5:
                    pct = 100.0*visited_count/max(1,fg_comp)
                    print(f"[{layer}]    visited {visited_count}/{fg_comp} px ({pct:4.1f}%)", flush=True)
                    last_print_t=time.perf_counter()
            if len(path)>=2:
                arr=np.array(path,dtype=np.int32).reshape(-1,1,2)
                paths.append(arr)

        # 2) cycles (no endpoints inside)
        ys, xs = np.nonzero((comp_mask==1) & (visited==0))
        for y0,x0 in zip(ys,xs):
            if visited[y0,x0]: continue
            path=[(x0,y0)]
            visited[y0,x0]=1; visited_global[y0,x0]=1; visited_count+=1
            px,py=x0,y0; prev=None
            step_guard=0
            while True:
                nb = [(nx,ny) for nx,ny in neighbors(px,py) if (nx,ny)!=prev and not visited[ny,nx]]
                if not nb:
                    # allow closing step into already visited start
                    nb = [(nx,ny) for nx,ny in neighbors(px,py) if (nx,ny)!=prev]
                    if not nb: break
                nx,ny = nb[0]
                path.append((nx,ny))
                if not visited[ny,nx]:
                    visited[ny,nx]=1; visited_global[ny,nx]=1; visited_count+=1
                prev=(px,py); px,py=nx,ny
                if (px,py)==(x0,y0): break
                step_guard+=1
                if step_guard>fg_comp*4: break
                if time.perf_counter()-last_print_t>1.5:
                    pct = 100.0*visited_count/max(1,fg_comp)
                    print(f"[{layer}]    visited {visited_count}/{fg_comp} px ({pct:4.1f}%)", flush=True)
                    last_print_t=time.perf_counter()

            arr=np.array(path,dtype=np.int32).reshape(-1,1,2)
            if len(arr)>=2:
                # close loop if ends match
                if np.hypot(arr[0,0,0]-arr[-1,0,0], arr[0,0,1]-arr[-1,0,1])<1.5:
                    arr=np.vstack([arr,arr[0:1]])
                paths.append(arr)

        print(f"[{layer}]  • Component {comp_id} done in {time.perf_counter()-t_comp:.2f}s, polylines so far: {len(paths)}", flush=True)
        print(f"[{layer}]  • Progress: components_fg {comp_fg_done}/{total_fg} px ({100.0*comp_fg_done/total_fg:4.1f}%)", flush=True)

    print(f"[{layer}] Trace done: {len(paths)} polylines in {time.perf_counter()-t0:.2f}s", flush=True)
    return paths

# ---------- per-layer worker (sequential, visible progress) ----------
def vectorize_layer(name: str, cfg: Config) -> Tuple[str, List[np.ndarray]]:
    edge_path = os.path.join(cfg.output_dir, name, "edges.png")
    edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise FileNotFoundError(f"Edges not found: {edge_path}")

    print(f"\n[{name}] Centerline vectorization started…", flush=True)
    skel = thinning_zhangsuen(edges, layer=name)
    paths = trace_centerlines(skel, layer=name)
    # filter very short paths
    paths = [p for p in paths if len(p) >= 5]

    out_path = os.path.join(cfg.output_dir, name, "contours.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(paths, f)
    print(f"[{name}] Saved contours: {len(paths)} → {out_path}", flush=True)
    return name, paths

def vectorize_all(cfg: Config):
    results = []
    for i, name in enumerate(cfg.color_names, 1):
        print(f"[4/10] ({i}/{len(cfg.color_names)}) {name}", flush=True)
        results.append(vectorize_layer(name, cfg))
    return dict(results)

if __name__ == "__main__":
    config = load_config()
    vectorize_all(config)
