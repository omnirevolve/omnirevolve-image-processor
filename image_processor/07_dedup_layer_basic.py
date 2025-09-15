# 07_dedup_layer_basic.py
# Intra-layer dedup (STRICT)
# Input : <layer>/contours_sorted.pkl  (from step 06; no fallbacks)
# Output: <layer>/lines_intra.pkl, <layer>/taps_intra.pkl
#
# Stage A: greedy virtual draw with global forbid mask + sliding tail.
# Stage B: cluster near-parallel leftovers → rasterize (small brush) →
#          thinning → ONE path per component, anchored to old endpoints when possible.
#          Short ticks are dropped early.
#
# Logs show Stage-A progress and Stage-B cluster progress.

from __future__ import annotations
import os, math, pickle
from collections import deque
from typing import List, Tuple, Dict

import numpy as np
import cv2

from config import load_config, Config

# ------------------------------ geometry helpers ------------------------------

def _poly_perimeter(poly: np.ndarray) -> float:
    p = np.asarray(poly).reshape(-1, 2).astype(np.float32)
    if len(p) < 2: return 0.0
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())

def _bbox(poly: np.ndarray) -> Tuple[int,int,int,int]:
    p = np.asarray(poly).reshape(-1, 2)
    return int(np.floor(p[:,0].min())), int(np.floor(p[:,1].min())), \
           int(np.ceil(p[:,0].max())),  int(np.ceil(p[:,1].max()))

def _bbox_expand(b: Tuple[int,int,int,int], m: int) -> Tuple[int,int,int,int]:
    x0,y0,x1,y1 = b; return x0-m, y0-m, x1+m, y1+m

def _bbox_union(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    return min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3])

def _bbox_overlap(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> bool:
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

def _is_closed(poly: np.ndarray) -> bool:
    p = np.asarray(poly).reshape(-1,2)
    return len(p) > 2 and np.all(p[0] == p[-1])

def _ensure_open(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly).reshape(-1,2)
    if len(p) >= 2 and np.all(p[0] == p[-1]): p = p[:-1]
    return p.reshape(-1,1,2).astype(np.int32)

def _resample_arclen(pts: np.ndarray, step: float) -> np.ndarray:
    p = np.asarray(pts).reshape(-1,2).astype(np.float32)
    if len(p) < 2: return p
    if _is_closed(pts): p = p[:-1]
    seg = np.linalg.norm(p[1:] - p[:-1], axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= step: return p
    t = np.arange(0.0, s[-1], step, dtype=np.float32)
    k = np.searchsorted(s, t, side="right") - 1
    k = np.clip(k, 0, len(p)-2)
    u = (t - s[k]) / np.maximum(1e-6, s[k+1]-s[k])
    return p[k]*(1.0-u[:,None]) + p[k+1]*u[:,None]

# ------------------------------ spatial hash ---------------------------------

class _PointHash:
    """Sparse grid for local radius queries (self-collision within a single source poly)."""
    def __init__(self, radius: float, cell: float | None = None):
        self.r = float(radius)
        self.cell = float(cell if cell and cell > 0 else max(4.0, radius))
        self.inv = 1.0 / self.cell
        self.g: Dict[Tuple[int,int], List[Tuple[float,float]]] = {}

    def _key(self, x: float, y: float) -> Tuple[int,int]:
        return (int(math.floor(x*self.inv)), int(math.floor(y*self.inv)))

    def _nbrs(self, x: float, y: float):
        cx,cy = self._key(x,y)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                yield (cx+dx, cy+dy)

    def near(self, x: float, y: float) -> bool:
        R2 = self.r*self.r
        for k in self._nbrs(x,y):
            arr = self.g.get(k)
            if not arr: continue
            for (px,py) in arr:
                dx,dy = px-x, py-y
                if dx*dx + dy*dy <= R2: return True
        return False

    def add(self, x: float, y: float) -> None:
        k = self._key(x,y)
        lst = self.g.get(k)
        if lst is None: self.g[k] = [(x,y)]
        else: lst.append((x,y))

# ------------------------------- canvas size ---------------------------------

def _target_size_px(cfg: Config) -> Tuple[int,int]:
    tw_px = int(getattr(cfg,"target_width_px",0) or 0)
    th_px = int(getattr(cfg,"target_height_px",0) or 0)
    if tw_px>0 and th_px>0: return tw_px, th_px
    tw_mm=float(getattr(cfg,"target_width_mm",0) or 0)
    th_mm=float(getattr(cfg,"target_height_mm",0) or 0)
    ppm  =int(getattr(cfg,"pixels_per_mm",0) or 0)
    if tw_mm>0 and th_mm>0 and ppm>0: return int(round(tw_mm*ppm)), int(round(th_mm*ppm))
    base=cv2.imread(os.path.join(cfg.output_dir,"resized.png"))
    if base is None: raise RuntimeError("Cannot infer target size.")
    h,w=base.shape[:2]; return w,h

# ------------------------- Stage A: greedy virtual draw ----------------------

def _virtual_draw_split_with_mask_and_tail(
    poly: np.ndarray,
    sample_step: float,
    tail_len_px: float,
    mask: np.ndarray,
    forbid_val: int,
    local_radius_px: float,
    hash_stride: float,
    brush_forbid: int,
) -> List[np.ndarray]:
    p = _ensure_open(poly).reshape(-1,2).astype(np.float32)
    if len(p) < 2: return []
    S = _resample_arclen(p, step=max(1.0, float(sample_step)))
    if len(S) < 2: return []

    H_local = _PointHash(radius=local_radius_px, cell=hash_stride)
    tail: deque[Tuple[float,float]] = deque(); tail_len = 0.0

    h,w = mask.shape
    segs: List[np.ndarray] = []; cur: List[Tuple[float,float]] = []
    last_old: Tuple[int,int] | None = None

    def _push_tail(xy):
        nonlocal tail_len
        if tail: tail_len += float(np.linalg.norm(np.array(xy)-np.array(tail[-1])))
        tail.append(xy)

    def _pop_old_if_needed():
        nonlocal tail_len, last_old
        while tail and tail_len > tail_len_px:
            xy_old = tail.popleft()
            H_local.add(xy_old[0], xy_old[1])
            if tail: tail_len -= float(np.linalg.norm(np.array(tail[0])-np.array(xy_old)))
            else: tail_len = 0.0
            xi,yi = int(round(xy_old[0])), int(round(xy_old[1]))
            if 0<=xi<w and 0<=yi<h:
                if last_old is not None:
                    cv2.line(mask, last_old, (xi,yi), forbid_val, thickness=brush_forbid, lineType=cv2.LINE_8)
                last_old = (xi,yi)

    for (x,y) in S:
        _push_tail((float(x),float(y)))
        _pop_old_if_needed()

        xi,yi = int(round(x)), int(round(y))
        if xi<0 or yi<0 or xi>=w or yi>=h:
            if len(cur)>=2: segs.append(np.array(cur,np.int32).reshape(-1,1,2))
            cur=[]; continue

        if mask[yi,xi]==forbid_val or H_local.near(x,y):
            if len(cur)>=2: segs.append(np.array(cur,np.int32).reshape(-1,1,2))
            cur=[]; continue

        cur.append((float(x),float(y)))

    # flush tail: stamp remaining old points
    _pop_old_if_needed()
    while tail:
        xy_old = tail.popleft()
        xi,yi = int(round(xy_old[0])), int(round(xy_old[1]))
        if 0<=xi<w and 0<=yi<h:
            if last_old is not None:
                cv2.line(mask, last_old, (xi,yi), forbid_val, thickness=brush_forbid, lineType=cv2.LINE_8)
            last_old = (xi,yi)

    if len(cur)>=2: segs.append(np.array(cur,np.int32).reshape(-1,1,2))
    return segs

def _split_on_long_jumps(poly: np.ndarray, max_jump: float) -> List[np.ndarray]:
    p = np.asarray(poly).reshape(-1,2).astype(np.float32)
    if len(p)<2: return []
    out=[]; cur=[tuple(p[0])]
    for i in range(1,len(p)):
        d = float(np.linalg.norm(p[i]-p[i-1]))
        if d>max_jump and len(cur)>=2:
            out.append(np.array(cur,np.int32).reshape(-1,1,2)); cur=[tuple(p[i])]
        else:
            cur.append(tuple(p[i]))
    if len(cur)>=2: out.append(np.array(cur,np.int32).reshape(-1,1,2))
    return out

def _split_small_and_taps(polys: List[np.ndarray],
                          tap_diam: float,
                          min_keep_diam: float,
                          tap_max_perimeter: float,
                          tap_max_vertices: int,
                          tap_max_dim: float) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
    kept: List[np.ndarray]=[]; taps_xy: List[Tuple[int,int]]=[]
    for c in polys:
        p = np.asarray(c).reshape(-1,2)
        if p.shape[0] < 2: continue
        x0,y0,x1,y1 = _bbox(c); d=float(max(x1-x0, y1-y0))
        if d<=tap_diam and d<=tap_max_dim:
            per=_poly_perimeter(c); verts=int(p.shape[0])
            if per<=tap_max_perimeter and verts<=tap_max_vertices:
                (x,y),_ = cv2.minEnclosingCircle(p.reshape(-1,1,2).astype(np.float32))
                taps_xy.append((int(round(x)), int(round(y)))); continue
        if d<min_keep_diam: continue
        kept.append(_ensure_open(c))
    return kept, taps_xy

# ------------------------------ simple reorder -------------------------------

def _ends(poly: np.ndarray):
    pts=np.asarray(poly).reshape(-1,2); return pts[0], pts[-1]

def _reorder_only(contours: List[np.ndarray]) -> List[np.ndarray]:
    if not contours: return []
    used=np.zeros(len(contours),dtype=bool); starts=[]; ends=[]
    for c in contours:
        s,e=_ends(c); starts.append(s); ends.append(e)
    starts=np.array(starts); ends=np.array(ends)
    lengths=[float(_poly_perimeter(c)) for c in contours]
    cur=int(np.argmax(lengths)); order=[cur]; flips=[False]; used[cur]=True; cur_end=ends[cur]
    while not np.all(used):
        idxs=np.flatnonzero(~used)
        d2s=np.sum((starts[idxs].astype(np.float32)-cur_end.astype(np.float32))**2,axis=1)
        d2e=np.sum((ends[idxs].astype(np.float32)-cur_end.astype(np.float32))**2,axis=1)
        best=-1; flip=False; bestd=1e30
        for k,i in enumerate(idxs):
            if d2s[k]<=d2e[k]:
                if d2s[k]<bestd: bestd=d2s[k]; best=i; flip=False
            else:
                if d2e[k]<bestd: bestd=d2e[k]; best=i; flip=True
        used[best]=True; order.append(best); flips.append(flip)
        cur_end = starts[best] if flip else ends[best]
    out=[]
    for i,f in zip(order,flips):
        pts=np.asarray(contours[i]).reshape(-1,2)
        if f: pts=pts[::-1].copy()
        out.append(pts.reshape(-1,1,2).astype(np.int32))
    return out

# ----------------------------- Stage B utilities -----------------------------

_OFFS = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

def _neighbors8(img: np.ndarray, y: int, x: int):
    h,w = img.shape
    for dy,dx in _OFFS:
        ny,nx = y+dy, x+dx
        if 0<=ny<h and 0<=nx<w and img[ny,nx]>0:
            yield ny,nx

def _bfs_path(img: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """BFS on a 1-px skeleton component (cheap)."""
    if start == goal: return [start]
    h,w = img.shape
    que=[start]; head=0
    prev = -np.ones((h,w,2), np.int32)
    seen = np.zeros((h,w), np.uint8); seen[start]=1
    while head < len(que):
        y,x = que[head]; head += 1
        if (y,x) == goal: break
        for ny,nx in _neighbors8(img,y,x):
            if seen[ny,nx]: continue
            seen[ny,nx]=1; prev[ny,nx]=(y,x); que.append((ny,nx))
    if prev[goal][0] == -1: return []
    path=[goal]; y,x=goal
    while (y,x) != start:
        py,px = prev[y,x]
        if py == -1: return []
        path.append((py,px)); y,x = py,px
    path.reverse(); return path

def _farthest(img: np.ndarray, src: Tuple[int,int]) -> Tuple[Tuple[int,int], int]:
    """Return farthest pixel and distance within a component."""
    h,w = img.shape
    que=[src]; head=0
    dist = -np.ones((h,w), np.int32); dist[src]=0
    last=src
    while head < len(que):
        y,x=que[head]; head+=1; last=(y,x)
        for ny,nx in _neighbors8(img,y,x):
            if dist[ny,nx] != -1: continue
            dist[ny,nx] = dist[y,x] + 1; que.append((ny,nx))
    return last, int(dist[last])

def _component_best_path(bin_comp: np.ndarray,
                         anchor_a: Tuple[int,int] | None,
                         anchor_b: Tuple[int,int] | None,
                         min_len: int) -> List[Tuple[int,int]]:
    """One path per component:
       - if both anchors are inside → shortest geodesic between them;
       - else → diameter via two farthest BFS runs."""
    img = (bin_comp>0).astype(np.uint8)
    ys,xs = np.where(img>0)
    if ys.size == 0: return []
    # try anchored
    if anchor_a is not None and anchor_b is not None:
        ya,xa = anchor_a; yb,xb = anchor_b
        if 0<=ya<img.shape[0] and 0<=xa<img.shape[1] and \
           0<=yb<img.shape[0] and 0<=xb<img.shape[1] and \
           img[ya,xa] and img[yb,xb]:
            path=_bfs_path(img,(ya,xa),(yb,xb))
            if len(path) >= max(2, min_len): return path
    # fallback: diameter
    seed=(int(ys[0]), int(xs[0]))
    u,_=_farthest(img, seed); v,_=_farthest(img, u)
    path=_bfs_path(img, u, v)
    return path if len(path) >= max(2, min_len) else []

def _cluster_by_overlap(bboxes: List[Tuple[int,int,int,int]]) -> List[List[int]]:
    """Naive O(n^2) is fine for ~1–2k lines and keeps code simple."""
    n=len(bboxes)
    if n==0: return []
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def unite(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i in range(n):
        bi=bboxes[i]
        for j in range(i+1,n):
            if _bbox_overlap(bi,bboxes[j]): unite(i,j)
    groups: Dict[int,List[int]]={}
    for i in range(n):
        r=find(i); groups.setdefault(r,[]).append(i)
    return list(groups.values())

# --------------------------- proper Zhang–Suen fallback ----------------------

def _zhang_suen_fast(bin_img: np.ndarray, max_iter: int = 48) -> np.ndarray:
    img = (bin_img > 0).astype(np.uint8)
    if img.size == 0:
        return bin_img
    img = np.pad(img, 1, mode='constant', constant_values=0)
    changed = True; it = 0
    while changed and it < max_iter:
        it += 1; changed = False
        P2 = img[:-2, 1:-1];  P3 = img[:-2, 2:];   P4 = img[1:-1, 2:]
        P5 = img[2:,  2:];    P6 = img[2:,  1:-1]; P7 = img[2:,  0:-2]
        P8 = img[1:-1, 0:-2]; P9 = img[0:-2,0:-2]; C = img[1:-1,1:-1]
        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m1 = (C==1)&(B>=2)&(B<=6)&(A==1)&((P2*P4*P6)==0)&((P4*P6*P8)==0)
        if m1.any(): C[m1] = 0; changed = True

        P2 = img[:-2, 1:-1];  P3 = img[:-2, 2:];   P4 = img[1:-1, 2:]
        P5 = img[2:,  2:];    P6 = img[2:,  1:-1]; P7 = img[2:,  0:-2]
        P8 = img[1:-1, 0:-2]; P9 = img[0:-2,0:-2]; C = img[1:-1,1:-1]
        B = (P2+P3+P4+P5+P6+P7+P8+P9)
        A = ((P2==0)&(P3==1)).astype(np.uint8)+((P3==0)&(P4==1)).astype(np.uint8)+\
            ((P4==0)&(P5==1)).astype(np.uint8)+((P5==0)&(P6==1)).astype(np.uint8)+\
            ((P6==0)&(P7==1)).astype(np.uint8)+((P7==0)&(P8==1)).astype(np.uint8)+\
            ((P8==0)&(P9==1)).astype(np.uint8)+((P9==0)&(P2==1)).astype(np.uint8)
        m2 = (C==1)&(B>=2)&(B<=6)&(A==1)&((P2*P4*P8)==0)&((P2*P6*P8)==0)
        if m2.any(): C[m2] = 0; changed = True

    return img[1:-1, 1:-1].astype(np.uint8)*255

# ----------------------------- Stage B (cheap fixes) -------------------------

def _post_skeleton_merge(lines: List[np.ndarray],
                         brush_px: int,
                         resample_step: float,
                         rdp_eps: float,
                         min_path_len_px: int) -> List[np.ndarray]:
    if not lines: return []
    exp = brush_px*2 + 6
    bxs = [_bbox_expand(_bbox(p), exp) for p in lines]
    groups = _cluster_by_overlap(bxs)

    merged: List[np.ndarray] = []
    print(f"[intra] post-merge: clusters={len(groups)}", flush=True)

    for gi, idxs in enumerate(groups, 1):
        # pick anchors from the longest original line in the cluster
        longest = max(idxs, key=lambda j: _poly_perimeter(lines[j]))
        lp = lines[longest].reshape(-1,2)
        a0_abs = (int(lp[0,0]),  int(lp[0,1]))   # (x,y)
        a1_abs = (int(lp[-1,0]), int(lp[-1,1]))

        # cluster ROI
        bx = bxs[idxs[0]]
        for j in idxs[1:]: bx = _bbox_union(bx, bxs[j])
        x0,y0,x1,y1 = bx; w=max(1,x1-x0); h=max(1,y1-y0)

        roi = np.zeros((h,w), np.uint8)
        for j in idxs:
            p = lines[j].reshape(-1,2).astype(np.int32)
            q = (p - np.array([x0,y0],np.int32)).reshape(-1,1,2)
            if len(q)>=2:
                cv2.polylines(roi,[q],False,255,thickness=max(1,brush_px),lineType=cv2.LINE_8)

        # thinning (correct dtype for ximgproc; fallback to Zhang–Suen)
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            sk = cv2.ximgproc.thinning(roi, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            sk = (sk > 0).astype(np.uint8) * 255
        else:
            sk = _zhang_suen_fast(roi)

        if sk.sum()==0: 
            if (gi % max(1, len(groups)//20)) == 0 or gi == len(groups):
                print(f"[intra] post-merge: {gi}/{len(groups)} clusters", flush=True)
            continue

        # connected components once
        num, lab = cv2.connectedComponents((sk>0).astype(np.uint8), connectivity=8)

        # map anchors to nearest skeleton pixels inside ROI
        ys, xs = np.where(sk>0)
        pts = np.stack([ys, xs], axis=1) if ys.size else np.zeros((0,2),np.int32)
        def _nearest(yx_abs: Tuple[int,int]) -> Tuple[int,int] | None:
            y_abs, x_abs = yx_abs[1], yx_abs[0]
            if pts.shape[0]==0: return None
            dy = pts[:,0] - (y_abs - y0)
            dx = pts[:,1] - (x_abs - x0)
            k  = int(np.argmin(dy*dy + dx*dx))
            return (int(pts[k,0]), int(pts[k,1]))

        a0 = _nearest((a0_abs[0], a0_abs[1]))
        a1 = _nearest((a1_abs[0], a1_abs[1]))

        for cc in range(1, int(num)):
            comp = (lab==cc).astype(np.uint8)*255
            # anchors that lie inside this component
            aa = a0 if (a0 is not None and comp[a0]) else None
            bb = a1 if (a1 is not None and comp[a1]) else None
            path = _component_best_path(comp, aa, bb, min_len=min_path_len_px)
            if len(path) < 2: 
                continue  # drop tiny ticks immediately

            arr = np.array([(x0+x, y0+y) for (y,x) in path], np.float32)
            # quick length check is already in min_len; resample+RDP afterwards
            rs  = _resample_arclen(arr, resample_step)
            if len(rs) < 2: 
                continue
            # RDP in absolute coords
            P = rs.astype(np.float32)
            stack=[(0,len(P)-1)]; keep=np.zeros(len(P),bool); keep[0]=keep[-1]=True
            while stack:
                s,e=stack.pop()
                if e <= s+1: continue
                a,b=P[s],P[e]; seg=b-a; seg_n=np.array([-seg[1],seg[0]],np.float32)
                seg_len=float(np.linalg.norm(seg))+1e-12
                d=np.abs((P[s+1:e]-a)@seg_n)/seg_len
                i=int(np.argmax(d))
                if d[i] > rdp_eps:
                    k=s+1+i; keep[k]=True; stack.append((s,k)); stack.append((k,e))
            simp = P[keep].astype(np.int32).reshape(-1,1,2)
            merged.append(simp)

        if (gi % max(1, len(groups)//20)) == 0 or gi == len(groups):
            print(f"[intra] post-merge: {gi}/{len(groups)} clusters", flush=True)

    return merged

# --------------------------------- I/O ---------------------------------------

def _load_source(layer_dir: str) -> List[np.ndarray]:
    src = os.path.join(layer_dir, "contours_sorted.pkl")
    if not os.path.exists(src):
        raise RuntimeError(f"[intra] missing input: {src}. Run step 06 first.")
    with open(src,"rb") as f: obj=pickle.load(f)
    if not isinstance(obj,list):
        raise RuntimeError(f"[intra] invalid pickle format: {src}")
    return obj

# --------------------------------- main --------------------------------------

def process_layer(layer_dir: str, cfg: Config) -> None:
    pen_diam   = float(getattr(cfg,"pen_width_px",60))
    pen_radius = float(getattr(cfg,"pen_radius_px",pen_diam/2.0))

    tap_diam    = float(getattr(cfg,"tap_diameter_px",pen_diam))
    tap_max_dim = float(getattr(cfg,"tap_max_dim",tap_diam))
    min_keep    = float(getattr(cfg,"min_keep_diameter_px",max(10.0, pen_radius*0.4)))
    tap_max_per = float(getattr(cfg,"tap_max_perimeter",2.5*tap_diam))
    tap_max_v   = int(getattr(cfg,"tap_max_vertices",50))

    sample_step = float(getattr(cfg,"dedup_sample_step",8))
    tail_len_px = float(getattr(cfg,"ignore_tail_len_px",
                        float(getattr(cfg,"ignore_tail_points_intra",120))))
    col_rad     = float(getattr(cfg,"collision_radius_intra_px",max(2*pen_radius,60.0)))
    grid_stride = float(getattr(cfg,"hash_stride_px",max(col_rad*0.8,18.0)))
    max_jump    = float(getattr(cfg,"max_join_jump_px",80.0))

    post_on     = bool(getattr(cfg,"intra_post_skeleton_enabled",True))
    post_brush  = int(getattr(cfg,"intra_post_brush_px",16))
    post_step   = float(getattr(cfg,"intra_post_resample_step_px",6))
    post_eps    = float(getattr(cfg,"intra_post_rdp_epsilon_px",max(1.0,0.08*post_brush)))
    post_minlen = int(getattr(cfg,"intra_post_min_path_len_px",max(2*post_brush,12)))

    W,H=_target_size_px(cfg)
    forbid=np.zeros((H,W),np.uint8); FVAL=255
    brush_forbid=max(1,int(round(2.0*col_rad)))

    polys=_load_source(layer_dir)
    if not polys:
        print(f"[intra] {os.path.basename(layer_dir)}: empty input."); return

    kept,taps=_split_small_and_taps(polys, tap_diam, min_keep, tap_max_per, tap_max_v, tap_max_dim)

    order=sorted(range(len(kept)), key=lambda i:_poly_perimeter(kept[i]), reverse=True)
    cleaned: List[np.ndarray]=[]
    total=len(order)
    if total==0:
        lines2, taps2 = [], taps
    else:
        chunk=max(1,total//20)
        for idx,i in enumerate(order,1):
            segs=_virtual_draw_split_with_mask_and_tail(
                poly=kept[i], sample_step=sample_step, tail_len_px=tail_len_px,
                mask=forbid, forbid_val=FVAL, local_radius_px=col_rad,
                hash_stride=grid_stride, brush_forbid=brush_forbid
            )
            if segs:
                for s in segs:
                    parts=_split_on_long_jumps(s, max_jump)
                    cleaned.extend(parts if parts else [s])
            if (idx%chunk)==0 or idx==total:
                print(f"[intra] {os.path.basename(layer_dir)}: {idx}/{total} ({(idx*100)//total}%)", flush=True)

        lines2, taps2 = _split_small_and_taps(cleaned, tap_diam, min_keep, tap_max_per, tap_max_v, tap_max_dim)
        taps = taps2 if len(taps)==0 else (taps + taps2)

    if post_on and len(lines2)>0:
        before=len(lines2)
        print(f"[intra] {os.path.basename(layer_dir)}: post-merge start — lines={before}", flush=True)
        lines2 = _post_skeleton_merge(lines2, post_brush, post_step, post_eps, post_minlen)
        print(f"[intra] {os.path.basename(layer_dir)}: post-merge {before} → {len(lines2)} lines", flush=True)

    lines2=_reorder_only(lines2)

    out_lines=os.path.join(layer_dir,"lines_intra.pkl")
    out_taps =os.path.join(layer_dir,"taps_intra.pkl")
    with open(out_lines,"wb") as f: pickle.dump(lines2,f)
    with open(out_taps,"wb")  as f: pickle.dump(taps,f)

    vin=sum(int(p.reshape(-1,2).shape[0]) for p in polys)
    vout=sum(int(p.reshape(-1,2).shape[0]) for p in lines2)
    print(f"[intra] {os.path.basename(layer_dir)}: lines={len(lines2)}, taps={len(taps)}, "
          f"vertices_in={vin}, vertices_out={vout}")
    print(f"[intra]   saved → {os.path.relpath(out_lines, layer_dir)}, {os.path.relpath(out_taps, layer_dir)}")

def main():
    cfg: Config = load_config()
    for name in cfg.color_names:
        process_layer(os.path.join(cfg.output_dir, name), cfg)

if __name__ == "__main__":
    main()
