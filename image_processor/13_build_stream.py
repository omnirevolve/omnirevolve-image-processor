# 13_build_stream.py
# Build binary stream from interleaved ops (plot_ops.pkl per layer) using shared helpers.
# Input : <output>/vector_manifest.json (from step 12)
# Output: <output>/plot_stream.bin (+ <output>/plot_stream.json)

from __future__ import annotations
import os, json, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import load_config, Config

# shared helpers
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "shared"))
if SHARED not in sys.path:
    sys.path.insert(0, SHARED)

from xyplotter_stream_creator_helper import (
    Config as StreamCfg,
    StreamWriter,
    travel_ramped,
    emit_polyline,
)

def _target_size_px(cfg: Config) -> Tuple[int,int]:
    tw = int(getattr(cfg, "target_width_px", 0) or 0)
    th = int(getattr(cfg, "target_height_px", 0) or 0)
    if tw > 0 and th > 0:
        return tw, th
    tw_mm = float(getattr(cfg, "target_width_mm", 0) or 0)
    th_mm = float(getattr(cfg, "target_height_mm", 0) or 0)
    ppm   = int(getattr(cfg, "pixels_per_mm", 0) or 0)
    if tw_mm > 0 and th_mm > 0 and ppm > 0:
        return int(round(tw_mm * ppm)), int(round(th_mm * ppm))
    raise RuntimeError("Cannot infer target size; set target_* or run earlier steps.")

def _first_op_xy(ops: List[Dict[str,Any]]) -> Tuple[int,int]:
    if not ops: return (0,0)
    o = ops[0]
    if o["type"] == "tap":
        return (int(o["x"]), int(o["y"]))
    p = o["points"]
    p = p.reshape(-1,2)
    return (int(p[0,0]), int(p[0,1]))

def _emit_layer(w: StreamWriter, cfg: StreamCfg, color_index: int, ops: List[Dict[str,Any]]):
    if not ops:
        return
    cur_x = cur_y = None

    # Move to first op BEFORE selecting color (pen activation on color switch)
    fx, fy = _first_op_xy(ops)
    cur_x, cur_y = fx, fy
    travel_ramped(w, 0, 0, cur_x, cur_y, cfg)  # from origin; upstream keeps pen up

    w.select_color(color_index)

    for o in ops:
        if o["type"] == "tap":
            tx, ty = int(o["x"]), int(o["y"])
            if (cur_x, cur_y) != (tx, ty):
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, tx, ty, cfg)
                cur_x, cur_y = tx, ty
            w.tap()
        else:
            pts = o["points"].reshape(-1,2).astype(int)
            sx, sy = int(pts[0,0]), int(pts[0,1])
            if (cur_x, cur_y) != (sx, sy):
                w.pen_up()
                travel_ramped(w, cur_x, cur_y, sx, sy, cfg)
                cur_x, cur_y = sx, sy
            w.pen_down()
            emit_polyline(w, cfg, [(int(x),int(y)) for x,y in pts])
            w.pen_up()
            cur_x, cur_y = int(pts[-1,0]), int(pts[-1,1])

def main():
    cfg: Config = load_config()
    outdir = cfg.output_dir
    man = Path(outdir) / "vector_manifest.json"
    if not man.exists():
        raise RuntimeError(f"Missing manifest: {man}. Run step 12 first.")

    W, H = _target_size_px(cfg)

    steps_per_mm = float(getattr(cfg, "pixels_per_mm", 40))
    invert_y = True

    scfg = StreamCfg(
        steps_per_mm=steps_per_mm,
        invert_y=invert_y,
        div_start=int(getattr(cfg, "stream_div_start", 64)),
        div_fast=int(getattr(cfg, "stream_div_fast", 20)),
        profile=str(getattr(cfg, "stream_profile", "triangle")),
        travel_div_fast=int(getattr(cfg, "stream_travel_div_fast", 10)),
        corner_deg=float(getattr(cfg, "stream_corner_deg", 85.0)),
        corner_div=int(getattr(cfg, "stream_corner_div", 64)),
        corner_window_steps=int(getattr(cfg, "stream_corner_window_steps",
                                        int(2.0 * steps_per_mm))),  # ≈2 cm
    )

    data = json.loads(man.read_text(encoding="utf-8"))
    layers = data.get("layers", [])

    w = StreamWriter()
    w.pen_up()

    for L in sorted(layers, key=lambda e: int(e.get("color_index", 0))):
        fp = Path(outdir) / L["file"]
        if not fp.exists():
            raise RuntimeError(f"Missing plot_ops: {fp}")
        blob = pickle.loads(fp.read_bytes())
        ops = blob.get("ops", [])
        # ensure ndarray for line points
        norm_ops: List[Dict[str,Any]] = []
        for o in ops:
            if o["type"] == "line":
                pts = np.asarray(o["points"]).reshape(-1,2).astype(np.int32)
                norm_ops.append({"type":"line","points":pts})
            else:
                norm_ops.append({"type":"tap","x":int(o["x"]),"y":int(o["y"])})
        _emit_layer(w, scfg, int(L.get("color_index", 0)), norm_ops)

    stream_bytes = w.finalize()
    out_bin = Path(outdir) / "plot_stream.bin"
    out_json = Path(outdir) / "plot_stream.json"
    out_bin.write_bytes(stream_bytes)
    out_json.write_text(json.dumps({
        "bytes": len(stream_bytes),
        "layers": len(layers),
        "target_steps": {"width": W, "height": H},
        "config": {k:getattr(scfg,k) for k in vars(scfg)}
    }, indent=2), encoding="utf-8")

    print(f"[stream] saved: {out_bin} ({len(stream_bytes)} bytes)")

if __name__ == "__main__":
    main()
