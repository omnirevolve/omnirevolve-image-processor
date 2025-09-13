#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный пайплайн дедупликации контуров для плоттера.
УПРОЩЕНО: работаем в одной системе координат — «пиксели = шаги».
Никаких пересчётов единиц, никаких масштабирований: входные слои уже
приведены normalize_image.py к целевому размеру (1 мм = 40 шагов/px).

Выход: превью, пакет .pkl по слоям и vector_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- внешние модули проекта ---
from contour_deduplicator import (  # type: ignore
    Tap,
    ProcessedContour,
    TapMerger,
    ContourAnalyzer,
    load_contours_from_layer,
    save_processing_stats,
    TapOptimizer,  # оптимизация порядка тапов
)
from contour_processor import (  # type: ignore
    SingleColorProcessor,
    ContourPathOptimizer,
)
from cross_color_deduplicator import (  # type: ignore
    ColorLayer,
    CrossColorDeduplicator,
    LuminanceCalculator,
)

# ============================= утилиты/вспомогалки =============================

def _ensure_xy(contour) -> np.ndarray:
    """Поддержка и numpy-контуров (N,2)/(N,1,2), и объектов с .points"""
    if hasattr(contour, "points"):
        pts = np.asarray(contour.points, dtype=np.float32)
    else:
        pts = np.asarray(contour, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)
    return pts


def _poly_length_xy(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    d = np.diff(pts.astype(np.float64), axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def self_dedup_polyline_segments(
    pts: np.ndarray, r_px: float, min_index_gap: int = 8
) -> List[np.ndarray]:
    """
    Внутриконтурная самодедупликация.
    Режем полилинию при подходе к ранее пройденным точкам ближе r_px
    (с индексной защитой от локальной кривизны).
    """
    n = int(pts.shape[0])
    if n < 2:
        return []

    cell = max(1, int(r_px))  # размер ячейки spatial hash
    r2 = float(r_px * r_px)

    grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def _key(x: float, y: float) -> Tuple[int, int]:
        return (int(x) // cell, int(y) // cell)

    def _put(i: int, x: float, y: float) -> None:
        k = _key(x, y)
        grid.setdefault(k, []).append((i, int(x), int(y)))

    def _hit(i: int, x: float, y: float) -> bool:
        cx, cy = _key(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j, qx, qy in grid.get((cx + dx, cy + dy), []):
                    if i - j <= min_index_gap:
                        continue
                    if (x - qx) * (x - qx) + (y - qy) * (y - qy) <= r2:
                        return True
        return False

    segs: List[np.ndarray] = []
    start = 0

    _put(0, pts[0, 0], pts[0, 1])
    i = 1
    while i < n:
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if _hit(i, x, y):
            # завершаем «хороший» сегмент до попадания
            if i - start >= 2:
                segs.append(pts[start:i].copy())
            # пропускаем зону конфликта пока не выйдем из радиуса
            i += 1
            while i < n:
                x2, y2 = float(pts[i, 0]), float(pts[i, 1])
                if not _hit(i, x2, y2):
                    start = i
                    _put(i, x2, y2)
                    i += 1
                    break
                i += 1
        else:
            _put(i, x, y)
            i += 1

    if n - start >= 2:
        segs.append(pts[start:n].copy())

    return segs


def segments_or_taps(
    segs: List[np.ndarray], tap_len_px: float
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Разделение сегментов на линии и «тапы».
    Если длина сегмента < tap_len_px — создаём тап в первой точке.
    """
    keep: List[np.ndarray] = []
    taps: List[Tuple[int, int]] = []
    for s in segs:
        L = _poly_length_xy(s)
        if L < tap_len_px:
            taps.append((int(s[0, 0]), int(s[0, 1])))
        else:
            keep.append(s)
    return keep, taps

# =============================== основной класс ===============================

class DeduplicationPipeline:
    """
    Оркестратор дедупликации (внутрицветовая + межцветовая),
    оптимизация порядка и сохранение вектора.
    Работает в координатах «пиксели = шаги».
    """

    def __init__(
        self,
        pen_diameter: int = 60,
        max_shift_ratio: float = 0.5,
        enable_self_dedup: bool = True,
        self_dedup_when: str = "end",        # "start" | "end"
        self_dedup_min_index_gap: int = 4,
        self_dedup_len_to_tap: float = 0.5,  # доля от R пера
        # Параметры ниже оставлены для CLI-совместимости, но не используются:
        target_width_steps: Optional[int] = None,
        target_height_steps: Optional[int] = None,
    ):
        self.pen_diameter = int(pen_diameter)
        self.pen_radius = self.pen_diameter // 2
        self.max_shift = self.pen_radius * float(max_shift_ratio)

        # параметры самодедупа
        self.enable_self_dedup = bool(enable_self_dedup)
        self.self_dedup_when = "start" if str(self_dedup_when).lower() == "start" else "end"
        self.self_dedup_min_index_gap = int(max(1, self_dedup_min_index_gap))
        self.self_dedup_len_px = float(self.pen_radius * max(0.0, self_dedup_len_to_tap))

        # подсистемы
        self.single_color = SingleColorProcessor(
            pen_radius=self.pen_radius, max_shift=self.max_shift
        )
        self.cross_color = CrossColorDeduplicator(pen_radius=self.pen_radius)
        self.path_optimizer = ContourPathOptimizer()
        self.lum = LuminanceCalculator()
        self.tap_optimizer = TapOptimizer(two_opt_passes=2)

        self.stats: Dict[str, Dict] = {}

    # --------------------------- самодедуп слоя ---------------------------

    def _self_dedup_layer(
        self, contours: List[ProcessedContour], taps: List[Tap]
    ) -> Tuple[List[ProcessedContour], List[Tap]]:
        if not contours:
            return contours, taps

        new_contours: List[ProcessedContour] = []
        new_taps: List[Tap] = list(taps)

        R = float(self.pen_radius)
        gap = int(self.self_dedup_min_index_gap)
        tap_len_px = float(self.self_dedup_len_px)

        tap_count = 0
        seg_count = 0

        for cnt in contours:
            pts = _ensure_xy(cnt).astype(np.float32)
            if pts.shape[0] < 2:
                if pts.shape[0] == 1:
                    new_taps.append(Tap(float(pts[0, 0]), float(pts[0, 1]), 1.0))
                    tap_count += 1
                continue

            segs = self_dedup_polyline_segments(pts, r_px=R, min_index_gap=gap)
            keep, taps_created = segments_or_taps(segs, tap_len_px=tap_len_px)

            for s in keep:
                new_contours.append(
                    ProcessedContour(points=s, color_idx=getattr(cnt, "color_idx", 0), is_closed=False)
                )
            for (tx, ty) in taps_created:
                new_taps.append(Tap(float(tx), float(ty), 1.0))

            tap_count += len(taps_created)
            seg_count += len(keep)

        print(
            f"  [self-dedup] kept {seg_count} segments, +{tap_count} taps "
            f"(R={R:.1f}, gap={gap}, tap_len={tap_len_px:.1f})"
        )
        return new_contours, new_taps

    # ------------------------------ загрузка слоёв ------------------------------

    def _extract_color_name(self, filename: str) -> str:
        parts = filename.split("_")
        if "layer" in parts:
            i = parts.index("layer")
            if i + 2 < len(parts):
                return "_".join(parts[i + 2 :])
        return parts[-1] if parts else "unknown"

    def _load_color_info(self, reference_file: Optional[str]) -> Optional[Dict]:
        if not reference_file:
            return None
        base_dir = Path(reference_file).parent
        colors_json = base_dir / "colors.json"
        if colors_json.exists():
            try:
                with open(colors_json, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _load_and_prepare_layers(
        self, layer_files: List[str], color_mapping: Dict[str, int]
    ) -> List[ColorLayer]:
        layers: List[ColorLayer] = []
        color_info = self._load_color_info(layer_files[0] if layer_files else None)

        for layer_path in layer_files:
            filename = Path(layer_path).stem
            color_name = self._extract_color_name(filename)

            if color_name not in color_mapping:
                print(f"Warning: Color {color_name} not in mapping, skip {layer_path}")
                continue

            color_idx = int(color_mapping[color_name])
            contours = load_contours_from_layer(layer_path, color_idx)

            img = cv2.imread(layer_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ih, iw = img.shape
            else:
                iw, ih = 800, 600

            if color_info and color_name in color_info:
                r, g, b = color_info[color_name]["rgb"]
                luminance = self.lum.rgb_to_luminance(r, g, b)
            else:
                luminance = color_idx * 50.0

            ly = ColorLayer(
                color_idx=color_idx,
                color_name=color_name,
                luminance=luminance,
                contours=contours,
                taps=[],
            )
            # Размер слоя уже «в шагах» (пикселях после нормализации)
            ly.source_size = (iw, ih)
            layers.append(ly)

            print(
                f"Loaded {color_name}: {len(contours)} contours, "
                f"size={iw}x{ih}, luminance={luminance:.1f}"
            )

        layers.sort(key=lambda L: L.luminance)
        return layers

    # ------------------------- (нет масштабирования) -------------------------

    def _confirm_canvas(self, layers: List['ColorLayer']) -> Tuple[int, int]:
        """
        Просто проверяем/возвращаем размер холста. Никакого масштабирования.
        """
        if not layers:
            return (0, 0)
        w, h = layers[0].source_size
        print(f"\nCanvas (steps): {w} x {h}  [identity]")
        return (w, h)

    # ------------------------------- сохранение -------------------------------

    def _draw_preview_layer(self, w: int, h: int, contours: List[ProcessedContour], taps: List[Tap]) -> np.ndarray:
        img = np.full((h, w), 255, dtype=np.uint8)
        for c in contours:
            pts = _ensure_xy(c).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, 0, 1, lineType=cv2.LINE_AA)
        for t in taps:
            cv2.circle(img, (int(t.x), int(t.y)), 2, 0, -1, lineType=cv2.LINE_AA)
        return img

    def _save_vector_bundle(self, final_layers: List['ColorLayer'], out_dir: Path, img_size: Tuple[int, int]) -> None:
        """
        Сохраняем векторный пакет. Координаты уже «в шагах» (пикселях целевого холста).
        Никаких units и original_size — только конечный размер.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "image_size": list(img_size),   # (width_steps, height_steps)
            "layers": [],
        }

        for i, L in enumerate(final_layers):
            layer_dict = {
                "color_idx": L.color_idx,
                "color_name": L.color_name,
                "luminance": float(L.luminance),
                "contours": [{"points": _ensure_xy(c).astype(np.float32)} for c in L.contours],
                "taps": [{"x": float(t.x), "y": float(t.y), "weight": getattr(t, "weight", 1.0)} for t in L.taps],
            }

            # сохраняем pkl
            pkl_name = f"layer_{i:02d}_{L.color_name}.pkl"
            pkl_path = out_dir / pkl_name
            with open(pkl_path, "wb") as f:
                pickle.dump(layer_dict, f)

            manifest["layers"].append({"file": pkl_name})

        with open(out_dir / "vector_manifest.json", "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _save_results(self, final_layers: List['ColorLayer'], out_dir: Path) -> Dict:
        iw, ih = final_layers[0].source_size if final_layers else (800, 600)
        # превью по слоям
        grid = []
        for L in final_layers:
            img = self._draw_preview_layer(iw, ih, L.contours, L.taps)
            grid.append(img)

        if grid:
            # 2-колоночная мозаика
            cols = 2
            rows = (len(grid) + cols - 1) // cols
            canvas = np.full((rows * ih, cols * iw), 255, dtype=np.uint8)
            for idx, img in enumerate(grid):
                r, c = idx // cols, idx % cols
                canvas[r * ih : (r + 1) * ih, c * iw : (c + 1) * iw] = img
            cv2.imwrite(str(out_dir / "preview_layers.png"), canvas)

        # векторный пакет
        self._save_vector_bundle(final_layers, out_dir, (iw, ih))

        # сводная статистика
        result = {
            "layers": [
                {
                    "color_name": L.color_name,
                    "color_idx": L.color_idx,
                    "contours_count": len(L.contours),
                    "taps_count": len(L.taps),
                }
                for L in final_layers
            ],
            "drawing_order": [L.color_name for L in final_layers],
        }
        return result

    # ------------------------------ основной прогон ------------------------------

    def process_image_layers(
        self, layer_files: List[str], color_mapping: Dict[str, int], output_dir: str
    ) -> Dict:
        t0 = time.time()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("DEDUPLICATION PIPELINE (self-dedup enabled)" if self.enable_self_dedup else "DEDUPLICATION PIPELINE")
        print("=" * 60)
        print(f"Pen diameter: {self.pen_diameter}px  (R = {self.pen_radius}px)")
        print(f"Max shift: {self.max_shift:.1f}px")
        print(f"Layers input: {len(layer_files)}")
        print("=" * 60)

        # 1) загрузка
        layers = self._load_and_prepare_layers(layer_files, color_mapping)

        # 2) подтверждаем размер (без масштабирования)
        w, h = self._confirm_canvas(layers)

        # 3) внутрицветовая обработка
        processed: List[ColorLayer] = []
        for L in layers:
            print(f"\nColor: {L.color_name} (idx {L.color_idx})")

            if self.enable_self_dedup and self.self_dedup_when == "start":
                sc, st = self._self_dedup_layer(L.contours, L.taps)
                L.contours, L.taps = sc, st
                if L.taps:
                    L.taps = self.tap_optimizer.optimize(L.taps, start=(0.0, 0.0))

            cont, taps = self.single_color.process_color_layer(L.contours, L.color_idx)

            if self.enable_self_dedup and self.self_dedup_when == "end":
                cont, taps = self._self_dedup_layer(cont, taps)
                if taps:
                    taps = self.tap_optimizer.optimize(taps, start=(0.0, 0.0))

            # оптимизация порядка контуров; тапы «как есть»
            cont, taps = self.path_optimizer.optimize_path(cont, taps)
            # дополнительно упорядочим тапы, минимизируя пробег от (0,0)
            if taps:
                taps = self.tap_optimizer.optimize(taps, start=(0.0, 0.0))

            L.contours, L.taps = cont, taps
            processed.append(L)

        # 4) межцветовая дедупликация (тёмные «побеждают»)
        print("\n=== Cross-color deduplication ===")
        final_layers = self.cross_color.process_all_colors(processed)

        # 5) финальная оптимизация
        print("\n=== Final path optimization ===")
        for L in final_layers:
            L.contours, L.taps = self.path_optimizer.optimize_path(L.contours, L.taps)
            if L.taps:
                L.taps = self.tap_optimizer.optimize(L.taps, start=(0.0, 0.0))

        # 6) сохранение результатов
        results = self._save_results(final_layers, out_dir)

        # 7) статистика
        elapsed = time.time() - t0
        self.stats["processing_time"] = elapsed
        self.stats["total_layers"] = len(final_layers)
        save_processing_stats(self.stats, str(out_dir / "deduplication_stats.json"))

        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETED in {elapsed:.1f}s")
        print(f"Results saved to: {out_dir}")
        print("=" * 60)
        return results


# =================================== CLI ===================================

def parse_mapping(s: str) -> Dict[str, int]:
    """
    'name1:0,name2:1,...' -> {'name1':0,'name2':1,...}
    """
    out: Dict[str, int] = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        name, idx = pair.split(":")
        out[name.strip()] = int(idx.strip())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplication pipeline for plotter (pixels == steps)")
    p.add_argument("layers", nargs="+", help="Edge layer PNG files")
    p.add_argument("-o", "--output", default="edges_dedup", help="Output directory")
    p.add_argument("-m", "--mapping", required=True, help="Color mapping name:idx,...")
    p.add_argument("--pen-diameter", type=int, default=60, help="Pen diameter in pixels (== steps)")
    p.add_argument("--max-shift-ratio", type=float, default=0.5, help="Max shift / pen radius")

    # Самодедуп — совместимые флаги
    p.add_argument("--self-dedup", action="store_true", help="Enable intra-contour deduplication")
    p.add_argument(
        "--self-dedup-when",
        choices=["start", "end"],
        default="end",
        help="Run self-dedup before ('start') or after ('end') within-color dedup",
    )
    p.add_argument("--self-dedup-min-index-gap", type=int, default=4, help="Ignore near-index points")
    p.add_argument(
        "--self-dedup-len-to-tap",
        type=float,
        default=0.5,
        help="Segment length (in radiuses) below which the fragment becomes a tap",
    )

    # Оставлено для совместимости, но не используется (normalize уже сделал «шаги»):
    p.add_argument("--target-width-steps", type=int, help="(compat) Target canvas width in steps")
    p.add_argument("--target-height-steps", type=int, help="(compat) Target canvas height in steps")

    args = p.parse_args()

    mapping = parse_mapping(args.mapping)

    pipe = DeduplicationPipeline(
        pen_diameter=args.pen_diameter,
        max_shift_ratio=args.max_shift_ratio,
        enable_self_dedup=bool(args.self_dedup),
        self_dedup_when=args.self_dedup_when,
        self_dedup_min_index_gap=args.self_dedup_min_index_gap,
        self_dedup_len_to_tap=args.self_dedup_len_to_tap,
        target_width_steps=args.target_width_steps,   # совместимость
        target_height_steps=args.target_height_steps, # совместимость
    )

    results = pipe.process_image_layers(args.layers, mapping, args.output)

    print("\n=== RESULTS ===")
    for L in results["layers"]:
        print(f"{L['color_name']}: {L['contours_count']} contours, {L['taps_count']} taps")
    print(f"\nDrawing order: {results['drawing_order']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
