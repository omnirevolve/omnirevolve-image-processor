#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработчик контуров внутри одного цвета с виртуальной отрисовкой,
включая самодедупликацию (удаление самоперекрытий внутри одного контура).
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from contour_deduplicator import (
    Tap,
    ProcessedContour,
    TapMerger,
    ContourAnalyzer,
    TapOptimizer,  # ← добавлено: оптимизация порядка тапов
)
# Упрощение контуров (как и раньше)
from contour_simplifier import ContourSimplifier


# --------------------------- сервисные структуры ---------------------------

@dataclass
class ConflictPoint:
    """Точка конфликта между КОНТУРАМИ (используется в межконтурной дедупликации)."""
    contour_idx: int
    point_idx: int
    position: Tuple[float, float]
    is_fixed: bool = False


@dataclass
class VirtualDrawingState:
    """Состояние виртуальной отрисовки для МЕЖКОНТУРНЫХ конфликтов."""
    drawn_points: List[Tuple[float, float]] = field(default_factory=list)
    drawn_contour_ids: List[int] = field(default_factory=list)
    fixed_points: Set[Tuple[int, int]] = field(default_factory=set)  # (contour_idx, point_idx)
    point_shifts: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)
    removed_points: Set[Tuple[int, int]] = field(default_factory=set)


# =============================== ядро класса ===============================

class SingleColorProcessor:
    """
    Обработчик контуров одного цвета:
      - извлечение/слияние тапов;
      - оптимизация порядка;
      - виртуальная межконтурная дедупликация;
      - САМОдедупликация контуров (новое).
    """

    def __init__(self,
                 pen_radius: float = 30.0,
                 max_shift: float = 15.0,
                 enable_simplification: bool = True,
                 simplifier: Optional[ContourSimplifier] = None,
                 self_dedup: str = "post",
                 self_dedup_min_gap: int = 8,
                 kdtree_rebuild_every: int = 512,   # пакетная пересборка KD-дерева
                 kdtree_tail_window: int = 256):     # «хвост» точек, не попавших в KD-дерево
        """
        Args:
            pen_radius: радиус пера (px)
            max_shift : предельный сдвиг "тянущихся" точек (px)
            enable_simplification: включить ли предварительное упрощение контуров
            simplifier: внешний упрощатель (если None — создадим дефолтный)
            self_dedup: 'off' | 'pre' | 'post' — где выполнять самодедупликацию
            self_dedup_min_gap: минимальный индексный разрыв для проверки на перекрытия
            kdtree_rebuild_every: перестраивать KD-дерево не чаще, чем раз в N добавленных точек
            kdtree_tail_window: гарантированно проверять последние N точек, не попавших в KD-дерево
        """
        self.pen_radius = float(pen_radius)
        self.max_shift = float(max_shift)
        self.self_dedup = self_dedup if self_dedup in ("off", "pre", "post") else "post"
        self.self_dedup_min_gap = int(self_dedup_min_gap)

        self.kdtree_rebuild_every = int(max(1, kdtree_rebuild_every))
        self.kdtree_tail_window = int(max(1, kdtree_tail_window))

        self.tap_merger = TapMerger(self.pen_radius)
        # Analyzer уже умеет решать «это тап/нет», а также разбивать по удалённым точкам
        self.analyzer = ContourAnalyzer(int(self.pen_radius * 2))

        self.enable_simplification = bool(enable_simplification)
        self.simplifier = simplifier or ContourSimplifier(
            epsilon_ratio=0.002,
            smoothing_window=5,
        )

        # Оптимизатор порядка выполнения тапов (без изменения другой логики)
        self.tap_optimizer = TapOptimizer(two_opt_passes=2)

    # ------------------------- публичный сценарий цвета -------------------------

    def process_color_layer(self,
                            contours: List[ProcessedContour],
                            color_idx: int) -> Tuple[List[ProcessedContour], List[Tap]]:
        """
        Полная обработка контуров одного цвета с усиленной фильтрацией тапов.
        Возвращает финальные «чистые» контуры и список тапов.
        """
        print(f"\nProcessing color {color_idx}: {len(contours)} contours")

        # 1) мелкие контуры -> тапы
        taps, remaining_contours = self._extract_initial_taps(contours)
        print(f"  Initial: {len(taps)} taps, {len(remaining_contours)} contours")

        # 2) схлопываем близкие тапы
        if taps:
            taps = self.tap_merger.merge_taps(taps)
            print(f"  After tap merging: {len(taps)} taps")

        # 2.5) (опционально) САМОдедупликация в самом начале
        if self.self_dedup == "pre" and remaining_contours:
            remaining_contours, new_taps = self._self_deduplicate_contours(remaining_contours)
            if new_taps:
                taps.extend(new_taps)
                taps = self.tap_merger.merge_taps(taps)

        # 3) порядок отрисовки (жадный ближайший сосед)
        remaining_contours = self._optimize_drawing_order(remaining_contours)

        # 4) итеративная межконтурная дедупликация (как раньше)
        iteration, max_iterations = 0, 10
        while iteration < max_iterations:
            iteration += 1
            print(f"  Deduplication iteration {iteration}")
            processed_contours, new_taps, changes_made = self._virtual_drawing_pass(
                remaining_contours, color_idx
            )
            taps.extend(new_taps)

            if not changes_made:
                print(f"  Converged after {iteration} iterations")
                remaining_contours = processed_contours
                break

            remaining_contours = self._optimize_drawing_order(processed_contours)

        # 5) финальная склейка тапов
        if taps:
            taps = self.tap_merger.merge_taps(taps)

        # 6) УСИЛЕННАЯ ФИЛЬТРАЦИЯ: удаляем тапы близкие к контурам
        if taps and remaining_contours:
            # Используем ДИАМЕТР пера как минимальное расстояние (требование задачи)
            taps = self.tap_merger.filter_taps_near_contours(
                taps, remaining_contours, min_distance=self.pen_radius * 2.0
            )

        # 6.5) НОВОЕ: фильтруем тапы, которые слишком близко друг к другу
        if taps:
            taps = self.tap_merger.filter_taps_mutual_distance(
                taps, min_distance=self.pen_radius * 2.0
            )

        # 6.7) Оптимизация порядка выполнения тапов (новое)
        if taps:
            taps = self.tap_optimizer.optimize(taps, start=(0.0, 0.0))

        # 7) (опционально) САМОдедупликация в самом конце (по умолчанию)
        if self.self_dedup == "post" and remaining_contours:
            remaining_contours, new_taps = self._self_deduplicate_contours(remaining_contours)
            if new_taps:
                taps.extend(new_taps)
                # Сначала объединяем близкие
                taps = self.tap_merger.merge_taps(taps)
                # Потом фильтруем от контуров с ДИАМЕТРОМ пера (как указано в задаче)
                if remaining_contours:
                    taps = self.tap_merger.filter_taps_near_contours(
                        taps, remaining_contours, min_distance=self.pen_radius * 2.0
                    )
                # И наконец убираем взаимно близкие тапы
                taps = self.tap_merger.filter_taps_mutual_distance(
                    taps, min_distance=self.pen_radius * 2.0
                )
                # И снова оптимизируем порядок
                if taps:
                    taps = self.tap_optimizer.optimize(taps, start=(0.0, 0.0))

        print(f"  Final: {len(remaining_contours)} contours, {len(taps)} taps")
        return remaining_contours, taps

    # ------------------------------- этапы цвета -------------------------------

    def _extract_initial_taps(self, contours: List[ProcessedContour]) -> Tuple[List[Tap], List[ProcessedContour]]:
        """Мелкие контуры -> тапы; крупные — по желанию упрощаем."""
        taps: List[Tap] = []
        remaining: List[ProcessedContour] = []
        for contour in contours:
            if self.analyzer.should_be_tap(contour):
                taps.append(self.analyzer.contour_to_tap(contour))
            else:
                if self.enable_simplification and len(contour.points) > 10:
                    simplified = self.simplifier.simplify_contour(contour.points, method="combined")
                    remaining.append(ProcessedContour(
                        points=simplified,
                        color_idx=contour.color_idx,
                        is_closed=contour.is_closed,
                        fixed_points=contour.fixed_points,
                    ))
                else:
                    remaining.append(contour)
        return taps, remaining

    def _optimize_drawing_order(self, contours: List[ProcessedContour]) -> List[ProcessedContour]:
        """Простая NNS-жадность: минимизируем холостой ход пера."""
        if len(contours) <= 1:
            return contours

        remaining = list(contours)
        ordered: List[ProcessedContour] = []
        current_pos = np.array([0.0, 0.0])  # стартовая позиция пера

        while remaining:
            best_idx = -1
            best_dist = float('inf')
            best_reverse = False

            for idx, cont in enumerate(remaining):
                pts = np.asarray(cont.points, dtype=float)
                if len(pts) == 0:
                    continue

                # расстояние до начала
                dist_start = np.linalg.norm(pts[0] - current_pos)
                if dist_start < best_dist:
                    best_dist = dist_start
                    best_idx = idx
                    best_reverse = False

                # расстояние до конца (для реверса)
                dist_end = np.linalg.norm(pts[-1] - current_pos)
                if dist_end < best_dist:
                    best_dist = dist_end
                    best_idx = idx
                    best_reverse = True

            if best_idx == -1:
                break

            cont = remaining.pop(best_idx)
            if best_reverse:
                cont.points = cont.points[::-1]

            ordered.append(cont)
            pts = np.asarray(cont.points, dtype=float)
            current_pos = pts[-1]

        return ordered

    def _virtual_drawing_pass(self, contours: List[ProcessedContour], color_idx: int) -> Tuple[List[ProcessedContour], List[Tap], bool]:
        """Проход виртуальной отрисовки: фиксируем конфликты и разрешаем их."""
        state = VirtualDrawingState()
        new_contours: List[ProcessedContour] = []
        new_taps: List[Tap] = []
        changes_made = False

        # ----- ускорение: пакетная пересборка KD-дерева + проверка «хвоста» -----
        tree: Optional[cKDTree] = None
        last_tree_size = 0  # сколько точек учтено в текущем KD-дереве

        # Порог пересборки выбираем так, чтобы все «неучтённые» точки всегда
        # попадали в хвостовую проверку (без потери корректности).
        rebuild_threshold = min(self.kdtree_rebuild_every, self.kdtree_tail_window)

        def ensure_tree():
            nonlocal tree, last_tree_size
            if not state.drawn_points:
                return
            need_rebuild = (tree is None) or (len(state.drawn_points) - last_tree_size >= rebuild_threshold)
            if need_rebuild:
                tree = cKDTree(state.drawn_points)
                last_tree_size = len(state.drawn_points)
        # -----------------------------------------------------------------------

        for cont_idx, contour in enumerate(contours):
            new_points = []
            prev_pos = None

            for pt_idx, point in enumerate(contour.points):
                pos = tuple(point)
                if (cont_idx, pt_idx) in contour.fixed_points:
                    state.fixed_points.add((cont_idx, pt_idx))
                    new_points.append(pos)
                    state.drawn_points.append(pos)
                    state.drawn_contour_ids.append(cont_idx)
                    prev_pos = pos
                    continue

                # Проверяем конфликт
                if not state.drawn_points:
                    new_points.append(pos)
                    state.drawn_points.append(pos)
                    state.drawn_contour_ids.append(cont_idx)
                    prev_pos = pos
                    continue

                # KD-дерево на "старых" точках
                ensure_tree()
                dist = float('inf')
                nearest_idx = -1
                if tree is not None:
                    d, ni = tree.query(pos)
                    dist = float(d)
                    nearest_idx = int(ni)

                # Быстрая проверка для «свежих» точек, добавленных после последней пересборки
                new_since = len(state.drawn_points) - last_tree_size
                if new_since > 0:
                    tail_pts = np.asarray(state.drawn_points[-new_since:], dtype=float)
                    dif = tail_pts - np.array(pos, dtype=float)
                    d2 = np.sum(dif * dif, axis=1)
                    j = int(np.argmin(d2))
                    d_tail = float(np.sqrt(d2[j]))
                    if d_tail < dist:
                        dist = d_tail
                        nearest_idx = len(state.drawn_points) - new_since + j

                if dist <= self.pen_radius and state.drawn_contour_ids[nearest_idx] != cont_idx:
                    changes_made = True
                    drawn_pos = state.drawn_points[nearest_idx]

                    # Если точка почти совпадает с уже проведённой линией (очень близко),
                    # удаляем её вместо попытки сдвига, чтобы не рисовать «параллельный дубль».
                    if dist <= self.pen_radius * 0.8:
                        state.removed_points.add((cont_idx, pt_idx))
                        continue

                    # Пытаемся сдвинуть
                    vector = np.array(pos) - np.array(drawn_pos)
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        shift_dir = vector / norm
                        new_pos = np.array(drawn_pos) + shift_dir * self.pen_radius
                        shift_dist = np.linalg.norm(new_pos - np.array(pos))
                        if shift_dist <= self.max_shift:
                            state.point_shifts[(cont_idx, pt_idx)] = tuple(new_pos)
                            new_points.append(tuple(new_pos))
                            state.drawn_points.append(tuple(new_pos))
                            state.drawn_contour_ids.append(cont_idx)
                            prev_pos = tuple(new_pos)
                            continue

                    # Если не сдвинуть - удаляем точку
                    state.removed_points.add((cont_idx, pt_idx))
                    continue

                new_points.append(pos)
                state.drawn_points.append(pos)
                state.drawn_contour_ids.append(cont_idx)
                prev_pos = pos

            if len(new_points) >= 2:
                new_cont = ProcessedContour(
                    points=np.array(new_points),
                    color_idx=color_idx,
                    is_closed=contour.is_closed,
                    fixed_points=contour.fixed_points
                )
                new_contours.append(new_cont)
            elif len(new_points) == 1:
                new_taps.append(Tap(x=new_points[0][0], y=new_points[0][1]))

        return new_contours, new_taps, changes_made

    def _seg_dist(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
        """Минимальное расстояние между двумя отрезками [a,b] и [c,d]."""
        def _point_to_seg_dist(p, s0, s1):
            v = s1 - s0
            w = p - s0
            c1 = np.dot(w, v)
            if c1 <= 0:
                return np.linalg.norm(p - s0)
            c2 = np.dot(v, v)
            if c2 <= c1:
                return np.linalg.norm(p - s1)
            b = c1 / c2
            pb = s0 + b * v
            return np.linalg.norm(p - pb)

        def _seg_to_seg_dist(p1, p2, q1, q2):
            d1 = _point_to_seg_dist(p1, q1, q2)
            d2 = _point_to_seg_dist(p2, q1, q2)
            d3 = _point_to_seg_dist(q1, p1, p2)
            d4 = _point_to_seg_dist(q2, p1, p2)
            return min(d1, d2, d3, d4)

        return _seg_to_seg_dist(a, b, c, d)

    def _self_deduplicate_contours(self, contours: List[ProcessedContour]) -> Tuple[List[ProcessedContour], List[Tap]]:
        """
        Самодедупликация: режем контуры, удаляя части, слишком близкие к ранее нарисованным отрезкам ЭТОГО ЖЕ контура.
        Короткие фрагменты => тапы.
        """
        R = float(self.pen_radius)
        cleaned: List[ProcessedContour] = []
        new_taps: List[Tap] = []

        for contour in contours:
            pts = np.asarray(contour.points, dtype=float)
            if len(pts) < 3:
                cleaned.append(contour)
                continue

            # Сканируем, параллельно индексируя уже пройденные сегменты
            segments: List[Tuple[np.ndarray, np.ndarray]] = []
            grid: Dict[Tuple[int, int], List[int]] = {}
            cell = max(R, 1.0)

            def _cells_for_seg(a: np.ndarray, b: np.ndarray) -> List[Tuple[int, int]]:
                x0, y0 = min(a[0], b[0]) - R, min(a[1], b[1]) - R
                x1, y1 = max(a[0], b[0]) + R, max(a[1], b[1]) + R
                ix0, iy0 = int(np.floor(x0 / cell)), int(np.floor(y0 / cell))
                ix1, iy1 = int(np.floor(x1 / cell)), int(np.floor(y1 / cell))
                out = []
                for ix in range(ix0, ix1 + 1):
                    for iy in range(iy0, iy1 + 1):
                        out.append((ix, iy))
                return out

            def _add_seg(a: np.ndarray, b: np.ndarray, idx: int):
                segments.append((a, b))
                for c in _cells_for_seg(a, b):
                    grid.setdefault(c, []).append(idx)

            def _candidates(a: np.ndarray, b: np.ndarray) -> List[int]:
                cand = set()
                for c in _cells_for_seg(a, b):
                    cand.update(grid.get(c, []))
                return list(cand)

            # собираем фрагменты с защитой от самоперекрытий
            fragments: List[np.ndarray] = []
            start = 0
            # сразу индексируем первый сегмент
            _add_seg(pts[0], pts[1], 0)

            for i in range(2, len(pts)):
                a = pts[i - 1]
                b = pts[i]
                # ищем пересечения с ранее добавленными сегментами (кроме соседних)
                hit = False
                for si in _candidates(a, b):
                    if si >= i - self.self_dedup_min_gap:  # игнор соседей
                        continue
                    s0, s1 = segments[si]
                    if self._seg_dist(a, b, s0, s1) <= R:
                        # обрываем фрагмент до точки a (i-1)
                        frag = pts[start:i - 1].copy()
                        if len(frag) >= 2:
                            seg_contour = ProcessedContour(points=frag,
                                                           color_idx=contour.color_idx,
                                                           is_closed=False,
                                                           fixed_points=set())
                            if self.analyzer.should_be_tap(seg_contour):
                                new_taps.append(self.analyzer.contour_to_tap(seg_contour))
                            else:
                                fragments.append(frag)
                        start = i  # новый фрагмент начнётся с b

                        # пропускаем точки, пока не выйдем из зоны конфликта
                        while i < len(pts) - 1:
                            next_a = pts[i]
                            next_b = pts[i + 1]
                            next_hit = False
                            for next_si in _candidates(next_a, next_b):
                                if next_si >= i - self.self_dedup_min_gap + 1:
                                    continue
                                next_s0, next_s1 = segments[next_si]
                                if self._seg_dist(next_a, next_b, next_s0, next_s1) <= R:
                                    next_hit = True
                                    break
                            if not next_hit:
                                break
                            i += 1

                        hit = True
                        break
                if not hit:
                    _add_seg(a, b, len(segments))

            # хвост
            if start < len(pts):
                frag = pts[start:].copy()
                if len(frag) >= 2:
                    seg_contour = ProcessedContour(points=frag,
                                                   color_idx=contour.color_idx,
                                                   is_closed=False,
                                                   fixed_points=set())
                    if self.analyzer.should_be_tap(seg_contour):
                        new_taps.append(self.analyzer.contour_to_tap(seg_contour))
                    else:
                        fragments.append(frag)

            if fragments:
                for f in fragments:
                    cleaned.append(ProcessedContour(points=f,
                                                    color_idx=contour.color_idx,
                                                    is_closed=False,
                                                    fixed_points=set()))
            else:
                # ничего не вырезалось
                cleaned.append(contour)

        if new_taps:
            print(f"  [self-dedup] produced {len(new_taps)} new taps")
        print(f"  [self-dedup] {len(contours)} -> {len(cleaned)} contours")
        return cleaned, new_taps


class ContourPathOptimizer:
    """
    Гридный (жадный) оптимизатор порядка отрисовки:
    выбирает следующий контур и его направление (прямой/реверс),
    минимизируя холостой пробег от текущего конца пера.
    Ничего не объединяет и не трогает тапы — только порядок и реверс.
    """

    def __init__(self, reverse_allowed: bool = True):
        self.reverse_allowed = bool(reverse_allowed)

    # Унифицировать доступ к точкам контура (ProcessedContour или numpy)
    def _pts(self, contour) -> np.ndarray:
        if hasattr(contour, "points"):
            pts = np.asarray(contour.points, dtype=np.float32)
        else:
            pts = np.asarray(contour, dtype=np.float32)
        if pts.ndim == 3 and pts.shape[1] == 1:
            # форматы OpenCV (N,1,2)
            pts = pts.reshape(-1, 2)
        return pts

    def optimize_path(self, contours: List, taps: List) -> Tuple[List, List]:
        """
        Возвращает (упорядоченные_контуры, тапы_как_есть).
        Нулевой старт (0,0) — ровно как ожидает пайплайн, никакой геометрии не меняем.
        """
        if not contours:
            return contours, taps

        remaining = list(contours)
        ordered: List = []
        cx, cy = 0.0, 0.0  # стартовая позиция пера

        total = len(remaining)
        printed = 0

        while remaining:
            # прогресс в stderr печатать тут не нужно — им занимается вызывающая сторона
            best_i = 0
            best_rev = False
            best_d2 = float("inf")

            for i, c in enumerate(remaining):
                pts = self._pts(c)
                if pts.size == 0:
                    continue
                # расстояние до начала и до конца
                dx0 = pts[0, 0] - cx
                dy0 = pts[0, 1] - cy
                d0 = dx0 * dx0 + dy0 * dy0

                if d0 < best_d2:
                    best_d2 = d0
                    best_i = i
                    best_rev = False

                if self.reverse_allowed:
                    dx1 = pts[-1, 0] - cx
                    dy1 = pts[-1, 1] - cy
                    d1 = dx1 * dx1 + dy1 * dy1
                    if d1 < best_d2:
                        best_d2 = d1
                        best_i = i
                        best_rev = True

            c = remaining.pop(best_i)
            if best_rev:
                if hasattr(c, "points"):
                    c.points = c.points[::-1]
                else:
                    c = c[::-1]

            ordered.append(c)
            pts = self._pts(c)
            if pts.size:
                cx, cy = float(pts[-1, 0]), float(pts[-1, 1])

        return ordered, taps
