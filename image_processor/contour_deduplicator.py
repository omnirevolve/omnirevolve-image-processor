#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# :contentReference[oaicite:0]{index=0}
"""
Модуль дедупликации контуров для плоттера с фломастерами.
Учитывает толщину пера и предотвращает повторные проходы.
"""

import cv2
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import json
from pathlib import Path

__all__ = [
    "Tap",
    "ProcessedContour",
    "TapMerger",
    "ContourAnalyzer",
    "load_contours_from_layer",
    "save_processing_stats",
    "TapOptimizer",
]


@dataclass
class Tap:
    """Точечный элемент (тап)"""
    x: float
    y: float
    weight: float = 1.0  # Вес на основе исходной площади контура
    color_idx: int = 0
    
    def distance_to(self, other: 'Tap') -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ProcessedContour:
    """Обработанный контур с метаданными"""
    points: np.ndarray  # Shape: (n_points, 2)
    color_idx: int
    is_closed: bool = False
    fixed_points: Set[int] = None  # Индексы "мертвых" точек
    
    def __post_init__(self):
        if self.fixed_points is None:
            self.fixed_points = set()
    
    def area(self) -> float:
        """Вычисляет площадь контура"""
        return float(cv2.contourArea(self.points.astype(np.float32)))
    
    def perimeter(self) -> float:
        """Вычисляет периметр контура"""
        return float(cv2.arcLength(self.points.astype(np.float32), self.is_closed))
    
    def bounding_box(self) -> Tuple[float, float]:
        """Возвращает размеры ограничивающего прямоугольника"""
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        width = float(np.max(x_coords) - np.min(x_coords))
        height = float(np.max(y_coords) - np.min(y_coords))
        return width, height


class TapMerger:
    """Класс для схлопывания близких тапов"""
    
    def __init__(self, merge_radius: float = 30.0):
        self.merge_radius = float(merge_radius)
    
    def merge_taps(self, taps: List[Tap]) -> List[Tap]:
        """
        Итеративно схлопывает близкие тапы в центры масс групп.
        
        Args:
            taps: Список тапов для обработки
            
        Returns:
            Список объединенных тапов
        """
        if len(taps) <= 1:
            return taps
        
        changed = True
        iteration = 0
        
        while changed and iteration < 100:  # Защита от бесконечного цикла
            changed = False
            iteration += 1
            
            # Строим пространственный индекс
            tap_coords = np.array([(t.x, t.y) for t in taps], dtype=np.float32)
            if tap_coords.size == 0:
                break
                
            tree = cKDTree(tap_coords)
            
            # Находим группы близких тапов
            groups = []
            used = set()
            
            for i, tap in enumerate(taps):
                if i in used:
                    continue
                
                # Находим всех соседей в радиусе
                neighbors = tree.query_ball_point([tap.x, tap.y], self.merge_radius)
                
                if len(neighbors) > 1:
                    group = [j for j in neighbors if j not in used]
                    if len(group) > 1:
                        groups.append(group)
                        used.update(group)
                        changed = True
                elif i not in used:
                    groups.append([i])
                    used.add(i)
            
            if not changed:
                break
            
            # Создаем новые тапы из групп
            new_taps = []
            for group in groups:
                if len(group) == 1:
                    new_taps.append(taps[group[0]])
                else:
                    # Взвешенный центр масс
                    total_weight = sum(taps[i].weight for i in group)
                    cx = sum(taps[i].x * taps[i].weight for i in group) / total_weight
                    cy = sum(taps[i].y * taps[i].weight for i in group) / total_weight
                    
                    # Берем цвет от самого тяжелого тапа
                    heaviest = max(group, key=lambda i: taps[i].weight)
                    color_idx = taps[heaviest].color_idx
                    
                    new_taps.append(Tap(cx, cy, total_weight, color_idx))
            
            taps = new_taps
            
            print(f"  Tap merge iteration {iteration}: {len(taps)} taps remaining")
        
        return taps
    
    def filter_taps_near_contours(self, taps: List[Tap], 
                                  contours: List[ProcessedContour],
                                  min_distance: float = None) -> List[Tap]:
        """
        Удаляет тапы, если в окружности РАДИУСОМ = D пера (то есть диаметром пера)
        от центра тапа есть ХОТЯ БЫ ОДНА ТОЧКА контура того же цвета.
        Проверка выполняется по ТОЧКАМ контура с KD-деревом.

        Args:
            taps: Список тапов
            contours: Список контуров ОДНОГО ЦВЕТА текущего слоя
            min_distance: Радиус проверки. Если не задан — используется
                          2 * merge_radius (то есть диаметр пера при
                          merge_radius = радиусу пера).
        Returns:
            Отфильтрованный список тапов
        """
        if not taps or not contours:
            return taps

        # По умолчанию — диаметр пера
        if min_distance is None:
            min_distance = 2.0 * self.merge_radius
        R = float(min_distance)

        # Собираем все точки контуров в слое
        clouds = []
        for contour in contours:
            if hasattr(contour, 'points'):
                pts = np.array(contour.points, dtype=np.float32)
            else:
                pts = np.array(contour, dtype=np.float32)
            if pts.ndim == 3 and pts.shape[1] == 1:
                pts = pts.reshape(-1, 2)
            if pts.size >= 2:
                clouds.append(pts)

        if not clouds:
            return taps

        all_pts = np.vstack(clouds)
        tree = cKDTree(all_pts)

        filtered: List[Tap] = []
        removed_count = 0

        for tap in taps:
            d, _ = tree.query([tap.x, tap.y], k=1)
            if float(d) <= R:
                removed_count += 1
            else:
                filtered.append(tap)

        if removed_count > 0:
            print(f"  Removed {removed_count} taps near contours (≤ {R:.1f}px to a contour POINT)")
        
        return filtered
    
    def filter_taps_mutual_distance(self, taps: List[Tap], min_distance: float = None) -> List[Tap]:
        """
        Удаляет тапы, которые слишком близко друг к другу.
        Приоритет отдается тапам с большим весом.
        
        Args:
            taps: Список тапов
            min_distance: Минимальное расстояние между тапами (по умолчанию = merge_radius)
            
        Returns:
            Отфильтрованный список тапов
        """
        if len(taps) <= 1:
            return taps
        
        if min_distance is None:
            min_distance = self.merge_radius
        
        # Сортируем тапы по весу (убывание) - более важные тапы имеют приоритет
        sorted_taps = sorted(taps, key=lambda t: t.weight, reverse=True)
        
        # Результирующий список
        filtered = []
        
        for tap in sorted_taps:
            # Проверяем расстояние до всех уже добавленных тапов
            too_close = False
            
            for existing in filtered:
                dist = float(np.hypot(tap.x - existing.x, tap.y - existing.y))
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(tap)
        
        removed_count = len(taps) - len(filtered)
        if removed_count > 0:
            print(f"  Removed {removed_count} redundant taps (mutual distance filter)")
        
        return filtered


class ContourAnalyzer:
    """Анализатор контуров для определения необходимости превращения в тапы"""
    
    def __init__(self, pen_diameter: int = 60):
        self.pen_diameter = int(pen_diameter)
        self.pen_radius = self.pen_diameter // 2
        self.min_area = float(np.pi * (self.pen_radius ** 2))  # Площадь круга с радиусом пера
    
    def should_be_tap(self, contour: ProcessedContour) -> bool:
        """
        Определяет, должен ли контур стать тапом.
        
        Args:
            contour: Контур для анализа
            
        Returns:
            True если контур должен стать тапом
        """
        # Проверка площади
        area = contour.area()
        if area > 0 and area < self.min_area:
            return True
        
        # Проверка размеров ограничивающего прямоугольника
        width, height = contour.bounding_box()
        if max(width, height) < self.pen_diameter:
            return True
        
        # Проверка периметра
        perimeter = contour.perimeter()
        if perimeter < self.pen_diameter:
            return True
        
        return False
    
    def contour_to_tap(self, contour: ProcessedContour) -> Tap:
        """
        Преобразует контур в тап.
        
        Args:
            contour: Контур для преобразования
            
        Returns:
            Тап в центре контура
        """
        # Находим центр масс контура
        M = cv2.moments(contour.points.astype(np.float32))
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            # Если моменты не вычисляются, берем среднее
            cx = float(np.mean(contour.points[:, 0]))
            cy = float(np.mean(contour.points[:, 1]))
        
        # Вес основан на площади
        weight = max(1.0, contour.area())
        
        return Tap(cx, cy, weight, contour.color_idx)
    
    def split_contour_at_removed_points(self, contour: ProcessedContour, 
                                       removed_indices: Set[int]) -> List[ProcessedContour]:
        """
        Разбивает контур на сегменты после удаления точек.
        
        Args:
            contour: Исходный контур
            removed_indices: Индексы удаленных точек
            
        Returns:
            Список новых контуров
        """
        if not removed_indices:
            return [contour]
        
        segments = []
        current_segment = []
        
        for i, point in enumerate(contour.points):
            if i not in removed_indices:
                current_segment.append(point)
            else:
                # Заканчиваем текущий сегмент
                if len(current_segment) >= 2:
                    new_contour = ProcessedContour(
                        points=np.array(current_segment),
                        color_idx=contour.color_idx,
                        is_closed=False
                    )
                    # Проверяем минимальную длину
                    if new_contour.perimeter() >= self.pen_radius * 2.0:
                        segments.append(new_contour)
                current_segment = []
        
        # Добавляем последний сегмент
        if len(current_segment) >= 2:
            new_contour = ProcessedContour(
                points=np.array(current_segment),
                color_idx=contour.color_idx,
                is_closed=False
            )
            if new_contour.perimeter() >= self.pen_radius * 2.0:
                segments.append(new_contour)
        
        return segments


def load_contours_from_layer(layer_path: str, color_idx: int, 
                            simplify: bool = True) -> List[ProcessedContour]:
    """
    Загружает контуры из файла слоя с опциональным упрощением.
    
    Args:
        layer_path: Путь к изображению слоя
        color_idx: Индекс цвета (0-3)
        simplify: Применять ли упрощение контуров
        
    Returns:
        Список контуров
    """
    img = cv2.imread(layer_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {layer_path}")
    
    # Извлекаем контуры
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Упрощаем контуры если нужно
    if simplify:
        try:
            from contour_simplifier import ContourSimplifier
            simplifier = ContourSimplifier(epsilon_ratio=0.001, smoothing_window=3)
        except ImportError:
            simplifier = None
            simplify = False
    
    processed = []
    for cnt in contours:
        if cnt is None or len(cnt) < 2:
            continue
        
        # Преобразуем в нужный формат
        points = cnt.reshape(-1, 2).astype(np.float32)
        
        # Упрощаем если включено и контур достаточно большой
        if simplify and simplifier and len(points) > 10:
            points = simplifier.simplify_contour(points, method='douglas_peucker')
        
        processed.append(ProcessedContour(
            points=points,
            color_idx=color_idx,
            is_closed=True
        ))
    
    return processed


def save_processing_stats(stats: Dict, output_path: str):
    """Сохраняет статистику обработки"""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {output_path}")


# -------------------------- Оптимизация порядка тапов --------------------------

class TapOptimizer:
    """
    Оптимизация порядка выполнения тапов без изменения другой логики.
    Алгоритм:
      1) строим начальный маршрут жадно (nearest-neighbor) от стартовой точки;
      2) улучшаем маршрут локально при помощи 2-opt (1–2 прохода).
    Использование: оптимизировать список тапов внутри одного слоя/цвета.
    """

    def __init__(self, two_opt_passes: int = 2):
        self.two_opt_passes = max(0, int(two_opt_passes))

    @staticmethod
    def _route_length(order: List[int], pts: np.ndarray, start: Tuple[float, float]) -> float:
        if not order:
            return 0.0
        x0, y0 = start
        length = float(np.hypot(pts[order[0], 0] - x0, pts[order[0], 1] - y0))
        for a, b in zip(order[:-1], order[1:]):
            length += float(np.hypot(pts[b, 0] - pts[a, 0], pts[b, 1] - pts[a, 1]))
        return length

    @staticmethod
    def _nearest_neighbor(pts: np.ndarray, start: Tuple[float, float]) -> List[int]:
        n = len(pts)
        if n == 0:
            return []
        unused = set(range(n))
        order: List[int] = []

        # старт — ближайшая точка к (x0,y0)
        x0, y0 = start
        d2 = (pts[:, 0] - x0) ** 2 + (pts[:, 1] - y0) ** 2
        i = int(np.argmin(d2))
        order.append(i)
        unused.remove(i)

        while unused:
            p = pts[order[-1]]
            cand = list(unused)
            sub = pts[cand] - p
            j = cand[int(np.argmin(np.sum(sub * sub, axis=1)))]
            order.append(j)
            unused.remove(j)

        return order

    @staticmethod
    def _two_opt_once(order: List[int], pts: np.ndarray, start: Tuple[float, float]) -> Tuple[List[int], bool]:
        """
        Один проход 2-opt. Возвращает (новый_порядок, было_улучшение).
        """
        if len(order) < 4:
            return order, False

        improved = False
        best = order
        n = len(order)

        # длины ребер из старта и между точками
        def seg_len(a, b):
            pa = pts[a]; pb = pts[b]
            return float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))

        def start_len(idx):
            p = pts[idx]
            return float(np.hypot(p[0] - start[0], p[1] - start[1]))

        best_len = TapOptimizer._route_length(best, pts, start)

        for i in range(n - 1):
            for k in range(i + 2, n):
                if i == 0:
                    old = start_len(best[i]) + seg_len(best[k - 1], best[k])
                    new = start_len(best[k - 1]) + seg_len(best[i], best[k])
                else:
                    old = seg_len(best[i - 1], best[i]) + seg_len(best[k - 1], best[k])
                    new = seg_len(best[i - 1], best[k - 1]) + seg_len(best[i], best[k])

                if new + 1e-9 < old:
                    candidate = best[:i] + best[i:k][::-1] + best[k:]
                    cand_len = TapOptimizer._route_length(candidate, pts, start)
                    if cand_len + 1e-9 < best_len:
                        best, best_len, improved = candidate, cand_len, True

        return best, improved

    def optimize(self, taps: List[Tap], start: Tuple[float, float] = (0.0, 0.0)) -> List[Tap]:
        """
        Возвращает переставленный список тапов в порядке минимального пробега пера.
        """
        if len(taps) <= 2:
            return taps

        pts = np.array([t.to_tuple() for t in taps], dtype=np.float64)

        # 1) начальный маршрут
        order = self._nearest_neighbor(pts, start)

        # 2) несколько проходов 2-opt
        for _ in range(self.two_opt_passes):
            order, improved = self._two_opt_once(order, pts, start)
            if not improved:
                break

        return [taps[i] for i in order]
