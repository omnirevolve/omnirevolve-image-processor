#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль дедупликации между разными цветами.
Обрабатывает от темных к светлым, удаляя конфликты.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from scipy.spatial import cKDTree
import copy

from contour_deduplicator import (
    Tap, ProcessedContour, TapMerger, ContourAnalyzer
)


@dataclass
class ColorLayer:
    """Слой одного цвета с контурами и тапами"""
    color_idx: int
    color_name: str
    luminance: float  # Яркость для сортировки
    contours: List[ProcessedContour]
    taps: List[Tap]
    
    def total_elements(self) -> int:
        return len(self.contours) + len(self.taps)


class CrossColorDeduplicator:
    """Дедупликатор между цветами"""
    
    def __init__(self, pen_radius: float = 30.0):
        self.pen_radius = pen_radius
        self.analyzer = ContourAnalyzer(pen_radius * 2)
        self.tap_merger = TapMerger(pen_radius)
    
    def process_all_colors(self, color_layers: List[ColorLayer]) -> List[ColorLayer]:
        """
        Обрабатывает все цветовые слои, удаляя конфликты между цветами.
        
        Args:
            color_layers: Список слоев, отсортированных по яркости (темные первые)
            
        Returns:
            Обработанные слои без межцветовых конфликтов
        """
        if len(color_layers) <= 1:
            return color_layers
        
        print("\n=== Cross-color deduplication ===")
        
        # Сортируем по яркости (темные первые)
        color_layers.sort(key=lambda x: x.luminance)
        
        processed_layers = []
        accumulated_elements = []  # Все элементы обработанных слоев
        
        for i, layer in enumerate(color_layers):
            print(f"\nProcessing layer {i+1}/{len(color_layers)}: {layer.color_name} "
                  f"(luminance={layer.luminance:.1f})")
            
            if i == 0:
                # Первый (самый темный) слой не изменяется
                processed_layers.append(layer)
                accumulated_elements.extend(self._get_all_points_from_layer(layer))
                print(f"  Base layer: {layer.total_elements()} elements")
            else:
                # Обрабатываем относительно всех предыдущих слоев
                processed_layer = self._process_layer_against_darker(
                    layer, accumulated_elements
                )
                processed_layers.append(processed_layer)
                
                # Добавляем обработанные элементы к накопленным
                accumulated_elements.extend(self._get_all_points_from_layer(processed_layer))
                
                print(f"  After dedup: {processed_layer.total_elements()} elements "
                      f"(was {layer.total_elements()})")
        
        return processed_layers
    
    def _process_layer_against_darker(self, layer: ColorLayer, 
                                     darker_points: List[Tuple[float, float]]) -> ColorLayer:
        """
        Обрабатывает светлый слой относительно более темных.
        
        Args:
            layer: Обрабатываемый (светлый) слой
            darker_points: Все точки более темных слоев
            
        Returns:
            Обработанный слой
        """
        if not darker_points:
            return layer
        
        # Строим пространственный индекс темных точек
        tree = cKDTree(np.array(darker_points))
        
        # Обрабатываем контуры
        processed_contours = []
        new_taps = []
        
        for contour in layer.contours:
            processed, contour_taps = self._process_contour_against_darker(
                contour, tree
            )
            processed_contours.extend(processed)
            new_taps.extend(contour_taps)
        
        # Объединяем тапы
        all_taps = layer.taps + new_taps
        
        # Фильтруем тапы относительно темных элементов
        filtered_taps = self._filter_taps_against_darker(all_taps, tree)
        
        # Схлопываем оставшиеся тапы
        if filtered_taps:
            filtered_taps = self.tap_merger.merge_taps(filtered_taps)
        
        return ColorLayer(
            color_idx=layer.color_idx,
            color_name=layer.color_name,
            luminance=layer.luminance,
            contours=processed_contours,
            taps=filtered_taps
        )
    
    def _process_contour_against_darker(self, contour: ProcessedContour,
                                       darker_tree: cKDTree) -> Tuple[List[ProcessedContour], List[Tap]]:
        """
        Обрабатывает контур, удаляя точки близкие к темным элементам.
        
        Args:
            contour: Обрабатываемый контур
            darker_tree: KD-дерево темных точек
            
        Returns:
            (список новых контуров, список новых тапов)
        """
        # Находим точки для удаления
        removed_indices = set()
        
        for i, point in enumerate(contour.points):
            # Проверяем расстояние до ближайшей темной точки
            dist, _ = darker_tree.query(point, k=1)
            
            if dist < self.pen_radius:
                removed_indices.add(i)
        
        if not removed_indices:
            # Контур не изменился
            return [contour], []
        
        # Разбиваем контур на сегменты
        segments = self.analyzer.split_contour_at_removed_points(contour, removed_indices)
        
        # Классифицируем сегменты
        final_contours = []
        new_taps = []
        
        for segment in segments:
            if self.analyzer.should_be_tap(segment):
                tap = self.analyzer.contour_to_tap(segment)
                new_taps.append(tap)
            else:
                final_contours.append(segment)
        
        return final_contours, new_taps
    
    def _filter_taps_against_darker(self, taps: List[Tap], 
                                   darker_tree: cKDTree) -> List[Tap]:
        """
        Удаляет тапы, находящиеся близко к темным элементам.
        
        Args:
            taps: Список тапов для фильтрации
            darker_tree: KD-дерево темных точек
            
        Returns:
            Отфильтрованный список тапов
        """
        if not taps:
            return []
        
        filtered = []
        removed_count = 0
        
        for tap in taps:
            dist, _ = darker_tree.query([tap.x, tap.y], k=1)
            
            if dist >= self.pen_radius:
                filtered.append(tap)
            else:
                removed_count += 1
        
        if removed_count > 0:
            print(f"    Removed {removed_count} taps near darker elements")
        
        return filtered
    
    def _get_all_points_from_layer(self, layer: ColorLayer) -> List[Tuple[float, float]]:
        """
        Извлекает все точки из слоя (контуры + тапы).
        
        Args:
            layer: Цветовой слой
            
        Returns:
            Список всех точек
        """
        points = []
        
        # Точки из контуров
        for contour in layer.contours:
            points.extend([tuple(p) for p in contour.points])
        
        # Точки из тапов
        for tap in layer.taps:
            points.append((tap.x, tap.y))
        
        return points


class LuminanceCalculator:
    """Вычислитель яркости цветов"""
    
    @staticmethod
    def rgb_to_luminance(r: int, g: int, b: int) -> float:
        """
        Вычисляет яркость по формуле ITU-R BT.709.
        
        Args:
            r, g, b: Компоненты цвета (0-255)
            
        Returns:
            Яркость (0-255)
        """
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    @staticmethod
    def calculate_opacity_factor(color_name: str) -> float:
        """
        Возвращает коэффициент непрозрачности для цвета.
        Некоторые цвета лучше перекрывают другие независимо от яркости.
        
        Args:
            color_name: Название цвета
            
        Returns:
            Коэффициент непрозрачности (0-1)
        """
        opacity_map = {
            'black': 1.0,
            'dark_blue': 0.95,
            'brown': 0.9,
            'dark_green': 0.9,
            'blue': 0.85,
            'green': 0.8,
            'red': 0.8,
            'purple': 0.75,
            'orange': 0.7,
            'pink': 0.6,
            'yellow': 0.5,
            'light_blue': 0.4,
            'light_green': 0.4,
            'gray': 0.7,
            'light_gray': 0.5,
        }
        
        # Поиск по частичному совпадению
        for key, value in opacity_map.items():
            if key in color_name.lower():
                return value
        
        return 0.7  # Значение по умолчанию
    
    @classmethod
    def calculate_effective_darkness(cls, color_rgb: Tuple[int, int, int], 
                                    color_name: str) -> float:
        """
        Вычисляет эффективную "темноту" с учетом яркости и непрозрачности.
        
        Args:
            color_rgb: RGB компоненты цвета
            color_name: Название цвета
            
        Returns:
            Эффективная темнота (чем меньше, тем темнее)
        """
        luminance = cls.rgb_to_luminance(*color_rgb)
        opacity = cls.calculate_opacity_factor(color_name)
        
        # Комбинируем яркость и непрозрачность
        # Меньшее значение = более темный/непрозрачный
        return luminance * (2.0 - opacity)


def sort_colors_by_darkness(color_info: List[Dict]) -> List[Dict]:
    """
    Сортирует цвета по эффективной темноте.
    
    Args:
        color_info: Список словарей с информацией о цветах
                   [{'name': str, 'rgb': [r,g,b], 'index': int}, ...]
    
    Returns:
        Отсортированный список (темные первые)
    """
    calculator = LuminanceCalculator()
    
    sorted_colors = sorted(
        color_info,
        key=lambda c: calculator.calculate_effective_darkness(
            tuple(c['rgb']), 
            c['name']
        )
    )
    
    return sorted_colors