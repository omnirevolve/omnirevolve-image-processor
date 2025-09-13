#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Палитра фломастеров Carioca и утилиты для работы с цветом.
Поддерживает как предустановленные цвета, так и загрузку из фото.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import json
from pathlib import Path

class CariocaPalette:
    """Палитра 24 стандартных цветов Carioca"""
    
    # Аппроксимация стандартных цветов Carioca Joy 24
    # Можно откалибровать по фото реальных фломастеров
    DEFAULT_COLORS = {
        # Базовые
        'black': (30, 30, 30),
        'brown': (101, 67, 33),
        'dark_gray': (128, 128, 128),
        'light_gray': (192, 192, 192),
        
        # Красные оттенки
        'red': (237, 28, 36),
        'dark_red': (157, 18, 23),
        'pink': (255, 174, 201),
        'magenta': (236, 0, 140),
        
        # Оранжевые/желтые
        'orange': (255, 127, 39),
        'light_orange': (255, 201, 14),
        'yellow': (255, 242, 0),
        'light_yellow': (255, 255, 153),
        
        # Зеленые
        'green': (34, 177, 76),
        'dark_green': (0, 100, 0),
        'light_green': (181, 230, 29),
        'lime': (153, 255, 153),
        
        # Синие
        'blue': (0, 162, 232),
        'dark_blue': (0, 71, 187),
        'light_blue': (153, 217, 234),
        'cyan': (0, 255, 255),
        
        # Фиолетовые
        'purple': (163, 73, 164),
        'violet': (127, 0, 255),
        
        # Телесные/натуральные
        'peach': (255, 218, 185),
        'skin': (255, 206, 180),
    }
    
    def __init__(self, colors_dict: Optional[Dict[str, Tuple[int, int, int]]] = None):
        """
        Args:
            colors_dict: Словарь {имя: (R,G,B)} или None для стандартной палитры
        """
        self.colors = colors_dict if colors_dict else self.DEFAULT_COLORS.copy()
        self._update_lab()
    
    def _update_lab(self):
        """Преобразует RGB в LAB для точного сравнения цветов"""
        self.colors_rgb = np.array(list(self.colors.values()), dtype=np.uint8)
        self.colors_names = list(self.colors.keys())
        
        # RGB -> LAB для перцептивно корректного сравнения
        rgb_reshaped = self.colors_rgb.reshape(1, -1, 3)
        self.colors_lab = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    @classmethod
    def from_photo(cls, photo_path: str, n_colors: int = 24) -> 'CariocaPalette':
        """
        Создает палитру из фото линий фломастеров.
        
        Args:
            photo_path: Путь к фото с образцами цветов
            n_colors: Количество цветов для извлечения
        """
        img = cv2.imread(photo_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Извлекаем доминирующие цвета через KMeans
        pixels = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        # Генерируем имена по оттенкам
        colors_dict = {}
        for i, color in enumerate(colors):
            hue = cls._get_hue_name(color)
            brightness = cls._get_brightness_name(color)
            name = f"{brightness}_{hue}_{i}" if brightness else f"{hue}_{i}"
            colors_dict[name] = tuple(color)
        
        return cls(colors_dict)
    
    @staticmethod
    def _get_hue_name(rgb: np.ndarray) -> str:
        """Определяет базовый оттенок цвета"""
        r, g, b = rgb
        
        if max(rgb) - min(rgb) < 30:  # Серый
            return 'gray'
        
        hsv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hue = hsv[0] * 2  # OpenCV использует H/2
        
        if hue < 15 or hue >= 345:
            return 'red'
        elif hue < 45:
            return 'orange'
        elif hue < 75:
            return 'yellow'
        elif hue < 150:
            return 'green'
        elif hue < 210:
            return 'cyan'
        elif hue < 270:
            return 'blue'
        elif hue < 330:
            return 'purple'
        else:
            return 'red'
    
    @staticmethod
    def _get_brightness_name(rgb: np.ndarray) -> str:
        """Определяет яркость цвета"""
        brightness = np.mean(rgb)
        if brightness < 80:
            return 'dark'
        elif brightness > 200:
            return 'light'
        return ''
    
    def find_closest(self, rgb: Tuple[int, int, int], n: int = 1) -> List[Tuple[str, float]]:
        """
        Находит ближайшие цвета из палитры.
        
        Args:
            rgb: Целевой цвет (R, G, B)
            n: Количество ближайших цветов
            
        Returns:
            Список [(имя_цвета, расстояние)]
        """
        target_rgb = np.array([[rgb]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        
        # Расстояния в LAB пространстве
        distances = cdist(target_lab, self.colors_lab, metric='euclidean')[0]
        
        # Индексы отсортированные по расстоянию
        indices = np.argsort(distances)[:n]
        
        result = []
        for idx in indices:
            name = self.colors_names[idx]
            dist = distances[idx]
            result.append((name, dist))
        
        return result
    
    def get_color_group(self, color_name: str, tolerance: float = 30.0) -> List[str]:
        """
        Находит группу похожих цветов для расширения покрытия.
        
        Args:
            color_name: Имя базового цвета
            tolerance: Максимальное LAB расстояние для включения в группу
            
        Returns:
            Список имен похожих цветов
        """
        if color_name not in self.colors:
            return []
        
        base_idx = self.colors_names.index(color_name)
        base_lab = self.colors_lab[base_idx:base_idx+1]
        
        distances = cdist(base_lab, self.colors_lab, metric='euclidean')[0]
        
        group = []
        for i, dist in enumerate(distances):
            if dist <= tolerance:
                group.append(self.colors_names[i])
        
        return group
    
    def recommend_set(self, img_path: str, n_colors: int = 4) -> List[str]:
        """
        Рекомендует оптимальный набор цветов для изображения.
        
        Args:
            img_path: Путь к изображению
            n_colors: Количество цветов для рекомендации
            
        Returns:
            Список рекомендованных цветов из палитры
        """
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Извлекаем доминирующие цвета изображения
        pixels = img_rgb.reshape(-1, 3)
        
        # Фильтруем белый/почти белый фон
        mask = np.any(pixels < 250, axis=1)
        pixels_filtered = pixels[mask]
        
        if len(pixels_filtered) < 100:
            pixels_filtered = pixels
        
        # Кластеризация
        kmeans = KMeans(n_clusters=min(n_colors*2, 8), random_state=42, n_init=10)
        kmeans.fit(pixels_filtered)
        
        # Взвешиваем кластеры по размеру
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        dominant_clusters = kmeans.cluster_centers_[np.argsort(-cluster_sizes)[:n_colors]]
        
        # Подбираем ближайшие цвета из палитры
        recommended = []
        used_indices = set()
        
        for cluster_center in dominant_clusters:
            rgb = tuple(cluster_center.astype(int))
            candidates = self.find_closest(rgb, n=len(self.colors))
            
            # Выбираем первый неиспользованный
            for name, dist in candidates:
                idx = self.colors_names.index(name)
                if idx not in used_indices:
                    recommended.append(name)
                    used_indices.add(idx)
                    break
        
        # Сортируем по яркости (светлые первыми)
        recommended.sort(key=lambda name: -np.mean(self.colors[name]))
        
        return recommended[:n_colors]
    
    def save(self, path: str):
        """Сохраняет палитру в JSON"""
        data = {
            'colors': {name: list(rgb) for name, rgb in self.colors.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CariocaPalette':
        """Загружает палитру из JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        colors = {name: tuple(rgb) for name, rgb in data['colors'].items()}
        return cls(colors)

# Сервисные байты для смены цвета
SERVICE_COLOR_0 = 0x04  # 0b00000100
SERVICE_COLOR_1 = 0x08  # 0b00001000
SERVICE_COLOR_2 = 0x0C  # 0b00001100
SERVICE_COLOR_3 = 0x10  # 0b00010000

def get_color_service_byte(color_index: int) -> int:
    """Возвращает сервисный байт для установки цвета 0-3"""
    if color_index == 0:
        return SERVICE_COLOR_0
    elif color_index == 1:
        return SERVICE_COLOR_1
    elif color_index == 2:
        return SERVICE_COLOR_2
    elif color_index == 3:
        return SERVICE_COLOR_3
    else:
        raise ValueError(f"Invalid color index: {color_index}")

if __name__ == "__main__":
    # Тест палитры
    palette = CariocaPalette()
    print(f"Loaded {len(palette.colors)} colors")
    
    # Пример поиска ближайшего цвета
    target = (200, 100, 50)
    closest = palette.find_closest(target, n=3)
    print(f"\nClosest to RGB{target}:")
    for name, dist in closest:
        print(f"  {name}: {palette.colors[name]} (dist: {dist:.1f})")