#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль упрощения контуров для устранения зигзагообразных артефактов.
Использует несколько методов: Douglas-Peucker, скользящее среднее и сплайны.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter


class ContourSimplifier:
    """Упрощение контуров различными методами"""
    
    def __init__(self, epsilon_ratio: float = 0.002, 
                 min_segment_length: int = 5,
                 smoothing_window: int = 5):
        """
        Args:
            epsilon_ratio: Коэффициент для Douglas-Peucker (доля от периметра)
            min_segment_length: Минимальная длина сегмента после упрощения
            smoothing_window: Размер окна для сглаживания
        """
        self.epsilon_ratio = epsilon_ratio
        self.min_segment_length = min_segment_length
        self.smoothing_window = smoothing_window
    
    def simplify_contour(self, contour: np.ndarray, 
                        method: str = 'combined',
                        preserve_ends: bool = True) -> np.ndarray:
        """
        Упрощает контур выбранным методом.
        
        Args:
            contour: Массив точек контура shape (n, 2)
            method: 'douglas_peucker', 'smooth', 'spline', 'combined'
            preserve_ends: Сохранять ли начальную и конечную точки
            
        Returns:
            Упрощенный контур
        """
        if len(contour) < 3:
            return contour
        
        if method == 'douglas_peucker':
            return self._douglas_peucker(contour)
        elif method == 'smooth':
            return self._smooth_contour(contour, preserve_ends)
        elif method == 'spline':
            return self._spline_simplify(contour, preserve_ends)
        elif method == 'combined':
            # Комбинированный подход: сначала DP, потом сглаживание
            simplified = self._douglas_peucker(contour)
            if len(simplified) > self.smoothing_window:
                simplified = self._smooth_contour(simplified, preserve_ends)
            return simplified
        else:
            return contour
    
    def _douglas_peucker(self, contour: np.ndarray) -> np.ndarray:
        """Алгоритм Douglas-Peucker для упрощения"""
        # Вычисляем epsilon на основе периметра контура
        perimeter = cv2.arcLength(contour.astype(np.float32), closed=False)
        epsilon = self.epsilon_ratio * perimeter
        
        # Применяем упрощение
        simplified = cv2.approxPolyDP(
            contour.astype(np.float32), 
            epsilon, 
            closed=False
        )
        
        # Преобразуем обратно в нужный формат
        result = simplified.reshape(-1, 2)
        
        # Убеждаемся, что не потеряли слишком много точек
        if len(result) < self.min_segment_length and len(contour) >= self.min_segment_length:
            # Если упростили слишком сильно, уменьшаем epsilon
            epsilon = epsilon * 0.5
            simplified = cv2.approxPolyDP(
                contour.astype(np.float32), 
                epsilon, 
                closed=False
            )
            result = simplified.reshape(-1, 2)
        
        return result
    
    def _smooth_contour(self, contour: np.ndarray, 
                       preserve_ends: bool = True) -> np.ndarray:
        """Сглаживание контура скользящим средним"""
        if len(contour) < self.smoothing_window:
            return contour
        
        x = contour[:, 0].astype(np.float64)
        y = contour[:, 1].astype(np.float64)
        
        # Применяем фильтр Савицкого-Голея для сглаживания
        window = min(self.smoothing_window, len(x))
        if window % 2 == 0:
            window -= 1  # Должно быть нечетным
        
        if window >= 3 and len(x) > window:
            polyorder = min(3, window - 1)
            x_smooth = savgol_filter(x, window, polyorder)
            y_smooth = savgol_filter(y, window, polyorder)
            
            if preserve_ends:
                # Сохраняем исходные концы
                x_smooth[0] = x[0]
                x_smooth[-1] = x[-1]
                y_smooth[0] = y[0]
                y_smooth[-1] = y[-1]
            
            return np.column_stack([x_smooth, y_smooth]).astype(np.float32)
        
        return contour
    
    def _spline_simplify(self, contour: np.ndarray, 
                        preserve_ends: bool = True) -> np.ndarray:
        """Упрощение через B-сплайны"""
        if len(contour) < 4:
            return contour
        
        # Параметрическое представление контура
        x = contour[:, 0]
        y = contour[:, 1]
        
        # Количество контрольных точек (меньше исходных)
        n_control = max(4, len(contour) // 5)
        
        try:
            # Создаем сплайн
            tck, u = splprep([x, y], s=len(contour)*0.1, k=3)
            
            # Генерируем новые точки
            u_new = np.linspace(0, 1, n_control)
            x_new, y_new = splev(u_new, tck)
            
            result = np.column_stack([x_new, y_new]).astype(np.float32)
            
            if preserve_ends:
                result[0] = contour[0]
                result[-1] = contour[-1]
            
            return result
        except:
            # Если сплайн не удался, возвращаем сглаженный контур
            return self._smooth_contour(contour, preserve_ends)
    
    def remove_zigzags(self, contour: np.ndarray, 
                      angle_threshold: float = 150) -> np.ndarray:
        """
        Удаляет зигзагообразные участки на основе анализа углов.
        
        Args:
            contour: Контур для обработки
            angle_threshold: Порог угла в градусах (зигзаг если угол > порога)
            
        Returns:
            Контур без резких зигзагов
        """
        if len(contour) < 3:
            return contour
        
        filtered_points = [contour[0]]
        
        for i in range(1, len(contour) - 1):
            # Векторы к предыдущей и следующей точке
            v1 = contour[i] - contour[i-1]
            v2 = contour[i+1] - contour[i]
            
            # Вычисляем угол между векторами
            angle = self._calculate_angle(v1, v2)
            
            # Если угол не слишком острый (не зигзаг), сохраняем точку
            if angle < angle_threshold:
                filtered_points.append(contour[i])
            # Иначе пропускаем точку (спрямляем)
        
        filtered_points.append(contour[-1])
        
        return np.array(filtered_points, dtype=np.float32)
    
    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Вычисляет угол между двумя векторами в градусах"""
        # Нормализуем векторы
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        
        # Вычисляем угол через скалярное произведение
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def adaptive_simplify(self, contour: np.ndarray, 
                         target_reduction: float = 0.5) -> np.ndarray:
        """
        Адаптивное упрощение для достижения целевого уровня редукции точек.
        
        Args:
            contour: Исходный контур
            target_reduction: Целевая доля оставшихся точек (0.5 = убрать 50%)
            
        Returns:
            Упрощенный контур
        """
        original_length = len(contour)
        target_length = int(original_length * target_reduction)
        target_length = max(target_length, self.min_segment_length)
        
        # Пробуем разные значения epsilon
        epsilon_low = 0.0001
        epsilon_high = 0.1
        
        for _ in range(10):  # Бинарный поиск
            epsilon_mid = (epsilon_low + epsilon_high) / 2
            self.epsilon_ratio = epsilon_mid
            
            simplified = self._douglas_peucker(contour)
            
            if len(simplified) > target_length:
                epsilon_low = epsilon_mid
            else:
                epsilon_high = epsilon_mid
            
            # Достаточно близко к цели
            if abs(len(simplified) - target_length) < 5:
                break
        
        return simplified


def process_layer_with_simplification(layer_path: str, 
                                     output_path: str,
                                     simplifier: ContourSimplifier) -> int:
    """
    Обрабатывает слой с упрощением контуров.
    
    Args:
        layer_path: Путь к изображению слоя
        output_path: Путь для сохранения результата
        simplifier: Объект упрощения
        
    Returns:
        Количество обработанных контуров
    """
    # Загружаем изображение
    img = cv2.imread(layer_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {layer_path}")
    
    # Извлекаем контуры
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Создаем новое изображение
    h, w = img.shape
    result = np.ones((h, w), dtype=np.uint8) * 255
    
    simplified_contours = []
    
    for contour in contours:
        if contour is None or len(contour) < 3:
            continue
        
        # Преобразуем в нужный формат
        points = contour.reshape(-1, 2)
        
        # Упрощаем контур
        simplified = simplifier.simplify_contour(points, method='combined')
        
        # Конвертируем обратно для OpenCV
        cv_contour = simplified.reshape(-1, 1, 2).astype(np.int32)
        simplified_contours.append(cv_contour)
    
    # Рисуем упрощенные контуры
    cv2.drawContours(result, simplified_contours, -1, 0, 1)
    
    # Сохраняем результат
    cv2.imwrite(output_path, result)
    
    return len(simplified_contours)


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplify contours to remove zigzag artifacts")
    parser.add_argument("input", help="Input image or directory with edge layers")
    parser.add_argument("-o", "--output", help="Output image or directory")
    parser.add_argument("--epsilon", type=float, default=0.002,
                       help="Douglas-Peucker epsilon ratio (default: 0.002)")
    parser.add_argument("--smooth-window", type=int, default=5,
                       help="Smoothing window size (default: 5)")
    parser.add_argument("--method", default="combined",
                       choices=["douglas_peucker", "smooth", "spline", "combined"],
                       help="Simplification method")
    
    args = parser.parse_args()
    
    from pathlib import Path
    
    simplifier = ContourSimplifier(
        epsilon_ratio=args.epsilon,
        smoothing_window=args.smooth_window
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Обработка одного файла
        output_path = args.output or str(input_path.with_suffix('.simplified.png'))
        count = process_layer_with_simplification(str(input_path), output_path, simplifier)
        print(f"Processed {count} contours")
        print(f"Saved to: {output_path}")
    else:
        # Обработка директории
        output_dir = Path(args.output or str(input_path) + "_simplified")
        output_dir.mkdir(exist_ok=True)
        
        for layer_file in sorted(input_path.glob("edges_layer_*.png")):
            output_file = output_dir / layer_file.name
            count = process_layer_with_simplification(
                str(layer_file), 
                str(output_file), 
                simplifier
            )
            print(f"{layer_file.name}: {count} contours -> {output_file.name}")


if __name__ == "__main__":
    main()