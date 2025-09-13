#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация композитного результата из векторных данных после дедупликации.
Работает напрямую с vector_manifest.json без извлечения контуров из растра.
"""

import cv2
import numpy as np
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

class CompositePreview:
    """Предпросмотр композитного изображения из векторных данных"""

    def __init__(self, pen_width_mm: float = 1.5, steps_per_mm: float = 40.0):
        """
        Args:
            pen_width_mm: Толщина пера в миллиметрах
            steps_per_mm: Шагов на миллиметр (разрешение плоттера)
        """
        self.pen_width_mm = pen_width_mm
        self.steps_per_mm = steps_per_mm
        self.pen_width_px = pen_width_mm * steps_per_mm

        self.layers = []  # Список векторных слоев
        self.colors = []  # RGB цвета
        self.color_names = []
        self.img_size = None

    def load_vector_data(self, manifest_path: str, color_info: Optional[Dict] = None) -> None:
        """
        Загружает векторные данные из манифеста.

        Args:
            manifest_path: Путь к vector_manifest.json
            color_info: Информация о цветах (из JSON analyze_colors)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Vector manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Размер изображения
        self.img_size = tuple(manifest.get('image_size', [800, 600]))
        base_dir = manifest_path.parent
        
        print(f"Loading vector data from {manifest_path}")
        print(f"Image size: {self.img_size[0]}x{self.img_size[1]}")
        
        self.layers = []
        self.colors = []
        self.color_names = []
        
        # Загружаем слои
        for layer_info in manifest['layers']:
            layer_file = base_dir / layer_info['file']
            
            if not layer_file.exists():
                print(f"Warning: Layer file not found: {layer_file}")
                continue
            
            # Загружаем pickle файл
            with open(layer_file, 'rb') as f:
                layer_data = pickle.load(f)
            
            # Добавляем слой
            self.layers.append(layer_data)
            self.color_names.append(layer_data['color_name'])
            
            # Определяем RGB цвет
            if color_info and 'recommended_colors' in color_info:
                color_rgb = None
                for rec_color in color_info['recommended_colors']:
                    if rec_color['name'] == layer_data['color_name']:
                        color_rgb = tuple(rec_color['rgb'])
                        break
                if not color_rgb:
                    # Генерируем цвет на основе индекса
                    hue = (layer_data['color_idx'] * 90) % 360
                    color_rgb = self._hsv_to_rgb(hue, 0.7, 0.7)
            else:
                # Генерируем цвет на основе индекса
                hue = (layer_data['color_idx'] * 90) % 360
                color_rgb = self._hsv_to_rgb(hue, 0.7, 0.7)
            
            self.colors.append(color_rgb)
            
            print(f"  Loaded {layer_data['color_name']}: "
                  f"{len(layer_data['contours'])} contours, {len(layer_data['taps'])} taps")
        
        if not self.layers:
            raise ValueError("No valid layers loaded")
        
        print(f"Loaded {len(self.layers)} layers")

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Конвертирует HSV в RGB"""
        hsv = np.array([[[h / 2, s * 255, v * 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        return tuple(rgb.tolist())

    def simulate_pen_stroke(self, img: np.ndarray, contours: List, taps: List,
                           pen_width_px: float) -> np.ndarray:
        """
        Рисует контуры и тапы с учетом толщины пера.
        """
        # Рисуем контуры
        for contour_data in contours:
            points = contour_data['points']
            if len(points) < 2:
                continue
            
            # Преобразуем в int для OpenCV
            points_int = points.astype(np.int32)
            
            # Рисуем линии с толщиной
            thickness = max(1, int(pen_width_px))
            for i in range(len(points_int) - 1):
                cv2.line(img, tuple(points_int[i]), tuple(points_int[i+1]), 255, thickness)
            
            # Если контур замкнутый
            if contour_data.get('is_closed', False) and len(points_int) > 2:
                cv2.line(img, tuple(points_int[-1]), tuple(points_int[0]), 255, thickness)
        
        # Рисуем тапы как круги
        for tap in taps:
            center = (int(tap['x']), int(tap['y']))
            radius = max(1, int(pen_width_px / 2))
            cv2.circle(img, center, radius, 255, -1)
        
        return img

    def create_composite(self,
                         simulate_pen: bool = True,
                         background: Tuple[int, int, int] = (255, 255, 255),
                         opacity: float = 0.8) -> np.ndarray:
        """
        Создает композитное изображение из векторных данных.
        """
        if not self.layers:
            raise ValueError("No layers loaded")

        h, w = self.img_size[1], self.img_size[0]  # Высота, ширина
        composite = np.full((h, w, 3), background, dtype=np.float32)

        # Обрабатываем слои в порядке рисования (светлые первыми)
        # Сортируем по luminance (светлые имеют большую luminance)
        sorted_layers = sorted(
            zip(self.layers, self.colors),
            key=lambda x: x[0].get('luminance', 0),
            reverse=True  # От светлых к темным
        )

        for layer_data, color in sorted_layers:
            # Создаем маску для этого слоя
            layer_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Рисуем контуры и тапы
            layer_mask = self.simulate_pen_stroke(
                layer_mask,
                layer_data['contours'],
                layer_data['taps'],
                self.pen_width_px if simulate_pen else 1
            )
            
            # Создаем маску (нормализованную)
            mask = (layer_mask > 127).astype(np.float32)
            
            # Расширяем маску до 3 каналов
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            
            # Создаем цветной слой
            color_layer = np.full((h, w, 3), color, dtype=np.float32)
            
            # Смешиваем с композитом с учетом прозрачности
            composite = composite * (1 - mask_3ch * opacity) + color_layer * mask_3ch * opacity

        # Конвертируем обратно в uint8
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        return composite

    def create_separated_view(self) -> np.ndarray:
        """Создает вид с раздельными слоями"""
        if not self.layers:
            raise ValueError("No layers loaded")

        h, w = self.img_size[1], self.img_size[0]
        n_layers = len(self.layers)

        # Создаем холст для всех слоев
        canvas_w = w * 2
        canvas_h = h * ((n_layers + 1) // 2)
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        for i, (layer_data, color, name) in enumerate(zip(self.layers, self.colors, self.color_names)):
            row = i // 2
            col = i % 2

            # Позиция на холсте
            y_start = row * h
            x_start = col * w

            # Создаем маску слоя
            layer_mask = np.zeros((h, w), dtype=np.uint8)
            layer_mask = self.simulate_pen_stroke(
                layer_mask,
                layer_data['contours'],
                layer_data['taps'],
                self.pen_width_px
            )

            # Создаем цветное изображение слоя
            layer_colored = np.full((h, w, 3), 255, dtype=np.uint8)
            mask = layer_mask > 127
            layer_colored[mask] = color

            # Помещаем на холст
            canvas[y_start:y_start + h, x_start:x_start + w] = layer_colored

            # Добавляем подпись
            cv2.putText(canvas, f"{i + 1}. {name}",
                        (x_start + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return canvas

    def estimate_drawing_stats(self) -> Dict:
        """Оценивает статистику рисования из векторных данных"""
        stats = {
            'total_contours': 0,
            'total_taps': 0,
            'total_length_mm': 0.0,
            'layers': []
        }

        for layer_data, name in zip(self.layers, self.color_names):
            contours_count = len(layer_data['contours'])
            taps_count = len(layer_data['taps'])
            
            # Подсчитываем длину контуров
            total_length_px = 0.0
            for contour_data in layer_data['contours']:
                points = contour_data['points']
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        dx = points[i+1][0] - points[i][0]
                        dy = points[i+1][1] - points[i][1]
                        total_length_px += np.sqrt(dx*dx + dy*dy)
            
            total_length_mm = total_length_px / self.steps_per_mm

            layer_stats = {
                'name': name,
                'contours': contours_count,
                'taps': taps_count,
                'length_mm': total_length_mm
            }

            stats['layers'].append(layer_stats)
            stats['total_contours'] += contours_count
            stats['total_taps'] += taps_count
            stats['total_length_mm'] += total_length_mm

        # Оценка времени
        drawing_speed_mm_s = 50.0  # мм/с
        stats['estimated_time_s'] = stats['total_length_mm'] / drawing_speed_mm_s
        stats['estimated_time_min'] = stats['estimated_time_s'] / 60.0

        return stats

    def visualize(self, save_path: Optional[str] = None):
        """Создает полную визуализацию"""
        composite = self.create_composite()
        separated = self.create_separated_view()
        stats = self.estimate_drawing_stats()

        fig = plt.figure(figsize=(16, 10))

        # Композитное изображение
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        ax1.set_title("Composite Result")
        ax1.axis('off')

        # Раздельные слои
        ax2 = plt.subplot(2, 3, (2, 3))
        ax2.imshow(cv2.cvtColor(separated, cv2.COLOR_BGR2RGB))
        ax2.set_title("Individual Layers")
        ax2.axis('off')

        # График длины по слоям
        ax3 = plt.subplot(2, 3, 4)
        layer_names = [s['name'] for s in stats['layers']]
        layer_lengths = [s['length_mm'] for s in stats['layers']]
        colors_rgb = [(r/255, g/255, b/255) for r, g, b in self.colors]

        bars = ax3.barh(range(len(layer_names)), layer_lengths, color=colors_rgb)
        ax3.set_yticks(range(len(layer_names)))
        ax3.set_yticklabels(layer_names)
        ax3.set_xlabel('Length (mm)')
        ax3.set_title('Drawing Length per Layer')
        ax3.invert_yaxis()
        for i, (bar, length) in enumerate(zip(bars, layer_lengths)):
            ax3.text(length, i, f' {length:.0f}mm', va='center')

        # Статистика по элементам
        ax4 = plt.subplot(2, 3, 5)
        elements_data = []
        elements_labels = []
        for s in stats['layers']:
            if s['contours'] > 0:
                elements_data.append(s['contours'])
                elements_labels.append(f"{s['name']} contours")
            if s['taps'] > 0:
                elements_data.append(s['taps'])
                elements_labels.append(f"{s['name']} taps")
        
        if elements_data:
            ax4.pie(elements_data, labels=elements_labels, autopct='%1.0f')
            ax4.set_title('Elements Distribution')
        else:
            ax4.text(0.5, 0.5, 'No elements', ha='center', va='center')
            ax4.set_title('Elements Distribution')

        # Информация
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')

        info_text = f"""Drawing Statistics:

Total contours: {stats['total_contours']}
Total taps: {stats['total_taps']}
Total length: {stats['total_length_mm']:.1f} mm
Estimated time: {stats['estimated_time_min']:.1f} min

Pen width: {self.pen_width_mm} mm
Resolution: {self.steps_per_mm} steps/mm

Layers: {len(self.layers)}
Drawing order: light → dark
"""
        ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes,
                 fontsize=11, verticalalignment='top', family='monospace')

        # Легенда цветов
        patches = [mpatches.Patch(color=(r/255, g/255, b/255), label=name)
                   for (r, g, b), name in zip(self.colors, self.color_names)]
        ax5.legend(handles=patches, loc='lower left', frameon=False)

        plt.suptitle('Plotter Preview - Vector Data Composite', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Preview composite result from vector data")
    parser.add_argument("manifest", help="Path to vector_manifest.json")
    parser.add_argument("-c", "--colors", help="Colors JSON file from analyze_colors.py")
    parser.add_argument("--pen-width", type=float, default=1.5, help="Pen width in mm")
    parser.add_argument("--steps-per-mm", type=float, default=40.0, help="Plotter resolution")
    parser.add_argument("-o", "--output", help="Save visualization to file")
    parser.add_argument("--no-pen-sim", action="store_true", help="Disable pen width simulation")
    parser.add_argument("--save-composite", help="Save composite image to file")
    parser.add_argument("--opacity", type=float, default=0.8, help="Layer opacity (0-1)")

    args = parser.parse_args()

    preview = CompositePreview(
        pen_width_mm=args.pen_width,
        steps_per_mm=args.steps_per_mm
    )

    # Загружаем информацию о цветах если есть
    color_info = None
    if args.colors and Path(args.colors).exists():
        with open(args.colors, 'r') as f:
            color_info = json.load(f)

    # Загружаем векторные данные
    preview.load_vector_data(args.manifest, color_info)

    # Создаем композит
    composite = preview.create_composite(
        simulate_pen=not args.no_pen_sim,
        opacity=args.opacity
    )

    # Сохраняем композит если нужно
    if args.save_composite:
        cv2.imwrite(args.save_composite, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        print(f"Composite saved to {args.save_composite}")

    # Выводим статистику
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)

    stats = preview.estimate_drawing_stats()
    print(f"\nTotal contours: {stats['total_contours']}")
    print(f"Total taps: {stats['total_taps']}")
    print(f"Total drawing length: {stats['total_length_mm']:.1f} mm")
    print(f"Estimated time: {stats['estimated_time_min']:.1f} minutes")

    print("\nPer layer:")
    for layer_stat in stats['layers']:
        print(f"  {layer_stat['name']:15s}: {layer_stat['length_mm']:7.1f} mm, "
              f"{layer_stat['contours']:4d} contours, {layer_stat['taps']:4d} taps")

    # Визуализация
    preview.visualize(args.output)

if __name__ == "__main__":
    main()