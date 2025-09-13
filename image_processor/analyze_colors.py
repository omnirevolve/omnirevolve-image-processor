#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ изображения и рекомендация оптимальных цветов из палитры Carioca.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

from color_palette import CariocaPalette

class ColorAnalyzer:
   """Анализатор цветового состава изображения"""
   
   def __init__(self, palette: CariocaPalette):
       self.palette = palette
       self.dominant_colors = []
       self.color_histogram = None
       self.recommendations = []
       
   def analyze(self, img_path: str, 
               n_clusters: int = 8,
               ignore_white: bool = True,
               white_threshold: int = 240) -> Dict:
       """
       Анализирует цветовой состав изображения.
       
       Args:
           img_path: Путь к изображению
           n_clusters: Количество кластеров для анализа
           ignore_white: Игнорировать белый/светлый фон
           white_threshold: Порог для определения белого
           
       Returns:
           Словарь с результатами анализа
       """
       # Загружаем изображение
       img = cv2.imread(img_path)
       if img is None:
           raise ValueError(f"Cannot load image: {img_path}")
       
       img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       h, w = img_rgb.shape[:2]
       
       # Подготавливаем пиксели
       pixels = img_rgb.reshape(-1, 3)
       
       # Фильтруем белый фон если нужно
       if ignore_white:
           # Маска для не-белых пикселей
           non_white_mask = np.any(pixels < white_threshold, axis=1)
           filtered_pixels = pixels[non_white_mask]
           
           if len(filtered_pixels) < 100:  # Если почти всё белое
               filtered_pixels = pixels
               print(f"Warning: Image is mostly white, using all pixels")
       else:
           filtered_pixels = pixels
       
       # Сэмплируем для ускорения если изображение большое
       if len(filtered_pixels) > 50000:
           indices = np.random.choice(len(filtered_pixels), 50000, replace=False)
           filtered_pixels = filtered_pixels[indices]
       
       # Кластеризация для поиска доминирующих цветов
       print(f"Clustering {len(filtered_pixels)} pixels into {n_clusters} groups...")
       kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
       labels = kmeans.fit_predict(filtered_pixels)
       
       # Центры кластеров - доминирующие цвета
       cluster_centers = kmeans.cluster_centers_.astype(int)
       
       # Размеры кластеров (важность цветов)
       cluster_sizes = np.bincount(labels)
       cluster_percentages = cluster_sizes / len(labels) * 100
       
       # Сортируем по важности
       sorted_indices = np.argsort(-cluster_sizes)
       
       self.dominant_colors = []
       for idx in sorted_indices:
           color_rgb = tuple(cluster_centers[idx])
           percentage = cluster_percentages[idx]
           
           # Находим ближайший цвет из палитры
           closest = self.palette.find_closest(color_rgb, n=1)[0]
           
           self.dominant_colors.append({
               'rgb': color_rgb,
               'percentage': percentage,
               'closest_palette': closest[0],
               'distance': closest[1]
           })
       
       # Строим гистограмму по оттенкам
       self._build_hue_histogram(filtered_pixels)
       
       # Результаты анализа
       results = {
           'image_size': (w, h),
           'total_pixels': len(pixels),
           'analyzed_pixels': len(filtered_pixels),
           'dominant_colors': self.dominant_colors,
           'hue_distribution': self.color_histogram
       }
       
       return results
   
   def _build_hue_histogram(self, pixels: np.ndarray):
       """Строит гистограмму распределения оттенков"""
       # Конвертируем в HSV для анализа оттенков
       pixels_rgb = pixels.reshape(-1, 1, 3).astype(np.uint8)
       pixels_hsv = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2HSV)
       
       hues = pixels_hsv[:, 0, 0]
       saturations = pixels_hsv[:, 0, 1]
       values = pixels_hsv[:, 0, 2]
       
       # Категоризируем цвета
       color_categories = {
           'red': 0, 'orange': 0, 'yellow': 0, 'green': 0,
           'cyan': 0, 'blue': 0, 'purple': 0, 'pink': 0,
           'brown': 0, 'gray': 0, 'black': 0
       }
       
       for h, s, v in zip(hues, saturations, values):
           # Определяем категорию
           if v < 50:  # Очень темный
               color_categories['black'] += 1
           elif s < 30:  # Низкая насыщенность
               if v < 128:
                   color_categories['gray'] += 1
               else:
                   color_categories['gray'] += 1
           else:
               # По оттенку (OpenCV использует H/2)
               h_full = h * 2
               if h_full < 15 or h_full >= 345:
                   color_categories['red'] += 1
               elif h_full < 25:
                   if s > 150 and v < 150:
                       color_categories['brown'] += 1
                   else:
                       color_categories['orange'] += 1
               elif h_full < 45:
                   color_categories['orange'] += 1
               elif h_full < 75:
                   color_categories['yellow'] += 1
               elif h_full < 150:
                   color_categories['green'] += 1
               elif h_full < 200:
                   color_categories['cyan'] += 1
               elif h_full < 270:
                   color_categories['blue'] += 1
               elif h_full < 330:
                   if s < 100:
                       color_categories['pink'] += 1
                   else:
                       color_categories['purple'] += 1
               else:
                   color_categories['pink'] += 1
       
       # Нормализуем в проценты
       total = sum(color_categories.values())
       if total > 0:
           self.color_histogram = {k: v/total*100 for k, v in color_categories.items()}
       else:
           self.color_histogram = color_categories
   
   def recommend_colors(self, n_colors: int = 4, 
                        coverage_boost: bool = True) -> List[Tuple[str, float]]:
       """
       Рекомендует оптимальный набор цветов.
       
       Args:
           n_colors: Количество цветов для рекомендации
           coverage_boost: Учитывать покрытие соседних цветов
           
       Returns:
           Список [(имя_цвета, покрытие_%)]
       """
       if not self.dominant_colors:
           raise ValueError("Run analyze() first")
       
       # Собираем все упоминания цветов палитры с весами
       color_scores = {}
       
       for dom_color in self.dominant_colors[:min(len(self.dominant_colors), 12)]:
           palette_color = dom_color['closest_palette']
           weight = dom_color['percentage']
           
           # Уменьшаем вес для далеких цветов
           if dom_color['distance'] > 50:
               weight *= 0.5
           
           if palette_color not in color_scores:
               color_scores[palette_color] = 0
           color_scores[palette_color] += weight
       
       # Если включен boost покрытия, добавляем похожие цвета
       if coverage_boost:
           boosted_scores = {}
           for color_name, score in color_scores.items():
               # Находим группу похожих цветов
               group = self.palette.get_color_group(color_name, tolerance=40)
               
               # Распределяем счет по группе
               for similar_color in group:
                   if similar_color not in boosted_scores:
                       boosted_scores[similar_color] = 0
                   # Основной цвет получает полный вес, похожие - частичный
                   if similar_color == color_name:
                       boosted_scores[similar_color] += score
                   else:
                       boosted_scores[similar_color] += score * 0.3
           
           color_scores = boosted_scores
       
       # Сортируем по счету
       sorted_colors = sorted(color_scores.items(), key=lambda x: -x[1])
       
       # Выбираем топ N, избегая слишком похожих
       selected = []
       selected_labs = []
       
       for color_name, score in sorted_colors:
           if len(selected) >= n_colors:
               break
           
           # Проверяем, не слишком ли похож на уже выбранные
           color_rgb = self.palette.colors[color_name]
           color_rgb_array = np.array([[color_rgb]], dtype=np.uint8)
           color_lab = cv2.cvtColor(color_rgb_array, cv2.COLOR_RGB2LAB).reshape(3)
           
           too_similar = False
           for lab in selected_labs:
               dist = np.linalg.norm(color_lab - lab)
               if dist < 30:  # Слишком похожи
                   too_similar = True
                   break
           
           if not too_similar:
               selected.append((color_name, score))
               selected_labs.append(color_lab)
       
       # Сортируем по яркости (светлые первыми для порядка рисования)
       selected.sort(key=lambda x: -np.mean(self.palette.colors[x[0]]))
       
       self.recommendations = selected
       return selected
   
   def visualize_analysis(self, save_path: str = None):
       """Визуализирует результаты анализа"""
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       # 1. Доминирующие цвета
       ax = axes[0, 0]
       colors = [d['rgb'] for d in self.dominant_colors[:8]]
       percentages = [d['percentage'] for d in self.dominant_colors[:8]]
       
       # Нормализуем RGB для matplotlib
       colors_norm = [(r/255, g/255, b/255) for r, g, b in colors]
       
       bars = ax.bar(range(len(colors)), percentages, color=colors_norm)
       ax.set_xlabel('Dominant Colors')
       ax.set_ylabel('Percentage (%)')
       ax.set_title('Dominant Colors in Image')
       ax.set_xticks(range(len(colors)))
       ax.set_xticklabels([f"C{i+1}" for i in range(len(colors))])
       
       # 2. Распределение оттенков
       ax = axes[0, 1]
       if self.color_histogram:
           categories = list(self.color_histogram.keys())
           values = list(self.color_histogram.values())
           
           # Сортируем по значению
           sorted_pairs = sorted(zip(categories, values), key=lambda x: -x[1])
           categories, values = zip(*sorted_pairs[:8])
           
           ax.pie(values, labels=categories, autopct='%1.1f%%')
           ax.set_title('Hue Distribution')
       
       # 3. Палитра соответствий
       ax = axes[1, 0]
       y_pos = 0
       for i, dom_color in enumerate(self.dominant_colors[:6]):
           # Исходный цвет
           orig_color = np.array(dom_color['rgb']).reshape(1, 1, 3)
           ax.imshow(orig_color, extent=[0, 1, y_pos, y_pos+0.8])
           
           # Ближайший из палитры
           palette_name = dom_color['closest_palette']
           palette_rgb = np.array(self.palette.colors[palette_name]).reshape(1, 1, 3)
           ax.imshow(palette_rgb, extent=[1.2, 2.2, y_pos, y_pos+0.8])
           
           # Подпись
           ax.text(2.4, y_pos+0.4, f"{palette_name}\n(dist: {dom_color['distance']:.1f})",
                  va='center', fontsize=9)
           
           y_pos += 1
       
       ax.set_xlim(-0.1, 4)
       ax.set_ylim(-0.5, y_pos)
       ax.set_title('Original → Palette Mapping')
       ax.set_xticks([0.5, 1.7])
       ax.set_xticklabels(['Original', 'Palette'])
       ax.set_yticks([])
       
       # 4. Рекомендации
       ax = axes[1, 1]
       if self.recommendations:
           rec_names = [r[0] for r in self.recommendations]
           rec_scores = [r[1] for r in self.recommendations]
           rec_colors = [self.palette.colors[name] for name in rec_names]
           rec_colors_norm = [(r/255, g/255, b/255) for r, g, b in rec_colors]
           
           bars = ax.barh(range(len(rec_names)), rec_scores, color=rec_colors_norm)
           ax.set_yticks(range(len(rec_names)))
           ax.set_yticklabels(rec_names)
           ax.set_xlabel('Coverage Score')
           ax.set_title(f'Recommended {len(rec_names)} Colors')
           ax.invert_yaxis()
       
       plt.suptitle('Color Analysis Results', fontsize=14, fontweight='bold')
       plt.tight_layout()
       
       if save_path:
           plt.savefig(save_path, dpi=150, bbox_inches='tight')
           print(f"Visualization saved to {save_path}")
       else:
           plt.show()

def main():
   parser = argparse.ArgumentParser(description="Analyze image colors and recommend Carioca markers")
   parser.add_argument("input", help="Input image path")
   parser.add_argument("-n", "--num-colors", type=int, default=4,
                      help="Number of colors to recommend (default: 4)")
   parser.add_argument("-c", "--clusters", type=int, default=8,
                      help="Number of color clusters for analysis (default: 8)")
   parser.add_argument("--palette", help="Custom palette JSON file")
   parser.add_argument("--no-boost", action="store_true",
                      help="Disable coverage boosting for similar colors")
   parser.add_argument("-v", "--visualize", action="store_true",
                      help="Show visualization")
   parser.add_argument("-o", "--output", help="Save visualization to file")
   parser.add_argument("--show-all", action="store_true",
                      help="Show all dominant colors with palette matches")
   
   args = parser.parse_args()
   
   # Загружаем палитру
   if args.palette:
       palette = CariocaPalette.load(args.palette)
       print(f"Loaded custom palette from {args.palette}")
   else:
       palette = CariocaPalette()
       print(f"Using default Carioca palette ({len(palette.colors)} colors)")
   
   # Анализируем изображение
   analyzer = ColorAnalyzer(palette)
   results = analyzer.analyze(args.input, n_clusters=args.clusters)
   
   print(f"\n{'='*50}")
   print(f"Image: {args.input}")
   print(f"Size: {results['image_size'][0]}x{results['image_size'][1]}")
   print(f"Analyzed pixels: {results['analyzed_pixels']:,}")
   
   # Показываем доминирующие цвета
   print(f"\n{'='*50}")
   print("Dominant colors:")
   for i, color in enumerate(results['dominant_colors'][:8], 1):
       print(f"  {i}. RGB{color['rgb']} ({color['percentage']:.1f}%)")
       print(f"     → {color['closest_palette']} (distance: {color['distance']:.1f})")
   
   # Рекомендации
   recommendations = analyzer.recommend_colors(
       n_colors=args.num_colors,
       coverage_boost=not args.no_boost
   )
   
   print(f"\n{'='*50}")
   print(f"RECOMMENDED MARKERS ({args.num_colors}):")
   print("Order: light → dark (for clean overlapping)")
   print("")
   
   for i, (color_name, score) in enumerate(recommendations, 1):
       rgb = palette.colors[color_name]
       print(f"  Position {i}: {color_name}")
       print(f"    RGB: {rgb}")
       print(f"    Coverage score: {score:.1f}")
       
       # Показываем похожие цвета для расширенного покрытия
       if not args.no_boost:
           group = palette.get_color_group(color_name, tolerance=35)
           if len(group) > 1:
               others = [c for c in group if c != color_name]
               print(f"    Also covers: {', '.join(others[:3])}")
       print()
   
   # Сохраняем рекомендации
   output_json = Path(args.input).stem + "_colors.json"
   recommendations_data = {
       'image': args.input,
       'recommended_colors': [
           {
               'position': i,
               'name': name,
               'rgb': list(palette.colors[name]),
               'coverage': score
           }
           for i, (name, score) in enumerate(recommendations, 1)
       ]
   }
   
   with open(output_json, 'w') as f:
       json.dump(recommendations_data, f, indent=2)
   print(f"Recommendations saved to {output_json}")
   
   # Визуализация
   if args.visualize or args.output:
       analyzer.visualize_analysis(args.output)

if __name__ == "__main__":
   main()