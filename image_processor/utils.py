import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import pickle

def load_config():
    """Загрузка конфигурации из временного файла или создание новой"""
    import os
    import pickle
    from config import Config
    
    if os.path.exists('config_temp.pkl'):
        with open('config_temp.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return Config()

def visualize_contours(contours: List[np.ndarray], 
                       image_size: Tuple[int, int],
                       title: str = "Contours") -> np.ndarray:
    """Визуализация контуров на изображении"""
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    img.fill(255)  # белый фон
    
    # Случайные цвета для каждого контура
    colors = [
        (np.random.randint(0, 255), 
         np.random.randint(0, 255), 
         np.random.randint(0, 255))
        for _ in range(len(contours))
    ]
    
    for contour, color in zip(contours, colors):
        cv2.drawContours(img, [contour], -1, color, 2)
    
    return img

def load_contours(filepath: str) -> List[np.ndarray]:
    """Загрузка контуров из pickle файла"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_contours(contours: List[np.ndarray], filepath: str):
    """Сохранение контуров в pickle файл"""
    with open(filepath, 'wb') as f:
        pickle.dump(contours, f)

def combine_layers_to_image(config):
    """Объединение всех слоев в одно изображение для визуализации"""
    import cv2
    
    # Получаем размер из масштабированного изображения
    img = cv2.imread(f"{config.output_dir}/resized.png")
    height, width = img.shape[:2]
    
    # Создаем пустое изображение
    combined = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Цвета для визуализации (RGB)
    viz_colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0)
    }
    
    # Рисуем контуры каждого слоя
    for color_name in config.color_names:
        try:
            contours = load_contours(f"{config.output_dir}/{color_name}/contours_final.pkl")
            color = viz_colors.get(color_name, (128, 128, 128))
            
            # Масштабируем контуры обратно для визуализации
            scale_back = width / (config.target_width_mm * config.pixels_per_mm)
            for contour in contours:
                scaled = (contour * scale_back).astype(np.int32)
                cv2.drawContours(combined, [scaled], -1, color, 1)
        except FileNotFoundError:
            print(f"Контуры для {color_name} не найдены")
    
    cv2.imwrite(f"{config.output_dir}/combined_result.png", combined)
    return combined

def analyze_results(config):
    """Анализ результатов пайплайна"""
    stats = {}
    
    for color_name in config.color_names:
        stats[color_name] = {}
        
        # Подсчет контуров на каждом этапе
        stages = [
            ('contours.pkl', 'initial'),
            ('contours_sorted.pkl', 'sorted'),
            ('contours_dedup_layer.pkl', 'dedup_layer'),
            ('contours_dedup_cross.pkl', 'dedup_cross'),
            ('contours_simplified.pkl', 'simplified'),
            ('contours_final.pkl', 'final')
        ]
        
        for filename, stage in stages:
            try:
                filepath = f"{config.output_dir}/{color_name}/{filename}"
                contours = load_contours(filepath)
                stats[color_name][stage] = len(contours)
                
                # Подсчет общего количества точек
                total_points = sum(len(c) for c in contours)
                stats[color_name][f'{stage}_points'] = total_points
                
            except FileNotFoundError:
                stats[color_name][stage] = 'N/A'
    
    # Вывод статистики
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПАЙПЛАЙНА")
    print("=" * 60)
    
    for color_name, color_stats in stats.items():
        print(f"\n{color_name.upper()}:")
        for stage, count in color_stats.items():
            if not stage.endswith('_points'):
                points = color_stats.get(f'{stage}_points', 'N/A')
                print(f"  {stage:15} : {count:5} контуров, {points:6} точек")
    
    return stats

def export_combined_svg(config):
    """Экспорт всех слоев в один SVG файл"""
    target_width = int(config.target_width_mm * config.pixels_per_mm)
    target_height = int(config.target_height_mm * config.pixels_per_mm)
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{target_width}" height="{target_height}" xmlns="http://www.w3.org/2000/svg">
'''
    
    for i, color_name in enumerate(config.color_names):
        try:
            contours = load_contours(f"{config.output_dir}/{color_name}/contours_final.pkl")
            
            # Конвертация BGR в RGB
            bgr = config.colors[i]
            rgb = (bgr[2], bgr[1], bgr[0])
            
            svg_content += f'  <g id="{color_name}" fill="none" stroke="rgb{rgb}" stroke-width="1" opacity="0.8">\n'
            
            for contour in contours:
                path_data = "M "
                for point in contour:
                    path_data += f"{point[0][0]},{point[0][1]} "
                path_data += "Z"
                svg_content += f'    <path d="{path_data}"/>\n'
            
            svg_content += '  </g>\n'
            
        except FileNotFoundError:
            print(f"Контуры для {color_name} не найдены")
    
    svg_content += '</svg>'
    
    with open(f"{config.output_dir}/combined_output.svg", 'w') as f:
        f.write(svg_content)
    
    print(f"Комбинированный SVG сохранен: {config.output_dir}/combined_output.svg")

if __name__ == "__main__":
    # Пример использования утилит
    from config import Config
    
    config = Config()
    
    # Анализ результатов
    stats = analyze_results(config)
    
    # Создание комбинированного изображения
    combined = combine_layers_to_image(config)
    
    # Экспорт в единый SVG
    export_combined_svg(config)