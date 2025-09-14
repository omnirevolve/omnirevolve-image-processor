import cv2
import numpy as np
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from config import Config

def load_config() -> Config:
    """
    Load configuration from JSON pointed to by PIPELINE_CONFIG env var;
    fallback to ./config.json if the variable is not set or file is missing.
    """
    import os, json
    cfg = Config()
    path = os.environ.get("PIPELINE_CONFIG", "config.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg
            
def extract_color_mask(img: np.ndarray, 
                       target_color: Tuple[int, int, int], 
                       tolerance: int) -> np.ndarray:
    """Извлекает маску для заданного цвета с допуском"""
    lower = np.array([max(0, c - tolerance) for c in target_color])
    upper = np.array([min(255, c + tolerance) for c in target_color])
    mask = cv2.inRange(img, lower, upper)
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def process_color(args):
    """Обработка одного цвета для параллельного выполнения"""
    img, color, color_name, tolerance, output_dir = args
    mask = extract_color_mask(img, color, tolerance)
    output_path = f"{output_dir}/{color_name}/mask.png"
    cv2.imwrite(output_path, mask)
    return color_name, mask

def extract_all_colors(img: np.ndarray, config: Config) -> dict:
    """Параллельное извлечение всех цветовых каналов"""
    masks = {}
    
    # Подготовка аргументов для параллельной обработки
    args_list = [
        (img, config.colors[i], config.color_names[i], 
         config.color_tolerance, config.output_dir)
        for i in range(len(config.colors))
    ]
    
    with ProcessPoolExecutor(max_workers=min(4, config.n_cores)) as executor:
        results = executor.map(process_color, args_list)
        for color_name, mask in results:
            masks[color_name] = mask
            print(f"Извлечен канал: {color_name}")
    
    return masks

from config import load_config

if __name__ == "__main__":
    config = load_config()
    img = cv2.imread(f"{config.output_dir}/resized.png")
    if img is None:
        raise ValueError(f"Failed to load resized image: {config.output_dir}/resized.png")
    masks = extract_all_colors(img, config)