#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер конфигураций для пайплайна плоттера.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class WorkspaceConfig:
   """Конфигурация рабочей области"""
   width_steps: int = 8400   # A4: 210mm * 40 steps/mm
   height_steps: int = 11880  # A4: 297mm * 40 steps/mm
   steps_per_mm: float = 40.0
   origin_x: int = 0
   origin_y: int = 0

@dataclass
class PenConfig:
   """Конфигурация пера"""
   width_mm: float = 1.5
   lift_height_mm: float = 5.0
   down_speed_mm_s: float = 10.0
   up_speed_mm_s: float = 20.0

@dataclass
class EdgeDetectionProfile:
   """Профиль детекции границ"""
   blur: int = 5
   canny_low: int = 50
   canny_high: int = 150
   dilate: int = 0
   erode: int = 0
   min_contour_length: int = 20
   close_gaps: bool = True
   gap_size: int = 3

@dataclass
class ColorQuantizationConfig:
   """Конфигурация квантизации цветов"""
   method: str = "closest"  # closest, kmeans, adaptive
   merge_threshold: float = 35.0
   min_area_pixels: int = 10
   edges_only: bool = True

@dataclass
class DrawingConfig:
   """Конфигурация рисования"""
   bridge_px: int = 20
   max_bridge_passes: int = 8
   min_line_px: int = 5
   enable_taps: bool = False
   optimize_path: bool = True
   drawing_order: str = "light_to_dark"  # light_to_dark, dark_to_light, custom

@dataclass
class PlotterConfig:
   """Полная конфигурация плоттера"""
   workspace: WorkspaceConfig
   pen: PenConfig
   edge_detection: Dict[str, EdgeDetectionProfile]
   color_quantization: ColorQuantizationConfig
   drawing: DrawingConfig
   
   # Дополнительные параметры
   palette_file: Optional[str] = None
   output_directory: str = "output"
   preview_scale: float = 1.0

class ConfigManager:
   """Менеджер конфигураций"""
   
   DEFAULT_CONFIG = {
       "workspace": {
           "width_steps": 8400,
           "height_steps": 11880,
           "steps_per_mm": 40.0,
           "origin_x": 0,
           "origin_y": 0
       },
       "pen": {
           "width_mm": 1.5,
           "lift_height_mm": 5.0,
           "down_speed_mm_s": 10.0,
           "up_speed_mm_s": 20.0
       },
       "edge_detection": {
           "default": {
               "blur": 5,
               "canny_low": 50,
               "canny_high": 150,
               "dilate": 0,
               "erode": 0,
               "min_contour_length": 20,
               "close_gaps": True,
               "gap_size": 3
           },
           "portrait": {
               "blur": 3,
               "canny_low": 30,
               "canny_high": 100,
               "dilate": 0,
               "erode": 1,
               "min_contour_length": 15,
               "close_gaps": True,
               "gap_size": 2
           },
           "landscape": {
               "blur": 7,
               "canny_low": 70,
               "canny_high": 200,
               "dilate": 1,
               "erode": 0,
               "min_contour_length": 25,
               "close_gaps": True,
               "gap_size": 4
           },
           "sketch": {
               "blur": 1,
               "canny_low": 20,
               "canny_high": 60,
               "dilate": 0,
               "erode": 0,
               "min_contour_length": 10,
               "close_gaps": False,
               "gap_size": 0
           },
           "technical": {
               "blur": 0,
               "canny_low": 100,
               "canny_high": 200,
               "dilate": 0,
               "erode": 0,
               "min_contour_length": 30,
               "close_gaps": True,
               "gap_size": 1
           }
       },
       "color_configs": {
           "black": {
               "blur": 3,
               "canny_low": 40,
               "canny_high": 120
           },
           "brown": {
               "blur": 5,
               "canny_low": 50,
               "canny_high": 150
           },
           "red": {
               "blur": 5,
               "canny_low": 45,
               "canny_high": 140
           },
           "blue": {
               "blur": 4,
               "canny_low": 35,
               "canny_high": 110
           },
           "green": {
               "blur": 5,
               "canny_low": 40,
               "canny_high": 130
           },
           "yellow": {
               "blur": 7,
               "canny_low": 60,
               "canny_high": 180
           },
           "gray": {
               "blur": 5,
               "canny_low": 50,
               "canny_high": 150
           },
           "pink": {
               "blur": 6,
               "canny_low": 55,
               "canny_high": 160
           }
       },
       "color_quantization": {
           "method": "closest",
           "merge_threshold": 35.0,
           "min_area_pixels": 10,
           "edges_only": True
       },
       "drawing": {
           "bridge_px": 20,
           "max_bridge_passes": 8,
           "min_line_px": 5,
           "enable_taps": False,
           "optimize_path": True,
           "drawing_order": "light_to_dark"
       },
       "output_directory": "output",
       "preview_scale": 1.0
   }
   
   def __init__(self, config_path: Optional[str] = None):
       """
       Инициализация менеджера конфигураций.
       
       Args:
           config_path: Путь к файлу конфигурации или None для дефолтной
       """
       self.config_path = config_path
       self.config_data = self.DEFAULT_CONFIG.copy()
       
       if config_path and Path(config_path).exists():
           self.load(config_path)
   
   def load(self, path: str) -> PlotterConfig:
       """Загружает конфигурацию из файла"""
       with open(path, 'r') as f:
           user_config = json.load(f)
       
       # Рекурсивное объединение с дефолтными значениями
       self.config_data = self._merge_configs(self.DEFAULT_CONFIG, user_config)
       self.config_path = path
       
       return self.to_plotter_config()
   
   def save(self, path: Optional[str] = None):
       """Сохраняет текущую конфигурацию"""
       save_path = path or self.config_path
       if not save_path:
           raise ValueError("No path specified for saving config")
       
       with open(save_path, 'w') as f:
           json.dump(self.config_data, f, indent=2)
       
       print(f"Configuration saved to {save_path}")
   
   def create_preset(self, name: str, base: str = "default") -> Dict[str, Any]:
       """
       Создает новый пресет на основе существующего.
       
       Args:
           name: Имя нового пресета
           base: Базовый пресет
           
       Returns:
           Словарь с настройками пресета
       """
       if base not in self.config_data["edge_detection"]:
           base = "default"
       
       preset = self.config_data["edge_detection"][base].copy()
       self.config_data["edge_detection"][name] = preset
       
       return preset
   
   def update_preset(self, name: str, **kwargs):
       """Обновляет параметры пресета"""
       if name not in self.config_data["edge_detection"]:
           self.create_preset(name)
       
       self.config_data["edge_detection"][name].update(kwargs)
   
   def get_edge_config(self, preset: str = "default", color: Optional[str] = None) -> EdgeDetectionProfile:
       """
       Получает конфигурацию детекции границ.
       
       Args:
           preset: Имя пресета (default, portrait, landscape, etc.)
           color: Имя цвета для специфичных настроек
           
       Returns:
           Профиль детекции границ
       """
       # Базовый пресет
       if preset in self.config_data["edge_detection"]:
           config = self.config_data["edge_detection"][preset].copy()
       else:
           config = self.config_data["edge_detection"]["default"].copy()
       
       # Переопределения для конкретного цвета
       if color and "color_configs" in self.config_data:
           if color in self.config_data["color_configs"]:
               config.update(self.config_data["color_configs"][color])
       
       return EdgeDetectionProfile(**config)
   
   def to_plotter_config(self) -> PlotterConfig:
       """Преобразует в структурированную конфигурацию"""
       # Workspace
       workspace = WorkspaceConfig(**self.config_data["workspace"])
       
       # Pen
       pen = PenConfig(**self.config_data["pen"])
       
       # Edge detection profiles
       edge_profiles = {}
       for name, profile_data in self.config_data["edge_detection"].items():
           if name != "color_configs":
               edge_profiles[name] = EdgeDetectionProfile(**profile_data)
       
       # Color quantization
       color_quant = ColorQuantizationConfig(**self.config_data["color_quantization"])
       
       # Drawing
       drawing = DrawingConfig(**self.config_data["drawing"])
       
       return PlotterConfig(
           workspace=workspace,
           pen=pen,
           edge_detection=edge_profiles,
           color_quantization=color_quant,
           drawing=drawing,
           palette_file=self.config_data.get("palette_file"),
           output_directory=self.config_data.get("output_directory", "output"),
           preview_scale=self.config_data.get("preview_scale", 1.0)
       )
   
   def _merge_configs(self, default: Dict, user: Dict) -> Dict:
       """Рекурсивное объединение конфигураций"""
       result = default.copy()
       
       for key, value in user.items():
           if key in result and isinstance(result[key], dict) and isinstance(value, dict):
               result[key] = self._merge_configs(result[key], value)
           else:
               result[key] = value
       
       return result
   
   def validate(self) -> bool:
       """Валидирует конфигурацию"""
       try:
           config = self.to_plotter_config()
           
           # Проверка workspace
           assert config.workspace.width_steps > 0
           assert config.workspace.height_steps > 0
           assert config.workspace.steps_per_mm > 0
           
           # Проверка pen
           assert config.pen.width_mm > 0
           
           # Проверка edge detection
           for name, profile in config.edge_detection.items():
               assert profile.blur >= 0
               assert profile.canny_low >= 0
               assert profile.canny_high > profile.canny_low
               
           return True
           
       except Exception as e:
           print(f"Configuration validation failed: {e}")
           return False
   
   def print_summary(self):
       """Выводит сводку конфигурации"""
       config = self.to_plotter_config()
       
       print("="*60)
       print("PLOTTER CONFIGURATION")
       print("="*60)
       
       print("\nWorkspace:")
       print(f"  Size: {config.workspace.width_steps}x{config.workspace.height_steps} steps")
       print(f"  Size: {config.workspace.width_steps/config.workspace.steps_per_mm:.1f}x"
             f"{config.workspace.height_steps/config.workspace.steps_per_mm:.1f} mm")
       print(f"  Resolution: {config.workspace.steps_per_mm} steps/mm")
       
       print("\nPen:")
       print(f"  Width: {config.pen.width_mm} mm")
       print(f"  Lift height: {config.pen.lift_height_mm} mm")
       
       print("\nEdge Detection Presets:")
       for name in config.edge_detection.keys():
           print(f"  - {name}")
       
       print("\nColor Quantization:")
       print(f"  Method: {config.color_quantization.method}")
       print(f"  Merge threshold: {config.color_quantization.merge_threshold}")
       
       print("\nDrawing:")
       print(f"  Bridge gaps: {config.drawing.bridge_px} px")
       print(f"  Path optimization: {config.drawing.optimize_path}")
       print(f"  Drawing order: {config.drawing.drawing_order}")
       
       if config.palette_file:
           print(f"\nPalette: {config.palette_file}")
       
       print(f"\nOutput directory: {config.output_directory}")

def main():
   import argparse
   
   parser = argparse.ArgumentParser(description="Configuration manager for plotter pipeline")
   parser.add_argument("-c", "--config", help="Configuration file path")
   parser.add_argument("-v", "--validate", action="store_true", help="Validate configuration")
   parser.add_argument("-s", "--save", help="Save configuration to file")
   parser.add_argument("--create-default", help="Create default configuration file")
   parser.add_argument("--preset", help="Select edge detection preset")
   parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), 
                      action="append", help="Set configuration value")
   
   args = parser.parse_args()
   
   # Создание дефолтной конфигурации
   if args.create_default:
       manager = ConfigManager()
       manager.save(args.create_default)
       print(f"Default configuration created: {args.create_default}")
       return
   
   # Загрузка конфигурации
   manager = ConfigManager(args.config)
   
   # Установка значений
   if args.set:
       for key, value in args.set:
           # Простая установка значений верхнего уровня
           try:
               # Пытаемся преобразовать в число
               if value.lower() in ['true', 'false']:
                   value = value.lower() == 'true'
               elif '.' in value:
                   value = float(value)
               else:
                   value = int(value)
           except ValueError:
               pass  # Оставляем как строку
           
           # Устанавливаем значение (упрощенно, только верхний уровень)
           keys = key.split('.')
           if len(keys) == 1:
               manager.config_data[key] = value
           elif len(keys) == 2:
               if keys[0] not in manager.config_data:
                   manager.config_data[keys[0]] = {}
               manager.config_data[keys[0]][keys[1]] = value
           
           print(f"Set {key} = {value}")
   
   # Валидация
   if args.validate:
       if manager.validate():
           print("Configuration is valid")
       else:
           print("Configuration has errors")
           return 1
   
   # Сохранение
   if args.save:
       manager.save(args.save)
   
   # Вывод сводки
   manager.print_summary()

if __name__ == "__main__":
   main()