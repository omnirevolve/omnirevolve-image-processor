#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт-оркестратор для полного пайплайна обработки изображений для плоттера.

УПРОЩЕНО: «пиксели = шаги»
- Нормализация сразу под жёсткий целевой холст (мм × 40 px/mm) с сохранением пропорций.
- Дальше по конвейеру никаких перескалирований и единиц — всё в пикселях (они же шаги).
"""

import argparse
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time
from PIL import Image

A_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "A6": (105.0, 148.0),
}

class PlotterPipeline:
    """Оркестратор пайплайна обработки"""
    def __init__(self, config_file: Optional[str] = None, verbose: bool = False):
        self.verbose = verbose
        self.config_file = config_file or "config.json"
        self.workspace = Path(".")
        self.results: Dict = {}

    def log(self, message: str, level: str = "INFO"):
        if self.verbose or level in ["ERROR", "WARNING"]:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {level:7s}: {message}")

    def run_command(self, cmd: List[str], check: bool = True, stream: bool = True) -> subprocess.CompletedProcess:
        """Запускает внешнюю команду с прямым выводом (по умолчанию)."""
        self.log(f"Running: {' '.join(cmd)}", "DEBUG")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        if stream:
            result = subprocess.run(cmd, env=env, check=False)
            if result.returncode != 0 and check:
                self.log(f"Command failed: {' '.join(cmd)} (see output above)", "ERROR")
                raise RuntimeError(f"Command failed with code {result.returncode}")
            return result
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            if result.returncode != 0 and check:
                self.log(f"Command failed: {' '.join(cmd)}", "ERROR")
                self.log(f"Error output: {result.stderr}", "ERROR")
                raise RuntimeError(f"Command failed with code {result.returncode}")
            return result

    # -------------------- Step 0: normalize --------------------
    def step_normalize_image(self, image_path: str, output_dir: str,
                             target_w_steps: int, target_h_steps: int,
                             no_upscale: bool = False) -> str:
        """
        Шаг 0: нормализация входного изображения на ЖЁСТКИЙ холст (pixels == steps).
        Масштаб с сохранением пропорций, белые поля.
        """
        self.log(f"Step 0: Normalizing input image to {target_w_steps}x{target_h_steps}...", "INFO")
        out_img = str(Path(output_dir) / "normalized" / (Path(image_path).stem + "_norm.png"))
        cmd = [
            "python3", "normalize_image.py",
            image_path,
            "-o", out_img,
            "--target-width", str(target_w_steps),
            "--target-height", str(target_h_steps),
        ]
        if no_upscale:
            cmd.append("--no-upscale")

        self.run_command(cmd)
        self.results["normalized_image"] = out_img
        self.results["target_steps"] = {"width": target_w_steps, "height": target_h_steps}
        return out_img

    # -------------------- Step 1: colors analysis --------------------
    def step_analyze_colors(self, image_path: str, num_colors: int = 4) -> Dict:
        self.log("Step 1: Analyzing colors...", "INFO")
        output_json = Path(image_path).stem + "_colors.json"

        cmd = ["python3", "analyze_colors.py", image_path, "-n", str(num_colors)]
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            if 'palette_file' in config:
                cmd.extend(["--palette", config['palette_file']])

        self.run_command(cmd)

        with open(output_json, 'r') as f:
            colors_data = json.load(f)
        self.results['colors'] = colors_data
        self.log(f"Recommended colors: {', '.join(c['name'] for c in colors_data['recommended_colors'])}", "INFO")
        return colors_data

    # -------------------- Step 2: layers --------------------
    def step_process_colors(self, image_path: str, output_dir: str = "layers") -> List[str]:
        self.log("Step 2: Processing color layers...", "INFO")
        cmd = [
            "python3", "process_colors.py",
            image_path,
            "-o", output_dir,
            "-m", "adaptive",
            "--edges-only"
        ]
        self.run_command(cmd)
        layer_files = sorted(Path(output_dir).glob("layer_*.png"))
        self.results['layers'] = [str(f) for f in layer_files]
        self.log(f"Created {len(layer_files)} layers", "INFO")
        return self.results['layers']

    # -------------------- Step 3: edges --------------------
    def step_detect_edges(self, layers_dir: str, output_dir: str = "edges", preset: str = "default") -> List[str]:
        self.log(f"Step 3: Detecting edges (preset: {preset})...", "INFO")
        cmd = ["python3", "detect_edges_multi.py", layers_dir, "-o", output_dir, "--composite"]

        if self.config_file:
            cmd.extend(["-c", self.config_file])

        if preset != "default" and self.config_file and Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            if preset in config.get('edge_detection', {}):
                preset_config = config['edge_detection'][preset]
                if 'blur' in preset_config:
                    cmd.extend(["--blur", str(preset_config['blur'])])
                if 'canny_low' in preset_config:
                    cmd.extend(["--low", str(preset_config['canny_low'])])
                if 'canny_high' in preset_config:
                    cmd.extend(["--high", str(preset_config['canny_high'])])

        self.run_command(cmd)
        edge_files = sorted(Path(output_dir).glob("edges_layer_*.png"))
        self.results['edges'] = [str(f) for f in edge_files]
        self.log(f"Processed {len(edge_files)} edge layers", "INFO")
        return self.results['edges']

    # -------------------- Step 3.5: Deduplication --------------------
    def step_deduplicate_edges(self, edges_dir: str, output_dir: str = "edges_dedup",
                               pen_diameter: int = 60, enable_dedup: bool = True) -> Optional[str]:
        """
        Шаг 3.5: Дедупликация контуров с учетом толщины пера.
        Возвращает путь к vector_manifest.json
        """
        if not enable_dedup:
            self.log("Step 3.5: Deduplication skipped (disabled)", "INFO")
            return None

        self.log(f"Step 3.5: Deduplicating edges (pen diameter={pen_diameter}px)...", "INFO")

        edge_files = sorted(Path(edges_dir).glob("edges_layer_*.png"))
        if not edge_files:
            self.log("No edge files found, skipping deduplication", "WARNING")
            return None

        # Извлекаем маппинг цветов из имен файлов
        parsed = []
        for f in edge_files:
            m = re.match(r"edges_layer_(\d+)_(.+)", Path(f).stem)
            if m:
                pos = int(m.group(1))
                name = m.group(2)
                parsed.append((pos, name))

        if not parsed:
            self.log("Cannot parse edge file names, skipping deduplication", "WARNING")
            return None

        # Формируем маппинг
        parsed.sort(key=lambda t: t[0])
        mapping_pairs = [f"{name}:{i}" for i, (_, name) in enumerate(parsed)]
        mapping_str = ",".join(mapping_pairs)

        # Команда дедупликации (никаких размеров и units — всё уже в целевом холсте)
        cmd = [
            "python3", "deduplication_pipeline.py"
        ] + [str(f) for f in edge_files] + [
            "-o", output_dir,
            "-m", mapping_str,
            "--pen-diameter", str(pen_diameter),
            "--self-dedup"
        ]

        # Добавляем max-shift если нужно (опционально)
        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            if 'deduplication' in config and 'max_shift_ratio' in config['deduplication']:
                cmd.extend(["--max-shift-ratio", str(config['deduplication']['max_shift_ratio'])])

        self.run_command(cmd)

        # Проверяем наличие vector_manifest.json
        manifest_path = Path(output_dir) / "vector_manifest.json"
        if not manifest_path.exists():
            self.log("Vector manifest not created, deduplication may have failed", "ERROR")
            return None

        self.results['vector_manifest'] = str(manifest_path)
        self.log(f"Deduplication complete, vector data saved", "INFO")

        return str(manifest_path)

    # -------------------- Step 4: composite preview --------------------
    def step_preview_composite(self, vector_manifest: str, output_file: Optional[str] = None) -> None:
        """Создает композитный превью из векторных данных"""
        self.log("Step 4: Creating composite preview...", "INFO")

        if not vector_manifest or not Path(vector_manifest).exists():
            self.log("No vector manifest for preview", "WARNING")
            return

        cmd = ["python3", "preview_composite.py", vector_manifest]
        if output_file:
            cmd.extend(["-o", output_file])

        self.run_command(cmd, check=False)

    # -------------------- Step 5: stream generation --------------------
    def step_generate_stream(self,
                             vector_manifest: str,
                             output_file: str,
                             target_w_steps: int,
                             target_h_steps: int,
                             steps_per_mm: float) -> str:
        """Шаг 5: Генерация потока нибблов из векторных данных."""
        self.log("Step 5: Generating nibble stream from vector data...", "INFO")

        if not vector_manifest or not Path(vector_manifest).exists():
            raise RuntimeError(f"Vector manifest not found: {vector_manifest}")

        manifest_dir = Path(vector_manifest).parent

        cmd = ["python3", "../shared/xyplotter_stream_creator.py",
            str(manifest_dir),
            "-o", output_file,
            "--target-width-steps", str(target_w_steps),
            "--target-height-steps", str(target_h_steps),
            "--steps-per-mm", str(steps_per_mm)]
        self.run_command(cmd)
        self.results['stream'] = output_file

        # Размер потока
        stream_size = Path(output_file).stat().st_size
        self.log(f"Stream generated: {stream_size:,} bytes", "INFO")

        # Запускаем превью
        try:
            self.log(f"Launching plotter_preview ({target_w_steps}x{target_h_steps})...", "INFO")
            self.run_command(
                ["python3", "../shared/xyplotter_stream_preview.py",
                 "--width", str(target_w_steps), "--height", str(target_h_steps),
                 output_file],
                check=False
            )
        except Exception as e:
            self.log(f"Preview launch failed: {e}", "WARNING")

        return output_file

    # -------------------- helpers --------------------
    @staticmethod
    def orient_mm_for_image(mm_w: float, mm_h: float, img_w: int, img_h: int) -> Tuple[float, float]:
        """Поворачиваем размеры бумаги под ориентацию изображения."""
        if (img_w >= img_h and mm_w < mm_h) or (img_w < img_h and mm_w > mm_h):
            return (mm_h, mm_w)
        return (mm_w, mm_h)

    # -------------------- main runner --------------------
    def run_full_pipeline(self,
                          image_path: str,
                          output_dir: str = "output",
                          preset: str = "default",
                          num_colors: int = 4,
                          paper: str = "A4",
                          custom_mm: Optional[Tuple[float, float]] = None,
                          steps_per_mm: float = 40.0,
                          no_upscale: bool = False,
                          pen_diameter: int = 60,
                          enable_dedup: bool = True,
                          skip_steps: List[str] = None) -> Dict:
        skip_steps = skip_steps or []
        start_time = time.time()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.results['input_image'] = image_path
        self.results['output_dir'] = output_dir

        print("\n" + "="*60)
        print("PLOTTER PIPELINE")
        print("="*60)
        print(f"Input: {image_path}")
        print(f"Output: {output_dir}")
        print(f"Preset: {preset}")
        print(f"Colors: {num_colors}")
        print(f"Paper: {paper}{' (custom)' if paper=='custom' else ''}")
        print(f"Steps/mm: {steps_per_mm}")
        print(f"Pen diameter: {pen_diameter}px")
        print(f"Deduplication: {'enabled' if enable_dedup else 'disabled'}")
        print("="*60 + "\n")

        try:
            # Вычисляем целевой холст (пиксели == шаги) из бумаги и 40 px/mm
            # Берём ориентацию по исходной картинке
            iw0, ih0 = Image.open(image_path).size
            if paper.lower() != "custom":
                mm_w, mm_h = A_SIZES_MM[paper.upper()]
            else:
                assert custom_mm is not None, "Custom paper selected but no size provided"
                mm_w, mm_h = custom_mm
            mm_w, mm_h = self.orient_mm_for_image(mm_w, mm_h, iw0, ih0)

            target_w_steps = int(round(mm_w * steps_per_mm))
            target_h_steps = int(round(mm_h * steps_per_mm))
            self.results["target_steps"] = {"width": target_w_steps, "height": target_h_steps}

            # Step 0: normalize ПОД ЖЁСТКИЙ ХОЛСТ
            norm_img = self.step_normalize_image(
                image_path, output_dir,
                target_w_steps=target_w_steps, target_h_steps=target_h_steps,
                no_upscale=no_upscale
            )

            # Step 1: analyze on normalized
            if 'analyze' not in skip_steps:
                self.step_analyze_colors(norm_img, num_colors)

            # Step 2: layers from normalized
            if 'layers' not in skip_steps:
                layers_dir = str(output_path / "layers")
                self.step_process_colors(norm_img, layers_dir)

            # Step 3: edges
            if 'edges' not in skip_steps:
                edges_dir = str(output_path / "edges")
                self.step_detect_edges(
                    layers_dir if 'layers' not in skip_steps else str(output_path / "layers"),
                    edges_dir,
                    preset
                )

            # Step 3.5: deduplication (все файлы уже в целевом размере; перескалирование не требуется)
            vector_manifest = None
            if 'dedup' not in skip_steps and enable_dedup:
                dedup_dir = str(output_path / "edges_dedup")
                vector_manifest = self.step_deduplicate_edges(
                    edges_dir if 'edges' not in skip_steps else str(output_path / "edges"),
                    dedup_dir,
                    pen_diameter=pen_diameter,
                    enable_dedup=enable_dedup
                )

            # Step 4: composite preview (если есть векторные данные)
            if 'preview' not in skip_steps and vector_manifest:
                preview_file = str(output_path / "composite_preview.png")
                self.step_preview_composite(vector_manifest, preview_file)

            # Step 5: stream generation
            if 'stream' not in skip_steps and vector_manifest:
                stream_file = str(output_path / "stream.bin")
                self.step_generate_stream(
                    vector_manifest,
                    stream_file,
                    target_w_steps=target_w_steps,
                    target_h_steps=target_h_steps,
                    steps_per_mm=steps_per_mm
                )

            # Step 6: optional saved preview
            if 'stream_preview' not in skip_steps and 'stream' not in skip_steps:
                stream_preview = str(output_path / "stream_preview.png")
                self.step_preview_stream(
                    self.results.get('stream', str(output_path / "stream.bin")),
                    width_steps=target_w_steps,
                    height_steps=target_h_steps,
                    preview_output=stream_preview
                )

            # Metadata
            metadata = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'input_image': image_path,
                'normalized_image': norm_img,
                'output_dir': output_dir,
                'preset': preset,
                'num_colors': num_colors,
                'pen_diameter': pen_diameter,
                'deduplication_enabled': enable_dedup,
                'vector_manifest': vector_manifest,
                'config_file': self.config_file,
                'results': self.results,
                'steps_per_mm': steps_per_mm,
                'processing_time': time.time() - start_time
            }
            metadata_file = output_path / "pipeline_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            elapsed = time.time() - start_time
            print("\n" + "="*60)
            print("PIPELINE COMPLETED")
            print("="*60)
            print(f"Time: {elapsed:.1f} seconds")
            print(f"Output directory: {output_dir}")
            print("\nGenerated files:")
            if 'stream' in self.results:
                print(f"  Stream: {self.results['stream']}")
                print(f"  Size: {Path(self.results['stream']).stat().st_size:,} bytes")
            if vector_manifest:
                print(f"  Vector data: {vector_manifest}")
            print(f"  Metadata: {metadata_file}")
            print("\nNext steps:")
            print(f"  1. Review: ../shared/xyplotter_stream_preview.py --width {target_w_steps} --height {target_h_steps} {self.results.get('stream', 'stream.bin')}")
            print(f"  2. Send stream to plotter")
            print("="*60 + "\n")
            return metadata

        except Exception as e:
            self.log(f"Pipeline failed: {e}", "ERROR")
            import traceback; traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Plotter pipeline orchestrator (pixels == steps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with deduplication, A4 by default, pen diameter 60px
  python plotter_pipeline.py image.jpg

  # A4 with custom pen diameter
  python plotter_pipeline.py image.jpg --paper A4 --pen-diameter 80

  # Disable deduplication
  python plotter_pipeline.py image.jpg --no-dedup

  # Custom size in millimeters (e.g., 180x240 mm)
  python plotter_pipeline.py image.jpg --paper custom --mm-width 180 --mm-height 240

  # Skip deduplication step
  python plotter_pipeline.py image.jpg --skip dedup

  # Verbose logs
  python plotter_pipeline.py image.jpg -v
        """
    )

    parser.add_argument("image", help="Input image file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-c", "--config", help="Configuration file")
    parser.add_argument("-p", "--preset", default="default",
                        choices=['default', 'portrait', 'landscape', 'sketch', 'technical'],
                        help="Edge detection preset")
    parser.add_argument("-n", "--colors", type=int, default=4, help="Number of colors (1-4)")
    parser.add_argument("--skip", nargs='+',
                        choices=['analyze', 'layers', 'edges', 'dedup', 'preview', 'stream', 'stream_preview'],
                        help="Skip pipeline steps")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Paper/size & steps/mm
    parser.add_argument("--paper", default="A4", choices=["A4", "A5", "A6", "custom"],
                        help="Target paper size (default: A4)")
    parser.add_argument("--mm-width", type=float, help="Custom width in mm (when --paper custom)")
    parser.add_argument("--mm-height", type=float, help="Custom height in mm (when --paper custom)")
    parser.add_argument("--steps-per-mm", type=float, default=40.0, help="Plotter steps per mm (default: 40)")

    # Deprecated normalize params (не используются, оставлены для совместимости CLI)
    parser.add_argument("--no-upscale", action="store_true", help="Do not upscale during normalization")
    parser.add_argument("--norm-long-side", type=int, default=2048, help=argparse.SUPPRESS)

    # Deduplication params
    parser.add_argument("--pen-diameter", type=int, default=60, 
                        help="Pen diameter in pixels (default: 60)")
    parser.add_argument("--no-dedup", action="store_true", 
                        help="Disable contour deduplication")

    args = parser.parse_args()

    # Проверяем входной файл
    if not Path(args.image).exists():
        print(f"Error: Input file not found: {args.image}")
        sys.exit(1)

    # Проверяем кастомные размеры
    custom_mm = None
    if args.paper.lower() == "custom":
        if args.mm_width is None or args.mm_height is None:
            print("Error: --paper custom requires --mm-width and --mm-height", file=sys.stderr)
            sys.exit(1)
        custom_mm = (float(args.mm_width), float(args.mm_height))

    skip_steps = args.skip or []

    pipeline = PlotterPipeline(
        config_file=args.config,
        verbose=args.verbose
    )

    pipeline.run_full_pipeline(
        image_path=args.image,
        output_dir=args.output,
        preset=args.preset,
        num_colors=args.colors,
        paper=args.paper,
        custom_mm=custom_mm,
        steps_per_mm=args.steps_per_mm,
        no_upscale=args.no_upscale,
        pen_diameter=args.pen_diameter,
        enable_dedup=not args.no_dedup,
        skip_steps=skip_steps
    )

if __name__ == "__main__":
    main()
