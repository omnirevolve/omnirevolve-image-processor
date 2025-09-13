#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нормализация входного изображения под ЖЁСТКИЙ целевой холст:
- «пиксели = шаги», никаких других единиц;
- если заданы --target-width/--target-height, то приводим к ЭТИМ размерам
  с сохранением пропорций и белыми полями (letterbox);
- для обратной совместимости оставлен режим --long-side (без холста).

Примеры:
  # A5 в мм → пиксели с шагом 40 px/mm считаем в пайплайне и передаем сюда
  python normalize_image.py in.jpg -o out.png --target-width 5920 --target-height 8400

  # Старый режим: длинная сторона = 2048 px (без жесткого холста)
  python normalize_image.py in.jpg -o out.png --long-side 2048
"""

import argparse
from pathlib import Path
from PIL import Image

# --------------------------- утилиты ---------------------------

def calc_new_size_by_long_side(w, h, long_side, allow_upscale=True):
    """Старый режим: масштаб по длинной стороне, без холста."""
    if w >= h:
        scale = long_side / w
    else:
        scale = long_side / h
    if not allow_upscale and scale > 1.0:
        return w, h, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return new_w, new_h, scale


def letterbox_to_canvas(img: Image.Image, tgt_w: int, tgt_h: int, allow_upscale: bool = True):
    """
    Масштабирует изображение с сохранением пропорций и вписывает в холст tgt_w × tgt_h
    с белыми полями. Возвращает (canvas, new_w, new_h, scale, offset_x, offset_y).
    """
    w, h = img.size
    # коэффициент масштаба по минимальному отношению сторон
    scale = min(tgt_w / max(1, w), tgt_h / max(1, h))
    if not allow_upscale and scale > 1.0:
        scale = 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img if (new_w, new_h) == (w, h) else img.resize((new_w, new_h), resample=Image.LANCZOS)

    # белый холст
    canvas = Image.new("RGB", (int(tgt_w), int(tgt_h)), (255, 255, 255))
    off_x = (tgt_w - new_w) // 2
    off_y = (tgt_h - new_h) // 2
    canvas.paste(img_resized, (off_x, off_y))

    return canvas, new_w, new_h, scale, off_x, off_y


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Normalize input image to a fixed target canvas (pixels=steps).")
    ap.add_argument("input", help="Input image")
    ap.add_argument("-o", "--output", required=True, help="Output image path (PNG/JPG)")
    # НОВОЕ: жёсткий холст в пикселях (т.е. в шагах)
    ap.add_argument("--target-width", type=int, help="Target canvas width in pixels (== steps)")
    ap.add_argument("--target-height", type=int, help="Target canvas height in pixels (== steps)")
    ap.add_argument("--no-upscale", action="store_true", help="Do not upscale image when fitting into target canvas")

    # Совместимость: старый режим без холста
    ap.add_argument("--long-side", type=int, default=None, help="Legacy mode: target long side in pixels")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(inp).convert("RGB")
    w, h = img.size

    # Режим 1: новый «жесткий холст»
    if args.target_width and args.target_height:
        tgt_w = int(args.target_width)
        tgt_h = int(args.target_height)

        canvas, new_w, new_h, scale, off_x, off_y = letterbox_to_canvas(
            img, tgt_w, tgt_h, allow_upscale=not args.no_upscale
        )

        # сохраняем
        if outp.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            outp = outp.with_suffix(".png")
        canvas.save(outp, optimize=True)

        print(
            f"Normalized to canvas: {inp} -> {outp}  "
            f"({w}x{h} -> {new_w}x{new_h} on {tgt_w}x{tgt_h}, scale={scale:.4f}, offset=({off_x},{off_y}))"
        )
        return

    # Режим 2: совместимость — только масштаб по длинной стороне
    if args.long_side is None:
        # По умолчанию, если ничего не задано, ведём себя как старый normalize с 2048
        args.long_side = 2048

    new_w, new_h, scale = calc_new_size_by_long_side(
        w, h, args.long_side, allow_upscale=not args.no_upscale
    )
    out_img = img if (new_w, new_h) == (w, h) else img.resize((new_w, new_h), resample=Image.LANCZOS)

    if outp.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        outp = outp.with_suffix(".png")
    out_img.save(outp, optimize=True)
    print(f"Normalized (legacy): {inp} -> {outp}  ({w}x{h} -> {new_w}x{new_h}, scale={scale:.4f})")


if __name__ == "__main__":
    main()
