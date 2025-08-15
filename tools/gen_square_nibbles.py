#!/usr/bin/env python3
import argparse
import math
from typing import List

# Ниббл биты (должны совпадать с прошивкой)
NIBBLE_X_STEP = 1 << 0
NIBBLE_X_DIR  = 1 << 1  # 1 = to MAX, 0 = to MIN
NIBBLE_Y_STEP = 1 << 2
NIBBLE_Y_DIR  = 1 << 3  # 1 = to MAX, 0 = to MIN

def pack_nibbles_to_bytes(nibbles: List[int]) -> bytes:
    out = bytearray()
    it = iter(nibbles)
    for lo in it:
        try:
            hi = next(it)
        except StopIteration:
            hi = 0
        out.append((hi << 4) | (lo & 0x0F))
    return bytes(out)

def gen_side_nibbles(
    steps_per_mm: float,
    side_mm: float,
    timer_hz: float,
    v_cruise_mm_s: float,
    v_min_mm_s: float,
    ramp_len_mm: float,
    move_axis: str,   # 'x' or 'y'
    dir_to_max: bool  # True => DIR=1, False => DIR=0
) -> List[int]:
    """
    Генерация нибблов для одной стороны (движение только по одной оси).
    Профиль: трапеция по скорости (или треугольник, если сторона короткая).
    """

    dt = 1.0 / timer_hz

    # Ограничения/проверки
    side_mm = float(side_mm)
    ramp_len_mm = float(ramp_len_mm)
    v_cruise_mm_s = float(v_cruise_mm_s)
    v_min_mm_s = float(v_min_mm_s)

    if side_mm <= 0.0:
        return []

    # Ускорение из требования пройти от Vmin до Vcruise на расстоянии ramp_len_mm
    if ramp_len_mm <= 0.0 or v_cruise_mm_s <= v_min_mm_s:
        a_mm_s2 = 0.0
    else:
        a_mm_s2 = (v_cruise_mm_s**2 - v_min_mm_s**2) / (2.0 * ramp_len_mm)

    # Проверяем, укладывается ли трапеция в длину стороны
    need_triangular = (2.0 * ramp_len_mm > side_mm) and (a_mm_s2 > 0.0)

    # Подготовка полей ниббла
    dir_bit = (NIBBLE_X_DIR if move_axis == 'x' else NIBBLE_Y_DIR)
    step_bit = (NIBBLE_X_STEP if move_axis == 'x' else NIBBLE_Y_STEP)
    fixed_dir_mask = dir_bit if dir_to_max else 0

    nibbles: List[int] = []

    # Состояние интеграции
    v = v_min_mm_s  # текущая скорость (мм/с)
    dist_mm = 0.0   # пройденная дистанция вдоль стороны (мм)
    acc_phase = True
    cruise_phase = False
    decel_phase = False

    # Для DDA по шагам
    steps_accum = 0.0  # накопитель дробных шагов
    step_len_mm = 1.0 / steps_per_mm
    target_mm = side_mm

    # Границы фаз
    if need_triangular:
        # Симметричная "гора": разгон до середины, затем торможение.
        accel_end_mm = side_mm * 0.5
        cruise_phase = False
    else:
        accel_end_mm = ramp_len_mm
        cruise_phase = True
    decel_start_mm = side_mm - ramp_len_mm if not need_triangular else side_mm * 0.5
    if decel_start_mm < accel_end_mm:
        decel_start_mm = accel_end_mm  # страховка от пересечений фаз

    while dist_mm < target_mm - 1e-9:
        # Определяем фазу
        if acc_phase and dist_mm >= accel_end_mm - 1e-12:
            acc_phase = False
        if cruise_phase and dist_mm >= decel_start_mm - 1e-12:
            cruise_phase = False
            decel_phase = True
        if need_triangular and acc_phase is False and decel_phase is False:
            # В триугольнике: после разгона сразу торможение
            decel_phase = True

        # Обновление скорости
        if a_mm_s2 > 0.0:
            if acc_phase:
                v = min(v + a_mm_s2 * dt, v_cruise_mm_s)
            elif decel_phase:
                v = max(v - a_mm_s2 * dt, v_min_mm_s)
            # cruise_phase: v держим как есть
        else:
            # Без ускорения/торможения: постоянная скорость = v_cruise_mm_s (или v_min если cruise<min)
            v = max(v_cruise_mm_s, v_min_mm_s)

        # Страховка на границах
        v = max(v_min_mm_s, min(v, v_cruise_mm_s))

        # Перевод в шаг/с
        step_rate = v * steps_per_mm  # steps per second
        steps_accum += step_rate * dt

        # По умолчанию шагов на этом тике нет
        step_flag = 0

        if steps_accum >= 1.0:
            # В наших скоростях шаг не более 1 за тик; если когда-то будет >1 — можно заменить на while
            steps_accum -= 1.0
            step_flag = 1
            dist_mm += step_len_mm
            if dist_mm > target_mm:
                dist_mm = target_mm  # не переливать

        # Собираем ниббл для текущего тика
        nib = 0
        if fixed_dir_mask:
            nib |= fixed_dir_mask
        if step_flag:
            nib |= step_bit
        nibbles.append(nib)

    return nibbles

def generate_square_stream(
    steps_per_mm: int = 40,
    outer_side_mm: float = 100.0,
    timer_hz: int = 10_000,
    v_cruise_mm_s: float = 20.0,
    v_min_mm_s: float = 2.0,
    ramp_len_mm: float = 20.0,
) -> bytes:
    """
    Квадрат со сторонами, параллельными осям.
    Порядок: +X, +Y, -X, -Y. Без пауз на поворотах.
    """
    nibbles: List[int] = []

    # Предупреждение о короткой стороне (stdout)
    if 2.0 * ramp_len_mm > outer_side_mm:
        print(f"[warn] side {outer_side_mm:.1f} mm is too short for 2*ramp {2*ramp_len_mm:.1f} mm: using triangular profile.")

    # +X
    nibbles += gen_side_nibbles(
        steps_per_mm, outer_side_mm, timer_hz,
        v_cruise_mm_s, v_min_mm_s, ramp_len_mm,
        move_axis='x', dir_to_max=True
    )
    # +Y
    nibbles += gen_side_nibbles(
        steps_per_mm, outer_side_mm, timer_hz,
        v_cruise_mm_s, v_min_mm_s, ramp_len_mm,
        move_axis='y', dir_to_max=True
    )
    # -X
    nibbles += gen_side_nibbles(
        steps_per_mm, outer_side_mm, timer_hz,
        v_cruise_mm_s, v_min_mm_s, ramp_len_mm,
        move_axis='x', dir_to_max=False
    )
    # -Y
    nibbles += gen_side_nibbles(
        steps_per_mm, outer_side_mm, timer_hz,
        v_cruise_mm_s, v_min_mm_s, ramp_len_mm,
        move_axis='y', dir_to_max=False
    )

    return pack_nibbles_to_bytes(nibbles)

def main():
    p = argparse.ArgumentParser(
        description="Generate raw nibble stream for square path (XY-plotter)."
    )
    p.add_argument("--steps-per-mm", type=int, default=40, help="шагов/мм (по умолчанию 40)")
    p.add_argument("--side-mm", type=float, default=100.0, help="сторона квадрата, мм (по умолчанию 100)")
    p.add_argument("--timer-hz", type=int, default=10_000, help="частота тиков, Гц (по умолчанию 10000)")
    p.add_argument("--v-cruise", type=float, default=20.0, help="маршевая скорость, мм/с (по умолчанию 20)")
    p.add_argument("--v-min", type=float, default=2.0, help="минимальная скорость на концах, мм/с (по умолчанию 2)")
    p.add_argument("--ramp-mm", type=float, default=20.0, help="длина разгона/торможения, мм (по умолчанию 20)")
    p.add_argument("-o", "--output", required=True, help="путь к выходному .bin файлу")

    args = p.parse_args()

    data = generate_square_stream(
        steps_per_mm=args.steps_per_mm,
        outer_side_mm=args.side_mm,
        timer_hz=args.timer_hz,
        v_cruise_mm_s=args.v_cruise,
        v_min_mm_s=args.v_min,
        ramp_len_mm=args.ramp_mm,
    )
    with open(args.output, "wb") as f:
        f.write(data)
    print(f"[ok] wrote {len(data)} bytes to {args.output}")

if __name__ == "__main__":
    main()
