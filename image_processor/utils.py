import cv2
import numpy as np
from typing import List, Tuple
import pickle


def load_config():
    """Load configuration from a temporary pickle file if present, otherwise create a new default Config()."""
    import os
    from config import Config

    if os.path.exists("config_temp.pkl"):
        with open("config_temp.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return Config()


def visualize_contours(
    contours: List[np.ndarray],
    image_size: Tuple[int, int],
    title: str = "Contours",
) -> np.ndarray:
    """Render contours onto a white RGB image and return the rendered image array."""
    img = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)

    colors = [
        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for _ in range(len(contours))
    ]

    for contour, color in zip(contours, colors):
        cv2.drawContours(img, [contour], -1, color, 2)

    return img


def load_contours(filepath: str) -> List[np.ndarray]:
    """Load contours list from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_contours(contours: List[np.ndarray], filepath: str):
    """Save contours list to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(contours, f)


def combine_layers_to_image(config):
    """Combine all layers into a single visualization image and save it to disk."""
    img = cv2.imread(f"{config.output_dir}/resized.png")
    if img is None:
        raise FileNotFoundError(f"Missing resized image: {config.output_dir}/resized.png")
    height, width = img.shape[:2]

    combined = np.ones((height, width, 3), dtype=np.uint8) * 255

    viz_colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
    }

    for color_name in config.color_names:
        try:
            contours = load_contours(f"{config.output_dir}/{color_name}/contours_final.pkl")
            color = viz_colors.get(color_name, (128, 128, 128))

            scale_back = width / (config.target_width_mm * config.pixels_per_mm)
            for contour in contours:
                scaled = (contour * scale_back).astype(np.int32)
                cv2.drawContours(combined, [scaled], -1, color, 1)
        except FileNotFoundError:
            print(f"[combine] Contours for '{color_name}' not found")

    out_path = f"{config.output_dir}/combined_result.png"
    ok = cv2.imwrite(out_path, combined)
    if not ok:
        raise RuntimeError(f"Failed to write combined image: {out_path}")
    return combined


def analyze_results(config):
    """Print per-layer statistics across pipeline stages; returns a nested dict with counts."""
    stats = {}

    for color_name in config.color_names:
        stats[color_name] = {}

        stages = [
            ("contours.pkl", "initial"),
            ("contours_sorted.pkl", "sorted"),
            ("contours_dedup_layer.pkl", "dedup_layer"),
            ("contours_dedup_cross.pkl", "dedup_cross"),
            ("contours_simplified.pkl", "simplified"),
            ("contours_final.pkl", "final"),
        ]

        for filename, stage in stages:
            try:
                filepath = f"{config.output_dir}/{color_name}/{filename}"
                contours = load_contours(filepath)
                stats[color_name][stage] = len(contours)
                total_points = sum(len(c) for c in contours)
                stats[color_name][f"{stage}_points"] = total_points
            except FileNotFoundError:
                stats[color_name][stage] = "N/A"

    print("\n" + "=" * 60)
    print("PIPELINE STATISTICS")
    print("=" * 60)

    for color_name, color_stats in stats.items():
        print(f"\n{color_name.upper()}:")
        for stage, count in color_stats.items():
            if not stage.endswith("_points"):
                points = color_stats.get(f"{stage}_points", "N/A")
                print(f"  {stage:15} : {count:5} contours, {points:6} points")

    return stats


def export_combined_svg(config):
    """Export all layers into a single SVG file (per-layer groups, 1px stroke)."""
    target_width = int(config.target_width_mm * config.pixels_per_mm)
    target_height = int(config.target_height_mm * config.pixels_per_mm)

    svg_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg width="{target_width}" height="{target_height}" '
        'xmlns="http://www.w3.org/2000/svg">\n'
    )

    for i, color_name in enumerate(config.color_names):
        try:
            contours = load_contours(f"{config.output_dir}/{color_name}/contours_final.pkl")

            bgr = config.colors[i]
            rgb = (bgr[2], bgr[1], bgr[0])

            svg_content += (
                f'  <g id="{color_name}" fill="none" stroke="rgb{rgb}" '
                'stroke-width="1" opacity="0.8">\n'
            )

            for contour in contours:
                path_data = "M "
                for point in contour:
                    path_data += f"{point[0][0]},{point[0][1]} "
                path_data += "Z"
                svg_content += f'    <path d="{path_data}"/>\n'

            svg_content += "  </g>\n"

        except FileNotFoundError:
            print(f"[svg] Contours for '{color_name}' not found")

    svg_content += "</svg>"

    out_svg = f"{config.output_dir}/combined_output.svg"
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"[svg] Combined SVG saved: {out_svg}")


if __name__ == "__main__":
    from config import Config

    config = Config()

    stats = analyze_results(config)
    combined = combine_layers_to_image(config)
    export_combined_svg(config)
