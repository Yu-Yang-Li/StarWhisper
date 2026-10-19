"""
在全天相机图像上标注地平坐标系。

该脚本复用 replan.py 中的新版解析逆投影，支持两种标定来源：
- pipeline 的 parameters.json + position_name
"""

from __future__ import annotations

import argparse
from datetime import timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

from replan import (
    DEFAULT_MASK_PATH,
    DEFAULT_PARAMETERS_PATH,
    DEFAULT_POSITION_NAME,
    PIPELINE_ROOT,
    altaz_to_pixel_fast,
    load_camera_model,
    parse_mask_time_beijing,
)


DEFAULT_OUTPUT_PATH = PIPELINE_ROOT / "output" / "debug_horizon_overlay.png"
DEFAULT_TIME = "2022-05-04 03:10:29"
BEIJING_TZ = timezone(timedelta(hours=8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在全天相机图像上标注地平坐标网格。")
    parser.add_argument("--image", default=str(DEFAULT_MASK_PATH), help="输入图像路径。")
    parser.add_argument(
        "--parameters",
        default=str(DEFAULT_PARAMETERS_PATH),
        help="pipeline parameters.json 路径。",
    )
    parser.add_argument(
        "--position-name",
        default=DEFAULT_POSITION_NAME,
        help="parameters.json 中使用的参数组名，如 position_2。",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="输出标注图路径。")
    parser.add_argument("--time0", default=DEFAULT_TIME, help="显示用观测时间字符串。")
    return parser.parse_args()


def read_image(image_path: Path) -> np.ndarray:
    with open(image_path, "rb") as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return img


def write_image(image_path: Path, image: np.ndarray) -> None:
    suffix = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码输出图像: {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(encoded.tobytes())


def draw_cross(
    image: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cv2.circle(image, (x, y), radius, color, 3)
    cv2.line(image, (x - radius - 10, y), (x + radius + 10, y), color, 3)
    cv2.line(image, (x, y - radius - 10), (x, y + radius + 10), color, 3)


def put_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 1.0,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def load_model(args: argparse.Namespace):
    return load_camera_model(
        Path(args.parameters).expanduser().resolve(),
        args.position_name,
    )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print("正在读取标定参数...")
    camera_model = load_model(args)
    print(f"图像中心: ({camera_model.cx:.2f}, {camera_model.cy:.2f})")

    print(f"\n正在读取图像: {image_path}")
    img = read_image(image_path)
    h, w = img.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    image_shape = img.shape
    img_annotated = img.copy()

    print("\n正在标注方位点...")
    directions = [
        (0, 30, "N", (0, 255, 0)),
        (90, 30, "E", (255, 255, 0)),
        (180, 30, "S", (0, 0, 255)),
        (270, 30, "W", (255, 0, 255)),
    ]
    for az, alt, label, color in directions:
        pos = altaz_to_pixel_fast(az, alt, camera_model, image_shape=image_shape)
        if pos is None:
            print(f"  {label} -> 未找到")
            continue
        x_pos, y_pos = pos
        draw_cross(img_annotated, x_pos, y_pos, color, radius=20)
        put_label(img_annotated, label, x_pos - 30, y_pos - 40, color, scale=1.6, thickness=3)
        print(f"  {label} (Az={az}°, Alt={alt}°) -> ({x_pos}, {y_pos})")

    print("\n正在标注天顶...")
    zenith_pos = altaz_to_pixel_fast(0, 90, camera_model, image_shape=image_shape)
    if zenith_pos is None:
        x_zen, y_zen = int(round(camera_model.cx)), int(round(camera_model.cy))
        print("  天顶投影失败，回退到图像中心")
    else:
        x_zen, y_zen = zenith_pos
    cv2.circle(img_annotated, (x_zen, y_zen), 30, (255, 255, 255), 4)
    cv2.circle(img_annotated, (x_zen, y_zen), 12, (255, 255, 255), -1)
    put_label(img_annotated, "Zenith", x_zen - 55, y_zen - 45, (255, 255, 255), scale=1.0)
    print(f"  天顶 -> ({x_zen}, {y_zen})")

    print("\n正在标注子午线、等高度圈和等方位角线...")
    for az, color in [(0, (0, 255, 0)), (180, (0, 0, 255))]:
        line_points = []
        for alt in range(10, 90, 10):
            pos = altaz_to_pixel_fast(float(az), float(alt), camera_model, image_shape=image_shape)
            if pos is not None:
                line_points.append(pos)
        if len(line_points) > 1:
            cv2.polylines(img_annotated, [np.array(line_points, dtype=np.int32)], False, color, 3)

    for alt in [20, 40, 60, 80]:
        circle_points = []
        for az in range(0, 360, 10):
            pos = altaz_to_pixel_fast(float(az), float(alt), camera_model, image_shape=image_shape)
            if pos is not None:
                circle_points.append(pos)
        if len(circle_points) > 2:
            pts = np.array(circle_points, dtype=np.int32)
            cv2.polylines(img_annotated, [pts], True, (100, 100, 255), 2)
            lx, ly = pts[0]
            put_label(img_annotated, f"{alt} deg", int(lx) + 8, int(ly), (100, 100, 255), scale=0.7)

    for az in [0, 45, 90, 135, 180, 225, 270, 315]:
        line_points = []
        for alt in range(10, 85, 5):
            pos = altaz_to_pixel_fast(float(az), float(alt), camera_model, image_shape=image_shape)
            if pos is not None:
                line_points.append(pos)
        if len(line_points) > 1:
            cv2.polylines(img_annotated, [np.array(line_points, dtype=np.int32)], False, (150, 150, 100), 1)

    display_time = args.time0
    try:
        display_time = f"BJT {parse_mask_time_beijing(image_path).strftime('%Y-%m-%d %H:%M:%S')}"
    except ValueError:
        pass
    put_label(img_annotated, "Horizontal Coordinate System", 30, 45, (255, 255, 255), scale=1.0)
    put_label(img_annotated, f"Time: {display_time}", 30, 80, (255, 255, 255), scale=0.75)

    write_image(output_path, img_annotated)
    print(f"\n输出完成: {output_path}")


if __name__ == "__main__":
    main()
