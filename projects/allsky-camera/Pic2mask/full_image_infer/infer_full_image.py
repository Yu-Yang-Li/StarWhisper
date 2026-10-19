from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "deeplabv3_best_loss_3_16.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不裁切，直接对整张图片执行 DeepLabV3 推理并导出二值掩码。",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="输入原图路径。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出掩码路径。",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="模型权重路径。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备，默认 auto。",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="额外保存 overlay 对比图。",
    )
    parser.add_argument(
        "--overlay-output",
        default=None,
        help="overlay 输出路径；不传时默认使用 output 同目录并追加 _overlay.png。",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 cuda，但当前环境不可用")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(weights_path: Path, device: torch.device):
    if not weights_path.is_file():
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_mask(image: Image.Image, model, image_transform, device: torch.device) -> Image.Image:
    rgb = image.convert("RGB")
    tensor = image_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)["out"][0]
    prediction = output.argmax(0).byte().cpu().numpy()
    mask_np = np.where(prediction == 1, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_np)


def build_overlay(original: Image.Image, mask: Image.Image) -> Image.Image:
    original_rgba = original.convert("RGBA")
    mask_arr = np.array(mask)
    alpha_arr = np.where(mask_arr == 255, 128, 0).astype(np.uint8)

    overlay = Image.new("RGBA", original_rgba.size, (255, 0, 0, 0))
    overlay.putalpha(Image.fromarray(alpha_arr))
    combined = Image.alpha_composite(original_rgba, overlay)

    comparison = Image.new("RGB", (original.width * 2, original.height))
    comparison.paste(original.convert("RGB"), (0, 0))
    comparison.paste(combined.convert("RGB"), (original.width, 0))
    return comparison


def default_overlay_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_overlay.png")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    device = resolve_device(args.device)

    if not image_path.is_file():
        raise FileNotFoundError(f"未找到输入图片: {image_path}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    model = load_model(weights_path, device)
    image_transform = build_transform()
    mask = predict_mask(image, model, image_transform, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)

    print(f"输入图片: {image_path}")
    print(f"图片尺寸: {image.size}")
    print(f"使用权重: {weights_path}")
    print(f"推理设备: {device}")
    print(f"掩码输出: {output_path}")

    if args.save_overlay:
        overlay = build_overlay(image, mask)
        overlay_path = (
            Path(args.overlay_output).expanduser().resolve()
            if args.overlay_output
            else default_overlay_path(output_path)
        )
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path)
        print(f"Overlay 输出: {overlay_path}")


if __name__ == "__main__":
    main()
