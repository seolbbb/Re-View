"""Draw detection boxes using 0-1000 normalized coordinates."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_code_fence(text: str) -> str:
    if "```" not in text:
        return text.strip()
    lines = [line.strip() for line in text.strip().splitlines()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "raw" in payload[0]:
        raw = payload[0].get("raw", "")
        cleaned = _strip_code_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict) and "raw" in payload:
        cleaned = _strip_code_fence(payload.get("raw", ""))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _extract_image_path(payload: Any) -> Optional[str]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        image_path = payload[0].get("image")
        if isinstance(image_path, str) and image_path:
            return image_path
    if isinstance(payload, dict):
        image_path = payload.get("image")
        if isinstance(image_path, str) and image_path:
            return image_path
    return None


def _extract_image_size(payload: Any) -> Optional[Tuple[int, int]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        image_size = payload[0].get("image_size")
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    if isinstance(payload, dict):
        image_size = payload.get("image_size")
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    return None


def _scale_box(
    box: List[float],
    base_size: Optional[Tuple[int, int]],
    actual_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    actual_w, actual_h = actual_size
    base_w, base_h = base_size if base_size else (actual_w, actual_h)
    max_coord = max(box)

    if max_coord <= 1.0:
        sx1 = box[0] * actual_w
        sy1 = box[1] * actual_h
        sx2 = box[2] * actual_w
        sy2 = box[3] * actual_h
    elif max_coord <= 1000:
        sx1 = box[0] / 1000.0 * actual_w
        sy1 = box[1] / 1000.0 * actual_h
        sx2 = box[2] / 1000.0 * actual_w
        sy2 = box[3] / 1000.0 * actual_h
    elif max_coord <= max(base_w, base_h):
        scale_x = actual_w / base_w if base_w else 1.0
        scale_y = actual_h / base_h if base_h else 1.0
        sx1 = box[0] * scale_x
        sy1 = box[1] * scale_y
        sx2 = box[2] * scale_x
        sy2 = box[3] * scale_y
    else:
        sx1, sy1, sx2, sy2 = box

    return (
        int(round(sx1)),
        int(round(sy1)),
        int(round(sx2)),
        int(round(sy2)),
    )


def _draw_boxes(
    image_path: Path,
    items: List[Dict[str, Any]],
    output_path: Path,
    base_size: Optional[Tuple[int, int]],
) -> None:
    image = Image.open(image_path).convert("RGB")
    actual_size = image.size

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for item in items:
        box = item.get("box") or item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        label = str(item.get("label") or "box")
        sx1, sy1, sx2, sy2 = _scale_box([float(v) for v in box], base_size, actual_size)

        draw.rectangle([sx1, sy1, sx2, sy2], outline=(255, 0, 0), width=3)
        text_bbox = draw.textbbox((sx1, sy1), label, font=font)
        pad = 4
        draw.rectangle(
            [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
            fill=(255, 0, 0),
        )
        draw.text((sx1, sy1), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw 0-1000 normalized boxes on an image.")
    parser.add_argument("--image", default=None, help="Path to the source image.")
    parser.add_argument("--json", required=True, help="Path to qwen3_detect JSON output.")
    parser.add_argument("--output", required=True, help="Path to save the boxed image.")
    args = parser.parse_args()

    payload = _load_json(Path(args.json))
    items = _extract_items(payload)
    base_size = _extract_image_size(payload)
    image_path = args.image or _extract_image_path(payload)
    if not image_path:
        raise ValueError("Image path is required (pass --image or include image in JSON).")
    _draw_boxes(Path(image_path), items, Path(args.output), base_size)
    print(f"[OK] saved to {args.output}")


if __name__ == "__main__":
    main()
