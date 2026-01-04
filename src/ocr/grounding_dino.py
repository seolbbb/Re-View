"""Grounding DINO wrapper via Replicate."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import replicate
from dotenv import load_dotenv


DEFAULT_MODEL = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
DEFAULT_QUERY = "illustration, plotted lines, axes, x-axis, y-axis, curve, plot area, figure region"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _resolve_image_input(image_arg: str) -> Tuple[Union[str, IO[bytes]], Optional[IO[bytes]]]:
    path = Path(image_arg).expanduser()
    if path.exists():
        handle = path.open("rb")
        return handle, handle
    return image_arg, None


def _load_image_for_crop(image_arg: str) -> Optional[np.ndarray]:
    path = Path(image_arg).expanduser()
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _extract_detections(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "detections" in payload and isinstance(payload["detections"], list):
        return [item for item in payload["detections"] if isinstance(item, dict)]
    if "output" in payload and isinstance(payload["output"], list):
        return [item for item in payload["output"] if isinstance(item, dict)]
    return []


def _normalize_bbox(
    bbox: List[float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if max(x1, y1, x2, y2) <= 1.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    x1, x2 = sorted([int(round(x1)), int(round(x2))])
    y1, y2 = sorted([int(round(y1)), int(round(y2))])
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
    return sanitized or "item"


def _save_crops(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_dir: Path,
    base_name: str,
) -> List[str]:
    height, width = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for idx, det in enumerate(detections, start=1):
        bbox = det.get("bbox")
        if not isinstance(bbox, list):
            continue
        normalized = _normalize_bbox(bbox, width, height)
        if not normalized:
            continue
        x1, y1, x2, y2 = normalized
        crop = image[y1:y2, x1:x2]
        label = _sanitize_label(str(det.get("label") or "item"))
        filename = f"{base_name}_crop_{idx:02d}_{label}.png"
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), crop)
        saved.append(str(out_path))
    return saved


def run_grounding_dino(
    *,
    image: str,
    query: str,
    box_threshold: float,
    text_threshold: float,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    image_input, handle = _resolve_image_input(image)
    try:
        payload = {
            "image": image_input,
            "query": query,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }
        output = replicate.run(model, input=payload)
    finally:
        if handle:
            handle.close()
    if isinstance(output, dict):
        return output
    return {"output": output}


def _load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grounding DINO on an image via Replicate.")
    parser.add_argument("--image", required=True, help="Image path or URL.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text query to ground.")
    parser.add_argument("--box-threshold", type=float, default=0.15, help="Box confidence threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.15, help="Text confidence threshold.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Replicate model reference.")
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout).")
    parser.add_argument("--crop-dir", default=None, help="Directory to save cropped detections.")
    return parser.parse_args()


def _json_fallback(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> None:
    _load_env()
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN is required.")

    args = _parse_args()
    result = run_grounding_dino(
        image=args.image,
        query=args.query,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        model=args.model,
    )
    detections = _extract_detections(result)
    crop_paths: List[str] = []
    if args.crop_dir:
        image = _load_image_for_crop(args.image)
        if image is None:
            print("[WARN] crop skipped (local image path required).")
        else:
            base_name = Path(args.image).expanduser().stem
            crop_paths = _save_crops(image, detections, Path(args.crop_dir).expanduser(), base_name)

    result_payload: Dict[str, Any] = dict(result)
    if crop_paths:
        result_payload["crops"] = crop_paths

    payload = json.dumps(result_payload, ensure_ascii=False, indent=2, default=_json_fallback)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"[OK] saved to {out_path.resolve()}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
