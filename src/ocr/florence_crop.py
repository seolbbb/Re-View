"""
Florence-2 cropping helper.

Uses phrase grounding to find specific visual elements (graphs, charts, etc.),
then exports crops and a JSON manifest.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def _load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _extract_bboxes(parsed: Any) -> Tuple[List[List[float]], List[str]]:
    """
    Extract bounding boxes and labels from Florence-2 outputs.
    Handles standard OD and phrase grounding formats.
    """
    bboxes: List[List[float]] = []
    labels: List[str] = []

    if isinstance(parsed, dict) and "bboxes" in parsed and "labels" in parsed:
        return parsed["bboxes"], parsed["labels"]

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if isinstance(value, list) and value and isinstance(value[0], list):
                for box in value:
                    bboxes.append(box)
                    labels.append(key)
            elif isinstance(value, dict):
                sub_bboxes, sub_labels = _extract_bboxes(value)
                bboxes.extend(sub_bboxes)
                labels.extend(sub_labels)

    return bboxes, labels


def _select_task_token(task_prompt: str) -> str:
    if "<CAPTION_TO_PHRASE_GROUNDING>" in task_prompt:
        return "<CAPTION_TO_PHRASE_GROUNDING>"
    if "<OD>" in task_prompt:
        return "<OD>"
    return "<OD>"


def run_florence(
    image: Image.Image,
    model_name: str,
    task_prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(
        device, torch_dtype
    )
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    task_token = _select_task_token(task_prompt)
    parsed = processor.post_process_generation(
        generated_text, task=task_token, image_size=(image.width, image.height)
    )

    return {"generated_text": generated_text, "parsed": parsed}


def _clamp_box(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def export_crops(
    image: Image.Image,
    parsed: Any,
    output_dir: Path,
    image_name: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bboxes, labels = _extract_bboxes(parsed)

    results = []
    for idx, box in enumerate(bboxes):
        label = labels[idx] if idx < len(labels) else "object"
        safe_label = label.replace(" ", "_").replace("/", "_")
        x1, y1, x2, y2 = _clamp_box(box, image.width, image.height)
        crop = image.crop((x1, y1, x2, y2))

        crop_path = output_dir / f"{image_name}_crop_{idx:03d}_{safe_label}.png"
        crop.save(crop_path)
        results.append(
            {
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "crop_path": str(crop_path),
            }
        )

    return {"items": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Florence-2 and export crops + JSON.")
    parser.add_argument("--image", required=True, help="Local image path.")
    parser.add_argument("--output-dir", default="ocr_crops", help="Directory for crops/JSON.")
    parser.add_argument("--model", default="microsoft/Florence-2-large", help="Model name.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation length.")
    args = parser.parse_args()

    image = _load_image(args.image)

    targets = "illustration, graphs, axes, diagrams, text blocks"
    task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {targets}"

    result = run_florence(
        image=image,
        model_name=args.model,
        task_prompt=task_prompt,
        max_new_tokens=args.max_new_tokens,
    )

    output_dir = Path(args.output_dir)
    image_name = Path(args.image).stem
    crops = export_crops(image, result["parsed"], output_dir, image_name=image_name)

    json_path = output_dir / "detections.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "image": args.image,
                "detections": crops["items"],
                "raw_parsed": result["parsed"],
            },
            handle,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Saved {len(crops['items'])} crops to {output_dir}")
    print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
