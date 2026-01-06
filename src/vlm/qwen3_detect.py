from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

SYSTEM_PROMPT = (
    "Output only JSON. Do not wrap the output in code fences. "
    "If nothing is found, output an empty array."
)

DEFAULT_DETECT_PROMPT = (
    "Detect graphs and handwritten sketches in the image. "
    "Return a JSON array of objects with keys: label, box, description. "
    "box must be [x1, y1, x2, y2] in 0-1000 normalized coordinates."
)

DEFAULT_REQUEST_PARAMS = {
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 512,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
    "stream": False,
    "stop": None,
}


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


class OpenRouterQwen3Detector:
    """OpenRouter Qwen3-VL detector using the same API pattern as vlm_engine."""

    def __init__(self) -> None:
        load_env()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("OPENROUTER_VLM_MODEL") or "qwen/qwen3-vl-32b-instruct"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.request_params = dict(DEFAULT_REQUEST_PARAMS)
        self.system_prompt = SYSTEM_PROMPT

    def _build_image_part(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        url = f"data:{mime_type};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": url}}

    def detect(self, image_path: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    self._build_image_part(image_path),
                ],
            },
        ]

        request_params = {k: v for k, v in self.request_params.items() if v is not None}
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_params,
        )
        return completion.choices[0].message.content or ""

    def extract_features(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: Optional[int] = None,
    ) -> List[dict]:
        if not image_paths:
            return []
        if batch_size is None:
            batch_size = len(image_paths) if len(image_paths) > 1 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        outputs: List[dict] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            for image_path in batch_paths:
                raw = self.detect(image_path, prompt)
                width = height = None
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception:
                    pass
                outputs.append(
                    {
                        "image": image_path,
                        "raw": raw,
                        "image_size": {"width": width, "height": height},
                        "bbox_format": "x1y1x2y2",
                        "coord_space": "norm_1000",
                    }
                )
        return outputs


def _write_json(output_path: Path, raw_text: str, image_path: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_path,
        "raw": raw_text,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenRouter Qwen3-VL detection (image -> JSON text).")
    parser.add_argument("--image", action="append", required=True, help="Path to a local image (repeatable).")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_DETECT_PROMPT,
        help="Detection prompt. Must instruct the model to return JSON.",
    )
    parser.add_argument("--model", default=None, help="Override the OpenRouter model name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for processing.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. If set, writes raw model text to JSON.",
    )
    args = parser.parse_args()

    detector = OpenRouterQwen3Detector()
    if args.model:
        detector.model_name = args.model

    outputs = detector.extract_features(args.image, args.prompt, batch_size=args.batch_size)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] saved to {output_path}")
    else:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
