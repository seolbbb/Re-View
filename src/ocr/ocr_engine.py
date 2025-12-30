"""
OpenRouter vision-based OCR extraction.

Inputs:
    - image_paths: List of slide image paths.
Outputs:
    - List[OcrResult]: Structured OCR detections per image.
"""

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.common.schemas import OcrBox, OcrResult

SYSTEM_PROMPT = (
    "Output only Markdown. Use Markdown tables when layout matters. "
    "Use LaTeX for equations (inline $...$ and block $$...$$). "
    "Do not wrap the output in code fences."
)

USER_PROMPT = (
    "이미지에 포함된 모든 텍스트와 수식을 가능한 한 원문 그대로 옮겨 적어라. "
    "원문 텍스트는 번역하지 말고 원문 언어를 유지하라. "
    "필요한 설명은 한국어로 간결히 작성하라. "
    "레이아웃이 중요하면 Markdown 표/목록을 사용하고, 수식은 LaTeX($...$, $$...$$)로 표기하라."
)

# LLM output is text-only, so we attach it to a full-image placeholder bbox.
FULL_IMAGE_BBOX = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

DEFAULT_REQUEST_PARAMS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
    "stream": False,
    "stop": None,
}


class OpenRouterOcrExtractor:
    """Extract OCR hints via a vision-capable OpenRouter model."""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("OPENROUTER_OCR_MODEL", "qwen/qwen3-vl-32b-instruct")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.request_params = dict(DEFAULT_REQUEST_PARAMS)

        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT

    def _build_image_part(self, image_path: str) -> dict:
        if image_path.startswith(("http://", "https://")):
            url = image_path
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("ascii")
            url = f"data:{mime_type};base64,{encoded}"

        return {"type": "image_url", "image_url": {"url": url}}

    def extract_features(self, image_paths: List[str]) -> List[OcrResult]:
        """
        Run OCR over provided images and keep text in a single bounding box.

        Returns:
            List[OcrResult]: Structured OCR results per image.
        """
        results: List[OcrResult] = []
        request_params = {k: v for k, v in self.request_params.items() if v is not None}

        for image_path in image_paths:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.user_prompt},
                        self._build_image_part(image_path),
                    ],
                },
            ]

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **request_params,
            )
            content = completion.choices[0].message.content or ""
            if content.strip():
                detections = [OcrBox(text=content, bbox=FULL_IMAGE_BBOX)]
            else:
                detections = []

            results.append(OcrResult(image_path=image_path, raw_results=detections))

        return results

def _write_markdown(results: List[OcrResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for idx, result in enumerate(results, start=1):
            output_file.write(f"## Image {idx}\n")
            output_file.write(f"`{result.image_path}`\n\n")
            if result.raw_results:
                output_file.write(result.raw_results[0].text)
            output_file.write("\n\n")


if __name__ == "__main__":
    # cd ./Lecture-Note-AI/
    # python -m src.ocr.ocr_engine --image <URL> --output ocr_output.md
    parser = argparse.ArgumentParser(description="Run OpenRouter OCR on one or more images.")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path or URL to an image (repeatable).",
    )
    parser.add_argument(
        "--output",
        default="ocr_output.md",
        help="Markdown output file path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the OpenRouter model name.",
    )
    args = parser.parse_args()

    extractor = OpenRouterOcrExtractor()
    if args.model:
        extractor.model_name = args.model

    ocr_results = extractor.extract_features(args.image)
    output_path = Path(args.output)
    _write_markdown(ocr_results, output_path)
    print(f"Saved output to {output_path}")
