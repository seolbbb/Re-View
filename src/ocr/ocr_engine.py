import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Optional

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

BATCH_SECTION_RE = re.compile(r"^##\s*Image\s+(\d+)\s*$", re.MULTILINE)


class OpenRouterOcrExtractor:
    """Extract OCR hints via a vision-capable OpenRouter model."""

    def __init__(self, video_name: Optional[str] = None, output_root: Path = Path("data/outputs")) -> None:
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
        self.video_name = video_name
        self.output_root = Path(output_root)

    def _build_batch_user_prompt(self, image_count: int) -> str:
        return (
            "여러 이미지를 순서대로 제공한다. "
            f"이미지는 총 {image_count}장이다. "
            "각 이미지 결과를 `## Image N` 제목으로 구분해 작성하라. "
            "제목은 반드시 `## Image 1`, `## Image 2`처럼 숫자를 붙여 순서대로 출력하고 "
            "이미지 수만큼 섹션을 만들어라.\n"
            f"{self.user_prompt}"
        )

    def get_output_path(self) -> Path:
        if not self.video_name:
            raise ValueError("video_name is required to build the output path.")
        return self.output_root / self.video_name / "vlm.json"

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

    def _split_batch_content(self, content: str, image_count: int) -> List[str]:
        if image_count < 1:
            return []
        matches = list(BATCH_SECTION_RE.finditer(content))
        if not matches:
            return [content.strip()] + [""] * (image_count - 1)

        sections = {}
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            section = content[start:end].strip()
            sections[int(match.group(1))] = section

        return [sections.get(i, "") for i in range(1, image_count + 1)]

    def _extract_batch(self, image_paths: List[str], request_params: dict) -> List[OcrResult]:
        content_parts = [{"type": "text", "text": self._build_batch_user_prompt(len(image_paths))}]
        for idx, image_path in enumerate(image_paths, start=1):
            content_parts.append({"type": "text", "text": f"Image {idx}:"})
            content_parts.append(self._build_image_part(image_path))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content_parts},
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **request_params,
        )
        content = completion.choices[0].message.content or ""
        sections = self._split_batch_content(content, len(image_paths))

        results: List[OcrResult] = []
        for image_path, section in zip(image_paths, sections):
            if section.strip():
                detections = [OcrBox(text=section, bbox=FULL_IMAGE_BBOX)]
            else:
                detections = []
            results.append(OcrResult(image_path=image_path, raw_results=detections))

        return results

    def extract_features(self, image_paths: List[str], batch_size: Optional[int] = None) -> List[OcrResult]:
        """
        Run OCR over provided images and keep text in a single bounding box.

        Args:
            batch_size: Number of images per API request. Defaults to all images per request.

        Returns:
            List[OcrResult]: Structured OCR results per image.
        """
        if not image_paths:
            return []
        if batch_size is None:
            batch_size = len(image_paths) if len(image_paths) > 1 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        results: List[OcrResult] = []
        request_params = {k: v for k, v in self.request_params.items() if v is not None}

        if batch_size > 1:
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                results.extend(self._extract_batch(batch_paths, request_params))
            return results

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

def _write_json(results: List[OcrResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenRouter OCR on one or more images.")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to a local image (repeatable).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the OpenRouter model name.",
    )
    parser.add_argument(
        "--video-name",
        required=True,
        help="Video name used to build data/outputs/{video_name}/vlm.json.",
    )
    args = parser.parse_args()

    extractor = OpenRouterOcrExtractor(video_name=args.video_name)
    if args.model:
        extractor.model_name = args.model

    ocr_results = extractor.extract_features(args.image)
    output_path = extractor.get_output_path()
    _write_json(ocr_results, output_path)
    print(f"Saved output to {output_path}")
