from __future__ import annotations

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

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

SYSTEM_PROMPT = (
    "Output only Markdown. Use Markdown tables when layout matters. "
    "Use LaTeX for equations (inline $...$ and block $$...$$). "
    "Do not wrap the output in code fences."
)

USER_PROMPT = (
    "이미지에 포함된 모든 텍스트와 수식을 가능한 한 원문 그대로 옮겨 적어라. "
    "원문 텍스트는 번역하지 말고 원문 언어를 유지하라. "
    "필요한 설명은 한국어로 간결히 작성하라. "
    "레이아웃이 중요하면 Markdown 표/목록을 사용하고, 수식은 LaTeX($...$, $$...$$)로 표기하라. "
    "텍스트가 거의 없거나 그림/그래프 위주라면 시각 요소를 구체적으로 설명하라"
)

FULL_IMAGE_BBOX = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

DEFAULT_REQUEST_PARAMS = {
    "temperature": 0.4,
    "top_p": 0.9,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.1,
    "seed": None,
    "stream": False,
    "stop": ["</s>", "<|endoftext|>"],
}

BATCH_SECTION_RE = re.compile(r"^##\s*Image\s+(\d+)\s*$", re.MULTILINE)


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


class OpenRouterVlmExtractor:
    """Vision-capable OpenRouter 모델로 이미지 텍스트/수식 힌트를 추출한다."""

    def __init__(self, video_name: Optional[str] = None, output_root: Path = Path("data/outputs")) -> None:
        load_env()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = "qwen/qwen3-vl-32b-instruct"
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
        return self.output_root / self.video_name / "vlm_raw.json"

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

    def extract_features(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[OcrResult]:
        """이미지 리스트에 대해 VLM 호출을 수행하고, 결과를 이미지 단위로 반환한다."""
        if not image_paths:
            return []
        if batch_size is None:
            batch_size = len(image_paths) if len(image_paths) > 1 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        results: List[OcrResult] = []
        request_params = {k: v for k, v in self.request_params.items() if v is not None}
        total_images = len(image_paths)

        if show_progress:
            print(
                f"[VLM] start: images={total_images}, batch_size={batch_size}, model={self.model_name}",
                flush=True,
            )
            print(f"[VLM] base_url: {self.base_url}", flush=True)

        if batch_size > 1:
            total_batches = (total_images + batch_size - 1) // batch_size
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                if show_progress:
                    batch_index = (start // batch_size) + 1
                    print(
                        f"[VLM] request batch {batch_index}/{total_batches} "
                        f"({len(batch_paths)} images)",
                        flush=True,
                    )
                results.extend(self._extract_batch(batch_paths, request_params))
                if show_progress:
                    print(f"[VLM] done batch {batch_index}/{total_batches}", flush=True)
            return results

        for idx, image_path in enumerate(image_paths, start=1):
            if show_progress:
                image_name = Path(image_path).name
                print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
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
            if show_progress:
                print(f"[VLM] done {idx}/{total_images}", flush=True)
            if content.strip():
                detections = [OcrBox(text=content, bbox=FULL_IMAGE_BBOX)]
            else:
                detections = []

            results.append(OcrResult(image_path=image_path, raw_results=detections))

        return results


def write_vlm_raw_json(results: List[OcrResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump() for result in results]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenRouter VLM 실행 (이미지 → Markdown 텍스트)")
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
        help="data/outputs/{video_name}/vlm_raw.json 경로를 만들 때 사용할 이름",
    )
    parser.add_argument(
        "--output-root",
        default="data/outputs",
        help="원시 결과 출력 베이스 디렉토리 (기본: data/outputs)",
    )
    args = parser.parse_args()

    extractor = OpenRouterVlmExtractor(video_name=args.video_name, output_root=Path(args.output_root))
    if args.model:
        extractor.model_name = args.model

    results = extractor.extract_features(args.image, show_progress=True)
    output_path = extractor.get_output_path()
    write_vlm_raw_json(results, output_path)
    print(f"[OK] saved to {output_path}")
