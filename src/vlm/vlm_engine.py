from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.common.schemas import OcrBox, OcrResult

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

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
PROVIDER_NAME_RE = re.compile(r'provider_name["\']?:\s*["\']([^"\']+)')
ERROR_CODE_RE = re.compile(r"Error code:\s*(\d+)")
PROMPT_SECTION_RE = re.compile(r"^##\s+.+?\(([^)]+)\)\s*$", re.MULTILINE)
PROMPT_BLOCK_RE = re.compile(
    r"^###\s*(SYSTEM|USER)\s*$\n```text\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)

PROMPT_VERSIONS_PATH = Path(__file__).resolve().with_name("prompt_versions.md")
DEFAULT_PROMPT_VERSION = "vlm_v1.1"


def _extract_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    message = str(exc)
    match = ERROR_CODE_RE.search(message)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_provider_name(message: str) -> Optional[str]:
    match = PROVIDER_NAME_RE.search(message)
    if match:
        return match.group(1)
    return None


def _is_service_unavailable_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code == 503:
        return True
    message = str(exc).lower()
    return "service_unavailable" in message or "service unavailable" in message


def _format_service_unavailable_message(exc: Exception) -> str:
    provider = _extract_provider_name(str(exc))
    if provider:
        return (
            f"VLM 제공자({provider})에서 503 오류가 발생했습니다. "
            "잠시 후 다시 시도하세요. 배치 크기와 동시성 값을 낮추면 도움이 될 수 있습니다."
        )
    return (
        "VLM 제공자에서 503 오류가 발생했습니다. "
        "잠시 후 다시 시도하세요. 배치 크기와 동시성 값을 낮추면 도움이 될 수 있습니다."
    )


def _parse_prompt_versions(content: str) -> Dict[str, Dict[str, str]]:
    versions: Dict[str, Dict[str, str]] = {}
    matches = list(PROMPT_SECTION_RE.finditer(content))
    for idx, match in enumerate(matches):
        version_id = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section = content[start:end]
        prompts: Dict[str, str] = {}
        for block in PROMPT_BLOCK_RE.finditer(section):
            name = block.group(1).strip().lower()
            text = block.group(2).strip()
            prompts[name] = text
        if prompts:
            versions[version_id] = prompts
    return versions


def load_prompt_bundle(
    *,
    prompt_version: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> Tuple[str, str]:
    version_id = (
        prompt_version
        or os.getenv("VLM_PROMPT_VERSION")
        or DEFAULT_PROMPT_VERSION
    )
    path = prompt_path or PROMPT_VERSIONS_PATH
    if not path.exists():
        raise FileNotFoundError(f"prompt_versions.md를 찾을 수 없습니다: {path}")
    content = path.read_text(encoding="utf-8")
    versions = _parse_prompt_versions(content)
    if version_id not in versions:
        available = ", ".join(sorted(versions.keys()))
        raise ValueError(f"프롬프트 버전을 찾을 수 없습니다: {version_id} (가능: {available})")
    prompt_pair = versions[version_id]
    if "system" not in prompt_pair or "user" not in prompt_pair:
        raise ValueError(f"프롬프트 블록이 누락되었습니다: {version_id}")
    return prompt_pair["system"], prompt_pair["user"]


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


class OpenRouterVlmExtractor:
    """Vision-capable OpenRouter 모델로 이미지 텍스트/수식 힌트를 추출한다."""

    def __init__(
        self,
        video_name: Optional[str] = None,
        output_root: Path = Path("data/outputs"),
        *,
        prompt_version: Optional[str] = None,
        prompt_path: Optional[Path] = None,
    ) -> None:
        load_env()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = "qwen/qwen3-vl-32b-instruct"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.request_params = dict(DEFAULT_REQUEST_PARAMS)
        self.system_prompt, self.user_prompt = load_prompt_bundle(
            prompt_version=prompt_version,
            prompt_path=prompt_path,
        )
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

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **request_params,
            )
        except Exception as exc:
            if _is_service_unavailable_error(exc):
                raise RuntimeError(_format_service_unavailable_message(exc)) from exc
            raise
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
        concurrency: int = 1,
    ) -> List[OcrResult]:
        """이미지 리스트에 대해 VLM 호출을 수행하고, 결과를 이미지 단위로 반환한다."""
        if not image_paths:
            return []
        if batch_size is None:
            batch_size = len(image_paths) if len(image_paths) > 1 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        results: List[OcrResult] = []
        request_params = {k: v for k, v in self.request_params.items() if v is not None}
        total_images = len(image_paths)

        if show_progress:
            print(
                f"[VLM] start: images={total_images}, batch_size={batch_size}, model={self.model_name}",
                flush=True,
            )
            print(f"[VLM] base_url: {self.base_url}", flush=True)
            print(f"[VLM] concurrency: {concurrency}", flush=True)

        if batch_size > 1:
            batches = [
                image_paths[start : start + batch_size]
                for start in range(0, len(image_paths), batch_size)
            ]
            total_batches = len(batches)
            if concurrency == 1 or total_batches == 1:
                for batch_index, batch_paths in enumerate(batches, start=1):
                    if show_progress:
                        print(
                            f"[VLM] request batch {batch_index}/{total_batches} "
                            f"({len(batch_paths)} images)",
                            flush=True,
                        )
                    results.extend(self._extract_batch(batch_paths, request_params))
                    if show_progress:
                        print(f"[VLM] done batch {batch_index}/{total_batches}", flush=True)
                return results

            results_by_index: dict[int, List[OcrResult]] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {}
                for batch_index, batch_paths in enumerate(batches, start=1):
                    if show_progress:
                        print(
                            f"[VLM] request batch {batch_index}/{total_batches} "
                            f"({len(batch_paths)} images)",
                            flush=True,
                        )
                    future = executor.submit(self._extract_batch, batch_paths, request_params)
                    future_map[future] = batch_index
                for future in as_completed(future_map):
                    batch_index = future_map[future]
                    results_by_index[batch_index] = future.result()
                    if show_progress:
                        print(f"[VLM] done batch {batch_index}/{total_batches}", flush=True)

            for batch_index in range(1, total_batches + 1):
                results.extend(results_by_index[batch_index])
            return results

        def _extract_single(image_path: str) -> OcrResult:
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

            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **request_params,
                )
            except Exception as exc:
                if _is_service_unavailable_error(exc):
                    raise RuntimeError(_format_service_unavailable_message(exc)) from exc
                raise
            content = completion.choices[0].message.content or ""
            if content.strip():
                detections = [OcrBox(text=content, bbox=FULL_IMAGE_BBOX)]
            else:
                detections = []
            return OcrResult(image_path=image_path, raw_results=detections)

        if concurrency == 1 or total_images == 1:
            for idx, image_path in enumerate(image_paths, start=1):
                if show_progress:
                    image_name = Path(image_path).name
                    print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
                result = _extract_single(image_path)
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)
                results.append(result)
            return results

        results_by_index = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {}
            for idx, image_path in enumerate(image_paths, start=1):
                if show_progress:
                    image_name = Path(image_path).name
                    print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
                future = executor.submit(_extract_single, image_path)
                future_map[future] = idx
            for future in as_completed(future_map):
                idx = future_map[future]
                results_by_index[idx] = future.result()
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)

        for idx in range(1, total_images + 1):
            results.append(results_by_index[idx])

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
        "--concurrency",
        type=int,
        default=1,
        help="Parallel batch requests (default: 1).",
    )
    parser.add_argument(
        "--video-name",
        required=True,
        help="data/outputs/{video_name}/vlm_raw.json 경로를 만들 때 사용할 이름",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="prompt_versions.md의 프롬프트 버전 ID (예: vlm_v1.0)",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="prompt_versions.md 경로 (기본: src/vlm/prompt_versions.md)",
    )
    parser.add_argument(
        "--output-root",
        default="data/outputs",
        help="원시 결과 출력 베이스 디렉토리 (기본: data/outputs)",
    )
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path) if args.prompt_path else None
    extractor = OpenRouterVlmExtractor(
        video_name=args.video_name,
        output_root=Path(args.output_root),
        prompt_version=args.prompt_version,
        prompt_path=prompt_path,
    )
    if args.model:
        extractor.model_name = args.model

    results = extractor.extract_features(
        args.image,
        show_progress=True,
        concurrency=args.concurrency,
    )
    output_path = extractor.get_output_path()
    write_vlm_raw_json(results, output_path)
    print(f"[OK] saved to {output_path}")
