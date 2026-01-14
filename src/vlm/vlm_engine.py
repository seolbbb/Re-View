from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import yaml

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm" / "prompts.yaml"
SETTINGS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm" / "settings.yaml"

BATCH_SECTION_RE = re.compile(r"^##\s*Image\s+(\d+)\s*$", re.MULTILINE)
PROVIDER_NAME_RE = re.compile(r'provider_name["\']?:\s*["\']([^"\']+)')
ERROR_CODE_RE = re.compile(r"Error code:\s*(\d+)")
DEFAULT_PROMPT_VERSION = "vlm_v1.1"
DEFAULT_MODEL_NAME = "qwen/qwen3-vl-32b-instruct"


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


def load_prompt_bundle(
    *,
    prompt_version: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """Load prompt text so the VLM engine can be updated without code changes."""
    version_id = prompt_version or DEFAULT_PROMPT_VERSION
    path = prompt_path or PROMPT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VLM 프롬프트 설정을 찾을 수 없습니다: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VLM 프롬프트 설정 형식이 올바르지 않습니다(맵이어야 함).")

    prompt_pair = payload.get(version_id)
    if not isinstance(prompt_pair, dict):
        available = ", ".join(sorted(payload.keys()))
        raise ValueError(f"프롬프트 버전을 찾을 수 없습니다: {version_id} (가능: {available})")
    system = prompt_pair.get("system")
    user = prompt_pair.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError(f"프롬프트 블록이 누락되었습니다: {version_id}")
    return system.strip(), user.strip()


def load_vlm_settings(*, settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load VLM settings from YAML so runtime behavior is configured in one place."""
    path = settings_path or SETTINGS_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VLM 설정 파일을 찾을 수 없습니다: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VLM 설정 형식이 올바르지 않습니다(맵이어야 함).")
    return payload

class OpenRouterVlmExtractor:
    """Vision-capable OpenRouter 모델로 이미지 텍스트/수식 힌트를 추출한다."""

    def __init__(
        self,
        video_name: Optional[str] = None,
        output_root: Path = Path("data/outputs"),
        *,
        prompt_version: Optional[str] = None,
        prompt_path: Optional[Path] = None,
        settings_path: Optional[Path] = None,
    ) -> None:
        if ENV_PATH.exists():
            load_dotenv(ENV_PATH)
        else:
            load_dotenv()
            
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        settings = load_vlm_settings(settings_path=settings_path)
        model_name = settings.get("model_name", DEFAULT_MODEL_NAME)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("VLM 설정의 model_name 형식이 올바르지 않습니다.")
        self.model_name = model_name
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        request_params = settings.get("request_params", {})
        if not isinstance(request_params, dict):
            raise ValueError("VLM 설정의 request_params 형식이 올바르지 않습니다(맵이어야 함).")
        self.request_params = request_params
        settings_prompt_version = settings.get("prompt_version")
        if settings_prompt_version is not None and not isinstance(settings_prompt_version, str):
            raise ValueError("VLM 설정의 prompt_version 형식이 올바르지 않습니다.")
        selected_prompt_version = (
            prompt_version
            or settings_prompt_version
            or DEFAULT_PROMPT_VERSION
        )
        self.system_prompt, self.user_prompt = load_prompt_bundle(
            prompt_version=selected_prompt_version,
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

    def _build_single_messages(self, image_path: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    self._build_image_part(image_path),
                ],
            },
        ]

    def _build_batch_messages(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        content_parts = [{"type": "text", "text": self._build_batch_user_prompt(len(image_paths))}]
        for idx, image_path in enumerate(image_paths, start=1):
            content_parts.append({"type": "text", "text": f"Image {idx}:"})
            content_parts.append(self._build_image_part(image_path))
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content_parts},
        ]

    def _request_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        label: str,
        request_params: Dict[str, Any],
    ) -> str:
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
        if not completion or not completion.choices:
            raise RuntimeError(f"OpenRouter API returned empty response for {label}: {completion}")
        return completion.choices[0].message.content or ""

    def _build_result(self, image_path: str, content: str) -> Dict[str, Any]:
        text = content.strip()
        detections = [{"text": text}] if text else []
        return {"image_path": image_path, "raw_results": detections}

    def _run_batch_request(
        self,
        batch_paths: List[str],
        *,
        label: str,
        request_params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        messages = self._build_batch_messages(batch_paths)
        content = self._request_completion(messages, label=label, request_params=request_params)
        sections = self._split_batch_content(content, len(batch_paths))
        return [
            self._build_result(image_path, section)
            for image_path, section in zip(batch_paths, sections)
        ]

    def _run_single_request(
        self,
        image_path: str,
        *,
        label: str,
        request_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = self._build_single_messages(image_path)
        content = self._request_completion(messages, label=label, request_params=request_params)
        return self._build_result(image_path, content)

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

    def extract_features(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run VLM extraction while keeping orchestration here for traceable flow."""
        if not image_paths:
            return []
        if batch_size is None:
            batch_size = len(image_paths) if len(image_paths) > 1 else 1
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        results: List[Dict[str, Any]] = []
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
                    results.extend(
                        self._run_batch_request(
                            batch_paths,
                            label=f"batch {batch_index}/{total_batches}",
                            request_params=request_params,
                        )
                    )
                    if show_progress:
                        print(f"[VLM] done batch {batch_index}/{total_batches}", flush=True)
                return results

            results_by_index: dict[int, List[Dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {}
                for batch_index, batch_paths in enumerate(batches, start=1):
                    if show_progress:
                        print(
                            f"[VLM] request batch {batch_index}/{total_batches} "
                            f"({len(batch_paths)} images)",
                            flush=True,
                        )
                    future = executor.submit(
                        self._run_batch_request,
                        batch_paths,
                        label=f"batch {batch_index}/{total_batches}",
                        request_params=request_params,
                    )
                    future_map[future] = batch_index
                for future in as_completed(future_map):
                    batch_index = future_map[future]
                    results_by_index[batch_index] = future.result()
                    if show_progress:
                        print(f"[VLM] done batch {batch_index}/{total_batches}", flush=True)

            for batch_index in range(1, total_batches + 1):
                results.extend(results_by_index[batch_index])
            return results

        if concurrency == 1 or total_images == 1:
            for idx, image_path in enumerate(image_paths, start=1):
                if show_progress:
                    image_name = Path(image_path).name
                    print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
                result = self._run_single_request(
                    image_path,
                    label=f"image {idx}/{total_images}",
                    request_params=request_params,
                )
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)
                results.append(result)
            return results

        results_by_index: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {}
            for idx, image_path in enumerate(image_paths, start=1):
                if show_progress:
                    image_name = Path(image_path).name
                    print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
                future = executor.submit(
                    self._run_single_request,
                    image_path,
                    label=f"image {idx}/{total_images}",
                    request_params=request_params,
                )
                future_map[future] = idx
            for future in as_completed(future_map):
                idx = future_map[future]
                results_by_index[idx] = future.result()
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)

        for idx in range(1, total_images + 1):
            results.append(results_by_index[idx])

        return results


def write_vlm_raw_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.vlm.cli import main

    main()
