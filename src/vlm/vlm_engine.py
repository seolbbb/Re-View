"""VLM 추출 엔진과 OpenRouter 연동 로직."""

from __future__ import annotations

import argparse
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
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
KEY_LIST_ENV = "OPENROUTER_API_KEYS"


def _extract_status_code(exc: Exception) -> Optional[int]:
    """예외 객체에서 HTTP 상태 코드를 추출한다."""
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
    """에러 메시지에서 제공자 이름을 추출한다."""
    match = PROVIDER_NAME_RE.search(message)
    if match:
        return match.group(1)
    return None


def _is_service_unavailable_error(exc: Exception) -> bool:
    """서비스 불가(503) 오류인지 판별한다."""
    status_code = _extract_status_code(exc)
    if status_code == 503:
        return True
    message = str(exc).lower()
    return "service_unavailable" in message or "service unavailable" in message


def _extract_error_code(error: Any) -> Optional[int]:
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, int):
            return code
        try:
            return int(code)
        except (TypeError, ValueError):
            return None
    return None


def _extract_provider_from_error(error: Any) -> Optional[str]:
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        provider = metadata.get("provider_name")
        if isinstance(provider, str) and provider.strip():
            return provider
    return None


def _format_openrouter_error(error: Any) -> str:
    if isinstance(error, dict):
        message = str(error.get("message", ""))
        code = error.get("code")
        provider = _extract_provider_from_error(error)
        parts = []
        if message:
            parts.append(message)
        if code is not None:
            parts.append(f"code={code}")
        if provider:
            parts.append(f"provider={provider}")
        return "OpenRouter error: " + ", ".join(parts) if parts else "OpenRouter error"
    return f"OpenRouter error: {error}"


def _format_service_unavailable_message(exc: Exception) -> str:
    """503 오류 메시지를 사용자 친화적으로 정리한다."""
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
    """프롬프트 설정 파일에서 시스템/사용자 프롬프트를 읽는다."""
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
    """VLM 설정 YAML을 로드한다."""
    path = settings_path or SETTINGS_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VLM 설정 파일을 찾을 수 없습니다: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("VLM 설정 형식이 올바르지 않습니다(맵이어야 함).")
    return payload


def _load_openrouter_keys() -> List[str]:
    keys: List[str] = []
    list_env = os.getenv(KEY_LIST_ENV, "")
    if list_env:
        for item in re.split(r"[,\s]+", list_env):
            if item.strip():
                keys.append(item.strip())
    index = 1
    while True:
        key = os.getenv(f"{DEFAULT_KEY_ENV}_{index}")
        if not key:
            break
        keys.append(key.strip())
        index += 1
    single_key = os.getenv(DEFAULT_KEY_ENV)
    if single_key:
        keys.append(single_key.strip())
    seen = set()
    deduped = []
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


class OpenRouterVlmExtractor:
    """OpenRouter 기반 VLM으로 이미지 텍스트를 추출한다."""

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

        self.api_keys = _load_openrouter_keys()
        if not self.api_keys:
            raise ValueError(
                "OPENROUTER_API_KEY(단일) 또는 OPENROUTER_API_KEYS(복수)를 설정해야 합니다."
            )

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        settings = load_vlm_settings(settings_path=settings_path)
        model_name = settings.get("model_name", DEFAULT_MODEL_NAME)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("VLM 설정의 model_name 형식이 올바르지 않습니다.")
        self.model_name = model_name
        self.clients = [
            OpenAI(base_url=self.base_url, api_key=api_key) for api_key in self.api_keys
        ]
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
        """배치 처리용 사용자 프롬프트를 구성한다."""
        return (
            "여러 이미지를 순서대로 제공한다. "
            f"이미지는 총 {image_count}장이다. "
            "각 이미지 결과를 `## Image N` 제목으로 구분해 작성하라. "
            "제목은 반드시 `## Image 1`, `## Image 2`처럼 숫자를 붙여 순서대로 출력하고 "
            "이미지 수만큼 섹션을 만들어라.\n"
            f"{self.user_prompt}"
        )

    def get_output_path(self) -> Path:
        """vlm_raw.json이 저장될 경로를 반환한다."""
        if not self.video_name:
            raise ValueError("video_name is required to build the output path.")
        return self.output_root / self.video_name / "vlm_raw.json"

    def _build_image_part(self, image_path: str) -> dict:
        """이미지 파일을 data URL 형태로 변환한다."""
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
        """단일 이미지 요청에 사용할 메시지를 구성한다."""
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
        """배치 요청에 사용할 메시지를 구성한다."""
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
        """OpenRouter 요청을 실행하고 응답 텍스트를 반환한다."""
        last_error: Optional[Exception] = None
        for idx, client in enumerate(self.clients, start=1):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **request_params,
                )
            except Exception as exc:
                last_error = exc
                if idx < len(self.clients):
                    continue
                if _is_service_unavailable_error(exc):
                    raise RuntimeError(_format_service_unavailable_message(exc)) from exc
                raise
            error = getattr(completion, "error", None)
            if error:
                last_error = RuntimeError(_format_openrouter_error(error))
                if idx < len(self.clients):
                    continue
                code = _extract_error_code(error)
                if code == 503:
                    message = _format_service_unavailable_message(
                        RuntimeError(str(error))
                    )
                    raise RuntimeError(message) from last_error
                raise last_error
            if not completion or not completion.choices:
                last_error = RuntimeError(
                    f"OpenRouter API returned empty response for {label}: {completion}"
                )
                if idx < len(self.clients):
                    continue
                raise last_error
            return completion.choices[0].message.content or ""

        if last_error:
            raise last_error
        raise RuntimeError(f"OpenRouter API returned empty response for {label}: none")

    def _build_result(self, image_path: str, content: str) -> Dict[str, Any]:
        """VLM 응답을 vlm_raw.json 형식의 결과로 변환한다."""
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
        """배치 요청을 실행하고 이미지별 결과를 반환한다."""
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
        """단일 이미지 요청을 실행하고 결과를 반환한다."""
        messages = self._build_single_messages(image_path)
        content = self._request_completion(messages, label=label, request_params=request_params)
        return self._build_result(image_path, content)

    def _split_batch_content(self, content: str, image_count: int) -> List[str]:
        """배치 응답을 이미지 수에 맞춰 섹션별로 분리한다."""
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
        """이미지 목록을 VLM에 전달해 결과를 순서대로 반환한다."""
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
    """VLM 결과를 vlm_raw.json 형태로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
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
        help="prompts.yaml의 프롬프트 버전 ID (예: vlm_v1.0)",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="prompts.yaml 경로 (기본: config/vlm/prompts.yaml)",
    )
    parser.add_argument(
        "--settings-path",
        default=None,
        help="settings.yaml 경로 (기본: config/vlm/settings.yaml)",
    )
    parser.add_argument(
        "--output-root",
        default="data/outputs",
        help="원시 결과 출력 베이스 디렉토리 (기본: data/outputs)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI에서 전달된 인자로 VLM 추출을 실행한다."""
    args = parse_args()

    prompt_path = Path(args.prompt_path) if args.prompt_path else None
    settings_path = Path(args.settings_path) if args.settings_path else None
    extractor = OpenRouterVlmExtractor(
        video_name=args.video_name,
        output_root=Path(args.output_root),
        prompt_version=args.prompt_version,
        prompt_path=prompt_path,
        settings_path=settings_path,
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


if __name__ == "__main__":
    main()
