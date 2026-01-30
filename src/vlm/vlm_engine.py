"""
VLM 추출 엔진과 OpenRouter 연동 로직
오류 처리 helper는 src/vlm/openrouter_errors.py에 분리되어 있다.
"""

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

from src.vlm.openrouter_errors import (
    extract_error_code,
    format_openrouter_error,
    format_service_unavailable_message,
    is_service_unavailable_error,
)

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm" / "prompts.yaml"
SETTINGS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "vlm" / "settings.yaml"

BATCH_SECTION_RE = re.compile(r"^##\s*Image\s+(\d+)\s*$", re.MULTILINE)
DEFAULT_PROMPT_VERSION = "vlm_v1.1"
DEFAULT_MODEL_NAME = "qwen/qwen3-vl-32b-instruct"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
KEY_LIST_ENV = "OPENROUTER_API_KEYS"


def load_prompt_bundle(
    *,
    prompt_version: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """프롬프트 설정 파일에서 시스템/사용자 프롬프트를 읽는다."""
    version_id = prompt_version or DEFAULT_PROMPT_VERSION
    path = prompt_path or PROMPT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VLM prompt config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid VLM prompt config format (must be a map).")

    prompt_pair = payload.get(version_id)
    if not isinstance(prompt_pair, dict):
        available = ", ".join(sorted(payload.keys()))
        raise ValueError(f"Prompt version not found: {version_id} (available: {available})")
    system = prompt_pair.get("system")
    user = prompt_pair.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError(f"Prompt block missing: {version_id}")
    return system.strip(), user.strip()


def load_vlm_settings(*, settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """VLM 설정 YAML을 로드한다."""
    path = settings_path or SETTINGS_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"VLM settings file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid VLM settings format (must be a map).")
    return payload


def _load_openrouter_keys() -> List[str]:
    """환경변수에서 OpenRouter API 키 목록을 구성한다."""
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
                "OPENROUTER_API_KEY (single) or OPENROUTER_API_KEYS (multiple) must be set."
            )

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        settings = load_vlm_settings(settings_path=settings_path)
        model_name = settings.get("model_name", DEFAULT_MODEL_NAME)
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Invalid model_name format in VLM settings.")
        self.model_name = model_name
        self.clients = [
            OpenAI(base_url=self.base_url, api_key=api_key) for api_key in self.api_keys
        ]
        request_params = settings.get("request_params", {})
        if not isinstance(request_params, dict):
            raise ValueError("Invalid request_params format in VLM settings (must be a map).")
        self.request_params = request_params
        settings_prompt_version = settings.get("prompt_version")
        if settings_prompt_version is not None and not isinstance(settings_prompt_version, str):
            raise ValueError("Invalid prompt_version format in VLM settings.")
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
        show_progress: bool,
    ) -> str:
        """OpenRouter 요청을 실행하고 응답 텍스트를 반환한다.

        - 여러 API 키가 있으면 순서대로 시도한다.
        - show_progress가 켜져 있으면 현재 키 인덱스와 실패 전환을 로그로 남긴다.
        - 502/503은 서비스 장애로 판단해 사용자 친화 메시지로 변환한다.
        - 응답이 비어 있거나 오류 payload가 있으면 예외를 던진다.
        - 정상 응답이면 첫 번째 choice의 content를 반환한다.
        """
        last_error: Optional[Exception] = None
        total_keys = len(self.clients)
        
        # 1. 설정된 모든 API 키를 순회하며 요청 시도 (Round-robin/Failover)
        for idx, client in enumerate(self.clients, start=1):
            if show_progress:
                print(f"[VLM] OpenRouter key {idx}/{total_keys}: {label}", flush=True)
                # [User Request] Detailed Logging
                print(f"[VLM] Request for {label}:", flush=True)
                # messages[0]은 system, messages[1]은 user라 가정 (vlm_engine 로직상)
                if len(messages) >= 2:
                    user_content = messages[1].get("content", "")
                    if isinstance(user_content, list):
                        # 텍스트 파트만 출력 (이미지 데이터는 방대하므로 제외)
                        text_only = [p.get("text") for p in user_content if p.get("type") == "text"]
                        print(f"      Prompt: {text_only}", flush=True)
                    else:
                        print(f"      Prompt: {user_content[:200]}...", flush=True)

            try:
                # 2. 실제 API 호출
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **request_params,
                )
            except Exception as exc:
                last_error = exc
                # 3. 실패 시 다음 키가 있으면 재시도, 없으면 예외 처리
                if idx < total_keys:
                    if show_progress:
                        print(
                            f"[VLM] OpenRouter key {idx} failed, trying next key",
                            flush=True,
                        )
                    continue
                # 4. 서비스 불가 에러(502/503)는 별도 메시지로 변환
                if is_service_unavailable_error(exc):
                    raise RuntimeError(format_service_unavailable_message(exc)) from exc
                raise
            
            # 5. 응답 에러 필드 확인
            error = getattr(completion, "error", None)
            if error:
                last_error = RuntimeError(format_openrouter_error(error))
                if idx < total_keys:
                    if show_progress:
                        print(
                            f"[VLM] OpenRouter key {idx} failed, trying next key",
                            flush=True,
                        )
                    continue
                code = extract_error_code(error)
                if code in (502, 503):
                    message = format_service_unavailable_message(
                        RuntimeError(str(error)),
                        status_code=code,
                    )
                    raise RuntimeError(message) from last_error
                raise last_error
            
            # 6. 빈 응답 처리
            if not completion or not completion.choices:
                last_error = RuntimeError(
                    f"OpenRouter API returned empty response for {label}: {completion}"
                )
                if idx < total_keys:
                    if show_progress:
                        print(
                            f"[VLM] OpenRouter key {idx} failed, trying next key",
                            flush=True,
                        )
                    continue
                raise last_error
            
            # 7. 성공 시 결과 반환
            content = completion.choices[0].message.content or ""
            if show_progress:
                 print(f"[VLM] Response for {label}:\n{content[:500]}...", flush=True)
            return content

        if last_error:
            raise last_error
        raise RuntimeError(f"OpenRouter API returned empty response for {label}: none")

    def _build_result(self, image_path: str, content: str) -> Dict[str, Any]:
        """Convert VLM response to vlm_raw.json format result."""
        text = content.strip()
        detections = [{"text": text}] if text else []
        return {"image_path": image_path, "raw_results": detections}

    def _run_batch_request(
        self,
        batch_paths: List[str],
        *,
        label: str,
        request_params: Dict[str, Any],
        show_progress: bool,
    ) -> List[Dict[str, Any]]:
        """배치 요청을 실행하고 이미지별 결과를 반환한다."""
        messages = self._build_batch_messages(batch_paths)
        content = self._request_completion(
            messages,
            label=label,
            request_params=request_params,
            show_progress=show_progress,
        )
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
        show_progress: bool,
    ) -> Dict[str, Any]:
        """단일 이미지 요청을 실행하고 결과를 반환한다."""
        messages = self._build_single_messages(image_path)
        content = self._request_completion(
            messages,
            label=label,
            request_params=request_params,
            show_progress=show_progress,
        )
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

        # [Case 1] 배치 모드 (batch_size > 1)
        if batch_size > 1:
            # 1. 이미지를 배치 단위로 분할
            batches = [
                image_paths[start : start + batch_size]
                for start in range(0, len(image_paths), batch_size)
            ]
            total_batches = len(batches)

            # 1-1. 순차 실행 (Concurrency 미사용)
            if concurrency == 1 or total_batches == 1:
                for batch_index, batch_paths in enumerate(batches, start=1):
                    if show_progress:
                        print(
                            f"[VLM] request group {batch_index}/{total_batches} "
                            f"({len(batch_paths)} images)",
                            flush=True,
                        )
                    results.extend(
                        self._run_batch_request(
                            batch_paths,
                            label=f"group {batch_index}/{total_batches}",
                            request_params=request_params,
                            show_progress=show_progress,
                        )
                    )
                    if show_progress:
                        print(f"[VLM] done group {batch_index}/{total_batches}", flush=True)
                return results

            # 1-2. 병렬 실행 (ThreadPoolExecutor 사용)
            results_by_index: dict[int, List[Dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_map = {}
                for batch_index, batch_paths in enumerate(batches, start=1):
                    if show_progress:
                        print(
                            f"[VLM] request group {batch_index}/{total_batches} "
                            f"({len(batch_paths)} images)",
                            flush=True,
                        )
                    future = executor.submit(
                        self._run_batch_request,
                        batch_paths,
                        label=f"group {batch_index}/{total_batches}",
                        request_params=request_params,
                        show_progress=show_progress,
                    )
                    future_map[future] = batch_index

                if show_progress:
                    print("-" * 50, flush=True)
                
                # 완료된 순서대로 결과 수집하되, 인덱스로 저장해 나중에 정렬
                for future in as_completed(future_map):
                    batch_index = future_map[future]
                    results_by_index[batch_index] = future.result()
                    if show_progress:
                        print(f"[VLM] done group {batch_index}/{total_batches}", flush=True)

            # 원래 배치 순서대로 결과 병합
            for batch_index in range(1, total_batches + 1):
                results.extend(results_by_index[batch_index])
            return results

        # [Case 2] 단일 모드 (batch_size == 1)
        # 2-1. 순차 실행
        if concurrency == 1 or total_images == 1:
            for idx, image_path in enumerate(image_paths, start=1):
                if show_progress:
                    image_name = Path(image_path).name
                    print(f"[VLM] request {idx}/{total_images}: {image_name}", flush=True)
                result = self._run_single_request(
                    image_path,
                    label=f"image {idx}/{total_images}",
                    request_params=request_params,
                    show_progress=show_progress,
                )
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)
                results.append(result)
            return results

        # 2-2. 병렬 실행
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
                    show_progress=show_progress,
                )
                future_map[future] = idx
            
            # 완료된 순서대로 결과 수집
            for future in as_completed(future_map):
                idx = future_map[future]
                results_by_index[idx] = future.result()
                if show_progress:
                    print(f"[VLM] done {idx}/{total_images}", flush=True)

        # 원래 이미지 순서대로 정렬하여 반환
        for idx in range(1, total_images + 1):
            results.append(results_by_index[idx])

        return results


def write_vlm_raw_json(results: List[Dict[str, Any]], output_path: Path) -> None:
    """VLM 결과를 vlm_raw.json 형태로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
