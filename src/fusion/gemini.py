"""Gemini API 연동을 위한 공통 유틸리티 모듈."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ConfigBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiClientBundle:
    """Gemini 클라이언트와 모델 메타데이터 묶음."""
    client: Any
    clients: List[Any]
    backend: str
    model: str


def _split_env_list(value: str) -> List[str]:
    """환경변수 문자열을 키 리스트로 분해한다."""
    items: List[str] = []
    for part in value.replace(",", " ").split():
        if part.strip():
            items.append(part.strip())
    return items


def _load_gemini_keys(env_candidates: List[str]) -> List[str]:
    """환경변수에서 Gemini API 키 목록을 구성한다."""
    keys: List[str] = []
    for base_name in env_candidates:
        if not isinstance(base_name, str):
            continue
        base = base_name.strip()
        if not base:
            continue

        list_env = os.getenv(f"{base}S", "")
        if list_env:
            keys.extend(_split_env_list(list_env))

        index = 1
        while True:
            key = os.getenv(f"{base}_{index}")
            if not key:
                break
            keys.append(key.strip())
            index += 1

        single_key = os.getenv(base)
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


def load_genai() -> Any:
    """google genai 모듈을 로드하고 오류를 명확히 알린다."""
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai package is required. Check requirements.txt."
        ) from exc
    return genai


def init_gemini_client(config: ConfigBundle) -> GeminiClientBundle:
    """fusion config 기준으로 Gemini 클라이언트를 생성한다."""
    genai = load_genai()
    llm_cfg = config.raw.llm_gemini
    
    if llm_cfg.backend == "developer_api":
        api_keys = _load_gemini_keys(llm_cfg.developer_api.api_key_env_candidates)
        if not api_keys:
            raise ValueError(
                "Developer API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        clients = [genai.Client(api_key=api_key) for api_key in api_keys]
        client = clients[0]
        return GeminiClientBundle(
            client=client,
            clients=clients,
            backend="developer_api",
            model=llm_cfg.model,
        )

    if llm_cfg.backend == "vertex_ai":
        project = llm_cfg.vertex_ai.project
        location = llm_cfg.vertex_ai.location
        if not project or not location:
            raise ValueError(
                "Vertex AI requires project/location. Check config."
            )
        if llm_cfg.vertex_ai.auth_mode == "adc":
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("API key for Vertex AI express_api_key mode is missing.")
            client = genai.Client(
                vertexai=True, project=project, location=location, api_key=api_key
            )
        clients = [client]
        return GeminiClientBundle(
            client=client,
            clients=clients,
            backend="vertex_ai",
            model=llm_cfg.model,
        )

    raise ValueError(f"Unsupported backend: {llm_cfg.backend}")


def extract_text_from_response(response: Any) -> str:
    """Gemini 응답에서 텍스트를 추출한다."""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text
    raise ValueError("Failed to extract text from Gemini response.")


def generate_content(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    *,
    client_override: Optional[Any] = None,
) -> str:
    """Gemini 호출 결과의 텍스트를 반환한다."""
    client = client_override or client_bundle.client
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
    }
    
    # timeout 인자 지원 여부에 따른 처리
    try:
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
            timeout=timeout_sec,
        )
    except TypeError:
        # 구버전이나 타입 에러 시 timeout 제외하고 재시도
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
        )
        
    return extract_text_from_response(response)


def run_with_retries(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: List[int],
) -> str:
    """오류 시 Gemini 호출을 재시도한다."""
    clients = getattr(client_bundle, "clients", None) or [client_bundle.client]
    total_clients = len(clients)
    last_error: Optional[Exception] = None

    for idx, client in enumerate(clients, start=1):
        attempt = 0
        while True:
            try:
                return generate_content(
                    client_bundle,
                    prompt,
                    response_schema,
                    temperature,
                    response_mime_type,
                    timeout_sec,
                    client_override=client,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    if total_clients > 1 and idx < total_clients:
                        logger.warning(
                            f"GenerateContent failed after {max_retries} retries on key "
                            f"{idx}/{total_clients}. Switching keys."
                        )
                    break

                sleep_for = (
                    backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
                )
                logger.warning(
                    f"GenerateContent failed (attempt {attempt+1}/{max_retries}). "
                    f"Retrying in {sleep_for}s. Error: {exc}"
                )
                time.sleep(max(sleep_for, 0))
                attempt += 1

    if last_error:
        logger.error(f"GenerateContent failed after {max_retries} retries: {last_error}")
        raise last_error
    raise RuntimeError("GenerateContent failed with no response.")
