"""Gemini API 연동을 위한 공통 유틸리티 모듈."""

from __future__ import annotations

import logging
import os
import time
import atexit
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ConfigBundle

logger = logging.getLogger(__name__)
REQUEST_HEARTBEAT_SEC = 5
REQUEST_EXECUTOR_WORKERS = max(
    4,
    int(os.getenv("GEMINI_HEARTBEAT_EXECUTOR_WORKERS", "16")),
)
_REQUEST_EXECUTOR = ThreadPoolExecutor(
    max_workers=REQUEST_EXECUTOR_WORKERS,
    thread_name_prefix="gemini-heartbeat",
)


def _shutdown_request_executor() -> None:
    _REQUEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_request_executor)


def _get_timestamp() -> str:
    """[YYYY-MM-DD | HH:MM:SS.mmm] 형식의 타임스탬프를 반환한다."""
    from datetime import datetime
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


# [TEST] Safety imports for fix
from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold


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


def _get_single_api_key(env_base: str) -> Optional[str]:
    keys = _load_gemini_keys([env_base])
    return keys[0] if keys else None


def init_gemini_client(config: ConfigBundle) -> GeminiClientBundle:
    """fusion config 기준으로 Gemini 클라이언트를 생성한다."""
    genai = load_genai()
    from google.genai import types as genai_types  # type: ignore

    llm_cfg = config.raw.llm_gemini
    timeout_ms = max(1, int(llm_cfg.timeout_sec * 1000))
    http_options = genai_types.HttpOptions(timeout=timeout_ms)
    
    if llm_cfg.backend == "developer_api":
        api_keys = _load_gemini_keys(llm_cfg.developer_api.api_key_env_candidates)
        if not api_keys:
            raise ValueError(
                "Developer API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        clients = [
            genai.Client(api_key=api_key, http_options=http_options)
            for api_key in api_keys
        ]
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
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=http_options,
            )
        else:
            api_key = _get_single_api_key(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("API key for Vertex AI express_api_key mode is missing.")
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                api_key=api_key,
                http_options=http_options,
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
) -> tuple[str, int, int]:
    """Gemini 호출 결과의 텍스트, 총 토큰 수, 캐시된 토큰 수를 반환한다."""
    client = client_override or client_bundle.client
    # timeout_sec is configured on Client(http_options=...) in init_gemini_client().
    _ = timeout_sec
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
    }

    # [TEST-START] Disable Safety Filters to fix 'Failed to extract text' error
    config['safety_settings'] = [
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
        SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    ]
    # [TEST-END]
    
    response = client.models.generate_content(
        model=client_bundle.model,
        contents=prompt,
        config=config,
    )
        
    text = extract_text_from_response(response)
    total_tokens = 0
    cached_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
        cached_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
    
    return text, total_tokens, cached_tokens


def run_with_retries(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: List[int],
    context: str = "",
    verbose: bool = False,
) -> tuple[str, int, int]:
    """오류 시 Gemini 호출을 재시도하며 결과, 총 토큰 수, 캐시된 토큰 수를 반환한다."""
    clients = getattr(client_bundle, "clients", None) or [client_bundle.client]
    total_clients = len(clients)
    last_error: Optional[Exception] = None

    for idx, client in enumerate(clients, start=1):
        attempt = 0
        while True:
            label = f" (Key {idx}/{total_clients})" if total_clients > 1 else ""
            indent = "" if context.lstrip().lower().startswith("batch") else "       "
            if verbose:
                print(f"{_get_timestamp()} {indent}[{context}] Sending request... Attempt {attempt+1}/{max_retries+1}{label}", flush=True)
            try:
                started_at = time.monotonic()
                future = _REQUEST_EXECUTOR.submit(
                    generate_content,
                    client_bundle,
                    prompt,
                    response_schema,
                    temperature,
                    response_mime_type,
                    timeout_sec,
                    client_override=client,
                )
                try:
                    while True:
                        elapsed_sec = time.monotonic() - started_at
                        remaining_sec = float(timeout_sec) - elapsed_sec
                        if remaining_sec <= 0:
                            future.cancel()
                            raise TimeoutError(
                                f"Safety timeout exceeded before response completion "
                                f"({timeout_sec}s)."
                            )
                        wait_sec = min(REQUEST_HEARTBEAT_SEC, remaining_sec)
                        try:
                            return future.result(timeout=wait_sec)
                        except FutureTimeoutError:
                            if verbose:
                                elapsed = int(time.monotonic() - started_at)
                                print(
                                    f"{_get_timestamp()} {indent}[{context}] Waiting... "
                                    f"{elapsed}s/{timeout_sec}s{label}",
                                    flush=True,
                                )
                finally:
                    if not future.done():
                        future.cancel()
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
                # Error message simplification
                error_msg = str(exc)
                if "429" in error_msg:
                    simplified_msg = "429 RESOURCE_EXHAUSTED"
                elif "503" in error_msg:
                    simplified_msg = "503 UNAVAILABLE"
                else:
                    try:
                        import ast
                        start = error_msg.find("{")
                        if start != -1:
                            err_dict = ast.literal_eval(error_msg[start:])
                            code = err_dict.get('error', {}).get('code', 'Unknown')
                            status = err_dict.get('error', {}).get('status', 'Unknown')
                            simplified_msg = f"{code} {status}"
                        else:
                            simplified_msg = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg
                    except:
                        simplified_msg = error_msg[:100] + "..." if len(error_msg) > 100 else error_msg

                if simplified_msg:
                    if verbose:
                        print(f"{_get_timestamp()} {indent}[{context}] Failed: {simplified_msg}. Retrying in {sleep_for}s...", flush=True)
                    else:
                        print(
                            f"{_get_timestamp()} ⚠️ {indent}[{context}] "
                            f"Retry {attempt+1}/{max_retries}: {simplified_msg} "
                            f"(backoff={sleep_for}s)",
                            flush=True,
                        )

                sleep_for_sec = max(0.0, float(sleep_for))
                if verbose and sleep_for_sec >= REQUEST_HEARTBEAT_SEC:
                    backoff_started = time.monotonic()
                    backoff_total = int(sleep_for_sec)
                    while True:
                        elapsed = time.monotonic() - backoff_started
                        remaining = sleep_for_sec - elapsed
                        if remaining <= 0:
                            break
                        time.sleep(min(REQUEST_HEARTBEAT_SEC, remaining))
                        elapsed_sec = int(min(sleep_for_sec, time.monotonic() - backoff_started))
                        if elapsed_sec < backoff_total:
                            print(
                                f"{_get_timestamp()} {indent}[{context}] Backoff waiting... "
                                f"{elapsed_sec}s/{backoff_total}s before Attempt "
                                f"{attempt+2}/{max_retries+1}{label}",
                                flush=True,
                            )
                else:
                    time.sleep(sleep_for_sec)
                attempt += 1

    if last_error:
        logger.error(f"GenerateContent failed after {max_retries} retries: {last_error}")
        raise last_error
    raise RuntimeError("GenerateContent failed with no response.")
