"""OpenRouter 오류 파싱과 메시지 정리를 담당한다."""

from __future__ import annotations

import re
from typing import Any, Optional

PROVIDER_NAME_RE = re.compile(r'provider_name["\']?:\s*["\']([^"\']+)')
ERROR_CODE_RE = re.compile(r"Error code:\s*(\d+)")


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


def _extract_provider_from_error(error: Any) -> Optional[str]:
    """OpenRouter error metadata에서 provider_name을 추출한다."""
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    if isinstance(metadata, dict):
        provider = metadata.get("provider_name")
        if isinstance(provider, str) and provider.strip():
            return provider
    return None


def is_service_unavailable_error(exc: Exception) -> bool:
    """502/503 계열 서비스 오류인지 판별한다."""
    status_code = _extract_status_code(exc)
    if status_code in (502, 503):
        return True
    message = str(exc).lower()
    return (
        "service_unavailable" in message
        or "service unavailable" in message
        or "bad_gateway" in message
        or "bad gateway" in message
    )


def format_service_unavailable_message(
    exc: Exception,
    status_code: Optional[int] = None,
) -> str:
    """502/503 오류를 사용자 메시지로 정리한다."""
    provider = _extract_provider_name(str(exc))
    code = status_code or _extract_status_code(exc)
    code_label = f"{code} " if code else ""
    if provider:
        return (
            f"VLM provider ({provider}) error {code_label}occurred. "
            "Please try again in a moment. Reducing batch size and concurrency may help."
        )
    return (
        f"VLM provider error {code_label}occurred. "
        "Please try again in a moment. Reducing batch size and concurrency may help."
    )


def format_openrouter_error(error: Any) -> str:
    """OpenRouter 오류 payload를 사용자 메시지로 정리한다."""
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


def extract_error_code(error: Any) -> Optional[int]:
    """OpenRouter error dict에서 code 값을 추출한다."""
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, int):
            return code
        try:
            return int(code)
        except (TypeError, ValueError):
            return None
    return None
