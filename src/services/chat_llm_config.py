from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass(frozen=True)
class ChatLLMSettings:
    timeout_sec: int
    max_retries: int
    backoff_sec: List[int]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def _env_int_list(name: str, default: str) -> List[int]:
    raw = os.getenv(name, default)
    values: List[int] = []
    normalized = raw.replace(";", ",").replace(" ", ",")
    for part in normalized.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value >= 0:
            values.append(value)
    return values


def _coerce_non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _coerce_backoff_list(value: Any, default: List[int]) -> List[int]:
    if isinstance(value, list):
        parsed = [_coerce_non_negative_int(item, -1) for item in value]
        parsed = [item for item in parsed if item >= 0]
        return parsed if parsed else default
    if isinstance(value, str):
        parsed: List[int] = []
        normalized = value.replace(";", ",").replace(" ", ",")
        for part in normalized.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                parsed_value = int(token)
            except ValueError:
                continue
            if parsed_value >= 0:
                parsed.append(parsed_value)
        return parsed if parsed else default
    return default


def load_chat_llm_settings(
    settings_path: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> ChatLLMSettings:
    default_timeout = max(1, _env_int("CHATBOT_LLM_TIMEOUT_SEC", 90))
    default_retries = max(0, _env_int("CHATBOT_LLM_MAX_RETRIES", 2))
    default_backoff = _env_int_list("CHATBOT_LLM_BACKOFF_SEC", "2,5,10") or [2, 5, 10]

    if not settings_path.exists():
        return ChatLLMSettings(
            timeout_sec=default_timeout,
            max_retries=default_retries,
            backoff_sec=default_backoff,
        )

    try:
        payload = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        if logger:
            logger.warning("Failed to read chat settings yaml (%s): %s", settings_path, exc)
        return ChatLLMSettings(
            timeout_sec=default_timeout,
            max_retries=default_retries,
            backoff_sec=default_backoff,
        )

    if not isinstance(payload, dict):
        if logger:
            logger.warning("Invalid chat settings format in %s. Falling back to defaults.", settings_path)
        return ChatLLMSettings(
            timeout_sec=default_timeout,
            max_retries=default_retries,
            backoff_sec=default_backoff,
        )

    llm_payload = payload.get("llm_gemini", {})
    if not isinstance(llm_payload, dict):
        llm_payload = {}

    timeout_sec = max(1, _coerce_non_negative_int(llm_payload.get("timeout_sec"), default_timeout))
    max_retries = max(0, _coerce_non_negative_int(llm_payload.get("max_retries"), default_retries))
    backoff_sec = _coerce_backoff_list(llm_payload.get("backoff_sec"), default_backoff)

    return ChatLLMSettings(
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )

