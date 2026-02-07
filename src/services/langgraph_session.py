from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import yaml

from src.services.summary_backend import ProcessApiBackend, SummaryBackend

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - handled at runtime
    END = None
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None

logger = logging.getLogger(__name__)
_TRACE_LOGGER: Optional[logging.Logger] = None
_HISTORY_LOGGER: Optional[logging.Logger] = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHAT_SETTINGS_PATH = _PROJECT_ROOT / "config" / "chat" / "settings.yaml"
_TRACE_LOG_PATH = Path(os.getenv("CHATBOT_TRACE_LOG", "logs/langgraph_trace.log"))
_HISTORY_LOG_PATH = Path("logs/history.log")

_TIME_TAG_RE = re.compile(r"\[time_ms=(\d+)\]")
_TIME_KR_RE = re.compile(r"(\d+)\s*분\s*(\d+)\s*초")
_TIME_MS_RE = re.compile(r"(\d+)\s*초")
_TIME_MMSS_RE = re.compile(r"\b(\d{1,2})\s*:\s*(\d{2})\b")
_TIME_MIN_ONLY_RE = re.compile(r"(\d+)\s*분")
_BACKEND_AUTHOR = "langgraph_chatbot"
_DEFAULT_MODEL = os.getenv("CHATBOT_LLM_MODEL", "gemini-3-flash-preview")
_DECISION_MODEL = os.getenv("CHATBOT_DECISION_MODEL", "gemini-2.5-flash")
_DEFAULT_TEMPERATURE = os.getenv("CHATBOT_LLM_TEMPERATURE", "0.2")
_DEFAULT_REASONING_MODE = os.getenv("CHATBOT_REASONING_MODE", "flash")
_DEFAULT_ROUTER_MODE = os.getenv("CHATBOT_ROUTER", "rules").strip().lower()
_DEFAULT_OUTPUT_BASE = Path(os.getenv("CHATBOT_OUTPUT_BASE", "data/outputs"))
_MAX_EVIDENCE_UNITS = int(os.getenv("CHATBOT_MAX_EVIDENCE_UNITS", "10"))
_HISTORY_LOG_FORMAT = os.getenv("CHATBOT_HISTORY_LOG_FORMAT", "pretty").strip().lower()
_TRACE_VERBOSE = True
_TRACE_HISTORY = True
_TRACE_HISTORY_FULL = False
_TRACE_SESSION_SEPARATOR = True
_TRACE_SEPARATOR_TEXT = "--------------------------------------------------------------------------------"
_TRACE_HISTORY_TAIL = 6


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


@dataclass(frozen=True)
class ChatLLMSettings:
    timeout_sec: int
    max_retries: int
    backoff_sec: List[int]


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


def _load_chat_llm_settings() -> ChatLLMSettings:
    default_timeout = max(1, _env_int("CHATBOT_LLM_TIMEOUT_SEC", 90))
    default_retries = max(0, _env_int("CHATBOT_LLM_MAX_RETRIES", 2))
    default_backoff = _env_int_list("CHATBOT_LLM_BACKOFF_SEC", "2,5,10") or [2, 5, 10]

    if not _CHAT_SETTINGS_PATH.exists():
        return ChatLLMSettings(
            timeout_sec=default_timeout,
            max_retries=default_retries,
            backoff_sec=default_backoff,
        )

    try:
        payload = yaml.safe_load(_CHAT_SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to read chat settings yaml (%s): %s", _CHAT_SETTINGS_PATH, exc)
        return ChatLLMSettings(
            timeout_sec=default_timeout,
            max_retries=default_retries,
            backoff_sec=default_backoff,
        )

    if not isinstance(payload, dict):
        logger.warning("Invalid chat settings format in %s. Falling back to defaults.", _CHAT_SETTINGS_PATH)
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


_CHAT_LLM_SETTINGS = _load_chat_llm_settings()
_CHAT_LLM_TIMEOUT_SEC = _CHAT_LLM_SETTINGS.timeout_sec
_CHAT_LLM_MAX_RETRIES = _CHAT_LLM_SETTINGS.max_retries
_CHAT_LLM_BACKOFF_SEC = _CHAT_LLM_SETTINGS.backoff_sec


_ENABLE_GRAPH_SUGGESTIONS = os.getenv("CHATBOT_ENABLE_GRAPH_SUGGESTIONS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
_SUGGESTION_MODEL = os.getenv("CHATBOT_SUGGESTION_MODEL", "gemini-2.5-flash")
_SUGGESTION_MAX_ITEMS = max(1, _env_int("CHATBOT_SUGGESTION_MAX_ITEMS", 2))
_SUGGESTION_MAX_CHARS = max(20, _env_int("CHATBOT_SUGGESTION_MAX_CHARS", 60))

_OUT_OF_RANGE_MESSAGE = (
    "죄송합니다. 해당 시간에 대한 요약 정보가 아직 업데이트되지 않았습니다. "
    "요약 작업은 완료되었지만, 특정 시간대의 상세 내용을 제공해 드리지 못하고 있습니다."
)
_NO_SUMMARY_MESSAGE = "아직 요약이 생성되지 않았습니다. 요약 작업을 먼저 실행해 주세요."


def _load_google_api_keys() -> List[str]:
    keys: List[str] = []
    primary = os.getenv("GOOGLE_API_KEY")
    if primary:
        keys.append(primary)
    suffix_items: List[Tuple[str, str]] = []
    for name, value in os.environ.items():
        if name.startswith("GOOGLE_API_KEY_") and value:
            suffix_items.append((name, value))

    def _sort_key(item: Tuple[str, str]) -> Tuple[int, object]:
        suffix = item[0].split("GOOGLE_API_KEY_", 1)[1]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    suffix_items.sort(key=_sort_key)
    for _, value in suffix_items:
        if value not in keys:
            keys.append(value)
    return keys


def _select_google_api_key(keys: List[str]) -> Optional[str]:
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    index = int(time.time()) % len(keys)
    return keys[index]


def _load_genai_modules() -> Tuple[Any, Any]:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is required for LangGraph chatbot.") from exc
    return genai, types


def _create_genai_client(genai: Any, types: Any, api_key: Optional[str]) -> Any:
    timeout_ms = max(1, int(_CHAT_LLM_TIMEOUT_SEC * 1000))
    try:
        http_options = types.HttpOptions(timeout=timeout_ms)
    except Exception as exc:
        raise RuntimeError(
            "Gemini SDK rejected HttpOptions(timeout). "
            "Cannot run chat LLM without timeout protection."
        ) from exc
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_options)
    return genai.Client(http_options=http_options)


def _chat_backoff_for_attempt(attempt: int) -> float:
    if not _CHAT_LLM_BACKOFF_SEC:
        return 1.0
    index = min(attempt, len(_CHAT_LLM_BACKOFF_SEC) - 1)
    return float(_CHAT_LLM_BACKOFF_SEC[index])


def _is_retriable_chat_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "429",
        "503",
        "resource_exhausted",
        "unavailable",
        "timeout",
        "timed out",
        "deadline exceeded",
        "readtimeout",
        "connecttimeout",
    )
    return any(marker in message for marker in markers)


def _get_trace_logger() -> logging.Logger:
    global _TRACE_LOGGER
    if _TRACE_LOGGER:
        return _TRACE_LOGGER
    trace_logger = logging.getLogger("langgraph_trace")
    if not trace_logger.handlers:
        _TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_TRACE_LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False
    _TRACE_LOGGER = trace_logger
    return trace_logger


def _get_history_logger() -> logging.Logger:
    global _HISTORY_LOGGER
    if _HISTORY_LOGGER:
        return _HISTORY_LOGGER
    history_logger = logging.getLogger("langgraph_history")
    if not history_logger.handlers:
        _HISTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_HISTORY_LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        history_logger.addHandler(handler)
    history_logger.setLevel(logging.INFO)
    history_logger.propagate = False
    _HISTORY_LOGGER = history_logger
    return history_logger


def _trace(event: str, **fields: Any) -> None:
    logger = _get_trace_logger()
    if event == "session.message":
        logger.info("=================================")
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    payload = ", ".join(parts)
    logger.info("%s | %s", event, payload)


def _history_log(event: str, **fields: Any) -> None:
    logger = _get_history_logger()
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    if _HISTORY_LOG_FORMAT == "jsonl":
        message = json.dumps(payload, ensure_ascii=False)
    else:
        message = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.info(message)


def _history_separator(text: str) -> None:
    logger = _get_history_logger()
    logger.info(text)


def _shorten(text: str, limit: int = 160) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


def _format_history_items(items: List[Dict[str, Any]], limit: int = 160) -> List[Dict[str, str]]:
    parts: List[Dict[str, str]] = []
    for item in items:
        role = item.get("role") or "unknown"
        content = item.get("content") or ""
        text = _shorten(str(content), limit=limit).replace("\n", " ").replace("\r", " ")
        parts.append({"role": role, "content": text})
    return parts


def _resolve_history(state: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], str]:
    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    history_key = f"history_{reasoning_mode}"
    history = state.get(history_key, [])
    if not isinstance(history, list):
        history = []
    if not history:
        legacy = state.get("history", [])
        if isinstance(legacy, list):
            history = legacy
    return history_key, history, reasoning_mode


def _trace_separator(session_id: str, *, kind: str) -> None:
    if not _TRACE_SESSION_SEPARATOR:
        return
    _history_separator(_TRACE_SEPARATOR_TEXT)


def _trace_history(session_id: str, state: Dict[str, Any], *, reason: str) -> None:
    if not _TRACE_HISTORY:
        return
    history_key, history, reasoning_mode = _resolve_history(state)
    history_len = len(history)
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "reason": reason,
        "reasoning_mode": reasoning_mode,
        "history_key": history_key,
        "history_len": history_len,
    }
    if history_len:
        if _TRACE_HISTORY_FULL:
            items = history
        else:
            tail_count = max(0, _TRACE_HISTORY_TAIL)
            items = history[-tail_count:] if tail_count else []
        if items:
            payload["tail_len"] = len(items)
            payload["tail"] = _format_history_items(items)
    _history_log("session.history", **payload)


class ChatState(TypedDict, total=False):
    message: str
    message_id: str
    cleaned_message: str
    time_ms: Optional[int]
    chat_mode: Optional[str]
    reasoning_mode: Optional[str]
    enrich_decision: str
    summary_decision: str
    response: str
    suggestions: List[str]
    suggestions_source: str
    suggestions_error: Optional[str]
    streaming: bool
    prompt: str
    answer_records: List[Dict[str, Any]]
    history: List[Dict[str, str]]
    history_flash: List[Dict[str, str]]
    history_thinking: List[Dict[str, str]]
    summary_cache: List[Dict[str, Any]]
    pending_updates: List[Dict[str, Any]]
    last_segment_id: int
    video_name: Optional[str]
    video_id: Optional[str]
    video_root: Optional[str]
    output_base: Optional[str]


@dataclass(frozen=True)
class LangGraphMessage:
    author: str
    text: str
    is_final: bool = True
    message_id: Optional[str] = None


def _normalize_reasoning_mode(text: str) -> Optional[str]:
    normalized = text.strip().lower()
    if normalized in {"flash", "flash mode"}:
        return "flash"
    if normalized in {"thinking", "thinking mode"}:
        return "thinking"
    return None


def _parse_input(state: ChatState) -> Dict[str, Any]:
    """사용자 입력을 파싱해 필요한 필드를 구성한다."""
    message = state.get("message", "") or ""
    time_ms: Optional[int] = None
    has_time_tag = False
    match = _TIME_TAG_RE.search(message)
    if match:
        has_time_tag = True
        try:
            time_ms = int(match.group(1))
        except ValueError:
            time_ms = None
        message = _TIME_TAG_RE.sub("", message).strip()
    if time_ms is None:
        match = _TIME_KR_RE.search(message)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            time_ms = (minutes * 60 + seconds) * 1000
        else:
            match = _TIME_MMSS_RE.search(message)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                time_ms = (minutes * 60 + seconds) * 1000
            else:
                match = _TIME_MS_RE.search(message)
                if match:
                    seconds = int(match.group(1))
                    time_ms = seconds * 1000
                else:
                    match = _TIME_MIN_ONLY_RE.search(message)
                    if match:
                        minutes = int(match.group(1))
                        time_ms = minutes * 60 * 1000
    cleaned = message.strip()
    _trace(
        "node.parse_input",
        has_time_tag=has_time_tag,
    )
    return {
        "cleaned_message": cleaned,
        "time_ms": time_ms,
    }


def _prepare_full(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    """요약 캐시/시간 필터를 준비해 answer_records를 구성한다."""
    _trace("node.prepare_full", video_id=state.get("video_id"), time_ms=state.get("time_ms"))
    started = time.monotonic()
    updates = backend.get_summary_updates(state)
    _trace("timing.summary_updates", duration_ms=int((time.monotonic() - started) * 1000))
    if not updates.get("success"):
        return {"response": updates.get("error", "요약 업데이트에 실패했습니다.")}

    time_ms = state.get("time_ms")
    started = time.monotonic()
    context = backend.get_summary_context(state, time_ms=time_ms)
    _trace("timing.summary_context", duration_ms=int((time.monotonic() - started) * 1000))
    if not context.get("success"):
        return {"response": context.get("error", "요약 컨텍스트를 불러오지 못했습니다.")}

    summary_cache = context.get("summary_cache", [])
    if not summary_cache:
        return {"response": _NO_SUMMARY_MESSAGE}

    if time_ms is not None and context.get("out_of_range"):
        return {"response": _OUT_OF_RANGE_MESSAGE}

    # 시간 태그가 있으면 해당 구간만 우선 사용
    matches = context.get("matches") or []
    records = matches or summary_cache
    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    _trace(
        "summary.records",
        total=len(summary_cache),
        matches=len(matches),
        used=len(records),
        reasoning_mode=reasoning_mode,
    )
    return {"answer_records": records}


def _route_after_prepare(state: ChatState) -> str:
    if state.get("response"):
        _trace("route.prepare_full", decision="respond")
        return "respond"
    _trace("route.prepare_full", decision="llm")
    return "llm"


def _resolve_video_root(state: ChatState) -> Optional[Path]:
    video_root = state.get("video_root")
    if video_root:
        return Path(video_root)
    video_name = state.get("video_name")
    if not video_name:
        return None
    output_base = state.get("output_base")
    base_path = Path(output_base) if output_base else _DEFAULT_OUTPUT_BASE
    return (base_path / video_name).resolve()


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    items.append(json.loads(stripped))
                except json.JSONDecodeError:
                    break
    except OSError:
        return []
    return items




def _normalize_evidence_refs(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        items: List[str] = []
        for key in ("transcript_unit_ids", "visual_unit_ids", "stt_ids", "vlm_ids", "cap_ids"):
            nested = value.get(key)
            if isinstance(nested, list):
                items.extend(str(item) for item in nested if str(item).strip())
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _collect_evidence_refs(value: Any) -> List[str]:
    refs: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence_refs":
                refs.extend(_normalize_evidence_refs(item))
            else:
                refs.extend(_collect_evidence_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_evidence_refs(item))
    return refs


def _collect_evidence_ids(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    stt_ids: List[str] = []
    cap_ids: List[str] = []
    seen_stt = set()
    seen_cap = set()
    for record in records:
        summary = record.get("summary") or {}
        for ref in _collect_evidence_refs(summary):
            if ref.startswith("stt_") and ref not in seen_stt:
                seen_stt.add(ref)
                stt_ids.append(ref)
            elif (ref.startswith("cap_") or ref.startswith("vlm_")) and ref not in seen_cap:
                seen_cap.add(ref)
                cap_ids.append(ref)
    return stt_ids, cap_ids


def _collect_source_ref_ids(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    stt_ids: List[str] = []
    cap_ids: List[str] = []
    seen_stt = set()
    seen_cap = set()
    for record in records:
        source_refs = record.get("source_refs") or {}
        stt_list = source_refs.get("stt_ids") or []
        vlm_list = source_refs.get("vlm_ids") or []
        for ref in stt_list:
            if ref and ref not in seen_stt:
                seen_stt.add(ref)
                stt_ids.append(ref)
        for ref in vlm_list:
            if ref and ref not in seen_cap:
                seen_cap.add(ref)
                cap_ids.append(ref)
    return stt_ids, cap_ids


def _format_stt_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": row.get("stt_id"),
        "start_ms": row.get("start_ms"),
        "end_ms": row.get("end_ms"),
        "text": row.get("transcript"),
    }


def _format_vlm_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": row.get("cap_id"),
        "timestamp_ms": row.get("timestamp_ms"),
        "text": row.get("extracted_text"),
    }


def _attach_db_evidence(
    records: List[Dict[str, Any]],
    stt_rows: List[Dict[str, Any]],
    vlm_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stt_map = {row.get("stt_id"): row for row in stt_rows if row.get("stt_id")}
    vlm_map = {row.get("cap_id"): row for row in vlm_rows if row.get("cap_id")}
    enriched: List[Dict[str, Any]] = []
    for record in records:
        summary = record.get("summary") or {}
        refs = _collect_evidence_refs(summary)
        stt_items: List[Dict[str, Any]] = []
        vlm_items: List[Dict[str, Any]] = []
        for ref in refs:
            if ref.startswith("stt_") and ref in stt_map:
                stt_items.append(_format_stt_evidence(stt_map[ref]))
            elif (ref.startswith("cap_") or ref.startswith("vlm_")) and ref in vlm_map:
                vlm_items.append(_format_vlm_evidence(vlm_map[ref]))

        evidence: Dict[str, Any] = {}
        if stt_items:
            evidence["stt"] = stt_items[:_MAX_EVIDENCE_UNITS]
        if vlm_items:
            evidence["vlm"] = vlm_items[:_MAX_EVIDENCE_UNITS]

        if evidence:
            updated = dict(record)
            updated["evidence"] = evidence
            enriched.append(updated)
        else:
            enriched.append(record)
    return enriched

def _format_records(records: List[Dict[str, Any]]) -> str:
    compact: List[Dict[str, Any]] = []
    for record in records:
        compact.append(
            {
                "segment_id": record.get("segment_id") or record.get("segment_index"),
                "start_ms": record.get("start_ms"),
                "end_ms": record.get("end_ms"),
                "summary": record.get("summary"),
                "source_refs": record.get("source_refs"),
                "evidence": record.get("evidence"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _strip_evidence_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_evidence_refs(v) for k, v in value.items() if k != "evidence_refs"}
    if isinstance(value, list):
        return [_strip_evidence_refs(item) for item in value]
    return value


def _sanitize_records_for_flash(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        if "summary" in updated:
            updated["summary"] = _strip_evidence_refs(updated.get("summary"))
        updated.pop("source_refs", None)
        evidence = updated.get("evidence") or {}
        sanitized_evidence: Dict[str, Any] = {}
        for key in ("stt", "vlm"):
            items = evidence.get(key) or []
            cleaned_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cleaned_items.append({k: v for k, v in item.items() if k != "unit_id"})
            if cleaned_items:
                sanitized_evidence[key] = cleaned_items
        if sanitized_evidence:
            updated["evidence"] = sanitized_evidence
        else:
            updated.pop("evidence", None)
        sanitized.append(updated)
    return sanitized


def _build_prompt(state: ChatState) -> str:
    question = state.get("cleaned_message", "").strip()
    records = state.get("answer_records") or []
    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    history_key = f"history_{reasoning_mode}"
    history = state.get(history_key, [])
    if not history:
        history = state.get("history", [])
    history_text = ""
    if isinstance(history, list) and history:
        tail = history[-6:]
        lines = [f"{item.get('role')}: {item.get('content')}" for item in tail]
        history_text = "\n".join(lines)
    records_for_prompt = records
    if reasoning_mode == "flash":
        records_for_prompt = _sanitize_records_for_flash(records)
    summary_json = _format_records(records_for_prompt)
    if reasoning_mode == "thinking":
        prompt = (
            "You are the Summary Chatbot.\n"
            "Always respond in Korean.\n"
            "Use only the provided summary records and evidence to answer.\n"
            "Explain evidence in natural language. Do not reveal internal IDs (e.g., segment_id, cap_001, stt_001).\n"
            "If summaries are missing but evidence is sufficient, provide a reasonable interpretation based on the evidence.\n"
            "When answering, connect claims to evidence and explain the reasoning briefly.\n"
            "If the answer is not present, say you cannot find it.\n"
        )
    else:
        prompt = (
            "You are the Summary Chatbot.\n"
            "Always respond in Korean.\n"
            "Use the provided summary records and evidence if available.\n"
            "Explain evidence in natural language. Do not reveal internal IDs (e.g., segment_id, cap_001, stt_001).\n"
            "If summaries are missing but evidence is sufficient, provide a reasonable interpretation based on the evidence.\n"
            "If the answer is not present, say you cannot find it.\n"
        )
    if history_text:
        prompt += f"\nRecent conversation:\n{history_text}\n"
    prompt += f"\nUser question:\n{question}\n\nSummary records:\n{summary_json}\n"
    return prompt


def _build_enrichment_prompt(state: ChatState) -> str:
    question = state.get("cleaned_message", "").strip()
    records = state.get("answer_records") or []
    condensed: List[Dict[str, Any]] = []
    for record in records[:6]:
        summary = record.get("summary") or {}
        bullets = summary.get("bullets") or []
        claims = [item.get("claim") for item in bullets if item.get("claim")]
        condensed.append(
            {
                "segment_id": record.get("segment_id") or record.get("segment_index"),
                "claims": claims[:6],
            }
        )
    payload = json.dumps(condensed, ensure_ascii=False, indent=2)
    return (
        "You are deciding whether the current summaries are sufficient to answer the user.\n"
        "If the summaries are likely insufficient and you should fetch extra evidence, reply with 'enrich'.\n"
        "If the summaries are sufficient, reply with 'answer'.\n"
        "Return only one word: enrich or answer.\n"
        "\n"
        f"Question:\n{question}\n\n"
        f"Summary claims:\n{payload}\n"
    )


def _build_summary_sufficiency_prompt(state: ChatState) -> str:
    """요약만으로 답변 가능 여부를 판단하기 위한 프롬프트."""
    question = state.get("cleaned_message", "").strip()
    records = state.get("answer_records") or []
    condensed: List[Dict[str, Any]] = []
    for record in records[:8]:
        summary = record.get("summary") or {}
        bullets = summary.get("bullets") or []
        claims = [item.get("claim") for item in bullets if item.get("claim")]
        condensed.append(
            {
                "segment_id": record.get("segment_id") or record.get("segment_index"),
                "claims": claims[:8],
            }
        )
    payload = json.dumps(condensed, ensure_ascii=False, indent=2)
    return (
        "Decide if the summaries alone are sufficient to answer the user.\n"
        "If sufficient, reply 'answer'. If not, reply 'need_evidence'.\n"
        "Return only one word: answer or need_evidence.\n"
        "\n"
        f"Question:\n{question}\n\n"
        f"Summary claims:\n{payload}\n"
    )


def _normalize_summary_sufficiency(value: str) -> str:
    cleaned = re.sub(r"[^a-z_]", "", value.strip().lower())
    if cleaned == "need_evidence":
        return "need_evidence"
    return "answer"


def _decide_summary_sufficiency(state: ChatState) -> Dict[str, Any]:
    """요약만으로 충분한지 판단해 summary_decision을 반환한다."""
    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    _trace("node.decide_summary", reasoning_mode=reasoning_mode)
    if reasoning_mode == "thinking":
        _trace("summary.decide", decision="need_evidence", reason="thinking_mode")
        return {"summary_decision": "need_evidence"}
    prompt = _build_summary_sufficiency_prompt(state)
    try:
        started = time.monotonic()
        raw = _generate_with_genai(prompt, model=_DECISION_MODEL)
        _trace(
            "timing.decide_summary",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        logger.warning("Summary sufficiency decision failed: %s", exc)
        _trace("summary.decide", decision="answer", error=str(exc))
        return {"summary_decision": "answer"}
    decision = _normalize_summary_sufficiency(raw)
    _trace("summary.decide", decision=decision, raw=raw)
    return {"summary_decision": decision}


def _route_after_summary_sufficiency(state: ChatState) -> str:
    if state.get("summary_decision") == "need_evidence":
        return "need_evidence"
    return "answer"


def _normalize_enrich_decision(value: str) -> str:
    cleaned = re.sub(r"[^a-z]", "", value.strip().lower())
    return "enrich" if cleaned == "enrich" else "answer"


def _decide_enrichment(state: ChatState) -> Dict[str, Any]:
    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    if reasoning_mode == "thinking":
        _trace("enrich.decide", decision="enrich", reason="thinking_mode")
        return {"enrich_decision": "enrich"}
    prompt = _build_enrichment_prompt(state)
    try:
        started = time.monotonic()
        raw = _generate_with_genai(prompt, model=_DECISION_MODEL)
        _trace(
            "timing.decide_enrich",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        logger.warning("Enrichment decision failed: %s", exc)
        _trace("enrich.decide", decision="answer", error=str(exc))
        return {"enrich_decision": "answer"}
    decision = _normalize_enrich_decision(raw)
    _trace("enrich.decide", decision=decision, raw=raw)
    return {"enrich_decision": decision}


def _route_after_enrich_decision(state: ChatState) -> str:
    if state.get("enrich_decision") == "enrich":
        return "enrich"
    return "answer"


def _enrich_with_db_evidence(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    """summary의 source_refs를 기반으로 STT/VLM evidence를 조회해 병합한다."""
    _trace("node.enrich_evidence")
    records = state.get("answer_records") or []
    stt_ids, cap_ids = _collect_source_ref_ids(records)
    if not stt_ids and not cap_ids:
        stt_ids, cap_ids = _collect_evidence_ids(records)
    if not stt_ids and not cap_ids:
        _trace("enrich.evidence", status="skip", stt_ids=0, cap_ids=0)
        return {}
    started = time.monotonic()
    evidence = backend.get_evidence(state, stt_ids=stt_ids, cap_ids=cap_ids)
    _trace("timing.enrich_evidence", duration_ms=int((time.monotonic() - started) * 1000))
    if not evidence.get("success"):
        logger.warning("Evidence lookup failed: %s", evidence.get("error"))
        _trace("enrich.evidence", status="error", error=evidence.get("error"))
        return {}
    stt_rows = len(evidence.get("stt") or [])
    vlm_rows = len(evidence.get("vlm") or [])
    if _TRACE_VERBOSE:
        _trace(
            "enrich.evidence",
            status="ok",
            stt_ids=len(stt_ids),
            cap_ids=len(cap_ids),
            stt_rows=stt_rows,
            vlm_rows=vlm_rows,
            stt_id_list=",".join(stt_ids),
            cap_id_list=",".join(cap_ids),
        )
    else:
        _trace(
            "enrich.evidence",
            status="ok",
            stt_ids=len(stt_ids),
            cap_ids=len(cap_ids),
            stt_rows=stt_rows,
            vlm_rows=vlm_rows,
        )
    enriched = _attach_db_evidence(
        records,
        evidence.get("stt", []),
        evidence.get("vlm", []),
    )
    return {"answer_records": enriched}


def _generate_with_genai(prompt: str, *, model: Optional[str] = None) -> str:
    genai, types = _load_genai_modules()
    api_keys = _load_google_api_keys()
    key_pool: List[Optional[str]] = api_keys if api_keys else [None]
    start_index = int(time.time()) % len(key_pool)
    try:
        temperature = float(_DEFAULT_TEMPERATURE)
    except ValueError:
        temperature = 0.2
    config = types.GenerateContentConfig(temperature=temperature)
    total_attempts = _CHAT_LLM_MAX_RETRIES + 1
    last_error: Optional[Exception] = None

    for attempt in range(total_attempts):
        pool_index = (start_index + attempt) % len(key_pool)
        api_key = key_pool[pool_index]
        client = _create_genai_client(genai, types, api_key)
        try:
            response = client.models.generate_content(
                model=model or _DEFAULT_MODEL,
                contents=prompt,
                config=config,
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip()
            return str(response)
        except Exception as exc:
            last_error = exc
            retriable = _is_retriable_chat_error(exc)
            if attempt >= total_attempts - 1 or not retriable:
                raise
            sleep_for = _chat_backoff_for_attempt(attempt)
            logger.warning(
                "LangGraph LLM call retry %s/%s due to %s. Backoff %.1fs.",
                attempt + 1,
                _CHAT_LLM_MAX_RETRIES,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_error:
        raise last_error
    raise RuntimeError("LangGraph LLM call failed with no response.")


def _extract_genai_text(response: Any) -> str:
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
    return ""


def _generate_with_genai_stream(prompt: str) -> Any:
    genai, types = _load_genai_modules()
    api_keys = _load_google_api_keys()
    key_pool: List[Optional[str]] = api_keys if api_keys else [None]
    start_index = int(time.time()) % len(key_pool)
    try:
        temperature = float(_DEFAULT_TEMPERATURE)
    except ValueError:
        temperature = 0.2
    config = types.GenerateContentConfig(temperature=temperature)
    total_attempts = _CHAT_LLM_MAX_RETRIES + 1
    last_error: Optional[Exception] = None

    for attempt in range(total_attempts):
        pool_index = (start_index + attempt) % len(key_pool)
        api_key = key_pool[pool_index]
        client = _create_genai_client(genai, types, api_key)
        yielded_any = False
        try:
            stream = client.models.generate_content_stream(
                model=_DEFAULT_MODEL,
                contents=prompt,
                config=config,
            )
            buffer = ""
            for chunk in stream:
                text = _extract_genai_text(chunk)
                if not text:
                    continue
                if buffer and text.startswith(buffer):
                    delta = text[len(buffer):]
                    buffer = text
                else:
                    delta = text
                    buffer += text
                if delta:
                    yielded_any = True
                    yield delta
            return
        except Exception as exc:
            last_error = exc
            if yielded_any:
                raise
            retriable = _is_retriable_chat_error(exc)
            if attempt >= total_attempts - 1 or not retriable:
                raise
            sleep_for = _chat_backoff_for_attempt(attempt)
            logger.warning(
                "LangGraph streaming LLM retry %s/%s due to %s. Backoff %.1fs.",
                attempt + 1,
                _CHAT_LLM_MAX_RETRIES,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_error:
        raise last_error
    raise RuntimeError("LangGraph streaming LLM call failed with no response.")


def _fallback_answer(records: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for record in records:
        summary = record.get("summary") or {}
        bullets = summary.get("bullets") or []
        for bullet in bullets:
            claim = bullet.get("claim")
            if claim:
                lines.append(f"- {claim}")
    if lines:
        snippet = "\n".join(lines[:10])
        return f"요약 기반 답변(간단):\n{snippet}"
    return "요약 데이터에서 답변할 근거를 찾지 못했습니다."


def _count_evidence(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    stt_total = 0
    vlm_total = 0
    for record in records:
        evidence = record.get("evidence") or {}
        stt_total += len(evidence.get("stt") or [])
        vlm_total += len(evidence.get("vlm") or [])
    return stt_total, vlm_total


def _generate_answer(state: ChatState) -> Dict[str, Any]:
    """LLM 호출을 수행하고 응답/히스토리를 갱신한다."""
    records = state.get("answer_records") or []
    stt_count, vlm_count = _count_evidence(records)
    started = time.monotonic()
    prompt = _build_prompt(state)
    _trace("timing.prompt_build", duration_ms=int((time.monotonic() - started) * 1000))
    if state.get("streaming"):
        _trace(
            "llm.prepare",
            streaming=True,
            summary_records=len(records),
            stt_evidence=stt_count,
            vlm_evidence=vlm_count,
        )
        return {"prompt": prompt, "answer_records": records}
    try:
        started = time.monotonic()
        response_text = _generate_with_genai(prompt)
        elapsed = time.monotonic() - started
    except Exception as exc:
        logger.warning("LangGraph LLM call failed: %s", exc)
        response_text = _fallback_answer(records)
        elapsed = None
    _trace(
        "llm.prepare",
        streaming=False,
        summary_records=len(records),
        stt_evidence=stt_count,
        vlm_evidence=vlm_count,
    )
    if elapsed is not None:
        _trace(
            "llm.call",
            streaming=False,
            duration_ms=int(elapsed * 1000),
            summary_records=len(records),
            stt_evidence=stt_count,
            vlm_evidence=vlm_count,
        )

    return {"response": response_text}


def _build_suggestions_prompt(state: ChatState) -> str:
    question = (state.get("cleaned_message") or "").strip()
    answer = (state.get("response") or "").strip()
    return (
        "You generate follow-up question chips for a chat UI.\n"
        "Return JSON ONLY in this exact format: {\"questions\": [\"question1\", \"question2\"]}.\n"
        "\n"
        "Rules:\n"
        "- Always write in Korean.\n"
        f"- Return 1 to {_SUGGESTION_MAX_ITEMS} concise questions.\n"
        f"- Keep each question within {_SUGGESTION_MAX_CHARS} characters.\n"
        "- Questions must be natural follow-ups based on the assistant's answer.\n"
        "- Avoid duplicates and generic prompts.\n"
        "\n"
        "Punctuation Rules (CRITICAL):\n"
        "1. Interrogative questions (의문문): Use ONLY '?' at the end\n"
        "   - Question words: 왜, 무엇을, 뭐, 어디서, 언제, 어떻게, 누가\n"
        "   - Question endings: ~인가요, ~나요, ~까요, ~인지, ~ㄹ까요\n"
        "   - Examples: \"왜 그런가요?\", \"어떻게 사용하나요?\", \"차이점이 뭐예요?\"\n"
        "2. Imperative/request sentences (명령문/청유문): Use ONLY '.' at the end\n"
        "   - Request endings: ~해 주세요, ~알려 주세요, ~설명해 주세요, ~비교해 주세요\n"
        "   - Examples: \"설명해 주세요.\", \"예시를 알려 주세요.\", \"비교해 주세요.\"\n"
        "3. NEVER use mixed punctuation like '.?', '?.', '??', or '..' at the end\n"
        "4. NEVER omit punctuation - every question must end with either '?' or '.'\n"
        "\n"
        "Few-shot Examples:\n"
        "\n"
        "Example 1:\n"
        "User: \"Stable Diffusion이 뭐야?\"\n"
        "Assistant: \"Stable Diffusion은 텍스트를 이미지로 변환하는 AI 모델입니다...\"\n"
        "Good Output:\n"
        "{\"questions\": [\"어떻게 사용하나요?\", \"다른 이미지 생성 AI와 비교해 주세요.\"]}\n"
        "Bad Output:\n"
        "{\"questions\": [\"어떻게 사용하나요??\", \"비교해 주세요?.\"]}\n"
        "\n"
        "Example 2:\n"
        "User: \"파이썬 리스트 컴프리헨션 설명해줘\"\n"
        "Assistant: \"리스트 컴프리헨션은 간결하게 리스트를 생성하는 문법입니다...\"\n"
        "Good Output:\n"
        "{\"questions\": [\"언제 사용하면 좋나요?\", \"실전 예제를 보여 주세요.\"]}\n"
        "Bad Output:\n"
        "{\"questions\": [\"언제 사용하면 좋나요.\", \"실전 예제를 보여 주세요?\"]}\n"
        "\n"
        "Example 3:\n"
        "User: \"Docker와 VM의 차이는?\"\n"
        "Assistant: \"Docker는 컨테이너 기반이고 VM은 하이퍼바이저 기반입니다...\"\n"
        "Good Output:\n"
        "{\"questions\": [\"어떤 상황에서 Docker를 쓰나요?\", \"성능 차이를 알려 주세요.\"]}\n"
        "Bad Output:\n"
        "{\"questions\": [\"어떤 상황에서 Docker를 쓰나요\", \"성능 차이를 알려 주세요??\"]}\n"
        "\n"
        f"User question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Generate follow-up questions in JSON format:\n"
    )


def _extract_questions_from_text(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    fence_match = re.search(r"```(?:json)?\s*(.+?)\s*```", stripped, re.IGNORECASE | re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else stripped

    parsed: Any = None
    brace_start = candidate.find("{")
    brace_end = candidate.rfind("}")
    bracket_start = candidate.find("[")
    bracket_end = candidate.rfind("]")
    for token in (
        candidate,
        candidate[brace_start : brace_end + 1] if brace_start != -1 and brace_end > brace_start else "",
        candidate[bracket_start : bracket_end + 1] if bracket_start != -1 and bracket_end > bracket_start else "",
    ):
        token = token.strip()
        if not token:
            continue
        try:
            parsed = json.loads(token)
            break
        except json.JSONDecodeError:
            continue

    raw_items: List[Any] = []
    if isinstance(parsed, dict):
        items = parsed.get("questions")
        if isinstance(items, list):
            raw_items = items
    elif isinstance(parsed, list):
        raw_items = parsed
    elif candidate:
        raw_items = [line.strip("- ").strip() for line in candidate.splitlines() if line.strip()]

    def _normalize_chip_text(value: str) -> str:
        """
        Minimal normalization for suggestion chips.
        Relies on LLM prompt to generate correct punctuation.
        This function acts as a safety net for edge cases.
        """
        # Basic whitespace normalization
        text_value = re.sub(r"\s+", " ", value.strip())

        # Remove surrounding quotes
        text_value = text_value.strip("'\"""''")

        if not text_value:
            return ""

        # Remove duplicate punctuation (e.g., "??" -> "?", ".." -> ".")
        text_value = re.sub(r"([?!.])\1+$", r"\1", text_value)

        # Enforce length limit
        if len(text_value) > _SUGGESTION_MAX_CHARS:
            text_value = text_value[:_SUGGESTION_MAX_CHARS].rstrip()

        # Fix mixed punctuation (safety net for extreme cases)
        if text_value.endswith(".?") or text_value.endswith("?."):
            # Remove all trailing punctuation and add '?'
            core = re.sub(r"[.!?]+$", "", text_value).strip()
            if not core:
                return ""
            logger.warning(f"Mixed punctuation detected in suggestion: '{value}' -> fixed to '{core}?'")
            return f"{core}?"

        # If punctuation is missing, log a warning but don't add it automatically
        # This helps us monitor LLM prompt effectiveness
        if not text_value.endswith(("?", "!", ".")):
            logger.warning(f"Missing punctuation in suggestion: '{text_value}' - relying on LLM to fix this")
            # Add a fallback punctuation to avoid breaking UI
            return f"{text_value}?"

        return text_value

    questions: List[str] = []
    seen = set()
    for item in raw_items:
        question = _normalize_chip_text(str(item or ""))
        if not question:
            continue
        if question in seen:
            continue
        seen.add(question)
        questions.append(question)
        if len(questions) >= _SUGGESTION_MAX_ITEMS:
            break
    return questions


def _generate_suggestions(state: ChatState) -> Dict[str, Any]:
    response_text = (state.get("response") or "").strip()
    if not response_text or not _ENABLE_GRAPH_SUGGESTIONS:
        return {"suggestions": [], "suggestions_source": "graph_node"}
    prompt = _build_suggestions_prompt(state)
    try:
        started = time.monotonic()
        raw = _generate_with_genai(prompt, model=_SUGGESTION_MODEL)
        _trace(
            "timing.suggestions",
            duration_ms=int((time.monotonic() - started) * 1000),
            model=_SUGGESTION_MODEL,
        )
    except Exception as exc:
        logger.warning("Suggestion generation failed: %s", exc)
        _trace("suggestions.generate", status="error", error=str(exc))
        return {
            "suggestions": [],
            "suggestions_source": "graph_node",
            "suggestions_error": str(exc),
        }

    questions = _extract_questions_from_text(raw)
    _trace("suggestions.generate", status="ok" if questions else "empty", count=len(questions))
    return {
        "suggestions": questions,
        "suggestions_source": "graph_node",
        "suggestions_error": None if questions else "empty",
    }


def _finalize_history(state: ChatState) -> Dict[str, Any]:
    question_text = (state.get("cleaned_message") or "").strip()
    response_text = (state.get("response") or "").strip()
    if not question_text or not response_text:
        return {}

    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    history_key = f"history_{reasoning_mode}"
    history = state.get(history_key, [])
    if not isinstance(history, list):
        history = []
    if len(history) >= 2:
        previous_user = history[-2]
        previous_assistant = history[-1]
        if (
            previous_user.get("role") == "user"
            and previous_assistant.get("role") == "assistant"
            and previous_user.get("content") == question_text
            and previous_assistant.get("content") == response_text
        ):
            return {history_key: history}

    history = history + [
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": response_text},
    ]
    return {history_key: history}


def _build_agent_graph(backend: SummaryBackend) -> Any:
    """full 전용 최소 그래프를 구성한다 (parse → 요약 준비 → 답변)."""
    if _LANGGRAPH_IMPORT_ERROR:
        raise RuntimeError(
            "langgraph is required for LangGraphSession. Install it or use CHATBOT_BACKEND=adk."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph: Any = StateGraph(ChatState)
    graph.add_node("parse_input", _parse_input)
    graph.add_node("prepare_full", lambda state: _prepare_full(state, backend))
    graph.add_node("decide_summary", _decide_summary_sufficiency)
    graph.add_node("enrich_evidence", lambda state: _enrich_with_db_evidence(state, backend))
    graph.add_node("generate_answer", _generate_answer)
    graph.add_node("generate_suggestions", _generate_suggestions)
    graph.add_node("finalize_history", _finalize_history)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "prepare_full")
    graph.add_conditional_edges(
        "prepare_full",
        _route_after_prepare,
        {
            "respond": "generate_suggestions",
            "llm": "decide_summary",
        },
    )
    graph.add_conditional_edges(
        "decide_summary",
        _route_after_summary_sufficiency,
        {
            "need_evidence": "enrich_evidence",
            "answer": "generate_answer",
        },
    )
    graph.add_edge("enrich_evidence", "generate_answer")
    graph.add_edge("generate_answer", "generate_suggestions")
    graph.add_edge("generate_suggestions", "finalize_history")
    graph.add_edge("finalize_history", END)
    return graph.compile()


def _build_post_answer_graph() -> Any:
    if _LANGGRAPH_IMPORT_ERROR:
        raise RuntimeError(
            "langgraph is required for LangGraphSession. Install it or use CHATBOT_BACKEND=adk."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph: Any = StateGraph(ChatState)
    graph.add_node("generate_suggestions", _generate_suggestions)
    graph.add_node("finalize_history", _finalize_history)
    graph.set_entry_point("generate_suggestions")
    graph.add_edge("generate_suggestions", "finalize_history")
    graph.add_edge("finalize_history", END)
    return graph.compile()


class LangGraphSession:
    """LangGraph 기반 챗봇 세션. 상태/그래프를 내부에 보관한다."""
    def __init__(
        self,
        *,
        app_name: str,
        user_id: str,
        backend: Optional[SummaryBackend] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._app_name = app_name
        self._user_id = user_id
        self._session_id = f"langgraph-{uuid.uuid4().hex}"
        self._backend = backend or ProcessApiBackend()
        state_seed = dict(initial_state or {})
        router_mode = (state_seed.get("router_mode") or _DEFAULT_ROUTER_MODE).strip().lower()
        self._graph = _build_agent_graph(self._backend)
        self._post_answer_graph = _build_post_answer_graph()
        self._last_suggestions: Optional[Dict[str, Any]] = None
        self._state = state_seed
        self._state.setdefault("summary_cache", [])
        self._state.setdefault("pending_updates", [])
        self._state.setdefault("last_segment_id", 0)
        self._state.setdefault("history", [])
        self._state.setdefault("history_flash", [])
        self._state.setdefault("history_thinking", [])
        self._state.setdefault("router_mode", router_mode)
        default_reasoning_mode = _normalize_reasoning_mode(_DEFAULT_REASONING_MODE) or "flash"
        self._state.setdefault("reasoning_mode", default_reasoning_mode)
        if _TRACE_HISTORY and _TRACE_SESSION_SEPARATOR:
            _trace_separator(self._session_id, kind="start")
        if _TRACE_HISTORY:
            history_key, history, reasoning_mode = _resolve_history(self._state)
            _history_log(
                "session.start",
                session_id=self._session_id,
                app_name=self._app_name,
                user_id=self._user_id,
                router=router_mode,
                reasoning_mode=reasoning_mode,
                history_key=history_key,
                history_len=len(history),
            )

    @property
    def app_name(self) -> str:
        return self._app_name

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def consume_latest_suggestions(self) -> Optional[Dict[str, Any]]:
        payload = self._last_suggestions
        self._last_suggestions = None
        return payload

    @staticmethod
    def _next_message_id() -> str:
        return f"msg-{uuid.uuid4().hex[:12]}"

    def _capture_suggestions(self, state: Dict[str, Any]) -> None:
        raw_questions = state.get("suggestions") or []
        if not isinstance(raw_questions, list):
            raw_questions = []
        questions = [str(item).strip() for item in raw_questions if str(item or "").strip()]
        if not questions:
            self._last_suggestions = None
            return
        message_id = str(state.get("message_id") or "").strip()
        if not message_id:
            self._last_suggestions = None
            return
        self._last_suggestions = {
            "message_id": message_id,
            "questions": questions[:_SUGGESTION_MAX_ITEMS],
            "source": state.get("suggestions_source") or "graph_node",
        }

    def _run_post_answer_graph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._post_answer_graph.invoke(state)
        except Exception as exc:
            logger.warning("Post-answer graph failed: %s", exc)
            fallback = dict(state)
            fallback.update(_finalize_history(fallback))
            fallback["suggestions"] = []
            fallback["suggestions_source"] = "graph_node"
            fallback["suggestions_error"] = str(exc)
            return fallback

    def send_message(self, text: str) -> List[LangGraphMessage]:
        started = time.monotonic()
        message_id = self._next_message_id()
        input_state = dict(self._state)
        input_state["message"] = text
        input_state["message_id"] = message_id
        _trace(
            "session.message",
            session_id=self._session_id,
            workflow="agent",
            router=self._state.get("router_mode"),
            streaming=False,
            message_id=message_id,
            question=_shorten(text),
        )
        result = self._graph.invoke(input_state)
        result["message_id"] = result.get("message_id") or message_id
        self._capture_suggestions(result)
        response_text = result.get("response", "")
        self._state = self._prune_state(result)
        _trace_history(self._session_id, self._state, reason="send_message")
        elapsed = time.monotonic() - started
        _trace(
            "session.complete",
            session_id=self._session_id,
            duration_ms=int(elapsed * 1000),
        )
        return [
            LangGraphMessage(
                author=_BACKEND_AUTHOR,
                text=response_text,
                is_final=True,
                message_id=result.get("message_id"),
            )
        ]

    def stream_message(self, text: str) -> Any:
        session_started = time.monotonic()
        message_id = self._next_message_id()
        input_state = dict(self._state)
        input_state["message"] = text
        input_state["message_id"] = message_id
        input_state["streaming"] = True
        _trace(
            "session.message",
            session_id=self._session_id,
            workflow="agent",
            router=self._state.get("router_mode"),
            streaming=True,
            message_id=message_id,
            question=_shorten(text),
        )
        result = self._graph.invoke(input_state)
        prompt = result.get("prompt")
        if not prompt:
            response_text = result.get("response", "")
            result["message_id"] = result.get("message_id") or message_id
            self._capture_suggestions(result)
            self._state = self._prune_state(result)
            _trace_history(self._session_id, self._state, reason="stream_message")
            elapsed = time.monotonic() - session_started
            _trace(
                "session.complete",
                session_id=self._session_id,
                duration_ms=int(elapsed * 1000),
            )
            yield LangGraphMessage(
                author=_BACKEND_AUTHOR,
                text=response_text,
                is_final=True,
                message_id=result.get("message_id"),
            )
            return

        stt_count, vlm_count = _count_evidence(result.get("answer_records") or [])
        response_text = ""
        try:
            llm_started = time.monotonic()
            first_token_ms: Optional[int] = None
            for chunk in _generate_with_genai_stream(prompt):
                if not chunk:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.monotonic() - llm_started) * 1000)
                    _trace("llm.first_token", duration_ms=first_token_ms)
                response_text += chunk
                yield LangGraphMessage(
                    author=_BACKEND_AUTHOR,
                    text=chunk,
                    is_final=False,
                    message_id=message_id,
                )
            elapsed = time.monotonic() - llm_started
        except Exception as exc:
            logger.warning("LangGraph streaming LLM call failed: %s", exc)
            response_text = _fallback_answer(result.get("answer_records") or [])
            result["response"] = response_text
            result["message_id"] = message_id
            result = self._run_post_answer_graph(result)
            self._capture_suggestions(result)
            self._state = self._prune_state(result)
            _trace_history(self._session_id, self._state, reason="stream_error")
            yield LangGraphMessage(
                author=_BACKEND_AUTHOR,
                text=response_text,
                is_final=True,
                message_id=message_id,
            )
            return
        if not response_text:
            response_text = _fallback_answer(result.get("answer_records") or [])
            yield LangGraphMessage(
                author=_BACKEND_AUTHOR,
                text=response_text,
                is_final=True,
                message_id=message_id,
            )
        else:
            _trace(
                "llm.call",
                streaming=True,
                duration_ms=int(elapsed * 1000),
                summary_records=len(result.get("answer_records") or []),
                stt_evidence=stt_count,
                vlm_evidence=vlm_count,
            )

        result["response"] = response_text
        result["message_id"] = message_id
        result = self._run_post_answer_graph(result)
        self._capture_suggestions(result)
        self._state = self._prune_state(result)
        _trace_history(self._session_id, self._state, reason="stream_message")
        total_elapsed = time.monotonic() - session_started
        _trace(
            "session.complete",
            session_id=self._session_id,
            duration_ms=int(total_elapsed * 1000),
        )

    def close(self) -> None:
        _trace_history(self._session_id, self._state, reason="close")
        _history_log("session.close", session_id=self._session_id)
        if _TRACE_HISTORY and _TRACE_SESSION_SEPARATOR:
            _trace_separator(self._session_id, kind="end")
        return None

    @staticmethod
    def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
        scratch_keys = {
            "message",
            "cleaned_message",
            "response",
            "time_ms",
            "answer_records",
            "streaming",
            "prompt",
            "enrich_decision",
            "summary_decision",
            "message_id",
            "suggestions",
            "suggestions_source",
            "suggestions_error",
        }
        return {k: v for k, v in state.items() if k not in scratch_keys}
