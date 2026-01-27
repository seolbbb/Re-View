from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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
_TRACE_LOG_PATH = Path(os.getenv("CHATBOT_TRACE_LOG", "logs/langgraph_trace.log"))

_TIME_TAG_RE = re.compile(r"\[time_ms=(\d+)\]")
_TIME_KR_RE = re.compile(r"(\d+)\s*분\s*(\d+)\s*초")
_TIME_MS_RE = re.compile(r"(\d+)\s*초")
_TIME_MMSS_RE = re.compile(r"\b(\d{1,2})\s*:\s*(\d{2})\b")
_TIME_MIN_ONLY_RE = re.compile(r"(\d+)\s*분")
_BACKEND_AUTHOR = "langgraph_chatbot"
_DEFAULT_MODEL = os.getenv("CHATBOT_LLM_MODEL", "gemini-2.5-flash")
_DECISION_MODEL = os.getenv("CHATBOT_DECISION_MODEL", "gemini-2.5-flash")
_DEFAULT_TEMPERATURE = os.getenv("CHATBOT_LLM_TEMPERATURE", "0.2")
_DEFAULT_REASONING_MODE = os.getenv("CHATBOT_REASONING_MODE", "flash")
_DEFAULT_ROUTER_MODE = os.getenv("CHATBOT_ROUTER", "rules").strip().lower()
_DEFAULT_OUTPUT_BASE = Path(os.getenv("CHATBOT_OUTPUT_BASE", "data/outputs"))
_MAX_EVIDENCE_UNITS = int(os.getenv("CHATBOT_MAX_EVIDENCE_UNITS", "10"))

_OUT_OF_RANGE_MESSAGE = (
    "죄송합니다. 해당 시간에 대한 요약 정보가 아직 업데이트되지 않았습니다. "
    "요약 작업은 완료되었지만, 특정 시간대의 상세 내용을 제공해 드리지 못하고 있습니다."
)
_NO_SUMMARY_MESSAGE = "아직 요약이 생성되지 않았습니다. 요약 작업을 먼저 실행해 주세요."
_NEED_MODE_MESSAGE = "먼저 모드를 선택해 주세요. (full 또는 partial)"
_PARTIAL_NOT_READY = "Partial summary chatbot is not implemented yet."
_NEED_REASONING_MODE_MESSAGE = "응답 모드를 선택해 주세요. (flash 또는 thinking)"


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


def _shorten(text: str, limit: int = 160) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


class ChatState(TypedDict, total=False):
    message: str
    cleaned_message: str
    time_ms: Optional[int]
    intent: str
    next_step: str
    selected_mode: Optional[str]
    selected_reasoning_mode: Optional[str]
    chat_mode: Optional[str]
    reasoning_mode: Optional[str]
    enrich_decision: str
    summary_decision: str
    response: str
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


def _normalize_mode(text: str) -> Optional[str]:
    normalized = text.strip().lower()
    if normalized in {"full", "전문 요약", "전문", "전체"}:
        return "full"
    if normalized in {"partial", "부분 요약", "부분"}:
        return "partial"
    return None


def _normalize_reasoning_mode(text: str) -> Optional[str]:
    normalized = text.strip().lower()
    if normalized in {"flash", "flash mode"}:
        return "flash"
    if normalized in {"thinking", "thinking mode"}:
        return "thinking"
    return None


def _parse_input(state: ChatState) -> Dict[str, Any]:
    """사용자 입력을 파싱해 라우팅에 필요한 필드를 구성한다."""
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
    selected_reasoning_mode = _normalize_reasoning_mode(cleaned)
    selected_mode = _normalize_mode(cleaned)
    if selected_reasoning_mode:
        intent = "reasoning_select"
    else:
        intent = "mode_select" if selected_mode else "message"
    _trace(
        "node.parse_input",
        has_time_tag=has_time_tag,
        selected_mode=selected_mode,
        selected_reasoning_mode=selected_reasoning_mode,
    )
    return {
        "cleaned_message": cleaned,
        "time_ms": time_ms,
        "selected_mode": selected_mode,
        "selected_reasoning_mode": selected_reasoning_mode,
        "intent": intent,
    }


def _route_after_parse(state: ChatState) -> str:
    """parse_input 결과를 바탕으로 다음 노드를 규칙 기반으로 결정한다."""
    if state.get("intent") == "reasoning_select":
        decision = "reasoning_select"
        _trace("route.rules", decision=decision)
        return decision
    if state.get("intent") == "mode_select":
        decision = "mode_select"
        _trace("route.rules", decision=decision)
        return decision
    mode = state.get("chat_mode")
    if mode == "partial":
        decision = "partial"
        _trace("route.rules", decision=decision)
        return decision
    if mode == "full":
        decision = "full"
        _trace("route.rules", decision=decision)
        return decision
    decision = "need_mode"
    _trace("route.rules", decision=decision)
    return decision


_ROUTER_LABELS = {"mode_select", "reasoning_select", "partial", "full", "need_mode"}


def _build_router_prompt(state: ChatState) -> str:
    """LLM 라우터용 프롬프트를 구성한다."""
    message = state.get("cleaned_message", "")
    return (
        "You are a routing assistant for a chatbot state machine.\n"
        "Choose exactly one label from: mode_select, reasoning_select, full, partial, need_mode.\n"
        "Rules:\n"
        "- If the user explicitly selects chat mode (full/partial), choose mode_select.\n"
        "- If the user explicitly selects reasoning mode (flash/thinking), choose reasoning_select.\n"
        "- If chat_mode is already set, choose full or partial to proceed.\n"
        "- If chat_mode is not set and the user asks a question, choose need_mode.\n"
        "- If unsure, choose need_mode.\n"
        "\n"
        f"message={message!r}\n"
        f"selected_mode={state.get('selected_mode')!r}\n"
        f"selected_reasoning_mode={state.get('selected_reasoning_mode')!r}\n"
        f"chat_mode={state.get('chat_mode')!r}\n"
        f"reasoning_mode={state.get('reasoning_mode')!r}\n"
        f"time_ms={state.get('time_ms')!r}\n"
        "\n"
        "Return only the label."
    )


def _generate_router_choice(prompt: str) -> str:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is required for LangGraph chatbot.") from exc

    client = genai.Client()
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=16)
    response = client.models.generate_content(
        model=_DECISION_MODEL,
        contents=prompt,
        config=config,
    )
    text = _extract_genai_text(response)
    return text.strip()


def _normalize_router_label(value: str) -> str:
    cleaned = re.sub(r"[^a-z_]", "", value.strip().lower())
    return cleaned


def _route_with_llm(state: ChatState) -> Dict[str, Any]:
    """LLM 라우팅 결과를 next_step에 기록한다."""
    if state.get("selected_mode") or state.get("selected_reasoning_mode"):
        next_step = _route_after_parse(state)
        _trace("route.llm", decision=next_step, reason="explicit_selection")
        return {"next_step": next_step}
    if state.get("chat_mode") in {"full", "partial"}:
        decision = state.get("chat_mode")
        _trace("route.llm", decision=decision, reason="chat_mode_set")
        return {"next_step": decision}
    prompt = _build_router_prompt(state)
    try:
        raw = _generate_router_choice(prompt)
    except Exception as exc:
        logger.warning("LLM router failed: %s", exc)
        _trace("route.llm", decision="fallback_rules", error=str(exc))
        return {"next_step": _route_after_parse(state)}
    label = _normalize_router_label(raw)
    if label not in _ROUTER_LABELS:
        _trace("route.llm", decision="fallback_rules", raw=raw, normalized=label)
        return {"next_step": _route_after_parse(state)}
    _trace("route.llm", decision=label, raw=raw)
    return {"next_step": label}


def _handle_mode_select(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    """사용자가 선택한 chat_mode를 반영하고 요약 상태를 확인한다."""
    selected_mode = state.get("selected_mode")
    _trace("node.mode_select", selected=selected_mode)
    if selected_mode not in {"full", "partial"}:
        return {"response": _NEED_MODE_MESSAGE}

    updates: Dict[str, Any] = {"chat_mode": selected_mode}
    if selected_mode == "partial":
        result = backend.partial_not_implemented(state)
        updates["response"] = result.get("error", _PARTIAL_NOT_READY)
        return updates

    result = backend.ensure_summary_exists(state)
    if not result.get("success"):
        updates["response"] = result.get("error", "요약 상태 확인에 실패했습니다.")
        return updates
    updates["response"] = result.get("message", "모드를 설정했습니다.")
    return updates


def _handle_reasoning_select(state: ChatState) -> Dict[str, Any]:
    """사용자가 선택한 reasoning_mode를 반영한다."""
    selected_mode = state.get("selected_reasoning_mode")
    _trace("node.reasoning_select", selected=selected_mode)
    if selected_mode not in {"flash", "thinking"}:
        return {"response": _NEED_REASONING_MODE_MESSAGE}
    return {
        "reasoning_mode": selected_mode,
        "response": f"응답 모드를 설정했습니다. ({selected_mode})",
    }


def _handle_partial(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    _trace("node.partial")
    result = backend.partial_not_implemented(state)
    return {"response": result.get("error", _PARTIAL_NOT_READY)}


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
    if reasoning_mode == "flash":
        records = _sanitize_records_for_flash(records)
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
    summary_json = _format_records(records)
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
            "Use only the provided summary records to answer.\n"
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
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is required for LangGraph chatbot.") from exc

    client = genai.Client()
    try:
        temperature = float(_DEFAULT_TEMPERATURE)
    except ValueError:
        temperature = 0.2
    config = types.GenerateContentConfig(temperature=temperature)
    response = client.models.generate_content(
        model=model or _DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response)


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
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-genai package is required for LangGraph chatbot.") from exc

    client = genai.Client()
    try:
        temperature = float(_DEFAULT_TEMPERATURE)
    except ValueError:
        temperature = 0.2
    config = types.GenerateContentConfig(temperature=temperature)
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
            yield delta


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

    reasoning_mode = _normalize_reasoning_mode(state.get("reasoning_mode") or "") or "flash"
    history_key = f"history_{reasoning_mode}"
    history = state.get(history_key, [])
    if not isinstance(history, list):
        history = []
    history = history + [
        {"role": "user", "content": state.get("cleaned_message", "")},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, history_key: history}


def _build_graph(backend: SummaryBackend, *, router_mode: str) -> Any:
    if _LANGGRAPH_IMPORT_ERROR:
        raise RuntimeError(
            "langgraph is required for LangGraphSession. Install it or use CHATBOT_BACKEND=adk."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph: Any = StateGraph(ChatState)
    graph.add_node("parse_input", _parse_input)
    graph.add_node("mode_select", lambda state: _handle_mode_select(state, backend))
    graph.add_node("reasoning_select", _handle_reasoning_select)
    graph.add_node("need_mode", lambda _: {"response": _NEED_MODE_MESSAGE})
    graph.add_node("partial", lambda state: _handle_partial(state, backend))
    graph.add_node("prepare_full", lambda state: _prepare_full(state, backend))
    graph.add_node("generate_answer", _generate_answer)

    graph.set_entry_point("parse_input")
    if router_mode == "llm":
        graph.add_node("route_with_llm", _route_with_llm)
        graph.add_edge("parse_input", "route_with_llm")
        graph.add_conditional_edges(
            "route_with_llm",
            lambda state: state.get("next_step") or "need_mode",
            {
                "mode_select": "mode_select",
                "reasoning_select": "reasoning_select",
                "partial": "partial",
                "full": "prepare_full",
                "need_mode": "need_mode",
            },
        )
    else:
        graph.add_conditional_edges(
            "parse_input",
            _route_after_parse,
            {
                "mode_select": "mode_select",
                "reasoning_select": "reasoning_select",
                "partial": "partial",
                "full": "prepare_full",
                "need_mode": "need_mode",
            },
        )
    graph.add_conditional_edges(
        "prepare_full",
        _route_after_prepare,
        {
            "respond": END,
            "llm": "generate_answer",
        },
    )
    graph.add_edge("mode_select", END)
    graph.add_edge("reasoning_select", END)
    graph.add_edge("need_mode", END)
    graph.add_edge("partial", END)
    graph.add_edge("generate_answer", END)
    return graph.compile()


def _build_agent_graph(backend: SummaryBackend, *, router_mode: str) -> Any:
    """에이전트형 그래프를 구성한다 (요약 판단 → RAG → evidence → 답변)."""
    if _LANGGRAPH_IMPORT_ERROR:
        raise RuntimeError(
            "langgraph is required for LangGraphSession. Install it or use CHATBOT_BACKEND=adk."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph: Any = StateGraph(ChatState)
    graph.add_node("parse_input", _parse_input)
    graph.add_node("mode_select", lambda state: _handle_mode_select(state, backend))
    graph.add_node("reasoning_select", _handle_reasoning_select)
    graph.add_node("need_mode", lambda _: {"response": _NEED_MODE_MESSAGE})
    graph.add_node("partial", lambda state: _handle_partial(state, backend))
    graph.add_node("prepare_full", lambda state: _prepare_full(state, backend))
    graph.add_node("decide_summary", _decide_summary_sufficiency)
    graph.add_node("enrich_evidence", lambda state: _enrich_with_db_evidence(state, backend))
    graph.add_node("generate_answer", _generate_answer)

    graph.set_entry_point("parse_input")
    if router_mode == "llm":
        graph.add_node("route_with_llm", _route_with_llm)
        graph.add_edge("parse_input", "route_with_llm")
        graph.add_conditional_edges(
            "route_with_llm",
            lambda state: state.get("next_step") or "need_mode",
            {
                "mode_select": "mode_select",
                "reasoning_select": "reasoning_select",
                "partial": "partial",
                "full": "prepare_full",
                "need_mode": "need_mode",
            },
        )
    else:
        graph.add_conditional_edges(
            "parse_input",
            _route_after_parse,
            {
                "mode_select": "mode_select",
                "reasoning_select": "reasoning_select",
                "partial": "partial",
                "full": "prepare_full",
                "need_mode": "need_mode",
            },
        )

    graph.add_conditional_edges(
        "prepare_full",
        _route_after_prepare,
        {
            "respond": END,
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
    graph.add_edge("mode_select", END)
    graph.add_edge("reasoning_select", END)
    graph.add_edge("need_mode", END)
    graph.add_edge("partial", END)
    graph.add_edge("generate_answer", END)
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
        self._graph = _build_agent_graph(self._backend, router_mode=router_mode)
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

    @property
    def app_name(self) -> str:
        return self._app_name

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def send_message(self, text: str) -> List[LangGraphMessage]:
        started = time.monotonic()
        input_state = dict(self._state)
        input_state["message"] = text
        _trace(
            "session.message",
            session_id=self._session_id,
            workflow="agent",
            router=self._state.get("router_mode"),
            streaming=False,
            question=_shorten(text),
        )
        result = self._graph.invoke(input_state)
        response_text = result.get("response", "")
        self._state = self._prune_state(result)
        elapsed = time.monotonic() - started
        _trace(
            "session.complete",
            session_id=self._session_id,
            duration_ms=int(elapsed * 1000),
        )
        return [LangGraphMessage(author=_BACKEND_AUTHOR, text=response_text, is_final=True)]

    def stream_message(self, text: str) -> Any:
        started = time.monotonic()
        input_state = dict(self._state)
        input_state["message"] = text
        input_state["streaming"] = True
        _trace(
            "session.message",
            session_id=self._session_id,
            workflow="agent",
            router=self._state.get("router_mode"),
            streaming=True,
            question=_shorten(text),
        )
        result = self._graph.invoke(input_state)
        prompt = result.get("prompt")
        if not prompt:
            response_text = result.get("response", "")
            self._state = self._prune_state(result)
            elapsed = time.monotonic() - started
            _trace(
                "session.complete",
                session_id=self._session_id,
                duration_ms=int(elapsed * 1000),
            )
            yield LangGraphMessage(author=_BACKEND_AUTHOR, text=response_text, is_final=True)
            return

        stt_count, vlm_count = _count_evidence(result.get("answer_records") or [])
        response_text = ""
        try:
            started = time.monotonic()
            first_token_ms: Optional[int] = None
            for chunk in _generate_with_genai_stream(prompt):
                if not chunk:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.monotonic() - started) * 1000)
                    _trace("llm.first_token", duration_ms=first_token_ms)
                response_text += chunk
                yield LangGraphMessage(author=_BACKEND_AUTHOR, text=chunk, is_final=False)
            elapsed = time.monotonic() - started
        except Exception as exc:
            logger.warning("LangGraph streaming LLM call failed: %s", exc)
            response_text = _fallback_answer(result.get("answer_records") or [])
            yield LangGraphMessage(author=_BACKEND_AUTHOR, text=response_text, is_final=True)
            result["response"] = response_text
            self._state = self._prune_state(result)
            return
        if not response_text:
            response_text = _fallback_answer(result.get("answer_records") or [])
            yield LangGraphMessage(author=_BACKEND_AUTHOR, text=response_text, is_final=True)
        else:
            _trace(
                "llm.call",
                streaming=True,
                duration_ms=int(elapsed * 1000),
                summary_records=len(result.get("answer_records") or []),
                stt_evidence=stt_count,
                vlm_evidence=vlm_count,
            )

        reasoning_mode = _normalize_reasoning_mode(result.get("reasoning_mode") or "") or "flash"
        history_key = f"history_{reasoning_mode}"
        history = result.get(history_key, [])
        if not isinstance(history, list):
            history = []
        history = history + [
            {"role": "user", "content": result.get("cleaned_message", "")},
            {"role": "assistant", "content": response_text},
        ]
        result[history_key] = history
        result["response"] = response_text
        self._state = self._prune_state(result)
        total_elapsed = time.monotonic() - started
        _trace(
            "session.complete",
            session_id=self._session_id,
            duration_ms=int(total_elapsed * 1000),
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
        scratch_keys = {
            "message",
            "cleaned_message",
            "intent",
            "next_step",
            "selected_mode",
            "selected_reasoning_mode",
            "response",
            "time_ms",
            "answer_records",
            "streaming",
            "prompt",
            "enrich_decision",
            "summary_decision",
        }
        return {k: v for k, v in state.items() if k not in scratch_keys}
