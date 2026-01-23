from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

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

_TIME_TAG_RE = re.compile(r"\[time_ms=(\d+)\]")
_BACKEND_AUTHOR = "langgraph_chatbot"
_DEFAULT_MODEL = os.getenv("CHATBOT_LLM_MODEL", "gemini-2.5-flash")
_DEFAULT_TEMPERATURE = os.getenv("CHATBOT_LLM_TEMPERATURE", "0.2")
_DEFAULT_OUTPUT_BASE = Path(os.getenv("CHATBOT_OUTPUT_BASE", "data/outputs"))
_DETAIL_KEYWORDS = (
    "자세히",
    "상세",
    "근거",
    "원문",
    "출처",
    "증거",
    "detail",
    "evidence",
    "source",
)
_MAX_EVIDENCE_UNITS = int(os.getenv("CHATBOT_MAX_EVIDENCE_UNITS", "10"))

_OUT_OF_RANGE_MESSAGE = (
    "죄송합니다. 해당 시간에 대한 요약 정보가 아직 업데이트되지 않았습니다. "
    "요약 작업은 완료되었지만, 특정 시간대의 상세 내용을 제공해 드리지 못하고 있습니다."
)
_NO_SUMMARY_MESSAGE = "아직 요약이 생성되지 않았습니다. 요약 작업을 먼저 실행해 주세요."
_NEED_MODE_MESSAGE = "먼저 모드를 선택해 주세요. (full 또는 partial)"
_PARTIAL_NOT_READY = "Partial summary chatbot is not implemented yet."


class ChatState(TypedDict, total=False):
    message: str
    cleaned_message: str
    time_ms: Optional[int]
    intent: str
    selected_mode: Optional[str]
    chat_mode: Optional[str]
    detail_requested: bool
    response: str
    answer_records: List[Dict[str, Any]]
    history: List[Dict[str, str]]
    summary_cache: List[Dict[str, Any]]
    pending_updates: List[Dict[str, Any]]
    last_segment_id: int
    video_name: Optional[str]
    video_id: Optional[str]
    video_root: Optional[str]
    output_base: Optional[str]
    unit_cache: Dict[str, Dict[str, Any]]


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


def _detail_requested(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _DETAIL_KEYWORDS)


def _parse_input(state: ChatState) -> Dict[str, Any]:
    message = state.get("message", "") or ""
    time_ms: Optional[int] = None
    match = _TIME_TAG_RE.search(message)
    if match:
        try:
            time_ms = int(match.group(1))
        except ValueError:
            time_ms = None
        message = _TIME_TAG_RE.sub("", message).strip()
    cleaned = message.strip()
    selected_mode = _normalize_mode(cleaned)
    intent = "mode_select" if selected_mode else "message"
    return {
        "cleaned_message": cleaned,
        "time_ms": time_ms,
        "selected_mode": selected_mode,
        "intent": intent,
        "detail_requested": _detail_requested(cleaned),
    }


def _route_after_parse(state: ChatState) -> str:
    if state.get("intent") == "mode_select":
        return "mode_select"
    mode = state.get("chat_mode")
    if mode == "partial":
        return "partial"
    if mode == "full":
        return "full"
    return "need_mode"


def _handle_mode_select(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    selected_mode = state.get("selected_mode")
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


def _handle_partial(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    result = backend.partial_not_implemented(state)
    return {"response": result.get("error", _PARTIAL_NOT_READY)}


def _prepare_full(state: ChatState, backend: SummaryBackend) -> Dict[str, Any]:
    updates = backend.get_summary_updates(state)
    if not updates.get("success"):
        return {"response": updates.get("error", "요약 업데이트에 실패했습니다.")}

    time_ms = state.get("time_ms")
    context = backend.get_summary_context(state, time_ms=time_ms)
    if not context.get("success"):
        return {"response": context.get("error", "요약 컨텍스트를 불러오지 못했습니다.")}

    summary_cache = context.get("summary_cache", [])
    if not summary_cache:
        return {"response": _NO_SUMMARY_MESSAGE}

    if time_ms is not None and context.get("out_of_range"):
        return {"response": _OUT_OF_RANGE_MESSAGE}

    matches = context.get("matches") or []
    records = matches or summary_cache
    detail_requested = bool(state.get("detail_requested"))
    if detail_requested:
        cache = _get_unit_cache(state)
        records = _attach_evidence(records, cache)
        return {"answer_records": records, "unit_cache": cache}
    return {"answer_records": records}


def _route_after_prepare(state: ChatState) -> str:
    if state.get("response"):
        return "respond"
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


def _load_units_from_segments_units(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    stt_units: Dict[str, Dict[str, Any]] = {}
    vlm_units: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for record in _iter_jsonl(path):
            for unit in record.get("transcript_units", []) or []:
                unit_id = unit.get("unit_id")
                if isinstance(unit_id, str) and unit_id not in stt_units:
                    stt_units[unit_id] = unit
            for unit in record.get("visual_units", []) or []:
                unit_id = unit.get("unit_id")
                if isinstance(unit_id, str) and unit_id not in vlm_units:
                    vlm_units[unit_id] = unit
    return {"stt": stt_units, "vlm": vlm_units}


def _load_stt_units(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, list):
        payload = payload.get("segments", []) if isinstance(payload, dict) else []
    for segment in payload or []:
        if not isinstance(segment, dict):
            continue
        unit_id = segment.get("id")
        if isinstance(unit_id, str) and unit_id not in cache:
            cache[unit_id] = {
                "unit_id": unit_id,
                "start_ms": segment.get("start_ms"),
                "end_ms": segment.get("end_ms"),
                "text": segment.get("text"),
            }


def _load_vlm_units(paths: List[Path], cache: Dict[str, Dict[str, Any]]) -> None:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            unit_id = item.get("id")
            if isinstance(unit_id, str) and unit_id not in cache:
                cache[unit_id] = {
                    "unit_id": unit_id,
                    "timestamp_ms": item.get("timestamp_ms"),
                    "text": item.get("extracted_text") or item.get("text"),
                }


def _build_unit_cache(state: ChatState) -> Dict[str, Dict[str, Any]]:
    video_root = _resolve_video_root(state)
    cache: Dict[str, Dict[str, Any]] = {"stt": {}, "vlm": {}}
    if not video_root:
        return cache

    segment_paths: List[Path] = []
    fusion_units = video_root / "fusion" / "segments_units.jsonl"
    if fusion_units.exists():
        segment_paths.append(fusion_units)
    batches_dir = video_root / "batches"
    if batches_dir.exists():
        segment_paths.extend(sorted(batches_dir.glob("batch_*/segments_units.jsonl")))

    if segment_paths:
        unit_data = _load_units_from_segments_units(segment_paths)
        cache["stt"].update(unit_data["stt"])
        cache["vlm"].update(unit_data["vlm"])

    stt_path = video_root / "stt.json"
    if stt_path.exists():
        _load_stt_units(stt_path, cache["stt"])

    vlm_paths: List[Path] = []
    vlm_root = video_root / "vlm.json"
    if vlm_root.exists():
        vlm_paths.append(vlm_root)
    if batches_dir.exists():
        vlm_paths.extend(sorted(batches_dir.glob("batch_*/vlm.json")))
    if vlm_paths:
        _load_vlm_units(vlm_paths, cache["vlm"])

    return cache


def _cache_key(state: ChatState) -> str:
    video_root = state.get("video_root")
    if video_root:
        return str(video_root)
    return state.get("video_name") or ""


def _get_unit_cache(state: ChatState) -> Dict[str, Dict[str, Any]]:
    cache = state.get("unit_cache")
    key = _cache_key(state)
    if not isinstance(cache, dict) or cache.get("_cache_key") != key:
        cache = _build_unit_cache(state)
        cache["_cache_key"] = key
    return cache


def _format_unit(unit_id: str, unit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "unit_id": unit_id,
        "start_ms": unit.get("start_ms"),
        "end_ms": unit.get("end_ms"),
        "timestamp_ms": unit.get("timestamp_ms"),
        "text": unit.get("text"),
    }


def _attach_evidence(
    records: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stt_cache = cache.get("stt", {})
    vlm_cache = cache.get("vlm", {})
    enriched: List[Dict[str, Any]] = []
    for record in records:
        source_refs = record.get("source_refs", {}) or {}
        stt_ids = source_refs.get("stt_ids") or []
        vlm_ids = source_refs.get("vlm_ids") or []

        evidence: Dict[str, Any] = {}
        if stt_ids:
            items: List[Dict[str, Any]] = []
            for unit_id in stt_ids[:_MAX_EVIDENCE_UNITS]:
                if not isinstance(unit_id, str):
                    continue
                unit = stt_cache.get(unit_id)
                if unit:
                    items.append(_format_unit(unit_id, unit))
            if items:
                evidence["stt"] = items
        if vlm_ids:
            items = []
            for unit_id in vlm_ids[:_MAX_EVIDENCE_UNITS]:
                if not isinstance(unit_id, str):
                    continue
                unit = vlm_cache.get(unit_id)
                if unit:
                    items.append(_format_unit(unit_id, unit))
            if items:
                evidence["vlm"] = items

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


def _build_prompt(state: ChatState) -> str:
    question = state.get("cleaned_message", "").strip()
    records = state.get("answer_records") or []
    history = state.get("history", [])
    history_text = ""
    if isinstance(history, list) and history:
        tail = history[-6:]
        lines = [f"{item.get('role')}: {item.get('content')}" for item in tail]
        history_text = "\n".join(lines)
    summary_json = _format_records(records)
    prompt = (
        "You are the Summary Chatbot.\n"
        "Always respond in Korean.\n"
        "Use only the provided summary records and evidence to answer.\n"
        "If the answer is not present, say you cannot find it.\n"
    )
    if history_text:
        prompt += f"\nRecent conversation:\n{history_text}\n"
    prompt += f"\nUser question:\n{question}\n\nSummary records:\n{summary_json}\n"
    return prompt


def _generate_with_genai(prompt: str) -> str:
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
        model=_DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    return str(response)


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


def _generate_answer(state: ChatState) -> Dict[str, Any]:
    records = state.get("answer_records") or []
    prompt = _build_prompt(state)
    try:
        response_text = _generate_with_genai(prompt)
    except Exception as exc:
        logger.warning("LangGraph LLM call failed: %s", exc)
        response_text = _fallback_answer(records)

    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    history = history + [
        {"role": "user", "content": state.get("cleaned_message", "")},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": history}


def _build_graph(backend: SummaryBackend) -> Any:
    if _LANGGRAPH_IMPORT_ERROR:
        raise RuntimeError(
            "langgraph is required for LangGraphSession. Install it or use CHATBOT_BACKEND=adk."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph: Any = StateGraph(ChatState)
    graph.add_node("parse_input", _parse_input)
    graph.add_node("mode_select", lambda state: _handle_mode_select(state, backend))
    graph.add_node("need_mode", lambda _: {"response": _NEED_MODE_MESSAGE})
    graph.add_node("partial", lambda state: _handle_partial(state, backend))
    graph.add_node("prepare_full", lambda state: _prepare_full(state, backend))
    graph.add_node("generate_answer", _generate_answer)

    graph.set_entry_point("parse_input")
    graph.add_conditional_edges(
        "parse_input",
        _route_after_parse,
        {
            "mode_select": "mode_select",
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
    graph.add_edge("need_mode", END)
    graph.add_edge("partial", END)
    graph.add_edge("generate_answer", END)
    return graph.compile()


class LangGraphSession:
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
        self._graph = _build_graph(self._backend)
        self._state: Dict[str, Any] = dict(initial_state or {})
        self._state.setdefault("summary_cache", [])
        self._state.setdefault("pending_updates", [])
        self._state.setdefault("last_segment_id", 0)
        self._state.setdefault("history", [])
        self._state.setdefault("unit_cache", {})

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
        input_state = dict(self._state)
        input_state["message"] = text
        result = self._graph.invoke(input_state)
        response_text = result.get("response", "")
        self._state = self._prune_state(result)
        return [LangGraphMessage(author=_BACKEND_AUTHOR, text=response_text, is_final=True)]

    def close(self) -> None:
        return None

    @staticmethod
    def _prune_state(state: Dict[str, Any]) -> Dict[str, Any]:
        scratch_keys = {
            "message",
            "cleaned_message",
            "intent",
            "selected_mode",
            "response",
            "time_ms",
            "detail_requested",
            "answer_records",
        }
        return {k: v for k, v in state.items() if k not in scratch_keys}
