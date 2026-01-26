from __future__ import annotations

import json
import re
import os
from typing import Any, Dict
from urllib import error as urllib_error
from urllib import request as urllib_request

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from src.adk_chatbot.paths import DEFAULT_OUTPUT_BASE
from src.adk_chatbot.store import VideoStore


def _process_api_base() -> str:
    return os.environ.get("PROCESS_API_URL", "http://localhost:8001").rstrip("/")


_TIME_TAG_RE = re.compile(r"\[time_ms=\d+\]\s*")


def _strip_time_tag(text: str) -> str:
    return _TIME_TAG_RE.sub("", text or "").strip()


def _call_process_api(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{_process_api_base()}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib_request.Request(
        url,
        data=data,
        method=method.upper(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib_request.urlopen(req, timeout=5) as resp:
            status_code = resp.status
            body = resp.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
        return {
            "success": False,
            "status_code": exc.code,
            "error": body or exc.reason,
        }
    except urllib_error.URLError as exc:
        return {"success": False, "error": str(exc.reason)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {"raw": body}
    return {"success": True, "status_code": status_code, "payload": payload}


def get_summary_status(tool_context: ToolContext) -> Dict[str, Any]:
    """비디오의 요약 상태를 조회합니다 (DB API 기반)."""
    video_id = tool_context.state.get("video_id")
    video_name = tool_context.state.get("video_name")
    
    if not video_id and not video_name:
        return {"success": False, "error": "video_name or video_id is not set"}

    # video_id가 있으면 새 DB API 사용
    if video_id:
        # 비디오 상태 조회
        status_response = _call_process_api("GET", f"/videos/{video_id}/status")
        if not status_response.get("success"):
            # 404면 비디오 없음
            if status_response.get("status_code") == 404:
                return {"success": False, "error": "Video not found"}
            return {
                "success": False,
                "error": status_response.get("error", "status API failed"),
            }
        
        status_data = status_response.get("payload", {})
        video_status = status_data.get("video_status")
        processing_job = status_data.get("processing_job", {})
        
        # 요약 결과 조회
        summary_response = _call_process_api("GET", f"/videos/{video_id}/summary")
        summary_data = summary_response.get("payload", {}) if summary_response.get("success") else {}
        has_summary = summary_data.get("has_summary", False)
        summary_status = summary_data.get("status") if has_summary else None
        
        # 상태 정규화
        if processing_job:
            job_status = processing_job.get("status")
            if job_status == "VLM_RUNNING":
                normalized_status = "running"
            elif job_status == "SUMMARY_RUNNING":
                normalized_status = "summarizing"
            elif job_status == "DONE":
                normalized_status = "completed"
            elif job_status == "FAILED":
                normalized_status = "error"
            else:
                normalized_status = job_status.lower() if job_status else "unknown"
        elif has_summary and summary_status == "DONE":
            normalized_status = "completed"
        elif has_summary and summary_status == "IN_PROGRESS":
            normalized_status = "in_progress"
        elif video_status == "PREPROCESS_DONE":
            normalized_status = "not_started"
        else:
            normalized_status = video_status.lower() if video_status else "unknown"
        
        return {
            "success": True,
            "status": normalized_status,
            "video_status": video_status,
            "has_summary": has_summary,
            "summary_status": summary_status,
            "processing_job": processing_job,
        }
    
    # video_name만 있으면 기존 로컬 파일 기반 로직 (fallback)
    response = _call_process_api("GET", f"/runs/{video_name}")
    if not response.get("success"):
        if response.get("status_code") == 404:
            return {"success": True, "status": "not_started", "has_summary": False}
        return {
            "success": False,
            "status": "error",
            "error": response.get("error", "process_api request failed"),
            "status_code": response.get("status_code"),
        }

    run_meta = response.get("payload", {})
    raw_status = run_meta.get("status")
    pipeline_type = None
    if isinstance(run_meta.get("args"), dict):
        pipeline_type = run_meta["args"].get("pipeline_type")
    pipeline_type = pipeline_type or run_meta.get("pipeline_type")

    if pipeline_type == "preprocess":
        if raw_status == "running":
            normalized_status = "running"
        elif raw_status == "error":
            normalized_status = "error"
        else:
            normalized_status = "not_started"
    else:
        normalized_status = raw_status
        if raw_status == "ok":
            normalized_status = "completed"
        elif raw_status is None:
            normalized_status = "unknown"

    store = VideoStore(output_base=DEFAULT_OUTPUT_BASE, video_name=video_name)
    has_summaries = store.segment_summaries_jsonl().exists()
    return {
        "success": True,
        "status": normalized_status,
        "raw_status": raw_status,
        "pipeline_type": pipeline_type,
        "has_summary": has_summaries,
        "processing_stats": run_meta.get("processing_stats"),
    }
def get_video_name(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name is not set"}
    return {"success": True, "video_name": video_name}


def get_summary_updates(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name is not set"}

    last_segment_id = tool_context.state.get("last_segment_id", -1)
    try:
        last_segment_id = int(last_segment_id)
    except (TypeError, ValueError):
        last_segment_id = -1

    # 1. DB API 시도
    video_id = tool_context.state.get("video_id")
    if video_id:
        query = f"/videos/{video_id}/summaries?after_segment_index={last_segment_id}&limit=200"
        response = _call_process_api("GET", query)
        if response.get("success"):
            items = response.get("payload", {}).get("items", [])
            new_updates = []
            max_segment_id = last_segment_id
            
            for item in items:
                try:
                    current_id = item.get("segment_index", -1)
                    if current_id > last_segment_id:
                        new_updates.append(item)
                        if current_id > max_segment_id:
                            max_segment_id = current_id
                except (TypeError, ValueError):
                    continue
            
            if new_updates:
                tool_context.state["pending_updates"] = tool_context.state.get("pending_updates", []) + new_updates
                tool_context.state["last_segment_id"] = max_segment_id
                
            return {
                "success": True, 
                "last_segment_id": max_segment_id, 
                "new_count": len(new_updates),
                "source": "db"
            }

    # 2. Local File Fallback (video_name only)
    store = VideoStore(output_base=DEFAULT_OUTPUT_BASE, video_name=video_name)
    summaries_path = store.segment_summaries_jsonl()
    
    if not summaries_path.exists():
        tool_context.state["pending_updates"] = []
        tool_context.state["last_segment_id"] = -1
        return {"success": True, "last_segment_id": -1, "new_count": 0}

    updates = []
    max_segment_id = last_segment_id
    with summaries_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                break
            try:
                segment_id = int(record.get("segment_id", -1))
            except (TypeError, ValueError):
                segment_id = -1
            if segment_id <= last_segment_id:
                continue
            updates.append(record)
            if segment_id > max_segment_id:
                max_segment_id = segment_id

    tool_context.state["last_segment_id"] = max_segment_id
    tool_context.state["pending_updates"] = updates
    return {"success": True, "last_segment_id": max_segment_id, "new_count": len(updates)}


def get_summary_context(
    tool_context: ToolContext,
    time_ms: int | None = None,
) -> Dict[str, Any]:
    summary_cache = tool_context.state.get("summary_cache", [])
    if not isinstance(summary_cache, list):
        summary_cache = []
    pending_updates = tool_context.state.get("pending_updates", [])
    if not isinstance(pending_updates, list):
        pending_updates = []

    if pending_updates:
        existing_ids = set()
        for record in summary_cache:
            try:
                # Handle both API (segment_index) and File (segment_id) formats
                sid = record.get("segment_index") if "segment_index" in record else record.get("segment_id")
                existing_ids.add(int(sid))
            except (TypeError, ValueError):
                continue
        for record in pending_updates:
            try:
                # DB API returns segment_index, File returns segment_id. Both used as ID.
                segment_id = int(record.get("segment_index") if "segment_index" in record else record.get("segment_id"))
            except (TypeError, ValueError):
                segment_id = None
            if segment_id is not None and segment_id in existing_ids:
                continue
            summary_cache.append(record)
            if segment_id is not None:
                existing_ids.add(segment_id)

    tool_context.state["summary_cache"] = summary_cache
    tool_context.state["pending_updates"] = []

    if time_ms is None:
        return {"success": True, "summary_cache": summary_cache}

    try:
        time_ms = int(time_ms)
    except (TypeError, ValueError):
        return {"success": False, "error": "time_ms must be an integer"}
    if time_ms < 0:
        return {"success": False, "error": "time_ms must be >= 0"}

    matches = []
    max_end_ms = None
    for record in summary_cache:
        try:
            start_ms = int(record.get("start_ms"))
            end_ms = int(record.get("end_ms"))
        except (TypeError, ValueError):
            continue
        if max_end_ms is None or end_ms > max_end_ms:
            max_end_ms = end_ms
        if start_ms <= time_ms <= end_ms:
            matches.append(record)

    out_of_range = max_end_ms is not None and time_ms > max_end_ms
    return {
        "success": True,
        "summary_cache": summary_cache,
        "matches": matches,
        "out_of_range": out_of_range,
        "max_end_ms": max_end_ms,
    }


def search_summary_context(
    tool_context: ToolContext,
    query: str,
    limit: int | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """RAG 기반 요약 검색 결과를 반환합니다."""
    query = _strip_time_tag(query)
    if not query:
        return {"success": False, "error": "query is empty"}

    video_id = tool_context.state.get("video_id")
    video_name = tool_context.state.get("video_name")
    if not video_id and not video_name:
        return {"success": False, "error": "video_name or video_id is not set"}

    if video_id:
        try:
            top_k = int(limit) if limit is not None else 5
        except (TypeError, ValueError):
            top_k = 5
        try:
            threshold_value = float(threshold) if threshold is not None else 0.4
        except (TypeError, ValueError):
            threshold_value = 0.4

        payload = {
            "query": query,
            "top_k": top_k,
            "threshold": threshold_value,
        }
        response = _call_process_api("POST", f"/videos/{video_id}/summaries/search", payload)
        if response.get("success"):
            items = response.get("payload", {}).get("items", [])
            tool_context.state["summary_cache"] = items
            tool_context.state["pending_updates"] = []
            return {
                "success": True,
                "summary_cache": items,
                "count": len(items),
                "source": "rag",
            }

    # Fallback to local file-based flow if DB search fails or video_id is missing
    fallback_updates = get_summary_updates(tool_context)
    if fallback_updates.get("success"):
        fallback_context = get_summary_context(tool_context)
        fallback_context["source"] = "local_fallback"
        return fallback_context

    return {
        "success": False,
        "error": "rag search failed",
        "details": fallback_updates.get("error"),
    }


summary_chat_agent = Agent(
    name="summary_chat_agent",
    model="gemini-2.5-flash",
    description="Chatbot for summary questions (API-backed).",
    instruction=(
        "You are the Summary Chatbot.\n"
        "Always respond in Korean.\n"
        "Workflow:\n"
        "1) Call get_summary_status when you need summary generation status.\n"
        "2) If the user asks which video is selected, call get_video_name and reply with it.\n"
        "3) If the user asks about summary generation status, reply with the status from get_summary_status.\n"
        "4) If the user explicitly asks to start or regenerate the summary, tell them to use the UI process_start button.\n"
        "5) For summary questions, first call get_summary_updates.\n"
        "   - If it returns 'video_name is not set', tell the user to select a video in the UI and stop.\n"
        "6) Then call get_summary_context to fetch summary_cache for answering.\n"
        "   - If the user message includes [time_ms=...], pass time_ms to get_summary_context.\n"
        "   - If matches is available, prioritize it and use summary_cache only for context.\n"
        "   - If out_of_range is true, say the summary is not updated for that time yet.\n"
        "   - If summary_cache is empty, say no summaries are available yet.\n"
        "7) Answer using ONLY summary_cache or matches. Do not guess beyond them."
    ),
    tools=[
        get_video_name,
        get_summary_status,
        get_summary_updates,
        get_summary_context,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)


summary_chat_rag_agent = Agent(
    name="summary_chat_rag_agent",
    model="gemini-2.5-flash",
    description="Chatbot for summary questions (RAG).",
    instruction=(
        "You are the Summary Chatbot (RAG).\n"
        "Always respond in Korean.\n"
        "Workflow:\n"
        "1) Call get_summary_status when you need summary generation status.\n"
        "2) If the user asks which video is selected, call get_video_name and reply with it.\n"
        "3) If the user asks about summary generation status, reply with the status from get_summary_status.\n"
        "4) If the user explicitly asks to start or regenerate the summary, tell them to use the UI process_start button.\n"
        "5) If the user message includes [time_ms=...], call get_summary_updates then get_summary_context with time_ms.\n"
        "6) For summary questions, call search_summary_context with query set to the user message.\n"
        "   - If it returns 'video_name is not set', tell the user to select a video in the UI and stop.\n"
        "   - If summary_cache is empty, say no relevant summaries are available.\n"
        "7) Answer using ONLY summary_cache or matches. Do not guess beyond them."
    ),
    tools=[
        get_video_name,
        get_summary_status,
        get_summary_updates,
        get_summary_context,
        search_summary_context,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)

root_agent = summary_chat_agent


__all__ = ["summary_chat_agent", "summary_chat_rag_agent", "root_agent"]
