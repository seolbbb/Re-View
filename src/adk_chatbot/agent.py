from __future__ import annotations

import json
from typing import Any, Dict

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from src.adk_pipeline.paths import DEFAULT_OUTPUT_BASE
from src.adk_pipeline.store import VideoStore


def select_chat_mode(tool_context: ToolContext, mode: str) -> Dict[str, Any]:
    if not mode or not str(mode).strip():
        return {"success": False, "error": "mode is required"}
    normalized = str(mode).strip().lower()
    if normalized not in {"full", "partial"}:
        return {"success": False, "error": "mode must be 'full' or 'partial'"}
    tool_context.state["chat_mode"] = normalized
    return {"success": True, "chat_mode": normalized}


def start_summary_job(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name is not set"}
    return {
        "success": False,
        "error": "Summary API is not wired yet.",
        "video_name": video_name,
    }


def get_summary_status(tool_context: ToolContext) -> Dict[str, Any]:
    return {"success": False, "error": "Summary API is not wired yet."}


def get_video_name(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name is not set"}
    return {"success": True, "video_name": video_name}


def get_summary_updates(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name is not set"}

    # Local file-based fallback until the summary API is wired.
    store = VideoStore(output_base=DEFAULT_OUTPUT_BASE, video_name=video_name)
    summaries_path = store.segment_summaries_jsonl()
    if not summaries_path.exists():
        tool_context.state["pending_updates"] = []
        tool_context.state["last_segment_id"] = 0
        return {"success": True, "last_segment_id": 0, "new_count": 0}

    last_segment_id = tool_context.state.get("last_segment_id", 0)
    try:
        last_segment_id = int(last_segment_id)
    except (TypeError, ValueError):
        last_segment_id = 0

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
                existing_ids.add(int(record.get("segment_id")))
            except (TypeError, ValueError):
                continue
        for record in pending_updates:
            try:
                segment_id = int(record.get("segment_id"))
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


def partial_not_implemented(tool_context: ToolContext) -> Dict[str, Any]:
    return {"success": False, "error": "Partial summary chatbot is not implemented yet."}


summary_chat_agent = Agent(
    name="summary_chat_agent",
    model="gemini-2.5-flash",
    description="Chatbot for summary questions (API-backed).",
    instruction=(
        "You are the Summary Chatbot.\n"
        "IMPORTANT: You MUST call get_summary_updates FIRST before responding to ANY user message.\n"
        "Workflow:\n"
        "1) Call get_summary_updates.\n"
        "   - If it returns 'video_name is not set', tell the user to select a video in the UI and stop.\n"
        "2) If the user asks which video is selected, call get_video_name and reply with it.\n"
        "3) Otherwise, call get_summary_context to fetch summary_cache for answering.\n"
        "   - If the user message includes [time_ms=...], pass time_ms to get_summary_context.\n"
        "   - If matches is available, prioritize it and use summary_cache only for context.\n"
        "   - If out_of_range is true, say the summary is not updated for that time yet.\n"
        "   - If summary_cache is empty, say no summaries are available yet.\n"
        "4) Answer using ONLY summary_cache or matches. Do not guess beyond them."
    ),
    tools=[
        get_video_name,
        start_summary_job,
        get_summary_status,
        get_summary_updates,
        get_summary_context,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)


partial_summary_chat_agent = Agent(
    name="partial_summary_chat_agent",
    model="gemini-2.5-flash",
    description="Placeholder for partial-summary chatbot.",
    instruction=(
        "You are the Partial Summary Chatbot.\n"
        "Always call partial_not_implemented and relay the message."
    ),
    tools=[partial_not_implemented],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)


root_agent = Agent(
    name="adk_chatbot_root",
    model="gemini-2.5-flash",
    description="Root agent that routes users to the proper chatbot.",
    instruction=(
        "You are the routing agent for the ADK chatbot.\n"
        "Workflow:\n"
        "1) Ask the user to choose a chat mode: 'full' or 'partial'.\n"
        "2) Call select_chat_mode with the user's choice.\n"
        "3) Transfer to the matching sub-agent.\n"
        "If the user asks to change modes, repeat the selection flow."
    ),
    tools=[select_chat_mode],
    sub_agents=[summary_chat_agent, partial_summary_chat_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
)


summary_chat_agent._sub_agents = [root_agent]
partial_summary_chat_agent._sub_agents = [root_agent]


__all__ = ["root_agent"]
