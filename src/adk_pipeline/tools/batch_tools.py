"""배치 처리 관리 도구들.

배치 모드에서 사용하는 도구:
- init_batch_mode: 배치 모드 초기화
- get_batch_info: 현재 배치 상태 조회
- get_current_batch_time_range: 현재 배치의 시간 범위 반환
- mark_batch_complete: 배치 완료 표시
- get_previous_context: 이전 배치 요약 context 반환
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..paths import DEFAULT_OUTPUT_BASE
from ..store import VideoStore

_OUTPUT_BASE = _PROJECT_ROOT / DEFAULT_OUTPUT_BASE


def _load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """manifest.json 로드."""
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_stt(stt_path: Path) -> Dict[str, Any]:
    """stt.json 로드."""
    if not stt_path.exists():
        return {}
    with open(stt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_context_from_summaries(
    summaries_path: Path,
    max_chars: int = 500,
) -> str:
    """요약 파일에서 핵심 context만 추출.

    각 segment의 첫 번째 bullet만 추출하여 context 생성.
    """
    if not summaries_path.exists():
        return ""

    context_parts: List[str] = []
    with open(summaries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                summary = json.loads(line)
                seg_id = summary.get("segment_id", "?")
                bullets = summary.get("summary", {}).get("bullets", [])
                if bullets:
                    claim = bullets[0].get("claim", "")
                    if claim:
                        context_parts.append(f"[Segment {seg_id}] {claim}")
            except json.JSONDecodeError:
                continue

    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    return context


def init_batch_mode(
    tool_context: ToolContext,
    batch_duration_ms: int = 200000,
    context_max_chars: int = 500,
) -> Dict[str, Any]:
    """배치 모드를 초기화합니다.

    manifest.json에서 duration을 계산하고 배치 개수를 결정합니다.

    Args:
        batch_duration_ms: 배치당 시간 길이 (밀리초, 기본 200000ms = 약 3.3분)
        context_max_chars: 이전 배치 context 최대 문자 수 (기본 500)

    Returns:
        success: 초기화 성공 여부
        total_duration_ms: 전체 비디오 길이
        total_batches: 전체 배치 개수
        batch_duration_ms: 배치당 시간 길이
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정. set_pipeline_config를 먼저 실행하세요."}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # manifest.json에서 duration 계산
    manifest = _load_manifest(store.manifest_json())
    if not manifest:
        return {"success": False, "error": "manifest.json이 없거나 비어있습니다."}

    # stt.json에서 duration_ms 확인
    stt = _load_stt(store.stt_json())
    stt_duration_ms = stt.get("duration_ms")

    # manifest의 마지막 캡처 타임스탬프
    max_timestamp_ms = max(
        item.get("timestamp_ms", item.get("start_ms", 0))
        for item in manifest
    )

    # duration 결정: stt_duration_ms가 있으면 사용, 없으면 manifest에서 추정
    if stt_duration_ms:
        total_duration_ms = int(stt_duration_ms)
    else:
        # manifest의 마지막 타임스탬프 + 여유 시간
        total_duration_ms = max_timestamp_ms + 10000

    # 배치 개수 계산
    total_batches = max(1, math.ceil(total_duration_ms / batch_duration_ms))

    # 캡처 개수
    total_captures = len(manifest)

    # batches 디렉토리 생성
    store.batches_dir().mkdir(parents=True, exist_ok=True)

    # state에 배치 정보 저장
    tool_context.state["batch_mode"] = True
    tool_context.state["batch_duration_ms"] = batch_duration_ms
    tool_context.state["total_duration_ms"] = total_duration_ms
    tool_context.state["total_batches"] = total_batches
    tool_context.state["current_batch_index"] = 0
    tool_context.state["completed_batches"] = []
    tool_context.state["context_max_chars"] = context_max_chars
    tool_context.state["previous_context"] = ""

    return {
        "success": True,
        "total_duration_ms": total_duration_ms,
        "total_batches": total_batches,
        "batch_duration_ms": batch_duration_ms,
        "total_captures": total_captures,
        "context_max_chars": context_max_chars,
        "message": f"배치 모드 초기화 완료. 총 {total_batches}개 배치로 처리합니다.",
    }


def get_batch_info(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 배치 상태를 조회합니다.

    Returns:
        batch_mode: 배치 모드 활성화 여부
        current_batch_index: 현재 처리 중인 배치 인덱스
        total_batches: 전체 배치 개수
        completed_batches: 완료된 배치 목록
        remaining_batches: 남은 배치 개수
    """
    batch_mode = tool_context.state.get("batch_mode", False)

    if not batch_mode:
        return {
            "batch_mode": False,
            "message": "배치 모드가 비활성화 상태입니다. init_batch_mode로 초기화하세요.",
        }

    current_batch_index = tool_context.state.get("current_batch_index", 0)
    total_batches = tool_context.state.get("total_batches", 0)
    completed_batches = tool_context.state.get("completed_batches", [])

    return {
        "batch_mode": True,
        "current_batch_index": current_batch_index,
        "total_batches": total_batches,
        "completed_batches": completed_batches,
        "remaining_batches": total_batches - len(completed_batches),
        "is_last_batch": current_batch_index >= total_batches - 1,
        "all_completed": len(completed_batches) >= total_batches,
    }


def get_current_batch_time_range(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 배치의 시간 범위를 반환합니다.

    Returns:
        start_ms: 배치 시작 시간 (밀리초)
        end_ms: 배치 종료 시간 (밀리초)
        batch_index: 현재 배치 인덱스
    """
    batch_mode = tool_context.state.get("batch_mode", False)

    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    current_batch_index = tool_context.state.get("current_batch_index", 0)
    batch_duration_ms = tool_context.state.get("batch_duration_ms", 200000)
    total_duration_ms = tool_context.state.get("total_duration_ms", 0)

    start_ms = current_batch_index * batch_duration_ms
    end_ms = min((current_batch_index + 1) * batch_duration_ms, total_duration_ms)

    return {
        "success": True,
        "batch_index": current_batch_index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
    }


def mark_batch_complete(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 배치를 완료 처리하고 다음 배치로 이동합니다.

    Returns:
        success: 완료 처리 성공 여부
        completed_batch_index: 완료된 배치 인덱스
        next_batch_index: 다음 배치 인덱스 (또는 None)
        all_completed: 모든 배치 완료 여부
    """
    batch_mode = tool_context.state.get("batch_mode", False)

    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    current_batch_index = tool_context.state.get("current_batch_index", 0)
    total_batches = tool_context.state.get("total_batches", 0)
    completed_batches = tool_context.state.get("completed_batches", [])
    context_max_chars = tool_context.state.get("context_max_chars", 500)

    # 완료 목록에 추가
    if current_batch_index not in completed_batches:
        completed_batches.append(current_batch_index)
        tool_context.state["completed_batches"] = completed_batches

    # 현재 배치의 요약에서 context 추출하여 누적
    current_summaries_path = store.batch_segment_summaries_jsonl(current_batch_index)
    if current_summaries_path.exists():
        new_context = _extract_context_from_summaries(
            current_summaries_path,
            max_chars=context_max_chars
        )
        previous_context = tool_context.state.get("previous_context", "")
        if previous_context:
            combined_context = f"{previous_context}\n{new_context}"
            # 전체 context 길이 제한
            if len(combined_context) > context_max_chars * 2:
                combined_context = combined_context[-(context_max_chars * 2):]
            tool_context.state["previous_context"] = combined_context
        else:
            tool_context.state["previous_context"] = new_context

    # 다음 배치로 이동
    next_batch_index = current_batch_index + 1
    all_completed = next_batch_index >= total_batches

    if not all_completed:
        tool_context.state["current_batch_index"] = next_batch_index

    return {
        "success": True,
        "completed_batch_index": current_batch_index,
        "next_batch_index": next_batch_index if not all_completed else None,
        "all_completed": all_completed,
        "total_completed": len(completed_batches),
        "total_batches": total_batches,
    }


def get_previous_context(tool_context: ToolContext) -> Dict[str, Any]:
    """이전 배치들의 요약 context를 반환합니다.

    이전 배치 요약에서 추출한 핵심 내용으로,
    다음 배치 요약 시 맥락 유지에 사용됩니다.

    Returns:
        context: 이전 배치들의 요약 context
        has_context: context가 있는지 여부
    """
    batch_mode = tool_context.state.get("batch_mode", False)

    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    previous_context = tool_context.state.get("previous_context", "")

    return {
        "success": True,
        "context": previous_context,
        "has_context": bool(previous_context),
        "context_length": len(previous_context),
    }


def reset_batch_mode(tool_context: ToolContext) -> Dict[str, Any]:
    """배치 모드를 초기화하고 처음부터 다시 시작합니다.

    Returns:
        success: 초기화 성공 여부
    """
    # 배치 관련 state 초기화
    tool_context.state["batch_mode"] = False
    tool_context.state["current_batch_index"] = 0
    tool_context.state["completed_batches"] = []
    tool_context.state["previous_context"] = ""

    return {
        "success": True,
        "message": "배치 모드가 초기화되었습니다.",
    }
