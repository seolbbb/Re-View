"""Root Agent 도구들.

Root Agent가 사용하는 도구:
- list_available_videos: 처리 가능한 비디오 목록 조회
- set_pipeline_config: 비디오 선택 및 설정
- get_pipeline_status: 현재 파이프라인 상태 조회
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..paths import DEFAULT_OUTPUT_BASE, sanitize_video_name
from ..store import VideoStore

_OUTPUT_BASE = _PROJECT_ROOT / DEFAULT_OUTPUT_BASE


def list_available_videos(tool_context: ToolContext) -> Dict[str, Any]:
    """data/outputs에서 처리 가능한 비디오 목록을 조회합니다.

    Pre-ADK 단계(STT, Capture)가 완료된 비디오만 표시합니다.

    Returns:
        videos: 비디오 목록 (이름, 각 단계 완료 여부)
        count: 처리 가능한 비디오 수
    """
    if not _OUTPUT_BASE.exists():
        return {"videos": [], "count": 0, "message": f"출력 디렉터리 없음: {_OUTPUT_BASE}"}

    videos: List[Dict[str, Any]] = []
    for video_dir in sorted(_OUTPUT_BASE.iterdir()):
        if not video_dir.is_dir():
            continue
        store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_dir.name)

        # Pre-ADK 산출물 확인
        stt_ok = store.stt_json().exists()
        manifest_ok = store.manifest_json().exists()
        captures_ok = store.captures_dir().exists()

        if stt_ok and manifest_ok and captures_ok:
            videos.append({
                "name": video_dir.name,
                "preprocessing_done": store.segments_units_jsonl().exists(),
                "summarize_done": store.segment_summaries_jsonl().exists(),
                "judge_done": (store.fusion_dir() / "judge.json").exists(),
            })

    return {
        "videos": videos,
        "count": len(videos),
        "message": f"{len(videos)}개의 처리 가능한 비디오가 있습니다." if videos else "처리 가능한 비디오가 없습니다.",
    }


def set_pipeline_config(
    tool_context: ToolContext,
    video_name: str,
    summarize_prompt: Optional[str] = None,
    max_reruns: int = 2,
    force_preprocessing: bool = False,
    vlm_batch_size: Optional[int] = 2,
    vlm_concurrency: int = 3,
    vlm_show_progress: bool = True,
<<<<<<< HEAD
) -> Dict[str, Any]:
    """파이프라인 설정을 변경합니다.

=======
    batch_mode: bool = True,
    batch_size: int = 4,
    context_max_chars: int = 500,
) -> Dict[str, Any]:
    """파이프라인 설정을 변경합니다.


>>>>>>> feat
    Args:
        video_name: 처리할 비디오 이름 (data/outputs 하위 디렉터리 이름)
        summarize_prompt: 요약 시 사용할 추가 프롬프트 (선택)
        max_reruns: Judge 실패 시 최대 재실행 횟수 (기본: 2)
        force_preprocessing: True면 기존 파일 삭제 후 Preprocessing 재실행 (기본: False)
        vlm_batch_size: VLM 배치 크기 (기본: 2, None이면 전체를 한 번에 요청)
        vlm_concurrency: VLM 병렬 요청 수 (기본: 3)
        vlm_show_progress: VLM 진행 로그 출력 여부 (기본: True)
<<<<<<< HEAD
=======
        batch_mode: 배치 처리 모드 활성화 (기본: True)
        batch_size: 배치당 캡처 개수 (기본: 4장)
        context_max_chars: 이전 배치 context 최대 문자 수 (기본: 500)
>>>>>>> feat

    Returns:
        success: 설정 성공 여부
        video_name: 설정된 비디오 이름
        status: 각 단계 완료 상태
    """
    sanitized = sanitize_video_name(video_name)
    video_root = _OUTPUT_BASE / sanitized

    if not video_root.exists():
        return {"success": False, "error": f"비디오 디렉터리 없음: {sanitized}"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=sanitized)

    if not store.stt_json().exists() or not store.manifest_json().exists():
        return {"success": False, "error": "Pre-ADK 산출물 없음 (stt.json, manifest.json 필요)"}

    # 세션 상태에 설정 저장
    tool_context.state["video_name"] = sanitized
    tool_context.state["summarize_prompt"] = summarize_prompt
    tool_context.state["max_reruns"] = max_reruns
    tool_context.state["current_rerun"] = 0
    tool_context.state["force_preprocessing"] = force_preprocessing
    tool_context.state["vlm_batch_size"] = vlm_batch_size
    tool_context.state["vlm_concurrency"] = vlm_concurrency
    tool_context.state["vlm_show_progress"] = vlm_show_progress

<<<<<<< HEAD
=======
    # 배치 모드 설정 (기본값: True)
    tool_context.state["batch_mode"] = batch_mode
    tool_context.state["batch_size"] = batch_size
    tool_context.state["context_max_chars"] = context_max_chars
    if batch_mode:
        # 배치 모드 관련 초기값 설정 (init_batch_mode에서 최종 설정됨)
        tool_context.state["current_batch_index"] = 0
        tool_context.state["completed_batches"] = []
        tool_context.state["previous_context"] = ""

>>>>>>> feat
    return {
        "success": True,
        "video_name": sanitized,
        "summarize_prompt": summarize_prompt,
        "max_reruns": max_reruns,
        "force_preprocessing": force_preprocessing,
        "vlm_batch_size": vlm_batch_size,
        "vlm_concurrency": vlm_concurrency,
        "vlm_show_progress": vlm_show_progress,
<<<<<<< HEAD
=======
        "batch_mode": batch_mode,
        "batch_size": batch_size,
        "context_max_chars": context_max_chars,
>>>>>>> feat
        "video_root": str(video_root),
        "status": {
            "preprocessing_done": store.segments_units_jsonl().exists(),
            "summarize_done": store.segment_summaries_jsonl().exists(),
            "judge_done": (store.fusion_dir() / "judge.json").exists(),
        },
    }


def get_pipeline_status(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 파이프라인 상태를 조회합니다.

    Returns:
        video_name: 현재 설정된 비디오 이름
        config: 현재 설정 값들
        status: 각 단계 완료 상태
        outputs: 생성된 산출물 경로들
    """
    video_name = tool_context.state.get("video_name")

    if not video_name:
        return {
            "video_name": None,
            "message": "video_name 미설정. set_pipeline_config로 비디오를 선택하세요.",
        }

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 존재하는 산출물 목록
    outputs = {}
    if store.vlm_json().exists():
        outputs["vlm_json"] = str(store.vlm_json())
    if store.segments_units_jsonl().exists():
        outputs["segments_units_jsonl"] = str(store.segments_units_jsonl())
    if store.segment_summaries_jsonl().exists():
        outputs["segment_summaries_jsonl"] = str(store.segment_summaries_jsonl())
    if store.final_outputs_dir().exists():
        final_files = list(store.final_outputs_dir().glob("final_summary_*.md"))
        if final_files:
            outputs["final_summaries"] = [str(f) for f in final_files]

<<<<<<< HEAD
=======
    # 배치 모드 상태
    batch_status = None
    batch_mode = tool_context.state.get("batch_mode", False)
    if batch_mode:
        batch_status = {
            "current_batch_index": tool_context.state.get("current_batch_index", 0),
            "total_batches": tool_context.state.get("total_batches", 0),
            "completed_batches": tool_context.state.get("completed_batches", []),
            "batch_size": tool_context.state.get("batch_size", 10),
        }

>>>>>>> feat
    return {
        "video_name": video_name,
        "video_root": str(store.video_root()),
        "config": {
            "summarize_prompt": tool_context.state.get("summarize_prompt"),
            "max_reruns": tool_context.state.get("max_reruns", 2),
            "current_rerun": tool_context.state.get("current_rerun", 0),
            "vlm_batch_size": tool_context.state.get("vlm_batch_size", 2),
            "vlm_concurrency": tool_context.state.get("vlm_concurrency", 3),
            "vlm_show_progress": tool_context.state.get("vlm_show_progress", True),
<<<<<<< HEAD
=======
            "batch_mode": batch_mode,
>>>>>>> feat
        },
        "status": {
            "preprocessing_done": store.segments_units_jsonl().exists(),
            "summarize_done": store.segment_summaries_jsonl().exists(),
            "judge_done": (store.fusion_dir() / "judge.json").exists(),
        },
<<<<<<< HEAD
=======
        "batch_status": batch_status,
>>>>>>> feat
        "outputs": outputs,
    }
