"""Summarize Agent 도구들.

Summarize Agent가 사용하는 도구:
- run_summarizer: 세그먼트별 요약 생성
- run_batch_summarizer: 배치별 세그먼트 요약 생성
- render_md: 마크다운 변환
- render_batch_md: 배치별 마크다운 변환
- write_final_summary: 최종 요약 생성
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from google.adk.tools import ToolContext

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..paths import DEFAULT_OUTPUT_BASE
from ..store import VideoStore

_OUTPUT_BASE = _PROJECT_ROOT / DEFAULT_OUTPUT_BASE


def run_summarizer(tool_context: ToolContext) -> Dict[str, Any]:
    """Gemini로 세그먼트별 요약을 생성합니다.

    Returns:
        success: 실행 성공 여부
        segment_summaries_jsonl: 생성된 파일 경로
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # Preprocessing이 먼저 완료되어야 함
    if not store.segments_units_jsonl().exists():
        return {"success": False, "error": "segments_units.jsonl이 없습니다. Preprocessing을 먼저 실행하세요."}

    # 재실행 시 이전 결과 아카이브
    current_rerun = tool_context.state.get("current_rerun", 0)
    if current_rerun > 0 and store.segment_summaries_jsonl().exists():
        from .internal.attempts import archive_summarize_outputs
        archive_summarize_outputs(fusion_dir=store.fusion_dir(), attempt_index=current_rerun)

    # 재실행 카운터 증가
    tool_context.state["current_rerun"] = current_rerun + 1

    try:
        from .internal.summarize import run_segment_summarizer
        run_segment_summarizer(
            fusion_config_path=store.fusion_config_yaml(),
            limit=None,
            dry_run=False,
        )

        return {
            "success": True,
            "attempt": current_rerun + 1,
            "segment_summaries_jsonl": str(store.segment_summaries_jsonl()),
        }
    except Exception as e:
        return {"success": False, "error": f"Summarizer 실행 실패: {e}"}


def render_md(tool_context: ToolContext) -> Dict[str, Any]:
    """요약을 마크다운으로 변환합니다.

    Returns:
        success: 실행 성공 여부
        segment_summaries_md: 생성된 파일 경로
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    fusion_dir = store.fusion_dir()

    # Summarizer가 먼저 완료되어야 함
    if not store.segment_summaries_jsonl().exists():
        return {"success": False, "error": "segment_summaries.jsonl이 없습니다. run_summarizer를 먼저 실행하세요."}

    try:
        from src.fusion.config import load_config
        config = load_config(str(store.fusion_config_yaml()))

        from .internal.render_md import render_md as _render_md
        _render_md(
            summaries_jsonl=fusion_dir / "segment_summaries.jsonl",
            output_md=fusion_dir / "segment_summaries.md",
            include_sources=config.raw.render.include_sources,
            sources_jsonl=fusion_dir / "segments_units.jsonl",
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=None,
        )

        return {
            "success": True,
            "segment_summaries_md": str(store.segment_summaries_md()),
        }
    except Exception as e:
        return {"success": False, "error": f"MD 렌더링 실패: {e}"}


def write_final_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """최종 요약을 생성합니다.

    Returns:
        success: 실행 성공 여부
        final_summary_dir: 생성된 파일들이 있는 디렉터리
        files: 생성된 파일 목록
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    fusion_dir = store.fusion_dir()

    # Summarizer가 먼저 완료되어야 함
    if not store.segment_summaries_jsonl().exists():
        return {"success": False, "error": "segment_summaries.jsonl이 없습니다. run_summarizer를 먼저 실행하세요."}

    try:
        from src.fusion.config import load_config
        config = load_config(str(store.fusion_config_yaml()))

        from .internal.final_summary import write_final_summaries as _write_final_summaries
        written = _write_final_summaries(
            summaries_jsonl=fusion_dir / "segment_summaries.jsonl",
            outputs_dir=fusion_dir / "outputs",
            generate_formats=[str(x) for x in config.raw.final_summary.generate_formats],
            max_chars_per_format=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=None,
        )

        return {
            "success": True,
            "final_summary_dir": str(store.final_outputs_dir()),
            "files": written,
        }
    except Exception as e:
        return {"success": False, "error": f"Final Summary 생성 실패: {e}"}


# ========== 배치 처리 도구 ==========


def run_batch_summarizer(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 배치의 세그먼트를 요약합니다.

    배치 모드에서 사용됩니다. 현재 배치의 segments_units.jsonl을 읽어
    이전 배치의 context를 반영하여 요약을 생성합니다.

    Returns:
        success: 실행 성공 여부
        batch_index: 처리된 배치 인덱스
        segment_summaries_jsonl: 생성된 파일 경로
        segments_count: 요약된 세그먼트 수
        context: 다음 배치에 전달할 context
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    batch_mode = tool_context.state.get("batch_mode", False)
    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 현재 배치 정보
    current_batch_index = tool_context.state.get("current_batch_index", 0)
    batch_dir = store.batch_dir(current_batch_index)

    # Sync가 먼저 완료되어야 함
    batch_segments_units = store.batch_segments_units_jsonl(current_batch_index)
    if not batch_segments_units.exists():
        return {
            "success": False,
            "error": f"배치 {current_batch_index}의 segments_units.jsonl이 없습니다. run_batch_sync를 먼저 실행하세요.",
        }

    # 이미 존재하면 스킵
    batch_summaries = store.batch_segment_summaries_jsonl(current_batch_index)
    if batch_summaries.exists():
        # 기존 파일에서 segment 수 확인
        segments_count = 0
        with open(batch_summaries, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    segments_count += 1

        # context 추출
        from src.fusion.summarizer import extract_batch_context
        context = extract_batch_context(batch_summaries, tool_context.state.get("context_max_chars", 500))

        return {
            "success": True,
            "skipped": True,
            "batch_index": current_batch_index,
            "segment_summaries_jsonl": str(batch_summaries),
            "segments_count": segments_count,
            "context": context,
            "message": f"배치 {current_batch_index}의 segment_summaries.jsonl이 이미 존재합니다. 스킵합니다.",
        }

    # 이전 context 가져오기
    previous_context = tool_context.state.get("previous_context", "")

    try:
        # Config 로드
        from src.fusion.config import load_config
        config = load_config(str(store.fusion_config_yaml()))

        # 배치 요약 실행
        from src.fusion.summarizer import run_batch_summarizer as _run_batch_summarizer
        result = _run_batch_summarizer(
            segments_units_jsonl=batch_segments_units,
            output_dir=batch_dir,
            config=config,
            previous_context=previous_context if previous_context else None,
        )

        # 다음 배치를 위해 context 업데이트
        new_context = result.get("context", "")
        # 이전 context와 새 context 결합 (최대 길이 제한)
        context_max_chars = tool_context.state.get("context_max_chars", 500)
        if previous_context and new_context:
            combined_context = f"{previous_context}\n{new_context}"
            if len(combined_context) > context_max_chars:
                # 너무 길면 최신 것만 유지
                combined_context = new_context[:context_max_chars]
            tool_context.state["previous_context"] = combined_context
        else:
            tool_context.state["previous_context"] = new_context

        return {
            "success": True,
            "skipped": False,
            "batch_index": current_batch_index,
            "segment_summaries_jsonl": result["segment_summaries_jsonl"],
            "segments_count": result["segments_count"],
            "context": new_context,
        }
    except Exception as e:
        return {"success": False, "error": f"배치 Summarizer 실행 실패: {e}"}


def render_batch_md(tool_context: ToolContext) -> Dict[str, Any]:
    """현재 배치의 요약을 마크다운으로 변환합니다.

    Returns:
        success: 실행 성공 여부
        batch_index: 처리된 배치 인덱스
        segment_summaries_md: 생성된 파일 경로
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    batch_mode = tool_context.state.get("batch_mode", False)
    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 현재 배치 정보
    current_batch_index = tool_context.state.get("current_batch_index", 0)
    batch_dir = store.batch_dir(current_batch_index)

    # Summarizer가 먼저 완료되어야 함
    batch_summaries = store.batch_segment_summaries_jsonl(current_batch_index)
    if not batch_summaries.exists():
        return {
            "success": False,
            "error": f"배치 {current_batch_index}의 segment_summaries.jsonl이 없습니다. run_batch_summarizer를 먼저 실행하세요.",
        }

    try:
        from src.fusion.config import load_config
        config = load_config(str(store.fusion_config_yaml()))

        from .internal.render_md import render_md as _render_md
        output_md = batch_dir / "segment_summaries.md"
        _render_md(
            summaries_jsonl=batch_summaries,
            output_md=output_md,
            include_sources=config.raw.render.include_sources,
            sources_jsonl=store.batch_segments_units_jsonl(current_batch_index),
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=None,
        )

        return {
            "success": True,
            "batch_index": current_batch_index,
            "segment_summaries_md": str(output_md),
        }
    except Exception as e:
        return {"success": False, "error": f"배치 MD 렌더링 실패: {e}"}
