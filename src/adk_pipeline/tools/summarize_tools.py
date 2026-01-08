"""Summarize Agent 도구들.

Summarize Agent가 사용하는 도구:
- run_summarizer: 세그먼트별 요약 생성
- render_md: 마크다운 변환
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
