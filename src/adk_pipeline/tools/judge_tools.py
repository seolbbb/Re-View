"""Judge Agent 도구들.

Judge Agent가 사용하는 도구:
- evaluate_summary: 요약 품질 평가 (PASS/FAIL)
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


def evaluate_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """요약 품질을 평가하고 PASS/FAIL을 반환합니다.

    현재는 stub으로 항상 PASS를 반환합니다.
    향후 실제 품질 평가 로직이 추가될 예정입니다.

    Returns:
        success: 실행 성공 여부
        result: PASS 또는 FAIL
        score: 평가 점수 (0.0 ~ 1.0)
        reason: 평가 이유
        can_rerun: 재실행 가능 여부
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # Summarize가 먼저 완료되어야 함
    if not store.segment_summaries_jsonl().exists():
        return {"success": False, "error": "segment_summaries.jsonl이 없습니다. Summarize를 먼저 실행하세요."}

    try:
        from .internal.judge_gemini import judge_stub_gemini
        result = judge_stub_gemini(fusion_dir=store.fusion_dir())

        # 재실행 가능 여부 확인
        current_rerun = tool_context.state.get("current_rerun", 1)
        max_reruns = tool_context.state.get("max_reruns", 2)
        passed = result.get("pass", True)
        can_rerun = not passed and current_rerun <= max_reruns

        return {
            "success": True,
            "result": "PASS" if passed else "FAIL",
            "score": result.get("score", 1.0),
            "reason": result.get("reason", "stub judge - 항상 통과"),
            "can_rerun": can_rerun,
            "attempt": current_rerun,
            "max_reruns": max_reruns,
            "judge_json": str(store.fusion_dir() / "judge.json"),
        }
    except Exception as e:
        return {"success": False, "error": f"Judge 실행 실패: {e}"}
