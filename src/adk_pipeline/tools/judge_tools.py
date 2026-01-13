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


def _read_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _read_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _read_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)



def evaluate_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """요약 품질을 평가하고 PASS/FAIL을 반환합니다.

    배치 모드일 때는 현재 배치의 summaries를 평가합니다.

    Returns:
        success: 실행 성공 여부
        result: PASS 또는 FAIL
        score: 평가 점수 (0.0 ~ 1.0)
        can_rerun: 재실행 가능 여부
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    batch_mode = tool_context.state.get("batch_mode", False)

    # 배치 모드: 현재 배치 평가
    if batch_mode:
        current_batch_index = tool_context.state.get("current_batch_index", 0)
        batch_dir = store.batch_dir(current_batch_index)
        batch_summaries = store.batch_segment_summaries_jsonl(current_batch_index)
        batch_segments = store.batch_segments_units_jsonl(current_batch_index)

        if not batch_summaries.exists():
            return {"success": False, "error": f"배치 {current_batch_index} segment_summaries.jsonl이 없습니다."}
        if not batch_segments.exists():
            return {"success": False, "error": f"배치 {current_batch_index} segments_units.jsonl이 없습니다."}

        batch_judge_json = batch_dir / "judge.json"

        # 이미 존재하면 스킵
        if batch_judge_json.exists():
            import json
            with open(batch_judge_json, "r", encoding="utf-8") as f:
                existing_result = json.load(f)
            passed = existing_result.get("pass", True)
            final_score = float(existing_result.get("final_score", 0.0))
            score = round(final_score / 10.0, 4)
            return {
                "success": True,
                "skipped": True,
                "batch_index": current_batch_index,
                "result": "PASS" if passed else "FAIL",
                "score": score,
                "judge_json": str(batch_judge_json),
            }

        try:
            min_score = _read_float(tool_context.state.get("judge_min_score"), 7.0)
            include_segments = _read_bool(
                tool_context.state.get("judge_include_segments"), False
            )

            from .internal.judge_gemini import run_judge_gemini
            result = run_judge_gemini(
                fusion_config_path=store.fusion_config_yaml(),
                fusion_dir=batch_dir,
                segments_units_path=batch_segments,
                segment_summaries_path=batch_summaries,
                batch_size=3,
                workers=1,
                json_repair_attempts=1,
                limit=None,
                verbose=False,
                min_score=min_score,
                include_segments=include_segments,
            )

            final_score = float(result.get("final_score", 0.0))
            score = round(final_score / 10.0, 4)
            passed = result.get("pass", True)

            current_rerun = tool_context.state.get("batch_rerun_count", 0)
            max_reruns = tool_context.state.get("max_reruns", 2)
            can_rerun = not passed and current_rerun < max_reruns

            return {
                "success": True,
                "batch_index": current_batch_index,
                "result": "PASS" if passed else "FAIL",
                "score": score,
                "can_rerun": can_rerun,
                "judge_json": str(batch_judge_json),
            }
        except Exception as e:
            return {"success": False, "error": f"배치 Judge 실행 실패: {e}"}

    # 일반 모드: 전체 평가
    if not store.segment_summaries_jsonl().exists():
        return {"success": False, "error": "segment_summaries.jsonl이 없습니다."}
    if not store.segments_units_jsonl().exists():
        return {"success": False, "error": "segments_units.jsonl이 없습니다."}
    if not store.fusion_config_yaml().exists():
        return {"success": False, "error": "config.yaml이 없습니다."}

    try:
        batch_size = _read_int(tool_context.state.get("judge_batch_size"), 3)
        workers = _read_int(tool_context.state.get("judge_workers"), 1)
        json_repair_attempts = _read_int(
            tool_context.state.get("judge_json_repair_attempts"), 1
        )
        limit_raw = tool_context.state.get("judge_limit")
        limit = _read_int(limit_raw, 0) if limit_raw is not None else None
        min_score = _read_float(tool_context.state.get("judge_min_score"), 7.0)
        verbose = _read_bool(tool_context.state.get("judge_verbose"), False)
        include_segments = _read_bool(
            tool_context.state.get("judge_include_segments"), False
        )

        from .internal.judge_gemini import run_judge_gemini
        result = run_judge_gemini(
            fusion_config_path=store.fusion_config_yaml(),
            fusion_dir=store.fusion_dir(),
            segments_units_path=store.segments_units_jsonl(),
            segment_summaries_path=store.segment_summaries_jsonl(),
            batch_size=batch_size,
            workers=workers,
            json_repair_attempts=json_repair_attempts,
            limit=limit,
            verbose=verbose,
            min_score=min_score,
            include_segments=include_segments,
        )

        final_score = float(result.get("final_score", 0.0))
        score = round(final_score / 10.0, 4)

        current_rerun = tool_context.state.get("current_rerun", 1)
        max_reruns = tool_context.state.get("max_reruns", 2)
        passed = result.get("pass", True)
        can_rerun = not passed and current_rerun <= max_reruns

        return {
            "success": True,
            "result": "PASS" if passed else "FAIL",
            "score": score,
            "can_rerun": can_rerun,
            "attempt": current_rerun,
            "max_reruns": max_reruns,
            "judge_json": str(store.fusion_dir() / "judge.json"),
        }
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"judge 옵션 형식이 올바르지 않습니다: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Judge 실행 실패: {e}"}
