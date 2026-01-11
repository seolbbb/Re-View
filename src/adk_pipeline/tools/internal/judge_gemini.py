"""Judge tool 구현.

Gemini를 사용해 요약 품질을 평가하고 judge.json을 생성한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.fusion.config import load_config
from src.fusion.io_utils import write_json
from src.judge.judge import run_judge


def run_judge_gemini(
    *,
    fusion_config_path: Path,
    fusion_dir: Path,
    segments_units_path: Path,
    segment_summaries_path: Path,
    batch_size: int,
    workers: int,
    json_repair_attempts: int,
    limit: Optional[int],
    verbose: bool,
    min_score: float,
    include_segments: bool,
) -> Dict[str, Any]:
    """Gemini로 요약 품질을 평가하고 judge.json을 반환한다."""
    config = load_config(str(fusion_config_path))
    report_path = fusion_dir / "judge" / "judge_report.json"
    segments_path = fusion_dir / "judge" / "judge_segment_reports.jsonl"

    judge_result = run_judge(
        config=config,
        segments_units_path=segments_units_path,
        segment_summaries_path=segment_summaries_path,
        output_report_path=report_path,
        output_segments_path=segments_path,
        batch_size=batch_size,
        workers=workers,
        json_repair_attempts=json_repair_attempts,
        limit=limit,
        verbose=verbose,
        write_outputs=False,
    )

    report = judge_result.get("report", {})
    segment_reports = judge_result.get("segment_reports", []) or []

    final_score = float(report.get("scores", {}).get("final", 0.0))
    passed = final_score >= min_score
    feedback = [
        {"segment_id": int(item.get("segment_id")), "feedback": str(item.get("feedback", "")).strip()}
        for item in segment_reports
        if item.get("segment_id") is not None
    ]
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "model": str(report.get("meta", {}).get("model", "")),
        "pass": passed,
        "final_score": final_score,
        "min_score": min_score,
        "prompt_version": str(report.get("meta", {}).get("prompt_version", "")),
        "generated_at_utc": str(report.get("meta", {}).get("generated_at_utc", "")),
        "feedback": feedback,
    }
    if include_segments:
        payload["segments"] = [
            {
                "segment_id": int(item.get("segment_id")),
                "scores": item.get("scores", {}),
            }
            for item in segment_reports
            if item.get("segment_id") is not None
        ]
    write_json(fusion_dir / "judge.json", payload)
    return payload
