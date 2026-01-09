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
    return_reasons: bool,
    verbose: bool,
    min_score: float,
) -> Dict[str, Any]:
    """Gemini로 요약 품질을 평가하고 judge.json을 반환한다."""
    config = load_config(str(fusion_config_path))
    judge_dir = fusion_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    report_path = judge_dir / "judge_report.json"
    segments_path = judge_dir / "judge_segment_reports.jsonl"

    report = run_judge(
        config=config,
        segments_units_path=segments_units_path,
        segment_summaries_path=segment_summaries_path,
        output_report_path=report_path,
        output_segments_path=segments_path,
        batch_size=batch_size,
        workers=workers,
        json_repair_attempts=json_repair_attempts,
        limit=limit,
        return_reasons=return_reasons,
        verbose=verbose,
    )

    final_score = float(report.get("scores", {}).get("final", 0.0))
    passed = final_score >= min_score
    result: Dict[str, Any] = {
        "schema_version": 1,
        "model": str(report.get("meta", {}).get("model", "")),
        "pass": passed,
        "score": round(final_score / 10.0, 4),
        "final_score": final_score,
        "min_score": min_score,
        "reason": f"최종 점수 {final_score:.2f} / 기준 {min_score:.2f}",
        "report_path": str(report_path),
        "segment_reports_path": str(segments_path),
    }
    write_json(fusion_dir / "judge.json", result)
    return result
