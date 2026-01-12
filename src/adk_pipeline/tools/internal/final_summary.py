"""최종 요약 생성 tool - final_summary_*.md 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.fusion.final_summary_composer import compose_final_summaries


def write_final_summaries(
    *,
    summaries_jsonl: Path,
    outputs_dir: Path,
    generate_formats: List[str],
    max_chars_per_format: int,
    include_timestamps: bool,
    limit: Optional[int],
) -> Dict[str, str]:
    summaries = compose_final_summaries(
        summaries_jsonl=summaries_jsonl,
        max_chars=max_chars_per_format,
        include_timestamps=include_timestamps,
        limit=limit,
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for fmt in generate_formats:
        if fmt not in summaries:
            continue
        path = outputs_dir / f"final_summary_{fmt}.md"
        path.write_text(summaries[fmt], encoding="utf-8")
        written[fmt] = str(path)
    return written
