"""Summarize 재실행을 위한 산출물 보관 도구."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List


def archive_summarize_outputs(*, fusion_dir: Path, attempt_index: int) -> List[str]:
    """현재 summarize 산출물을 attempts/{attempt_index:02d}/ 아래로 복사한다."""

    attempts_dir = fusion_dir / "attempts" / f"attempt_{attempt_index:02d}"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []

    candidate_files = [
        fusion_dir / "segment_summaries.jsonl",
        fusion_dir / "segment_summaries.md",
        fusion_dir / "judge.json",
    ]

    outputs_dir = fusion_dir / "outputs"
    if outputs_dir.exists():
        for path in outputs_dir.glob("final_summary_*.md"):
            candidate_files.append(path)

    for path in candidate_files:
        if not path.exists():
            continue
        target = attempts_dir / path.name
        shutil.copy2(path, target)
        copied.append(str(target))

    return copied
