"""요약하기 tool - fusion summarizer 실행."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.fusion.config import load_config
from src.fusion.summarizer import run_summarizer, run_batch_summarizer


def run_segment_summarizer(
    *,
    fusion_config_path: Path,
    limit: Optional[int],
    dry_run: bool,
) -> Dict[str, str]:
    config = load_config(str(fusion_config_path))
    run_summarizer(config, limit=limit, dry_run=dry_run)

    output_dir = config.paths.output_root / "fusion"
    return {"segment_summaries_jsonl": str(output_dir / "segment_summaries.jsonl")}
