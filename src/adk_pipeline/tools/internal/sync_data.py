"""Sync data tool - fusion sync engine 실행."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.fusion.config import load_config
from src.fusion.sync_engine import run_sync_engine, run_batch_sync_engine


def run_sync(
    *,
    fusion_config_path: Path,
    limit: Optional[int],
) -> Dict[str, str]:
    config = load_config(str(fusion_config_path))
    run_sync_engine(config, limit=limit, dry_run=False)

    output_dir = config.paths.output_root / "fusion"
    return {
        "segments_jsonl": str(output_dir / "segments.jsonl"),
        "segments_units_jsonl": str(output_dir / "segments_units.jsonl"),
    }
