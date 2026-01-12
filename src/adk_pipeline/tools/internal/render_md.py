"""md로 변환 tool - segment_summaries.jsonl -> segment_summaries.md"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.fusion.renderer import render_segment_summaries_md


def render_md(
    *,
    summaries_jsonl: Path,
    output_md: Path,
    include_sources: bool,
    sources_jsonl: Optional[Path],
    md_wrap_width: int,
    limit: Optional[int],
) -> None:
    render_segment_summaries_md(
        summaries_jsonl=summaries_jsonl,
        output_md=output_md,
        include_sources=include_sources,
        sources_jsonl=sources_jsonl,
        md_wrap_width=md_wrap_width,
        limit=limit,
    )
