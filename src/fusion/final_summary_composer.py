"""최종 요약 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .io_utils import format_ms, read_jsonl


def _truncate_lines(lines: List[str], max_chars: int) -> str:
    if max_chars <= 0:
        return "\n".join(lines)
    joined = "\n".join(lines)
    if len(joined) <= max_chars:
        return joined
    suffix = "...(이하 생략)"
    truncated = lines[:]
    while truncated and len("\n".join(truncated + [suffix])) > max_chars:
        truncated.pop()
    if not truncated:
        return suffix[:max_chars]
    truncated.append(suffix)
    return "\n".join(truncated)


def _collect_segments(
    summaries_jsonl: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    processed = 0
    for row in read_jsonl(summaries_jsonl):
        if limit is not None and processed >= limit:
            break
        processed += 1
        segment_id = int(row.get("segment_id"))
        start_ms = int(row.get("start_ms"))
        end_ms = int(row.get("end_ms"))
        summary = row.get("summary", {}) or {}
        segments.append(
            {
                "segment_id": segment_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "summary": summary,
            }
        )
    return segments


def _append_segment_details(
    lines: List[str],
    segment: Dict[str, object],
    include_timestamps: bool,
) -> None:
    segment_id = int(segment["segment_id"])
    start_ms = int(segment["start_ms"])
    end_ms = int(segment["end_ms"])
    summary = segment.get("summary", {}) or {}
    bullets = summary.get("bullets", []) or []
    definitions = summary.get("definitions", []) or []
    explanations = summary.get("explanations", []) or []
    open_questions = summary.get("open_questions", []) or []

    if include_timestamps:
        lines.append(
            f"#### Segment {segment_id} ({format_ms(start_ms)}–{format_ms(end_ms)})"
        )
    else:
        lines.append(f"#### Segment {segment_id}")

    if bullets:
        lines.append("- 요약")
        for bullet in bullets:
            claim = str(bullet.get("claim", "")).strip()
            if not claim:
                continue
            bullet_id = str(bullet.get("bullet_id", "")).strip()
            if bullet_id:
                lines.append(f"  - ({bullet_id}) {claim}")
            else:
                lines.append(f"  - {claim}")

    if definitions:
        lines.append("- 정의")
        for item in definitions:
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            if term and definition:
                lines.append(f"  - {term}: {definition}")

    if explanations:
        lines.append("- 해설")
        for item in explanations:
            point = str(item.get("point", "")).strip()
            if point:
                lines.append(f"  - {point}")

    if open_questions:
        lines.append("- 열린 질문")
        for item in open_questions:
            question = str(item.get("question", "")).strip()
            if question:
                lines.append(f"  - {question}")

    lines.append("")


def build_summary_timeline(
    segments: List[Dict[str, object]], include_timestamps: bool
) -> str:
    lines = ["# Final Summary (시간 순 상세)"]
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        _append_segment_details(lines, segment, include_timestamps)
    return "\n".join(lines).strip()


def build_summary_tldr_timeline(
    segments: List[Dict[str, object]], include_timestamps: bool
) -> str:
    lines = ["# Final Summary (TL;DR + 시간 순)"]
    lines.append("## TL;DR")
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        summary = segment.get("summary", {}) or {}
        bullets = summary.get("bullets", []) or []
        for bullet in bullets:
            claim = str(bullet.get("claim", "")).strip()
            if not claim:
                continue
            bullet_id = str(bullet.get("bullet_id", "")).strip()
            if bullet_id:
                lines.append(f"- ({bullet_id}) {claim}")
            else:
                lines.append(f"- {claim}")

    lines.append("")
    lines.append("## 시간 순 요약")
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        _append_segment_details(lines, segment, include_timestamps)

    return "\n".join(lines).strip()


def compose_final_summaries(
    summaries_jsonl: Path,
    max_chars: int,
    include_timestamps: bool,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    segments = _collect_segments(summaries_jsonl, limit=limit)

    summary_timeline = _truncate_lines(
        build_summary_timeline(segments, include_timestamps).splitlines(), max_chars
    )
    summary_tldr_timeline = _truncate_lines(
        build_summary_tldr_timeline(segments, include_timestamps).splitlines(),
        max_chars,
    )

    return {"timeline": summary_timeline, "tldr_timeline": summary_tldr_timeline}
