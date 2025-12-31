"""segment_summaries.jsonl -> Markdown 렌더러."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import format_ms, read_jsonl


def _wrap_line(prefix: str, text: str, width: int) -> List[str]:
    line = f"{prefix}{text}"
    if width <= 0:
        return [line]
    return textwrap.fill(
        text,
        width=width,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
    ).splitlines()


def _load_sources_map(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    mapping: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = int(row.get("segment_id"))
        mapping[segment_id] = {
            "transcript_units": row.get("transcript_units", []),
            "visual_units": row.get("visual_units", []),
        }
    return mapping


def _split_evidence_refs(evidence: Any) -> tuple[List[str], List[str]]:
    items: List[str] = []
    if isinstance(evidence, list):
        items = [str(item) for item in evidence]
    elif isinstance(evidence, dict):
        items.extend([str(item) for item in evidence.get("transcript_unit_ids", []) or []])
        items.extend([str(item) for item in evidence.get("visual_unit_ids", []) or []])
    elif isinstance(evidence, str):
        items = [evidence]

    t_refs: List[str] = []
    v_refs: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        if item.startswith("t"):
            t_refs.append(item)
        elif item.startswith("v"):
            v_refs.append(item)
    return t_refs, v_refs


def render_segment_summaries_md(
    summaries_jsonl: Path,
    output_md: Path,
    include_sources: bool = False,
    sources_jsonl: Optional[Path] = None,
    md_wrap_width: int = 0,
    limit: Optional[int] = None,
) -> None:
    sources_map = _load_sources_map(sources_jsonl) if include_sources else {}
    output_md.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_md.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(summaries_jsonl):
            if limit is not None and processed >= limit:
                break
            processed += 1

            segment_id = int(row.get("segment_id"))
            start_ms = int(row.get("start_ms"))
            end_ms = int(row.get("end_ms"))
            summary = row.get("summary", {}) or {}

            handle.write(f"### Segment {segment_id} ({format_ms(start_ms)}–{format_ms(end_ms)})\n")
            handle.write("- 요약\n")

            bullets = summary.get("bullets", []) or []
            for bullet in bullets:
                bullet_id = bullet.get("bullet_id", f"{segment_id}-?")
                claim = str(bullet.get("claim", "")).strip()
                for line in _wrap_line(f"  - ({bullet_id}) ", claim, md_wrap_width):
                    handle.write(line + "\n")

                t_refs_list, v_refs_list = _split_evidence_refs(bullet.get("evidence_refs"))
                t_refs = ",".join(t_refs_list)
                v_refs = ",".join(v_refs_list)
                handle.write(f"    - evidence: transcript=[{t_refs}], visual=[{v_refs}]\n")
                handle.write(f"    - confidence: {bullet.get('confidence', '')}\n")
                notes = str(bullet.get("notes", "")).strip()
                if notes:
                    handle.write(f"    - notes: {notes}\n")

            definitions = summary.get("definitions", []) or []
            if definitions:
                handle.write("- 정의\n")
                for item in definitions:
                    term = str(item.get("term", "")).strip()
                    definition = str(item.get("definition", "")).strip()
                    for line in _wrap_line("  - ", f"{term}: {definition}", md_wrap_width):
                        handle.write(line + "\n")
                    t_refs_list, v_refs_list = _split_evidence_refs(item.get("evidence_refs"))
                    t_refs = ",".join(t_refs_list)
                    v_refs = ",".join(v_refs_list)
                    handle.write(f"    - evidence: transcript=[{t_refs}], visual=[{v_refs}]\n")
                    confidence = str(item.get("confidence", "")).strip()
                    if confidence:
                        handle.write(f"    - confidence: {confidence}\n")
                    notes = str(item.get("notes", "")).strip()
                    if notes:
                        handle.write(f"    - notes: {notes}\n")

            explanations = summary.get("explanations", []) or []
            if explanations:
                handle.write("- 해설\n")
                for item in explanations:
                    if isinstance(item, dict):
                        point = str(item.get("point", "")).strip()
                        evidence = item.get("evidence_refs")
                        confidence = str(item.get("confidence", "")).strip()
                        notes = str(item.get("notes", "")).strip()
                    else:
                        point = str(item).strip()
                        evidence = None
                        confidence = ""
                        notes = ""
                    if not point:
                        continue
                    for line in _wrap_line("  - ", point, md_wrap_width):
                        handle.write(line + "\n")
                    t_refs_list, v_refs_list = _split_evidence_refs(evidence)
                    t_refs = ",".join(t_refs_list)
                    v_refs = ",".join(v_refs_list)
                    handle.write(f"    - evidence: transcript=[{t_refs}], visual=[{v_refs}]\n")
                    if confidence:
                        handle.write(f"    - confidence: {confidence}\n")
                    if notes:
                        handle.write(f"    - notes: {notes}\n")

            questions = summary.get("open_questions", []) or []
            if questions:
                handle.write("- 확인 불가/열린 질문\n")
                for question in questions:
                    if isinstance(question, dict):
                        text = str(question.get("question", "")).strip()
                        evidence = question.get("evidence_refs")
                        confidence = str(question.get("confidence", "")).strip()
                        notes = str(question.get("notes", "")).strip()
                    else:
                        text = str(question).strip()
                        evidence = None
                        confidence = ""
                        notes = ""
                    if not text:
                        continue
                    for line in _wrap_line("  - ", text, md_wrap_width):
                        handle.write(line + "\n")
                    t_refs_list, v_refs_list = _split_evidence_refs(evidence)
                    t_refs = ",".join(t_refs_list)
                    v_refs = ",".join(v_refs_list)
                    handle.write(f"    - evidence: transcript=[{t_refs}], visual=[{v_refs}]\n")
                    if confidence:
                        handle.write(f"    - confidence: {confidence}\n")
                    if notes:
                        handle.write(f"    - notes: {notes}\n")

            if include_sources:
                sources = sources_map.get(segment_id, {})
                t_units = sources.get("transcript_units", []) or []
                v_units = sources.get("visual_units", []) or []
                handle.write("- 근거 텍스트\n")
                if t_units:
                    transcript_parts = [f"[{u.get('unit_id')}] {u.get('text', '')}" for u in t_units]
                    handle.write("  - Transcript Units: " + ", ".join(transcript_parts) + "\n")
                if v_units:
                    visual_parts = [f"[{u.get('unit_id')}] {u.get('text', '')}" for u in v_units]
                    handle.write("  - Visual Units: " + ", ".join(visual_parts) + "\n")

            handle.write("\n")
