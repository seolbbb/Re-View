"""segment_summaries.jsonl -> Markdown renderer."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import format_ms, read_jsonl


def _wrap_line(prefix: str, text: str, width: int) -> List[str]:
    """텍스트를 주어진 너비에 맞춰 줄바꿈 처리한다."""
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
    """근거 텍스트가 포함된 JSONL 파일을 로드하여 세그먼트 ID별 맵을 생성한다."""
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
    """근거 참조(evidence) 데이터를 transcript(t)와 visual(v) ID 리스트로 분리한다."""
    items: List[str] = []
    if isinstance(evidence, list):
        items = [str(item) for item in evidence]
    elif isinstance(evidence, dict):
        items.extend(
            [str(item) for item in evidence.get("transcript_unit_ids", []) or []]
        )
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
        if item.startswith("stt_"):
            t_refs.append(item)
        elif item.startswith("vlm") or item.startswith("cap_"):
            v_refs.append(item)
    return t_refs, v_refs


def render_summary_to_text(summary: Dict[str, Any]) -> str:
    """JSONB 요약 데이터를 시맨틱 검색용 평문 텍스트로 변환합니다.
    
    임베딩 벡터 생성에 적합한 형식으로 요약 내용을 라벨링하여 변환합니다.
    마크다운 기호를 제거하고, 의미 구분을 위한 태그([핵심], [정의] 등)를 사용합니다.
    
    Args:
        summary: segment_summaries.jsonl의 summary 필드 (JSONB 구조)
            - bullets: 핵심 내용 리스트
            - definitions: 정의 리스트
            - explanations: 해설 리스트
            - open_questions: 열린 질문 리스트
        
    Returns:
        str: 시맨틱 검색에 최적화된 라벨링된 평문 텍스트
        
    Example:
        >>> summary = {"bullets": [{"claim": "ELBO는..."}], "definitions": [...]}
        >>> render_summary_to_text(summary)
        "[핵심] ELBO는...\\n[정의] KL Divergence: ..."
    """
    if not summary:
        return ""
    
    lines: List[str] = []
    
    # 1. Bullets (핵심 내용)
    for bullet in summary.get("bullets", []) or []:
        claim = str(bullet.get("claim", "")).strip()
        if claim:
            lines.append(f"[핵심] {claim}")
    
    # 2. Definitions (정의)
    for defn in summary.get("definitions", []) or []:
        term = str(defn.get("term", "")).strip()
        definition = str(defn.get("definition", "")).strip()
        if term and definition:
            lines.append(f"[정의] {term}: {definition}")
    
    # 3. Explanations (해설)
    for expl in summary.get("explanations", []) or []:
        if isinstance(expl, dict):
            point = str(expl.get("point", "")).strip()
            notes = str(expl.get("notes", "")).strip()
            if point:
                text = f"[해설] {point}"
                if notes:
                    text += f" - {notes}"
                lines.append(text)
        elif isinstance(expl, str) and expl.strip():
            lines.append(f"[해설] {expl.strip()}")
    
    # 4. Open Questions (질문)
    for q in summary.get("open_questions", []) or []:
        if isinstance(q, dict):
            question = str(q.get("question", "")).strip()
        else:
            question = str(q).strip()
        if question:
            lines.append(f"[질문] {question}")
    
    return "\n".join(lines)


def render_segment_summaries_md(
    summaries_jsonl: Path,
    output_md: Path,
    include_sources: bool = False,
    sources_jsonl: Optional[Path] = None,
    md_wrap_width: int = 0,
    limit: Optional[int] = None,
    group_order: Optional[List[str]] = None,
    group_headers: Optional[Dict[str, str]] = None,
    fusion_prompt_version: Optional[str] = None,
    judge_prompt_version: Optional[str] = None,
    # New metadata fields
    execution_time: Optional[Dict[str, float]] = None,
    batch_config: Optional[Dict[str, Any]] = None,
    judge_stats: Optional[Dict[str, Any]] = None,
    token_usage: Optional[Dict[str, int]] = None,
) -> None:
    """세그먼트 요약 JSONL을 읽어 상세 마크다운 리포트를 생성한다. (Standardized V2 Format)"""
    sources_map = _load_sources_map(sources_jsonl) if include_sources else {}
    output_md.parent.mkdir(parents=True, exist_ok=True)

    # Defaults
    if group_order is None:
        group_order = ["direct", "background", "inferred"]
    
    if group_headers is None:
        group_headers = {
            "direct": "핵심 내용 (Direct / Recall)",
            "inferred": "심화/추론 (Inferred / Logic)",
            "background": "배경 지식 (Background)"
        }

    # Read all lines first to count total
    rows = list(read_jsonl(summaries_jsonl))
    if limit is not None:
        rows = rows[:limit]

    with output_md.open("w", encoding="utf-8") as handle:
        # Header
        handle.write(f"# Segment Summaries Report\n\n")
        handle.write(f"| Property | Value |\n")
        handle.write(f"|:---|:---|\n")
        handle.write(f"| Source File | `{summaries_jsonl.name}` |\n")
        handle.write(f"| Total Segments | {len(rows)} |\n")
        
        # Version Info
        if fusion_prompt_version:
            handle.write(f"| Fusion Prompt | `{fusion_prompt_version}` |\n")
        if judge_prompt_version:
            handle.write(f"| Judge Prompt | `{judge_prompt_version}` |\n")

        # Judge Score
        if judge_stats and "final_score" in judge_stats:
            score = judge_stats["final_score"]
            pass_status = "PASS" if judge_stats.get("passed") else "FAIL"
            handle.write(f"| Judge Score | **{score:.2f}** ({pass_status}) |\n")
            
            # Category-wise scores (5 categories)
            category_scores = judge_stats.get("category_scores", {})
            if category_scores:
                score_parts = []
                # Display in a readable order
                for key in ["compliance", "groundedness", "note_quality", "multimodal_use"]:
                    if key in category_scores:
                        # Shorten key names for display
                        short_name = {
                            "compliance": "Comp",
                            "groundedness": "Ground",
                            "note_quality": "Quality",
                            "multimodal_use": "Multimodal",
                        }.get(key, key.capitalize())
                        score_parts.append(f"{short_name}={category_scores[key]:.1f}")
                if score_parts:
                    handle.write(f"| Category Scores | {', '.join(score_parts)} |\n")

        # Batch Config
        if batch_config:
            b_size = batch_config.get("batch_size", "-")
            workers = batch_config.get("workers", "-")
            handle.write(f"| Batch Settings | Batch={b_size}, Workers={workers} |\n")

        # Execution Time (Summarizer & Judge)
        if execution_time:
            sum_time = execution_time.get("summarizer", 0.0)
            judge_time = execution_time.get("judge", 0.0)
            total_fusion = sum_time + judge_time
            time_str = f"Total: {total_fusion:.1f}s (Sum: {sum_time:.1f}s, Judge: {judge_time:.1f}s)"
            handle.write(f"| Execution Time | {time_str} |\n")

        # Token Usage
        if token_usage:
            sum_tokens = token_usage.get("summarizer", 0)
            judge_tokens = token_usage.get("judge", 0)
            total_tokens = sum_tokens + judge_tokens
            handle.write(f"| Token Usage | Total: {total_tokens:,} (Sum: {sum_tokens:,}, Judge: {judge_tokens:,}) |\n")
            
        handle.write("\n")

        processed = 0
        for row in rows:
            processed += 1
            segment_id = int(row.get("segment_id"))
            start_ms = int(row.get("start_ms", 0))
            end_ms = int(row.get("end_ms", 0))
            summary = row.get("summary", {}) or {}

            # Time formatting
            start_sec = start_ms // 1000
            end_sec = end_ms // 1000
            start_str = f"{start_sec//60:02d}:{start_sec%60:02d}"
            end_str = f"{end_sec//60:02d}:{end_sec%60:02d}"

            handle.write(
                f"### Segment {segment_id} ({start_str}-{end_str})\n"
            )

            # Resolve Valid IDs for Evidence Validation
            row_refs = row.get("source_refs", {})
            valid_stt = set(row_refs.get("stt_ids", []))
            valid_vlm = set(row_refs.get("vlm_ids", []))

            # If not in row, check sources_map
            if not valid_stt and not valid_vlm and segment_id in sources_map:
                s_data = sources_map[segment_id]
                t_units = s_data.get("transcript_units", [])
                v_units = s_data.get("visual_units", [])
                valid_stt = set(u.get("unit_id", "") for u in t_units if u.get("unit_id"))
                valid_vlm = set(u.get("unit_id", "") for u in v_units if u.get("unit_id"))

            def _validate_and_format(refs):
                if not refs:
                    return ""
                
                t_refs = []
                v_refs = []
                
                for r in refs:
                    r_str = str(r)
                    if r_str in valid_stt:
                        t_refs.append(r_str)
                    elif r_str in valid_vlm:
                        v_refs.append(r_str)
                    else:
                        # Heuristic fallback
                        if r_str.startswith("stt"):
                            t_refs.append(r_str)
                        elif r_str.startswith("vlm") or r_str.startswith("cap_"):
                            v_refs.append(r_str)
                
                parts = []
                t_refs_str = str(t_refs).replace("'", "")
                v_refs_str = str(v_refs).replace("'", "")
                parts.append(f"text_ids : {t_refs_str}")
                parts.append(f"vlm_ids : {v_refs_str}")
                return ", ".join(parts)

            # Aggregation by Source Type
            items_by_source: Dict[str, List[Dict[str, Any]]] = {}
            for k in group_order:
                items_by_source[k] = []
            
            # Default bucket for unknown types
            items_by_source["unknown"] = []

            def add_item(s_type, text, refs):
                 # Normalize source type
                s_type = str(s_type).lower().strip()
                # Map known variations if needed, or just default
                if s_type not in items_by_source:
                    # Try to match case-insensitive or exact
                    matched = False
                    for key in items_by_source.keys():
                        if key == s_type:
                            items_by_source[key].append({
                                "text": text,
                                "refs": refs
                            })
                            matched = True
                            break
                    if not matched:
                         # Fallback to first group if "direct" exists, or just append to unknown?
                         # Let's map unknown to "direct" purely as a fallback if it exists, else 'unknown'
                         if "direct" in items_by_source:
                             items_by_source["direct"].append({"text": text, "refs": refs})
                         else:
                             items_by_source["unknown"].append({"text": text, "refs": refs})
                else:
                    items_by_source[s_type].append({
                        "text": text,
                        "refs": refs
                    })

            # 1. Bullets
            bullets = summary.get("bullets", []) or []
            for i, b in enumerate(bullets, 1):
                claim = str(b.get("claim", "")).strip()
                s_type = b.get("source_type", "direct")
                notes = str(b.get("notes", "")).strip()
                refs = b.get("evidence_refs", [])
                
                text = f"({segment_id}-{i}) {claim}"
                if notes:
                    text += f"\n    - notes: {notes}"
                add_item(s_type, text, refs)

            # 2. Definitions
            definitions = summary.get("definitions", []) or []
            for d in definitions:
                term = str(d.get("term", "")).strip()
                defin = str(d.get("definition", "")).strip()
                s_type = d.get("source_type", "background")
                refs = d.get("evidence_refs", [])
                
                text = f"**{term}**: {defin}"
                add_item(s_type, text, refs)

            # 3. Explanations
            explanations = summary.get("explanations", []) or []
            for e in explanations:
                # Handle both dict and string (though v2 prompt usually gives dict)
                if isinstance(e, dict):
                     point = str(e.get("point", "")).strip()
                     s_type = e.get("source_type", "inferred")
                     refs = e.get("evidence_refs", [])
                else:
                     point = str(e).strip()
                     s_type = "inferred"
                     refs = []
                
                add_item(s_type, point, refs)
            
            # 4. Open Questions (Optional - mostly inferred)
            questions = summary.get("open_questions", []) or []
            for q in questions:
                 if isinstance(q, dict):
                     text = str(q.get("question", "")).strip()
                     s_type = q.get("source_type", "inferred") 
                     refs = q.get("evidence_refs", [])
                 else:
                     text = str(q).strip()
                     s_type = "inferred"
                     refs = []
                 # Label as Question
                 add_item(s_type, f"[Question] {text}", refs)

            # Render Groups
            for key in group_order:
                items = items_by_source.get(key, [])
                if not items:
                    continue
                
                header_title = group_headers.get(key, key.capitalize())
                handle.write(f"- [{key}] {header_title}\n")
                
                for item in items:
                    handle.write(f"  - {item['text']}\n")
                    ev_str = _validate_and_format(item['refs'])
                    if ev_str:
                         handle.write(f"    - evidence: {ev_str}\n")
            
            # Explicitly handle unknown if not empty and not in order
            if items_by_source.get("unknown"):
                 handle.write(f"- [Other] Additional Items\n")
                 for item in items_by_source["unknown"]:
                     handle.write(f"  - {item['text']}\n")
                     ev = _validate_and_format(item['refs'])
                     if ev: handle.write(f"    - evidence: {ev}\n")

            handle.write("\n")



def _truncate_lines(lines: List[str], max_chars: int) -> str:
    """텍스트가 최대 길이를 초과하면 잘라내고 생략 표시를 추가한다."""
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
    """요약 JSONL 파일에서 모든 세그먼트 데이터를 수집한다."""
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
    """단일 세그먼트의 상세 내용을 마크다운 라인 리스트에 추가한다."""
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
    """시간 순서대로 정렬된 상세 타임라인 요약문을 생성한다."""
    lines = ["# Final Summary (시간 순 상세)"]
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        _append_segment_details(lines, segment, include_timestamps)
    return "\n".join(lines).strip()


def build_summary_tldr_timeline(
    segments: List[Dict[str, object]], include_timestamps: bool
) -> str:
    """TL;DR 섹션이 포함된 시간 순 타임라인 요약문을 생성한다."""
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


def build_summary_tldr(
    segments: List[Dict[str, object]]
) -> str:
    """TL;DR 요약문만 생성한다."""
    lines = ["# Final Summary (TL;DR)"]
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
    return "\n".join(lines).strip()


def compose_final_summaries(
    summaries_jsonl: Path,
    max_chars: int,
    include_timestamps: bool,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """최종 요약본(타임라인, TL;DR 포함)을 생성하여 딕셔너리로 반환한다."""
    segments = _collect_segments(summaries_jsonl, limit=limit)

    summary_timeline = _truncate_lines(
        build_summary_timeline(segments, include_timestamps).splitlines(), max_chars
    )
    summary_tldr_timeline = _truncate_lines(
        build_summary_tldr_timeline(segments, include_timestamps).splitlines(),
        max_chars,
    )
    summary_tldr = _truncate_lines(
        build_summary_tldr(segments).splitlines(),
        max_chars,
    )

    return {
        "timeline": summary_timeline,
        "tldr_timeline": summary_tldr_timeline,
        "tldr": summary_tldr,
    }
