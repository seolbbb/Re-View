"""최종 요약(A/B/C) 생성."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import format_ms, read_jsonl


STOPWORDS = {
    "그리고",
    "그러나",
    "그래서",
    "또한",
    "때문에",
    "합니다",
    "입니다",
    "있다",
    "없다",
    "것",
    "수",
    "등",
    "더",
    "대한",
    "관련",
    "필요",
    "가능",
    "정도",
    "내용",
    "개념",
    "핵심",
    "중요",
    "정리",
    "요약",
}


@dataclass(frozen=True)
class BulletItem:
    segment_id: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class ExplanationItem:
    segment_id: int
    start_ms: int
    end_ms: int
    point: str


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _merge_similar_tokens(
    tokens: List[str], freq: Counter, threshold: float = 0.6
) -> List[str]:
    merged: List[str] = []
    for token in tokens:
        placed = False
        for idx, existing in enumerate(merged):
            if _jaccard(_char_ngrams(token), _char_ngrams(existing)) >= threshold:
                if freq[token] > freq[existing] or (
                    freq[token] == freq[existing] and token < existing
                ):
                    merged[idx] = token
                placed = True
                break
        if not placed:
            merged.append(token)
    return merged


def _build_topics(
    bullets: List[BulletItem], definitions: List[BulletItem]
) -> List[str]:
    counter: Counter = Counter()
    for item in bullets + definitions:
        counter.update(_tokenize(item.text))
    if not counter:
        return []
    sorted_tokens = [t for t, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
    merged = _merge_similar_tokens(sorted_tokens, counter)
    if len(merged) <= 4:
        return merged
    return merged[: min(8, len(merged))]


def _assign_topics(
    topics: List[str],
    bullets: List[BulletItem],
    max_topics: int = 8,
) -> Dict[str, List[BulletItem]]:
    assignments: Dict[str, List[BulletItem]] = {topic: [] for topic in topics}
    other_bucket: List[BulletItem] = []

    for bullet in bullets:
        tokens = set(_tokenize(bullet.text))
        best_topic = None
        best_score = 0.0
        for topic in topics:
            score = 1.0 if topic in tokens else 0.0
            if score == 0.0:
                score = max(
                    (
                        _jaccard(_char_ngrams(topic), _char_ngrams(tok))
                        for tok in tokens
                    ),
                    default=0.0,
                )
            if score > best_score:
                best_score = score
                best_topic = topic
        if best_topic and best_score > 0.0:
            assignments[best_topic].append(bullet)
        else:
            other_bucket.append(bullet)

    if other_bucket:
        if len(topics) < max_topics:
            assignments["기타"] = other_bucket
        elif topics:
            assignments[topics[0]].extend(other_bucket)
    return assignments


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


def _collect_items(
    summaries_jsonl: Path,
    limit: Optional[int] = None,
) -> Tuple[
    List[BulletItem], List[BulletItem], List[ExplanationItem], List[Dict[str, object]]
]:
    bullets: List[BulletItem] = []
    definitions: List[BulletItem] = []
    explanations: List[ExplanationItem] = []
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
        for bullet in summary.get("bullets", []) or []:
            text = str(bullet.get("claim", "")).strip()
            if text:
                bullets.append(BulletItem(segment_id, start_ms, end_ms, text))
        for item in summary.get("definitions", []) or []:
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            if term and definition:
                definitions.append(
                    BulletItem(segment_id, start_ms, end_ms, f"{term}: {definition}")
                )
        for item in summary.get("explanations", []) or []:
            point = str(item.get("point", "")).strip()
            if point:
                explanations.append(
                    ExplanationItem(segment_id, start_ms, end_ms, point)
                )
    return bullets, definitions, explanations, segments


def build_summary_a(segments: List[Dict[str, object]], include_timestamps: bool) -> str:
    lines = ["# Final Summary A (시간 순 타임라인)"]
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        segment_id = int(segment["segment_id"])
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        summary = segment.get("summary", {}) or {}
        bullets = summary.get("bullets", []) or []
        explanations = summary.get("explanations", []) or []
        if include_timestamps:
            lines.append(
                f"#### Segment {segment_id} ({format_ms(start_ms)}–{format_ms(end_ms)})"
            )
        else:
            lines.append(f"#### Segment {segment_id}")
        # 핵심 포인트 (bullets)
        for bullet in bullets[:3]:
            claim = str(bullet.get("claim", "")).strip()
            if claim:
                lines.append(f"- {claim}")
        # 상세 설명 (explanations) - 처음 보는 사람 이해를 위해 추가
        if explanations:
            lines.append("")
            lines.append("**해설:**")
            for explanation in explanations[:2]:
                point = str(explanation.get("point", "")).strip()
                if point:
                    lines.append(f"  - {point}")
            lines.append("")
    return "\n".join(lines)


def build_summary_b(
    bullets: List[BulletItem],
    definitions: List[BulletItem],
    include_timestamps: bool,
) -> str:
    topics = _build_topics(bullets, definitions)
    assignments = _assign_topics(topics, bullets + definitions)

    lines = ["# Final Summary B (주제별 재구성)"]
    for topic in topics + (
        ["기타"] if "기타" in assignments and "기타" not in topics else []
    ):
        items = assignments.get(topic, [])
        if not items:
            continue
        lines.append(f"## {topic}")
        segment_ids = [item.segment_id for item in items]
        for item in items:
            if include_timestamps:
                lines.append(
                    f"- ({format_ms(item.start_ms)}–{format_ms(item.end_ms)}) {item.text}"
                )
            else:
                lines.append(f"- {item.text}")
        lines.append(f"- 근거 segment 범위: {min(segment_ids)}–{max(segment_ids)}")
    return "\n".join(lines)


def build_summary_c(segments: List[Dict[str, object]], include_timestamps: bool) -> str:
    lines = ["# Final Summary C (TL;DR + 시간 순)"]
    lines.append("## TL;DR")
    tldr_lines = []
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        summary = segment.get("summary", {}) or {}
        bullets = summary.get("bullets", []) or []
        if bullets:
            claim = str(bullets[0].get("claim", "")).strip()
            if claim:
                tldr_lines.append(f"- {claim}")
        if len(tldr_lines) >= 8:
            break
    lines.extend(tldr_lines)

    lines.append("")
    lines.append("## 시간 순 요약")
    for segment in sorted(segments, key=lambda x: int(x["segment_id"])):
        segment_id = int(segment["segment_id"])
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        summary = segment.get("summary", {}) or {}
        bullets = summary.get("bullets", []) or []
        explanations = summary.get("explanations", []) or []
        if include_timestamps:
            lines.append(
                f"#### Segment {segment_id} ({format_ms(start_ms)}–{format_ms(end_ms)})"
            )
        else:
            lines.append(f"#### Segment {segment_id}")
        # 핵심 포인트
        for bullet in bullets[:2]:
            claim = str(bullet.get("claim", "")).strip()
            if claim:
                lines.append(f"- {claim}")
        # 핵심 해설 1개 추가 (이해 도움)
        if explanations:
            point = str(explanations[0].get("point", "")).strip()
            if point:
                lines.append(f"  > {point}")
        lines.append("")

    return "\n".join(lines)


def compose_final_summaries(
    summaries_jsonl: Path,
    max_chars: int,
    include_timestamps: bool,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    bullets, definitions, explanations, segments = _collect_items(
        summaries_jsonl, limit=limit
    )

    summary_a = _truncate_lines(
        build_summary_a(segments, include_timestamps).splitlines(), max_chars
    )
    summary_b = _truncate_lines(
        build_summary_b(bullets, definitions, include_timestamps).splitlines(),
        max_chars,
    )
    summary_c = _truncate_lines(
        build_summary_c(segments, include_timestamps).splitlines(), max_chars
    )

    return {"A": summary_a, "B": summary_b, "C": summary_c}
