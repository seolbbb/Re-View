"""STT/VLM 동기화 및 세그먼트 생성."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ConfigBundle
from .io_utils import compute_run_id, ensure_output_root, read_json, write_json


@dataclass(frozen=True)
class SegmentWindow:
    """세그먼트 시간 구간을 표현한다."""
    start_ms: int
    end_ms: int


def _load_stt_segments(path: Path) -> Tuple[List[Dict[str, object]], Optional[int]]:
    """stt.json을 표준 segment 리스트로 정리한다."""
    payload = read_json(path, "stt.json")
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"Invalid stt.json segments format: {path}")

    normalized: List[Dict[str, object]] = []
    for seg in segments:
        try:
            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
            text = str(seg["text"]).strip()
        except KeyError as exc:
            raise ValueError(f"Missing required keys in stt.json: {seg}") from exc
        if end_ms < start_ms:
            raise ValueError(f"Invalid segment duration in stt.json: {seg}")
        if text:
            normalized.append(
                {"start_ms": start_ms, "end_ms": end_ms, "text": text, "id": seg.get("id")}
            )
    duration_ms = payload.get("duration_ms")
    duration_ms = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
    return sorted(normalized, key=lambda x: (x["start_ms"], x["end_ms"])), duration_ms


def _load_vlm_items(path: Path) -> Tuple[List[Dict[str, object]], Optional[int]]:
    """vlm.json을 표준 item 리스트로 정리한다."""
    payload = read_json(path, "vlm.json")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid vlm.json items format: {path}")

    normalized: List[Dict[str, object]] = []
    for item in items:
        try:
            timestamp_ms = int(item["timestamp_ms"])
            extracted_text = str(item.get("extracted_text", "")).strip()
        except KeyError as exc:
            raise ValueError(f"Missing required keys in vlm.json: {item}") from exc
        normalized.append({"timestamp_ms": timestamp_ms, "extracted_text": extracted_text, "id": item.get("id")})
    duration_ms = payload.get("duration_ms")
    duration_ms = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
    return sorted(normalized, key=lambda x: x["timestamp_ms"]), duration_ms


def _load_manifest_scores(
    path: Optional[Path] = None,
    captures_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[int, float]:
    """manifest.json 또는 captures_data에서 diff_score 기준 점수 맵을 만든다.
    
    Args:
        path: manifest.json 파일 경로 (선택)
        captures_data: DB에서 가져온 captures 리스트 (선택, path보다 우선)
    """
    # captures_data가 제공되면 우선 사용
    if captures_data is not None:
        scores: Dict[int, float] = {}
        for item in captures_data:
            diff_score = float(item.get("diff_score", 0.0))
            
            # time_ranges 지원
            time_ranges = item.get("time_ranges")
            if isinstance(time_ranges, list) and time_ranges:
                for rng in time_ranges:
                     if isinstance(rng, dict) and "start_ms" in rng:
                        try:
                            start_ms = int(rng["start_ms"])
                            if start_ms not in scores or diff_score > scores[start_ms]:
                                scores[start_ms] = diff_score
                        except (TypeError, ValueError):
                            continue
            
            # fallback: top-level start_ms
            if "start_ms" in item:
                try:
                    start_ms = int(item["start_ms"])
                    if start_ms not in scores or diff_score > scores[start_ms]:
                        scores[start_ms] = diff_score
                except (TypeError, ValueError):
                    pass
        return scores
    
    # 파일 경로 기반 로드
    if not path:
        return {}
    if not path.exists():
        return {}
    payload = read_json(path, "captures/manifest.json")
    if not isinstance(payload, list):
        raise ValueError(f"Invalid manifest.json format: {path}")
    scores = {}
    for item in payload:
        diff_score = float(item.get("diff_score", 0.0))

        # time_ranges 지원
        time_ranges = item.get("time_ranges")
        if isinstance(time_ranges, list) and time_ranges:
            for rng in time_ranges:
                    if isinstance(rng, dict) and "start_ms" in rng:
                        try:
                            start_ms = int(rng["start_ms"])
                            if start_ms not in scores or diff_score > scores[start_ms]:
                                scores[start_ms] = diff_score
                        except (TypeError, ValueError):
                            continue

        # fallback: top-level start_ms
        if "start_ms" in item:
            try:
                start_ms = int(item["start_ms"])
                if start_ms not in scores or diff_score > scores[start_ms]:
                    scores[start_ms] = diff_score
            except (TypeError, ValueError):
                continue
    return scores


def _compute_duration_ms(
    stt_segments: List[Dict[str, object]],
    vlm_items: List[Dict[str, object]],
    stt_duration_ms: Optional[int],
    vlm_duration_ms: Optional[int],
) -> int:
    """STT/VLM 길이 정보를 조합해 전체 길이를 산출한다."""
    provided = [d for d in [stt_duration_ms, vlm_duration_ms] if d]
    if provided:
        return int(max(provided))
    max_stt = max([seg["end_ms"] for seg in stt_segments], default=0)
    max_vlm = max([item["timestamp_ms"] for item in vlm_items], default=0)
    if max_stt == 0 and max_vlm == 0:
        raise ValueError("Cannot compute duration_ms: stt/vlm inputs are empty.")
    return int(max(max_stt, max_vlm) + 1000)


def _build_initial_segments(
    vlm_items: List[Dict[str, object]],
    duration_ms: int,
    min_segment_ms: int,
) -> List[SegmentWindow]:
    """VLM 타임스탬프를 기준으로 초기 세그먼트를 만든다."""
    timestamps = sorted({int(item["timestamp_ms"]) for item in vlm_items if item["timestamp_ms"] >= 0})
    boundaries = [0] + [ts for ts in timestamps if ts < duration_ms] + [duration_ms]

    merged: List[SegmentWindow] = []
    start = boundaries[0]
    for boundary in boundaries[1:]:
        if boundary - start < min_segment_ms:
            continue
        merged.append(SegmentWindow(start_ms=start, end_ms=boundary))
        start = boundary
    if start < boundaries[-1]:
        merged.append(SegmentWindow(start_ms=start, end_ms=boundaries[-1]))
    return merged


def _segments_in_range(
    stt_segments: List[Dict[str, object]], start_ms: int, end_ms: int
) -> List[Dict[str, object]]:
    """지정 구간과 겹치는 STT 세그먼트를 반환한다."""
    return [
        seg
        for seg in stt_segments
        if seg["end_ms"] > start_ms and seg["start_ms"] < end_ms
    ]


def _compute_transcript_chars(stt_segments: List[Dict[str, object]], start_ms: int, end_ms: int) -> int:
    """구간 내 STT 텍스트 길이를 합산한다."""
    selected = _segments_in_range(stt_segments, start_ms, end_ms)
    if not selected:
        return 0
    return sum(len(seg["text"]) for seg in selected) + (len(selected) - 1)


def _choose_split_point(
    stt_segments: List[Dict[str, object]],
    start_ms: int,
    end_ms: int,
    silence_gap_ms: int,
) -> int:
    """침묵 구간 또는 경계에 기반해 분할 지점을 선택한다."""
    midpoint = (start_ms + end_ms) / 2
    gap_candidates: List[int] = []
    for idx in range(len(stt_segments) - 1):
        gap_start = int(stt_segments[idx]["end_ms"])
        gap_end = int(stt_segments[idx + 1]["start_ms"])
        if gap_end - gap_start >= silence_gap_ms:
            candidate = (gap_start + gap_end) // 2
            if start_ms < candidate < end_ms:
                gap_candidates.append(candidate)
    if gap_candidates:
        return min(gap_candidates, key=lambda x: (abs(x - midpoint), x))

    boundary_candidates: List[int] = []
    for seg in stt_segments:
        seg_start = int(seg["start_ms"])
        seg_end = int(seg["end_ms"])
        if start_ms < seg_start < end_ms:
            boundary_candidates.append(seg_start)
        if start_ms < seg_end < end_ms:
            boundary_candidates.append(seg_end)
    if boundary_candidates:
        return min(boundary_candidates, key=lambda x: (abs(x - midpoint), x))

    return (start_ms + end_ms) // 2


def _split_segment_recursive(
    segment: SegmentWindow,
    stt_segments: List[Dict[str, object]],
    max_segment_ms: int,
    max_transcript_chars: int,
    silence_gap_ms: int,
) -> List[SegmentWindow]:
    """세그먼트 길이/텍스트 제한을 만족할 때까지 재귀 분할한다."""
    if segment.end_ms - segment.start_ms <= 1:
        return [segment]

    transcript_chars = _compute_transcript_chars(stt_segments, segment.start_ms, segment.end_ms)
    if (segment.end_ms - segment.start_ms) <= max_segment_ms and transcript_chars <= max_transcript_chars:
        return [segment]

    selected = _segments_in_range(stt_segments, segment.start_ms, segment.end_ms)
    split_point = _choose_split_point(selected, segment.start_ms, segment.end_ms, silence_gap_ms)
    if split_point <= segment.start_ms or split_point >= segment.end_ms:
        return [segment]

    left = SegmentWindow(segment.start_ms, split_point)
    right = SegmentWindow(split_point, segment.end_ms)
    return _split_segment_recursive(left, stt_segments, max_segment_ms, max_transcript_chars, silence_gap_ms) + _split_segment_recursive(
        right, stt_segments, max_segment_ms, max_transcript_chars, silence_gap_ms
    )


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """중복 제거용 n-gram을 만든다."""
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """두 집합의 자카드 유사도를 계산한다."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _select_vlm_items(
    vlm_items: List[Dict[str, object]],
    manifest_scores: Dict[int, float],
    start_ms: int,
    end_ms: int,
    max_visual_items: int,
) -> List[Dict[str, object]]:
    """구간 내 VLM 아이템을 선택하고 우선순위를 적용한다."""
    candidates = [item for item in vlm_items if start_ms <= item["timestamp_ms"] < end_ms]
    if not candidates:
        if not vlm_items:
            return []
        closest = min(vlm_items, key=lambda x: (abs(x["timestamp_ms"] - start_ms), x["timestamp_ms"]))
        return [closest]

    if manifest_scores:
        candidates = sorted(
            candidates,
            key=lambda x: (-manifest_scores.get(int(x["timestamp_ms"]), 0.0), abs(x["timestamp_ms"] - start_ms), x["timestamp_ms"]),
        )
    else:
        candidates = sorted(candidates, key=lambda x: (abs(x["timestamp_ms"] - start_ms), x["timestamp_ms"]))
    return candidates[:max_visual_items]


def _extract_visual_units(
    selected_items: List[Dict[str, object]],
    dedup_threshold: float,
    max_visual_chars: int,
) -> Tuple[List[Dict[str, object]], str]:
    """VLM 텍스트를 중복 제거해 visual_units를 만든다."""
    deduped_lines: List[str] = []
    item_buffers: List[Dict[str, object]] = []
    for item in selected_items:
        timestamp_ms = int(item["timestamp_ms"])
        extracted_text = str(item.get("extracted_text", ""))
        kept_lines: List[str] = []
        for raw_line in extracted_text.splitlines():
            text = raw_line.strip()
            if not text or len(text) < 3:
                continue
            line_ngrams = _char_ngrams(text)
            if any(_jaccard(line_ngrams, _char_ngrams(prev)) >= dedup_threshold for prev in deduped_lines):
                continue
            kept_lines.append(text)
            deduped_lines.append(text)
        if kept_lines:
            item_buffers.append({"timestamp_ms": timestamp_ms, "lines": kept_lines, "id": item.get("id")})

    visual_units: List[Dict[str, object]] = []
    for idx, item in enumerate(item_buffers, start=1):
        visual_units.append(
            {
                "unit_id": item.get("id") or f"v{idx}",
                "timestamp_ms": int(item["timestamp_ms"]),
                "text": "\n".join(item["lines"]),
            }
        )

    def total_len(items: List[Dict[str, object]]) -> int:
        if not items:
            return 0
        return sum(len(item["text"]) for item in items) + (len(items) - 1)

    if max_visual_chars > 0:
        while visual_units and total_len(visual_units) > max_visual_chars:
            visual_units.pop()

    visual_text = "\n".join(unit["text"] for unit in visual_units)
    return visual_units, visual_text


def _build_transcript_units(
    stt_segments: List[Dict[str, object]], start_ms: int, end_ms: int
) -> Tuple[List[Dict[str, object]], str]:
    """STT 세그먼트를 단위별로 재구성한다."""
    selected = _segments_in_range(stt_segments, start_ms, end_ms)
    transcript_units: List[Dict[str, object]] = []
    for idx, seg in enumerate(selected, start=1):
        transcript_units.append(
            {
                "unit_id": seg.get("id") or f"t{idx}",
                "start_ms": int(seg["start_ms"]),
                "end_ms": int(seg["end_ms"]),
                "text": seg["text"],
            }
        )
    transcript_text = "\n".join(unit["text"] for unit in transcript_units)
    return transcript_units, transcript_text


def run_sync_engine(config: ConfigBundle, limit: Optional[int] = None) -> None:
    """STT/VLM을 동기화해 segments_units.jsonl을 만든다."""
    paths = config.paths
    ensure_output_root(paths.output_root)

    # 입력 로드
    stt_segments, stt_duration_ms = _load_stt_segments(paths.stt_json)
    vlm_items, vlm_duration_ms = _load_vlm_items(paths.vlm_json)
    manifest_scores = _load_manifest_scores(paths.captures_manifest_json)

    # 전체 길이와 분할 기준 준비
    duration_ms = _compute_duration_ms(stt_segments, vlm_items, stt_duration_ms, vlm_duration_ms)
    min_segment_ms = config.raw.sync_engine.min_segment_sec * 1000
    max_segment_ms = config.raw.sync_engine.max_segment_sec * 1000

    # VLM 타임스탬프 기반 초기 세그먼트 생성
    initial_segments = _build_initial_segments(vlm_items, duration_ms, min_segment_ms)
    refined_segments: List[SegmentWindow] = []
    for segment in initial_segments:
        refined_segments.extend(
            _split_segment_recursive(
                segment,
                stt_segments,
                max_segment_ms,
                config.raw.sync_engine.max_transcript_chars,
                config.raw.sync_engine.silence_gap_ms,
            )
        )

    refined_segments = sorted(refined_segments, key=lambda x: (x.start_ms, x.end_ms))
    if limit is not None:
        refined_segments = refined_segments[:limit]

    # 출력 경로 준비
    run_id = compute_run_id(config.config_path, paths.stt_json, paths.vlm_json, paths.captures_manifest_json)
    output_dir = paths.output_root / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # segments_units.jsonl 생성
    segments_units_handle = (output_dir / "segments_units.jsonl").open("w", encoding="utf-8")

    try:
        for idx, segment in enumerate(refined_segments, start=1):
            # STT 유닛 구성
            transcript_units, transcript_text = _build_transcript_units(
                stt_segments, segment.start_ms, segment.end_ms
            )
            # 시각 정보 후보 선택
            selected_vlm_items = _select_vlm_items(
                vlm_items,
                manifest_scores,
                segment.start_ms,
                segment.end_ms,
                config.raw.sync_engine.max_visual_items,
            )
            # 시각 텍스트 중복 제거/제한 적용
            visual_units, visual_text = _extract_visual_units(
                selected_vlm_items,
                config.raw.sync_engine.dedup_similarity_threshold,
                config.raw.sync_engine.max_visual_chars,
            )

            stt_ids = [u["unit_id"] for u in transcript_units if isinstance(u["unit_id"], str) and u["unit_id"].startswith("stt_")]
            vlm_ids = [u["unit_id"] for u in visual_units if isinstance(u["unit_id"], str) and (u["unit_id"].startswith("vlm_") or u["unit_id"].startswith("cap_"))]

            segments_units_handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "segment_id": idx,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "transcript_units": transcript_units,
                        "visual_units": visual_units,
                        "source_refs": {
                            "stt_ids": stt_ids,
                            "vlm_ids": vlm_ids
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=False,
                )
                + "\n"
            )
    finally:
        segments_units_handle.close()


def run_batch_sync_engine(
    *,
    stt_json: Path,
    vlm_json: Path,
    manifest_json: Optional[Path] = None,
    captures_data: Optional[List[Dict[str, Any]]] = None,
    output_dir: Path,
    time_range: Tuple[int, int],
    sync_config: Dict[str, object],
    segment_id_offset: int = 0,
) -> Dict[str, object]:
    """배치 단위로 Sync를 실행합니다.

    특정 시간 범위의 데이터만 처리하여 배치별 segments_units.jsonl을 생성합니다.

    Args:
        stt_json: stt.json 경로
        vlm_json: 배치별 vlm.json 경로
        manifest_json: manifest.json 경로 (선택, captures_data가 없을 때 사용)
        captures_data: DB에서 가져온 captures 리스트 (선택, manifest_json보다 우선)
        output_dir: 출력 디렉토리 (배치별 디렉토리)
        time_range: (start_ms, end_ms) 시간 범위
        sync_config: sync_engine 설정 (min_segment_sec, max_segment_sec 등)
        segment_id_offset: segment_id 오프셋 (배치 병합 시 사용)

    Returns:
        segments_count: 생성된 segment 수
        segments_units_jsonl: 생성된 파일 경로
    """
    start_ms, end_ms = time_range

    # STT 로드 및 시간 범위 필터링
    stt_segments, stt_duration_ms = _load_stt_segments(stt_json)
    filtered_stt_segments = [
        seg for seg in stt_segments
        if seg["end_ms"] > start_ms and seg["start_ms"] < end_ms
    ]

    # VLM 로드 (이미 배치별로 필터링된 상태)
    vlm_items, vlm_duration_ms = _load_vlm_items(vlm_json)

    # manifest scores 로드 (captures_data 우선, 없으면 manifest_json 사용)
    manifest_scores = _load_manifest_scores(path=manifest_json, captures_data=captures_data)

    # 설정 값 추출
    min_segment_ms = int(sync_config.get("min_segment_sec", 30)) * 1000
    max_segment_ms = int(sync_config.get("max_segment_sec", 120)) * 1000
    max_transcript_chars = int(sync_config.get("max_transcript_chars", 1000))
    silence_gap_ms = int(sync_config.get("silence_gap_ms", 500))
    max_visual_items = int(sync_config.get("max_visual_items", 10))
    max_visual_chars = int(sync_config.get("max_visual_chars", 3000))
    dedup_similarity_threshold = float(sync_config.get("dedup_similarity_threshold", 0.9))

    # duration 계산 (배치 범위 내)
    duration_ms = end_ms - start_ms

    # VLM/STT가 모두 없으면 빈 결과 반환
    if not vlm_items and not filtered_stt_segments:
        output_dir.mkdir(parents=True, exist_ok=True)
        segments_units_path = output_dir / "segments_units.jsonl"
        segments_units_path.write_text("", encoding="utf-8")
        return {
            "segments_count": 0,
            "segments_units_jsonl": str(segments_units_path),
        }

    # 초기 세그먼트 생성 (VLM 타임스탬프 기반)
    # 시간 범위를 배치 내부로 조정
    adjusted_vlm_items = [
        {**item, "timestamp_ms": int(item["timestamp_ms"]) - start_ms}
        for item in vlm_items
        if start_ms <= int(item["timestamp_ms"]) < end_ms
    ]

    if not adjusted_vlm_items:
        # VLM 아이템이 없으면 전체 범위를 하나의 세그먼트로
        initial_segments = [SegmentWindow(start_ms=0, end_ms=duration_ms)]
    else:
        initial_segments = _build_initial_segments(adjusted_vlm_items, duration_ms, min_segment_ms)

    # 세그먼트 분할
    # STT 세그먼트도 배치 시간 기준으로 조정
    adjusted_stt_segments = [
        {
            **seg,
            "start_ms": max(0, int(seg["start_ms"]) - start_ms),
            "end_ms": min(duration_ms, int(seg["end_ms"]) - start_ms),
        }
        for seg in filtered_stt_segments
    ]

    refined_segments: List[SegmentWindow] = []
    for segment in initial_segments:
        refined_segments.extend(
            _split_segment_recursive(
                segment,
                adjusted_stt_segments,
                max_segment_ms,
                max_transcript_chars,
                silence_gap_ms,
            )
        )

    refined_segments = sorted(refined_segments, key=lambda x: (x.start_ms, x.end_ms))

    # run_id 생성
    run_id = compute_run_id(None, stt_json, vlm_json, manifest_json)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_units_path = output_dir / "segments_units.jsonl"

    # 결과 생성
    with open(segments_units_path, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(refined_segments, start=1):
            # 배치 내 조정된 시간을 원래 시간으로 복원
            actual_start_ms = segment.start_ms + start_ms
            actual_end_ms = segment.end_ms + start_ms

            transcript_units, transcript_text = _build_transcript_units(
                adjusted_stt_segments, segment.start_ms, segment.end_ms
            )
            # transcript_units의 시간도 원래 시간으로 복원
            for unit in transcript_units:
                unit["start_ms"] = int(unit["start_ms"]) + start_ms
                unit["end_ms"] = int(unit["end_ms"]) + start_ms

            # VLM 아이템 선택 (조정된 시간 기준)
            selected_vlm_items = _select_vlm_items(
                adjusted_vlm_items,
                {},  # manifest_scores는 원래 시간 기준이라 사용 안 함
                segment.start_ms,
                segment.end_ms,
                max_visual_items,
            )
            visual_units, visual_text = _extract_visual_units(
                selected_vlm_items,
                dedup_similarity_threshold,
                max_visual_chars,
            )
            # visual_units의 시간도 원래 시간으로 복원
            for unit in visual_units:
                unit["timestamp_ms"] = int(unit["timestamp_ms"]) + start_ms

            stt_ids = [u["unit_id"] for u in transcript_units if isinstance(u["unit_id"], str) and u["unit_id"].startswith("stt_")]
            vlm_ids = [u["unit_id"] for u in visual_units if isinstance(u["unit_id"], str) and (u["unit_id"].startswith("vlm_") or u["unit_id"].startswith("cap_"))]

            record = {
                "run_id": run_id,
                "segment_id": idx + segment_id_offset,
                "start_ms": actual_start_ms,
                "end_ms": actual_end_ms,
                "transcript_units": transcript_units,
                "visual_units": visual_units,
                "source_refs": {
                    "stt_ids": stt_ids,
                    "vlm_ids": vlm_ids
                },
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")

    return {
        "segments_count": len(refined_segments),
        "segments_units_jsonl": str(segments_units_path),
    }
