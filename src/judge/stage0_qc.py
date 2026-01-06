"""Stage0 rule-based QC for fused segments."""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.io_utils import read_json, read_jsonl, write_json


@dataclass(frozen=True)
class Stage0Thresholds:
    max_empty_transcript_ratio: float = 0.1
    max_empty_visual_ratio: float = 0.5
    max_invalid_time_ratio: float = 0.0
    max_out_of_range_ratio: float = 0.2
    min_stt_text_coverage_ratio: float = 0.98
    min_stt_segment_coverage_ratio: float = 0.98
    min_vlm_mapped_ratio: float = 0.98
    allow_overlaps: bool = False
    max_list_items: int = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _limit_items(values: List[int], max_items: int) -> List[int]:
    if max_items <= 0 or len(values) <= max_items:
        return values
    return values[:max_items]


def _summarize_ids(values: List[int], max_items: int) -> Dict[str, Any]:
    return {
        "count": len(values),
        "segment_ids": _limit_items(values, max_items),
    }


def _load_segments_units(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def _load_stt_segments(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path, "stt.json")
    if payload.get("schema_version") != 1:
        raise ValueError(f"stt.json schema_version is not 1: {path}")
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"stt.json segments is not a list: {path}")
    normalized: List[Dict[str, Any]] = []
    for seg in segments:
        start_ms = _safe_int(seg.get("start_ms"))
        end_ms = _safe_int(seg.get("end_ms"))
        text = str(seg.get("text", "") or "").strip()
        if start_ms is None or end_ms is None:
            continue
        normalized.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return normalized


def _load_vlm_items(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path, "vlm.json")
    if payload.get("schema_version") != 1:
        raise ValueError(f"vlm.json schema_version is not 1: {path}")
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"vlm.json items is not a list: {path}")
    normalized: List[Dict[str, Any]] = []
    for item in items:
        timestamp_ms = _safe_int(item.get("timestamp_ms"))
        extracted_text = str(item.get("extracted_text", "") or "").strip()
        if timestamp_ms is None:
            continue
        normalized.append({"timestamp_ms": timestamp_ms, "extracted_text": extracted_text})
    return normalized


def _analyze_segments(
    segments: List[Dict[str, Any]],
    *,
    min_segment_ms: Optional[int],
    max_segment_ms: Optional[int],
    stt_segments: Optional[List[Dict[str, Any]]],
    vlm_items: Optional[List[Dict[str, Any]]],
    thresholds: Stage0Thresholds,
) -> Dict[str, Any]:
    total = len(segments)
    if total == 0:
        return {
            "metrics": {"total_segments": 0},
            "violations": [
                {
                    "rule_id": "segments.empty",
                    "message": "no segments found",
                    "count": 0,
                    "segment_ids": [],
                }
            ],
            "decision": "fail",
            "reasons": ["no segments found"],
        }

    empty_transcript_ids: List[int] = []
    empty_visual_ids: List[int] = []
    invalid_time_ids: List[int] = []
    short_segment_ids: List[int] = []
    long_segment_ids: List[int] = []
    order_violation_ids: List[int] = []
    overlap_ids: List[int] = []
    malformed_units_ids: List[int] = []
    missing_segment_id_rows: List[int] = []

    segment_ids: List[int] = []
    durations: List[int] = []

    timeline: List[Tuple[int, int, int]] = []
    prev_start_ms: Optional[int] = None

    valid_windows: List[Tuple[int, int, int]] = []

    for row_idx, seg in enumerate(segments, start=1):
        seg_id = _safe_int(seg.get("segment_id"))
        row_id = seg_id if seg_id is not None else row_idx
        if seg_id is None:
            missing_segment_id_rows.append(row_idx)
        else:
            segment_ids.append(seg_id)

        start_ms = _safe_int(seg.get("start_ms"))
        end_ms = _safe_int(seg.get("end_ms"))
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            invalid_time_ids.append(row_id)
        else:
            duration = end_ms - start_ms
            durations.append(duration)
            timeline.append((row_id, start_ms, end_ms))
            valid_windows.append((row_id, start_ms, end_ms))
            if min_segment_ms is not None and duration < min_segment_ms:
                short_segment_ids.append(row_id)
            if max_segment_ms is not None and duration > max_segment_ms:
                long_segment_ids.append(row_id)
            if prev_start_ms is not None and start_ms < prev_start_ms:
                order_violation_ids.append(row_id)
            prev_start_ms = start_ms

        transcript_units = seg.get("transcript_units", [])
        if transcript_units is None:
            transcript_units = []
        if not isinstance(transcript_units, list):
            malformed_units_ids.append(row_id)
            transcript_units = []
        has_transcript = _has_text(seg.get("transcript_text")) or bool(transcript_units)
        if not has_transcript:
            empty_transcript_ids.append(row_id)

        visual_units = seg.get("visual_units", [])
        if visual_units is None:
            visual_units = []
        if not isinstance(visual_units, list):
            malformed_units_ids.append(row_id)
            visual_units = []
        has_visual = _has_text(seg.get("visual_text")) or bool(visual_units)
        if not has_visual:
            empty_visual_ids.append(row_id)

    duplicate_ids: List[int] = []
    missing_ids: List[int] = []
    if segment_ids:
        counts: Dict[int, int] = {}
        for seg_id in segment_ids:
            counts[seg_id] = counts.get(seg_id, 0) + 1
        duplicate_ids = sorted([seg_id for seg_id, count in counts.items() if count > 1])
        expected = set(range(1, max(segment_ids) + 1))
        missing_ids = sorted(expected - set(segment_ids))

    if timeline:
        timeline_sorted = sorted(timeline, key=lambda item: (item[1], item[2], item[0]))
        prev_end_ms: Optional[int] = None
        for row_id, start_ms, end_ms in timeline_sorted:
            if prev_end_ms is not None and start_ms < prev_end_ms:
                overlap_ids.append(row_id)
            prev_end_ms = max(prev_end_ms or end_ms, end_ms)

    empty_transcript_ratio = len(empty_transcript_ids) / total
    empty_visual_ratio = len(empty_visual_ids) / total
    invalid_time_ratio = len(invalid_time_ids) / total
    out_of_range_ratio = (len(short_segment_ids) + len(long_segment_ids)) / total

    metrics: Dict[str, Any] = {
        "total_segments": total,
        "transcript_coverage_ratio": round(1.0 - empty_transcript_ratio, 4),
        "visual_coverage_ratio": round(1.0 - empty_visual_ratio, 4),
        "min_segment_ms": min(durations) if durations else None,
        "max_segment_ms": max(durations) if durations else None,
        "avg_segment_ms": round(sum(durations) / len(durations), 2) if durations else None,
        "empty_transcript": _summarize_ids(empty_transcript_ids, thresholds.max_list_items),
        "empty_visual": _summarize_ids(empty_visual_ids, thresholds.max_list_items),
        "invalid_time": _summarize_ids(invalid_time_ids, thresholds.max_list_items),
        "short_segments": _summarize_ids(short_segment_ids, thresholds.max_list_items),
        "long_segments": _summarize_ids(long_segment_ids, thresholds.max_list_items),
    }

    stt_coverage: Dict[str, Any] = {}
    if stt_segments is not None:
        stt_segments_with_text = [seg for seg in stt_segments if _has_text(seg.get("text"))]
        stt_total = len(stt_segments_with_text)
        stt_total_chars = sum(len(seg["text"]) for seg in stt_segments_with_text)
        transcript_text_total = sum(len(str(seg.get("transcript_text", "") or "")) for seg in segments)

        stt_unmapped_ids: List[int] = []
        for idx, seg in enumerate(stt_segments_with_text, start=1):
            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
            if not any(end_ms > w_start and start_ms < w_end for _, w_start, w_end in valid_windows):
                stt_unmapped_ids.append(idx)

        stt_mapped = stt_total - len(stt_unmapped_ids)
        stt_segment_coverage_ratio = (stt_mapped / stt_total) if stt_total else None
        if stt_total_chars > 0:
            stt_text_ratio = transcript_text_total / stt_total_chars
            stt_text_coverage_ratio = round(min(stt_text_ratio, 1.0), 4)
        else:
            stt_text_coverage_ratio = None

        stt_coverage = {
            "stt_total_segments": stt_total,
            "stt_total_chars": stt_total_chars,
            "stt_segment_coverage_ratio": round(stt_segment_coverage_ratio, 4) if stt_segment_coverage_ratio is not None else None,
            "stt_text_coverage_ratio": stt_text_coverage_ratio,
            "stt_unmapped_segments": _summarize_ids(stt_unmapped_ids, thresholds.max_list_items),
        }
        metrics["stt_coverage"] = stt_coverage

    vlm_coverage: Dict[str, Any] = {}
    if vlm_items is not None:
        vlm_total = len(vlm_items)
        vlm_total_chars = sum(len(item.get("extracted_text", "")) for item in vlm_items)
        vlm_unmapped_ids: List[int] = []
        for idx, item in enumerate(vlm_items, start=1):
            timestamp_ms = int(item["timestamp_ms"])
            if not any(w_start <= timestamp_ms < w_end for _, w_start, w_end in valid_windows):
                vlm_unmapped_ids.append(idx)

        vlm_mapped = vlm_total - len(vlm_unmapped_ids)
        vlm_mapped_ratio = (vlm_mapped / vlm_total) if vlm_total else None
        visual_units_total = sum(len(seg.get("visual_units", []) or []) for seg in segments)
        visual_text_total = sum(len(str(seg.get("visual_text", "") or "")) for seg in segments)
        vlm_selected_ratio = (visual_units_total / vlm_total) if vlm_total else None
        if vlm_total_chars > 0:
            vlm_text_ratio = visual_text_total / vlm_total_chars
            vlm_text_coverage_ratio = round(min(vlm_text_ratio, 1.0), 4)
        else:
            vlm_text_coverage_ratio = None

        vlm_coverage = {
            "vlm_total_items": vlm_total,
            "vlm_total_chars": vlm_total_chars,
            "vlm_mapped_ratio": round(vlm_mapped_ratio, 4) if vlm_mapped_ratio is not None else None,
            "vlm_selected_ratio": round(vlm_selected_ratio, 4) if vlm_selected_ratio is not None else None,
            "vlm_text_coverage_ratio": vlm_text_coverage_ratio,
            "vlm_unmapped_items": _summarize_ids(vlm_unmapped_ids, thresholds.max_list_items),
            "visual_units_total": visual_units_total,
        }
        metrics["vlm_coverage"] = vlm_coverage

    violations: List[Dict[str, Any]] = []

    if missing_segment_id_rows:
        violations.append(
            {
                "rule_id": "segment.missing_id",
                "message": "segment_id is missing",
                "count": len(missing_segment_id_rows),
                "segment_ids": _limit_items(missing_segment_id_rows, thresholds.max_list_items),
            }
        )
    if duplicate_ids:
        violations.append(
            {
                "rule_id": "segment.duplicate_id",
                "message": "duplicate segment_id detected",
                "count": len(duplicate_ids),
                "segment_ids": _limit_items(duplicate_ids, thresholds.max_list_items),
            }
        )
    if missing_ids:
        violations.append(
            {
                "rule_id": "segment.missing_sequence",
                "message": "segment_id sequence has gaps",
                "count": len(missing_ids),
                "segment_ids": _limit_items(missing_ids, thresholds.max_list_items),
            }
        )
    if invalid_time_ids:
        violations.append(
            {
                "rule_id": "segment.invalid_time",
                "message": "start_ms/end_ms missing or invalid",
                "count": len(invalid_time_ids),
                "segment_ids": _limit_items(invalid_time_ids, thresholds.max_list_items),
            }
        )
    if malformed_units_ids:
        violations.append(
            {
                "rule_id": "segment.malformed_units",
                "message": "transcript_units or visual_units is not a list",
                "count": len(malformed_units_ids),
                "segment_ids": _limit_items(malformed_units_ids, thresholds.max_list_items),
            }
        )
    if order_violation_ids:
        violations.append(
            {
                "rule_id": "segment.order_violation",
                "message": "segments are not ordered by start_ms",
                "count": len(order_violation_ids),
                "segment_ids": _limit_items(order_violation_ids, thresholds.max_list_items),
            }
        )
    if overlap_ids and not thresholds.allow_overlaps:
        violations.append(
            {
                "rule_id": "segment.overlap",
                "message": "segments overlap in time",
                "count": len(overlap_ids),
                "segment_ids": _limit_items(overlap_ids, thresholds.max_list_items),
            }
        )

    decision = "pass"
    reasons: List[str] = []

    def _fail(reason: str) -> None:
        nonlocal decision
        decision = "fail"
        reasons.append(reason)

    if invalid_time_ratio > thresholds.max_invalid_time_ratio:
        _fail("invalid_time_ratio exceeds threshold")
    if empty_transcript_ratio > thresholds.max_empty_transcript_ratio:
        _fail("empty_transcript_ratio exceeds threshold")
    if empty_visual_ratio > thresholds.max_empty_visual_ratio:
        _fail("empty_visual_ratio exceeds threshold")
    if out_of_range_ratio > thresholds.max_out_of_range_ratio:
        _fail("out_of_range_ratio exceeds threshold")
    if duplicate_ids or missing_ids:
        _fail("segment_id sequence is invalid")
    if order_violation_ids:
        _fail("segments are not ordered by start_ms")
    if overlap_ids and not thresholds.allow_overlaps:
        _fail("segment overlaps detected")
    if malformed_units_ids:
        _fail("malformed units detected")
    if stt_coverage:
        ratio = stt_coverage.get("stt_text_coverage_ratio")
        if ratio is not None and ratio < thresholds.min_stt_text_coverage_ratio:
            _fail("stt_text_coverage_ratio below threshold")
        ratio = stt_coverage.get("stt_segment_coverage_ratio")
        if ratio is not None and ratio < thresholds.min_stt_segment_coverage_ratio:
            _fail("stt_segment_coverage_ratio below threshold")
    if vlm_coverage:
        ratio = vlm_coverage.get("vlm_mapped_ratio")
        if ratio is not None and ratio < thresholds.min_vlm_mapped_ratio:
            _fail("vlm_mapped_ratio below threshold")

    return {
        "metrics": metrics,
        "violations": violations,
        "decision": decision,
        "reasons": reasons,
    }


def run_stage0(
    *,
    segments_units_path: Path,
    stt_path: Optional[Path],
    vlm_path: Optional[Path],
    output_path: Path,
    min_segment_ms: Optional[int],
    max_segment_ms: Optional[int],
    thresholds: Stage0Thresholds,
) -> Dict[str, Any]:
    segments = _load_segments_units(segments_units_path)
    stt_segments = _load_stt_segments(stt_path) if stt_path else None
    vlm_items = _load_vlm_items(vlm_path) if vlm_path else None
    analysis = _analyze_segments(
        segments,
        min_segment_ms=min_segment_ms,
        max_segment_ms=max_segment_ms,
        stt_segments=stt_segments,
        vlm_items=vlm_items,
        thresholds=thresholds,
    )

    report = {
        "schema_version": 1,
        "decision": analysis["decision"],
        "metrics": analysis["metrics"],
        "violations": analysis["violations"],
        "reasons": analysis["reasons"],
        "thresholds": {
            "max_empty_transcript_ratio": thresholds.max_empty_transcript_ratio,
            "max_empty_visual_ratio": thresholds.max_empty_visual_ratio,
            "max_invalid_time_ratio": thresholds.max_invalid_time_ratio,
            "max_out_of_range_ratio": thresholds.max_out_of_range_ratio,
            "min_stt_text_coverage_ratio": thresholds.min_stt_text_coverage_ratio,
            "min_stt_segment_coverage_ratio": thresholds.min_stt_segment_coverage_ratio,
            "min_vlm_mapped_ratio": thresholds.min_vlm_mapped_ratio,
            "min_segment_ms": min_segment_ms,
            "max_segment_ms": max_segment_ms,
            "allow_overlaps": thresholds.allow_overlaps,
        },
        "meta": {
            "stage": "stage0",
            "generated_at_utc": _utc_now_iso(),
            "segments_units_path": str(segments_units_path),
            "stt_path": str(stt_path) if stt_path else None,
            "vlm_path": str(vlm_path) if vlm_path else None,
        },
    }

    write_json(output_path, report)
    return report


def _resolve_segments_units(config_path: Optional[str], explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    if config_path:
        config = _load_config(config_path)
        return (config.paths.output_root / "fusion" / "segments_units.jsonl").resolve()
    raise ValueError("segments_units.jsonl path or --config is required.")


def _resolve_output_path(segments_units_path: Path, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (segments_units_path.parent / "judge" / "stage0_report.json").resolve()


def _load_config(config_path: str):
    from src.fusion.config import load_config

    return load_config(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage0 rule-based QC for fused segments")
    parser.add_argument("--config", default=None, help="fusion config YAML path")
    parser.add_argument("--segments-units", default=None, help="segments_units.jsonl path")
    parser.add_argument("--stt", default=None, help="stt.json path")
    parser.add_argument("--vlm", default=None, help="vlm.json path")
    parser.add_argument("--output", default=None, help="output report path")
    parser.add_argument("--min-segment-sec", type=float, default=None, help="min segment length (sec)")
    parser.add_argument("--max-segment-sec", type=float, default=None, help="max segment length (sec)")
    parser.add_argument("--max-empty-transcript-ratio", type=float, default=0.1)
    parser.add_argument("--max-empty-visual-ratio", type=float, default=0.5)
    parser.add_argument("--max-invalid-time-ratio", type=float, default=0.0)
    parser.add_argument("--max-out-of-range-ratio", type=float, default=0.2)
    parser.add_argument("--min-stt-text-coverage-ratio", type=float, default=0.98)
    parser.add_argument("--min-stt-segment-coverage-ratio", type=float, default=0.98)
    parser.add_argument("--min-vlm-mapped-ratio", type=float, default=0.98)
    parser.add_argument("--allow-overlaps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-list-items", type=int, default=20)
    args = parser.parse_args()

    try:
        segments_units_path = _resolve_segments_units(args.config, args.segments_units)
        stt_path = Path(args.stt).expanduser().resolve() if args.stt else None
        vlm_path = Path(args.vlm).expanduser().resolve() if args.vlm else None

        min_segment_ms = None
        max_segment_ms = None
        if args.min_segment_sec is not None:
            min_segment_ms = int(args.min_segment_sec * 1000)
        if args.max_segment_sec is not None:
            max_segment_ms = int(args.max_segment_sec * 1000)

        if args.config and (min_segment_ms is None or max_segment_ms is None or stt_path is None or vlm_path is None):
            config = _load_config(args.config)
            if min_segment_ms is None:
                min_segment_ms = int(config.raw.sync_engine.min_segment_sec * 1000)
            if max_segment_ms is None:
                max_segment_ms = int(config.raw.sync_engine.max_segment_sec * 1000)
            if stt_path is None:
                stt_path = config.paths.stt_json
            if vlm_path is None:
                vlm_path = config.paths.vlm_json

        output_path = _resolve_output_path(segments_units_path, args.output)

        thresholds = Stage0Thresholds(
            max_empty_transcript_ratio=args.max_empty_transcript_ratio,
            max_empty_visual_ratio=args.max_empty_visual_ratio,
            max_invalid_time_ratio=args.max_invalid_time_ratio,
            max_out_of_range_ratio=args.max_out_of_range_ratio,
            min_stt_text_coverage_ratio=args.min_stt_text_coverage_ratio,
            min_stt_segment_coverage_ratio=args.min_stt_segment_coverage_ratio,
            min_vlm_mapped_ratio=args.min_vlm_mapped_ratio,
            allow_overlaps=args.allow_overlaps,
            max_list_items=args.max_list_items,
        )

        report = run_stage0(
            segments_units_path=segments_units_path,
            stt_path=stt_path,
            vlm_path=vlm_path,
            output_path=output_path,
            min_segment_ms=min_segment_ms,
            max_segment_ms=max_segment_ms,
            thresholds=thresholds,
        )
        print(f"[OK] stage0 report: {output_path}")
        print(f"[OK] decision: {report['decision']}")
    except Exception as exc:
        print(f"[ERROR] stage0 QC failed: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
