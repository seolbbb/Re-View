from __future__ import annotations

import json
from pathlib import Path

from src.run_process_pipeline import (
    _compute_skip_captures,
    _detect_completed_batches,
    _rebuild_fusion_accumulators,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_detect_completed_batches_counts_consecutive_pass(tmp_path: Path) -> None:
    video_root = tmp_path

    # batch_1: PASS
    _write_json(video_root / "batches" / "batch_1" / "judge.json", {"pass": True})
    (video_root / "batches" / "batch_1" / "fusion").mkdir(parents=True, exist_ok=True)
    (video_root / "batches" / "batch_1" / "fusion" / "segment_summaries.jsonl").write_text(
        '{"segment_id": 1}\n', encoding="utf-8"
    )
    (video_root / "batches" / "batch_1" / "fusion" / "segments_units.jsonl").write_text(
        '{"segment_id": 1}\n', encoding="utf-8"
    )

    # batch_2: PASS
    _write_json(video_root / "batches" / "batch_2" / "judge.json", {"pass": True})
    (video_root / "batches" / "batch_2" / "fusion").mkdir(parents=True, exist_ok=True)
    (video_root / "batches" / "batch_2" / "fusion" / "segment_summaries.jsonl").write_text(
        '{"segment_id": 2}\n', encoding="utf-8"
    )
    (video_root / "batches" / "batch_2" / "fusion" / "segments_units.jsonl").write_text(
        '{"segment_id": 2}\n', encoding="utf-8"
    )

    # batch_3: FAIL (should stop here)
    _write_json(video_root / "batches" / "batch_3" / "judge.json", {"pass": False})
    (video_root / "batches" / "batch_3" / "fusion").mkdir(parents=True, exist_ok=True)
    (video_root / "batches" / "batch_3" / "fusion" / "segment_summaries.jsonl").write_text(
        '{"segment_id": 3}\n', encoding="utf-8"
    )
    (video_root / "batches" / "batch_3" / "fusion" / "segments_units.jsonl").write_text(
        '{"segment_id": 3}\n', encoding="utf-8"
    )

    assert _detect_completed_batches(video_root) == 2


def test_compute_skip_captures_uses_vlm_items(tmp_path: Path) -> None:
    video_root = tmp_path

    # batch_1: 4 captures
    _write_json(
        video_root / "batches" / "batch_1" / "vlm.json",
        {"items": [{"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}]},
    )

    # batch_2: 2 captures
    _write_json(
        video_root / "batches" / "batch_2" / "vlm.json",
        {"items": [{"i": 1}, {"i": 2}]},
    )

    assert _compute_skip_captures(video_root, completed_batches=2, batch_size=4) == 6


def test_rebuild_fusion_accumulators_concatenates_jsonl(tmp_path: Path) -> None:
    video_root = tmp_path

    b1 = video_root / "batches" / "batch_1" / "fusion"
    b2 = video_root / "batches" / "batch_2" / "fusion"
    b1.mkdir(parents=True, exist_ok=True)
    b2.mkdir(parents=True, exist_ok=True)

    (b1 / "segment_summaries.jsonl").write_text("a\n", encoding="utf-8")
    (b1 / "segments_units.jsonl").write_text("u1\n", encoding="utf-8")

    # no trailing newline on purpose to test newline normalization
    (b2 / "segment_summaries.jsonl").write_text("b", encoding="utf-8")
    (b2 / "segments_units.jsonl").write_text("u2", encoding="utf-8")

    _rebuild_fusion_accumulators(video_root, completed_batches=2)

    fusion_dir = video_root / "fusion"
    assert (fusion_dir / "segment_summaries.jsonl").read_text(encoding="utf-8") == "a\nb\n"
    assert (fusion_dir / "segments_units.jsonl").read_text(encoding="utf-8") == "u1\nu2\n"
