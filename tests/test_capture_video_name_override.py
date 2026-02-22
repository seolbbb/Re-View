from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_run_capture_forwards_video_name_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.pipeline import stages

    got: dict[str, object] = {}

    def fake_process_single_video_capture(*args, **kwargs):
        got.update(kwargs)
        return [{"id": "cap_001", "file_name": "cap_001.jpg", "time_ranges": []}]

    monkeypatch.setattr(stages, "process_single_video_capture", fake_process_single_video_capture)

    out = stages.run_capture(
        Path("C:/tmp/input.mp4"),
        tmp_path,
        threshold=0.1,
        dedupe_threshold=42.0,
        min_interval=0.5,
        verbose=False,
        video_name="sanitized_video_name",
        write_manifest=False,
    )

    assert isinstance(out, list)
    assert got.get("video_name_override") == "sanitized_video_name"


def test_process_single_video_capture_uses_override_for_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.capture import process_content

    fake_settings = SimpleNamespace(
        persistence_drop_ratio=0.1,
        sample_interval_sec=0.5,
        persistence_threshold=10,
        min_orb_features=200,
        dedup_phash_threshold=8,
        dedup_orb_distance=50,
        dedup_sim_threshold=0.7,
        enable_roi_detection=False,
        roi_padding=0,
        enable_smart_roi=False,
        roi_warmup_frames=0,
        enable_adaptive_resize=False,
    )
    monkeypatch.setattr(process_content, "get_capture_settings", lambda: fake_settings)

    captured: dict[str, object] = {}

    class FakeExtractor:
        def __init__(self, *, video_path: str, output_dir: str, **kwargs) -> None:
            captured["video_path"] = video_path
            captured["output_dir"] = output_dir
            captured["kwargs"] = kwargs

        def process(self, *, video_name: str):
            captured["process_video_name"] = video_name
            return [{"id": "cap_001", "file_name": "cap_001.jpg", "time_ranges": []}]

    monkeypatch.setattr(process_content, "HybridSlideExtractor", FakeExtractor)

    results = process_content.process_single_video_capture(
        video_path="C:/tmp/녹화_2022_05_08_16_49_55_421.mp4",
        output_base=str(tmp_path),
        write_manifest=True,
        video_name_override="timestamp_2022_05_08_16_49_55_421",
    )

    assert isinstance(results, list)

    expected_video_root = tmp_path / "timestamp_2022_05_08_16_49_55_421"
    expected_captures_dir = expected_video_root / "captures"
    expected_manifest = expected_video_root / "manifest.json"

    assert expected_captures_dir.exists()
    assert expected_manifest.exists()
    assert captured.get("output_dir") == str(expected_captures_dir)
    assert captured.get("process_video_name") == "timestamp_2022_05_08_16_49_55_421"

