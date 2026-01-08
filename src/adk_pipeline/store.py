"""DB(파일시스템) 아티팩트 경로 규약.

기존 파이프라인과 동일하게 `data/outputs/{video_name}` 아래에 저장한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoStore:
    output_base: Path
    video_name: str

    def video_root(self) -> Path:
        return self.output_base / self.video_name

    def stt_json(self) -> Path:
        return self.video_root() / "stt.json"

    def manifest_json(self) -> Path:
        return self.video_root() / "manifest.json"

    def captures_dir(self) -> Path:
        return self.video_root() / "captures"

    def vlm_raw_json(self) -> Path:
        return self.video_root() / "vlm_raw.json"

    def vlm_json(self) -> Path:
        return self.video_root() / "vlm.json"

    def fusion_config_yaml(self) -> Path:
        return self.video_root() / "config.yaml"

    def fusion_dir(self) -> Path:
        return self.video_root() / "fusion"

    def segments_jsonl(self) -> Path:
        return self.fusion_dir() / "segments.jsonl"

    def segments_units_jsonl(self) -> Path:
        return self.fusion_dir() / "segments_units.jsonl"

    def segment_summaries_jsonl(self) -> Path:
        return self.fusion_dir() / "segment_summaries.jsonl"

    def segment_summaries_md(self) -> Path:
        return self.fusion_dir() / "segment_summaries.md"

    def final_outputs_dir(self) -> Path:
        return self.fusion_dir() / "outputs"

    def pipeline_run_json(self) -> Path:
        return self.video_root() / "adk_pipeline_run.json"

    def attempts_dir(self) -> Path:
        return self.fusion_dir() / "attempts"
