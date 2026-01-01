"""데모 입력 생성 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config
from src.fusion.io_utils import write_json


def _write_if_not_dry_run(path: Path, payload: object, dry_run: bool) -> None:
    if dry_run:
        return
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="데모 stt/vlm/manifest 입력 생성")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="config YAML 경로")
    parser.add_argument("--dry_run", action="store_true", help="출력 미생성 모드")
    args = parser.parse_args()

    config = load_config(args.config)
    stt_path = config.paths.stt_json
    vlm_path = config.paths.vlm_json
    manifest_path = config.paths.captures_manifest_json

    stt_payload = {
        "schema_version": 1,
        "segments": [
            {"start_ms": 0, "end_ms": 8000, "text": "안녕하세요. 오늘은 데모 파이프라인을 설명합니다."},
            {"start_ms": 8000, "end_ms": 18000, "text": "첫 번째 주제는 동기화 규칙과 세그먼트 분할입니다."},
            {"start_ms": 20000, "end_ms": 30000, "text": "두 번째 주제는 Gemini 요약 출력 형식입니다."},
            {"start_ms": 32000, "end_ms": 45000, "text": "마지막으로 최종 요약 A/B/C 형식을 확인합니다."},
        ],
    }
    vlm_payload = {
        "schema_version": 1,
        "items": [
            {"timestamp_ms": 0, "extracted_text": "슬라이드 1: 파이프라인 개요\n- STT/VLM 동기화\n- 요약 생성"},
            {"timestamp_ms": 15000, "extracted_text": "슬라이드 2: 세그먼트 규칙\n- min/max 길이\n- 침묵 구간 분할"},
            {"timestamp_ms": 28000, "extracted_text": "슬라이드 3: 요약 스키마\n- bullets/definitions\n- evidence_refs"},
            {"timestamp_ms": 42000, "extracted_text": "슬라이드 4: 최종 요약\n- A/B/C 형식"},
        ],
    }

    _write_if_not_dry_run(stt_path, stt_payload, args.dry_run)
    _write_if_not_dry_run(vlm_path, vlm_payload, args.dry_run)

    if manifest_path:
        manifest_payload = [
            {"timestamp_ms": 0, "diff_score": 0.4},
            {"timestamp_ms": 15000, "diff_score": 0.55},
            {"timestamp_ms": 28000, "diff_score": 0.6},
            {"timestamp_ms": 42000, "diff_score": 0.35},
        ]
        _write_if_not_dry_run(manifest_path, manifest_payload, args.dry_run)

    if args.dry_run:
        print("[DRY RUN] 데모 입력 출력 미생성")


if __name__ == "__main__":
    main()
