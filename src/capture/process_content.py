"""
캡쳐 단계에서 비디오 1건을 처리해 슬라이드 캡쳐와 manifest.json을 만든다.
run_preprocess_pipeline에서 호출되며 설정은 config/capture/settings.yaml에서 로드한다.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.capture.settings import get_capture_settings
from src.capture.tools.hybrid_extractor import HybridSlideExtractor


def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: Optional[float] = None,
    dedupe_threshold: Optional[float] = None,
    min_interval: Optional[float] = None,
    dedup_enabled: bool = True,
    write_manifest: bool = True,
) -> list:
    """
    run_preprocess_pipeline.py에서 호출되는 캡쳐 인터페이스.

    지연 저장 로직으로 슬라이드를 추출한다.

    video_path: 처리할 비디오 파일 경로.
    output_base: 출력 기본 디렉터리.
    scene_threshold: 장면 전환 감지 임계값(None이면 설정값 사용).
    dedupe_threshold: 미사용 파라미터(호환 유지용).
    min_interval: 최소 캡처 간격(None이면 설정값 사용).
    dedup_enabled: 중복 제거 활성화 여부 (기본 True).
    write_manifest: manifest.json 저장 여부.

    반환: 추출된 슬라이드 메타데이터 리스트.
    """
    settings = get_capture_settings()
    resolved_scene_threshold = settings.sensitivity_diff if scene_threshold is None else scene_threshold
    resolved_min_interval = settings.min_interval if min_interval is None else min_interval
    video_name = Path(video_path).stem
    output_root = Path(output_base) / video_name
    captures_dir = output_root / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Capture] Processing: {video_name}")

    extractor = HybridSlideExtractor(
        video_path,
        output_dir=str(captures_dir),
        sensitivity_diff=resolved_scene_threshold,
        sensitivity_sim=settings.sensitivity_sim,
        min_interval=resolved_min_interval,
        sample_interval_sec=settings.sample_interval_sec,
        buffer_duration_sec=settings.buffer_duration_sec,
        transition_timeout_sec=settings.transition_timeout_sec,
        dedup_enabled=dedup_enabled,
    )

    start_time = time.time()
    slides = extractor.process(video_name=video_name)
    elapsed = time.time() - start_time

    # Assign IDs
    for idx, slide in enumerate(slides, 1):
        slide["id"] = f"cap_{idx:03d}"

    if write_manifest:
        manifest_path = output_root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(slides, handle, ensure_ascii=False, indent=2)

    print(f"[Capture] Completed: {len(slides)} slides in {elapsed:.2f}s")
    return slides
