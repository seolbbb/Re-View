"""
[Intent]
단일 비디오 파일에 대해 전체 캡처 파이프라인(전처리, 특징점 추출, 슬라이드 저장, 매니페스트 생성)을
조율하고 실행하는 인터페이스 모듈입니다.
캡처 설정(settings.py)과 실제 추출 엔진(hybrid_extractor.py)을 연결하는 브릿지 역할을 합니다.

[Usage]
- src/pipeline/stages.py: run_capture() → process_single_video_capture() 호출
- src/run_preprocess_pipeline.py: 'Capture' 단계 실행 시 메인 프로세스로 호출
- src/pipeline/producers_async.py: asyncio.to_thread(run_capture, ...) → 내부에서 호출

[Usage Method]
- process_single_video_capture(video_path, output_base, ...) 함수 호출
- 내부에서 get_capture_settings()로 config/capture/settings.yaml의 12개 파라미터를 로드
- scene_threshold 인자가 주어지면 persistence_drop_ratio를 오버라이드
- HybridSlideExtractor에 모든 설정을 주입하고 process()를 실행
- 결과 manifest.json을 {output_base}/{video_name}/ 에 저장
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

from .settings import get_capture_settings
from .tools.hybrid_extractor import HybridSlideExtractor

def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: Optional[float] = None,
    min_interval: Optional[float] = None,
    write_manifest: bool = True,
    callback: Optional[Any] = None,
) -> List[dict]:
    """
    [Usage File]
    - src/pipeline/stages.py: run_capture() 내부에서 호출
    - src/run_preprocess_pipeline.py: Capture 단계 실행 시 직접 호출

    [Purpose]
    - 단일 비디오 파일을 입력받아 슬라이드 추출 전체 과정을 수행합니다.
    - 설정 로드, 출력 디렉토리 생성, Extractor 초기화 및 실행, 결과 매니페스트 저장을 담당합니다.

    [Connection]
    - get_capture_settings(): config/capture/settings.yaml → CaptureSettings 로드
    - HybridSlideExtractor: CaptureSettings의 12개 파라미터를 주입받아 슬라이드 추출 수행
    - 결과: {output_base}/{video_name}/captures/ 에 이미지 저장, manifest.json 생성

    [Args]
    - video_path (str): 처리할 입력 비디오의 절대 경로
    - output_base (str): 결과물(이미지, JSON)이 저장될 최상위 출력 디렉토리
    - scene_threshold (Optional[float]): persistence_drop_ratio 오버라이드 (None이면 settings.yaml 값 사용)
    - min_interval (Optional[float]): 미사용 (하위 호환용으로 시그니처에만 존재)
    - write_manifest (bool): 처리 완료 후 manifest.json 파일 생성 여부
    - callback (Optional[Any]): 슬라이드 감지 시 호출되는 콜백 ("new"/"update" 이벤트와 slide_data 전달)

    [Returns]
    - List[dict]: 추출된 슬라이드들의 메타데이터 리스트 (id, file_name, time_ranges)
    """
    settings = get_capture_settings()
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # 출력 경로 설정: {output_base}/{video_name}/captures
    video_output_dir = Path(output_base) / video_name
    captures_dir = video_output_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    # 파라미터 결정 (전달된 인자가 없으면 settings.yaml 기본값 사용)
    resolved_drop_ratio = settings.persistence_drop_ratio if scene_threshold is None else scene_threshold
    
    timestamp = datetime.now().strftime('%Y-%m-%d | %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] [Capture] Processing: {video_name}")
    print(f"   - Config: drop={resolved_drop_ratio}, persist={settings.persistence_threshold}, "
          f"min_feat={settings.min_orb_features}, "
          f"dist={settings.dedup_orb_distance}, sim={settings.dedup_sim_threshold})")
    
    # 추출기 초기화
    extractor = HybridSlideExtractor(
        video_path=video_path,
        output_dir=str(captures_dir),
        persistence_drop_ratio=resolved_drop_ratio,
        sample_interval_sec=settings.sample_interval_sec,
        persistence_threshold=settings.persistence_threshold,
        min_orb_features=settings.min_orb_features,
        dedup_phash_threshold=settings.dedup_phash_threshold,
        dedup_orb_distance=settings.dedup_orb_distance,
        dedup_sim_threshold=settings.dedup_sim_threshold,
        enable_roi_detection=settings.enable_roi_detection,
        roi_padding=settings.roi_padding,
        enable_smart_roi=settings.enable_smart_roi,
        roi_warmup_frames=settings.roi_warmup_frames,
        enable_adaptive_resize=settings.enable_adaptive_resize,
        callback=callback
    )

    start_time = time.time()
    
    # 실제 추출 프로세스 실행
    results = extractor.process(video_name=video_name)
    
    # 개별 비디오별 manifest 파일 저장 (필요 시)
    if write_manifest and results:
        manifest_path = video_output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
