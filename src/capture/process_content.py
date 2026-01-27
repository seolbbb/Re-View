"""
[Intent]
단일 비디오 파일에 대해 캡처 파이프라인(추출, 저장, 매니페스트 생성)을 
실행하는 고수준 인터페이스 모듈입니다. 

[Usage]
- run_preprocess_pipeline.py에서 'Capture' 단계 실행 시 메인 프로세스로 호출됩니다.
- 캡처 설정(settings.py)과 실제 추출 엔진(hybrid_extractor.py)을 연결하는 가교 역할을 합니다.

[Usage Method]
- process_single_video_capture() 함수를 비디오 경로와 함께 호출하여 
  해당 비디오의 모든 주요 장면을 이미지로 추출하고 결과를 리스트 형태로 반환받습니다.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from .settings import get_capture_settings
from .tools.hybrid_extractor import HybridSlideExtractor


def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: Optional[float] = None,
    dedupe_threshold: Optional[float] = None,
    min_interval: Optional[float] = None,
    write_manifest: bool = True,
) -> List[dict]:
    """
    [Usage File] run_preprocess_pipeline.py
    [Purpose] 단일 비디오에 대해 슬라이드 추출 프로세스를 수행하고 결과 메타데이터를 반환합니다.
    [Connection] HybridSlideExtractor 클래스와 통신하여 실제 연산 수행
    
    [Args]
    - video_path (str): 입력 비디오 절대 경로
    - output_base (str): 결과 파일이 저장될 부모 디렉토리
    - scene_threshold (Optional[float]): 슬라이드 전환 민감도 (None일 경우 설정 파일 값 사용)
    - dedupe_threshold (Optional[float]): 중복 제거 민감도 (현재 로직에서는 Persistence에 통합됨)
    - min_interval (Optional[float]): 캡처 간 최소 간격
    - write_manifest (bool): 처리 완료 후 개별 manifest 파일을 생성할지 여부
    
    [Returns]
    - List[dict]: 추출된 모든 슬라이드의 정보 (timestamp, image_path 등)
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
    
    print(f"[Capture] Processing: {video_name}")
    print(f"   - Config: drop_ratio={resolved_drop_ratio}, threshold={settings.persistence_threshold}, sample={settings.sample_interval_sec}s")
    
    # 추출기 초기화
    extractor = HybridSlideExtractor(
        video_path=video_path,
        output_dir=str(captures_dir),
        persistence_drop_ratio=resolved_drop_ratio,
        sample_interval_sec=settings.sample_interval_sec,
        persistence_threshold=settings.persistence_threshold,
        min_orb_features=settings.min_orb_features
    )

    start_time = time.time()
    # 실제 추출 프로세스 실행
    results = extractor.process(video_name=video_name)
    elapsed = time.time() - start_time

    # 개별 비디오별 manifest 파일 저장 (필요 시)
    if write_manifest and results:
        manifest_path = video_output_dir / "capture_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
