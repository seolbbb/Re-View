"""
================================================================================
process_original.py - Legacy 슬라이드 캡처 파이프라인 (VideoProcessor 기반)
================================================================================

[역할]
    VideoProcessor + UltimateSlideExtractor 기반의 레거시 캡처 파이프라인.
    2-Pass 방식으로 더 정밀한 슬라이드 검출을 수행합니다.
    
    ※ 참고: 현재 production에서는 process_content.py (Hybrid)를 사용합니다.
            이 파일은 하위 호환성 및 비교 테스트 목적으로 유지됩니다.

[파이프라인 아키텍처]
    +-------------------------------------------------------------------------+
    |  입력: MP4 비디오 파일 (src/data/input/*.mp4)                            |
    +-------------------------------------------------------------------------+
                                    |
                                    v
    +-------------------------------------------------------------------------+
    |  1단계: VideoProcessor (1차 정제)                                        |
    |  -----------------------------------------------------------------------+
    |  - dHash 기반 장면 전환 감지                                             |
    |  - 시간 축 Median 필터 (마우스/사람 제거)                                 |
    |  - 초기 키프레임 후보 추출                                               |
    +-------------------------------------------------------------------------+
                                    |
                                    v
    +-------------------------------------------------------------------------+
    |  2단계: 중복 제거 (2차 정제)                                             |
    |  -----------------------------------------------------------------------+
    |  - ORB 특징점 기반 유사도 분석                                           |
    |  - THRESHOLD_DEDUPE 이상 차이가 나야 저장                                |
    +-------------------------------------------------------------------------+
                                    |
                                    v
    +-------------------------------------------------------------------------+
    |  출력: 캡처된 슬라이드 이미지 (src/data/output/{video}_frames/*.jpg)      |
    +-------------------------------------------------------------------------+

[임계값 설정]
    THRESHOLD_SCENE = 8     # 1차 정제: 장면 전환 감지 임계값
                             # - 낮음(5): 작은 변화도 감지 -> 더 많은 후보
                             # - 높음(15): 큰 변화만 감지 -> 더 적은 후보
    
    THRESHOLD_DEDUPE = 3.0  # 2차 정제: 중복 제거 임계값
                             # - 낮음(2.0): 거의 동일해야 중복 처리
                             # - 높음(5.0): 조금만 비슷해도 중복 처리
    
    MIN_INTERVAL = 0.5      # 최소 캡처 간격 (초)

[사용 방법]
    - 직접 실행: python src/capture/process_original.py
    - (현재 run_video_pipeline.py는 process_content.py를 사용합니다)

[최종 수정] 2026-01-07
================================================================================
"""

import os
import sys
import glob
import time
import logging
from pathlib import Path

# ============================================================
# 프로젝트 경로 구성
# Python import 시스템에서 프로젝트 루트를 인식하도록 설정
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/capture/
src_dir = os.path.dirname(current_dir)                      # src/
project_root = os.path.dirname(src_dir)                     # Screentime-MVP/

if project_root not in sys.path:
    sys.path.append(project_root)

# Legacy 캡처 엔진 임포트
from src.capture.tools import VideoProcessor, UltimateSlideExtractor
from src.capture.tools.json_parser import JsonParser

# ============================================================
# 설정 파라미터 (1차/2차 정제 임계값)
# 이 값들을 조정하여 캡처 민감도를 튜닝할 수 있습니다.
# ============================================================
THRESHOLD_SCENE = 8     # 1차 정제: 장면 전환 감지 임계값 (낮을수록 민감)
THRESHOLD_DEDUPE = 3.0  # 2차 정제: 중복 제거 임계값 (높을수록 엄격)
MIN_INTERVAL = 0.5      # 최소 캡처 간격 (초)


# ============================================================
# 핵심 기능: process_single_video_capture
# (Legacy 인터페이스 - 현재 사용되지 않음)
# ============================================================
def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: int = THRESHOLD_SCENE,
    dedupe_threshold: float = THRESHOLD_DEDUPE,
    min_interval: float = MIN_INTERVAL
) -> list:
    """
    단일 비디오에 대해 Legacy 캡처 파이프라인을 실행합니다.
    
    ※ 참고: 현재 run_video_pipeline.py는 process_content.py를 사용합니다.
            이 함수는 하위 호환성을 위해 유지됩니다.
    
    Args:
        video_path: 입력 비디오 파일의 절대 경로
        output_base: 출력 베이스 디렉토리
        scene_threshold: 1차 정제 - 장면 전환 감지 임계값
        dedupe_threshold: 2차 정제 - 중복 제거 임계값
        min_interval: 최소 캡처 간격 (초)
    
    Returns:
        list: 캡처된 슬라이드 메타데이터 리스트
    """
    video_name = Path(video_path).stem
    # 출력 경로 구조: {output_base}/{video_name}/captures
    video_output_root = os.path.join(output_base, video_name)
    captures_dir = os.path.join(video_output_root, "captures")
    os.makedirs(captures_dir, exist_ok=True)

    print(f"\n[Capture] Processing (Legacy): {video_name}")
    vp = VideoProcessor()
    
    # VideoProcessor를 사용한 키프레임 추출
    metadata = vp.extract_keyframes(
        video_path, 
        output_dir=captures_dir,
        threshold=scene_threshold,      # 1차: 장면 전환 감지
        min_interval=min_interval,       # 최소 캡처 간격
        dedupe_threshold=dedupe_threshold,  # 2차: 중복 제거
        video_name=video_name
    )
    
    return metadata


def run_full_processing(video_path: str, input_dir: str, output_dir: str) -> list:
    """
    비디오 캡처 + STT JSON 파싱을 함께 수행합니다.
    
    Args:
        video_path: 입력 비디오 파일 경로
        input_dir: 입력 폴더 (비디오 + JSON)
        output_dir: 출력 폴더
    
    Returns:
        list: 캡처된 슬라이드 메타데이터
    """
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]
    
    # ---------------------------------------------------------
    # 1단계: Capture (슬라이드 캡처)
    # ---------------------------------------------------------
    capture_output_dir = os.path.join(output_dir, f"{video_name}_frames")
    print(f"\n[1/2] Capturing Slides: {filename}")
    
    vp = VideoProcessor()
    metadata = vp.extract_keyframes(
        video_path,
        output_dir=capture_output_dir,
        threshold=THRESHOLD_SCENE,       # 1차: 장면 전환 감지
        min_interval=MIN_INTERVAL,        # 최소 캡처 간격
        dedupe_threshold=THRESHOLD_DEDUPE,  # 2차: 중복 제거
        verbose=True
    )
    
    # manifest.json 저장 (캡처 메타데이터)
    import json
    manifest_path = os.path.join(capture_output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[Manifest] Saved: {manifest_path}")

    # ---------------------------------------------------------
    # 2단계: JSON Parsing (ClovaSpeech STT 결과)
    # ---------------------------------------------------------
    print(f"\n[2/2] Parsing STT JSON: {video_name}.json")
    json_p = JsonParser(input_dir, output_dir)
    json_path = os.path.join(input_dir, f"{video_name}.json")
    
    if os.path.exists(json_path):
        json_p.parse_clova_speech(f"{video_name}.json")
    else:
        print(f"Warning: STT JSON file not found, skipping: {json_path}")

    return metadata


def main() -> None:
    """
    메인 엔트리포인트: src/data/input/ 폴더의 모든 MP4 파일을 처리합니다.
    
    사용법:
        python src/capture/process_original.py
    """
    # 기본 경로 설정
    INPUT_DIR = os.path.join(src_dir, "data", "input")
    OUTPUT_DIR = os.path.join(src_dir, "data", "output")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 입력 폴더에서 MP4 파일 검색
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not video_files:
        print("Warning: No video files to process (check src/data/input)")
        return

    print("="*60)
    print("Lecture Note AI: Video & Content Processor (Legacy)")
    print("="*60)

    # 각 비디오 파일 처리
    for video_path in video_files:
        run_full_processing(video_path, INPUT_DIR, OUTPUT_DIR)

    print("\n[Done] All tasks completed successfully.")


if __name__ == "__main__":
    main()
