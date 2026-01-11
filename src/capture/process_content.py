"""
================================================================================
process_content.py - Hybrid 슬라이드 캡처 파이프라인 (Production Default)
================================================================================

[역할]
    강의 영상에서 슬라이드 전환을 감지하고 깨끗한 이미지를 추출하는 메인 파이프라인.
    HybridSlideExtractor를 사용하여 단일 패스로 고효율 캡처를 수행합니다.

[파이프라인 아키텍처]
    +-------------------------------------------------------------------------+
    |  입력: MP4 비디오 파일 (src/data/input/*.mp4)                            |
    +-------------------------------------------------------------------------+
                                    |
                                    v
    +-------------------------------------------------------------------------+
    |  1단계: HybridSlideExtractor (Hybrid Single-Pass)                       |
    |  -----------------------------------------------------------------------+
    |  - 프레임 단위 분석: 연속 프레임 간 픽셀 차이(dHash) 감지                 |
    |  - 전환 감지: 프레임 차이 > SENSITIVITY_DIFF -> 전환 시작                 |
    |  - 안정성 확인: 2.5초간 변화 없으면 -> 슬라이드 안정 상태로 판정           |
    |  - 마우스 제거: 2.5초 버퍼 -> Median 기준 Best Frame 선택                 |
    |  - 중복 제거: ORB 특징 + RANSAC 기하 검증 + Sim<0.5 구조변경 오버라이드   |
    +-------------------------------------------------------------------------+
                                    |
                                    v
    +-------------------------------------------------------------------------+
    |  출력: 캡처된 슬라이드 이미지 (src/data/output/{video}/captures/*.jpg)    |
    |        manifest.json (run_video_pipeline.py 연동용)                      |
    +-------------------------------------------------------------------------+

[임계값 설정]
    SENSITIVITY_DIFF = 3.0   # 픽셀 차이 민감도 (낮을수록 민감, 권장: 2.0~5.0)
                              # - 낮음(2.0): 작은 변화도 감지 -> 더 많은 캡처
                              # - 높음(5.0): 큰 변화만 감지 -> 더 적은 캡처
    
    SENSITIVITY_SIM = 0.8    # ORB 구조 유사도 (높을수록 엄격, 권장: 0.7~0.9)
                              # - 높음(0.9): 거의 동일해야 중복 처리
                              # - 낮음(0.7): 조금만 비슷해도 중복 처리
    
    MIN_INTERVAL = 0.5       # 최소 캡처 간격 (초)
                              # - 연속 슬라이드 전환 시 최소 대기 시간

[연동]
    - 직접 실행: python src/capture/process_content.py
    - 파이프라인 호출: run_video_pipeline.py -> process_single_video_capture()
"""

import os
import sys
import glob
import time
import logging
import shutil
import json
import cv2
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

# Hybrid 캡처 엔진 임포트
from src.capture.tools import HybridSlideExtractor

# ============================================================
# 설정 파라미터 (임계값)
# 이 값들을 조정하여 캡처 민감도를 튜닝할 수 있습니다.
# ============================================================
INPUT_DIR = os.path.join(src_dir, "data", "input")   # 입력 비디오 폴더
OUTPUT_DIR = os.path.join(src_dir, "data", "output") # 출력 결과 폴더

SENSITIVITY_DIFF = 3.0   # 픽셀 차이 민감도 (장면 전환 감지)
SENSITIVITY_SIM = 0.8    # ORB 구조 유사도 (중복 제거 엄격도)
MIN_INTERVAL = 0.5       # 최소 캡처 간격 (초)


def print_summary_table(title: str, metrics: dict) -> None:
    """
    처리 결과를 정렬된 테이블 형태로 출력합니다.
    
    Args:
        title: 테이블 제목
        metrics: 출력할 메트릭 딕셔너리 {키: 값}
    """
    print("\n" + "="*60)
    print(f"Result: {title}")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"   {key:<20}: {value}")
    print("="*60)


# ============================================================
# run_video_pipeline.py 호환 인터페이스
# run_video_pipeline.py가 이 함수를 직접 호출합니다.
# ============================================================
# ============================================================
# Core Logic (Refactored)
# ============================================================
def _process_video_core(
    video_path: str,
    output_base: str,
    scene_threshold: float,
    sensitivity_sim: float,
    min_interval: float,
    is_standalone: bool = False
) -> list:
    """
    [Core] 비디오 캡처 및 메타데이터 생성을 수행하는 핵심 로직입니다.
    """
    video_name = Path(video_path).stem
    
    # 출력 경로 설정
    if is_standalone:
        # Standalone: {output_root}/{video_name}/captures
        video_output_root = os.path.join(output_base, video_name)
    else:
        # Pipeline: {output_base}/{video_name} (Caller가 base를 어디까지 주느냐에 따라 다름)
        # run_video_pipeline.py는 output_base를 root로 전달함 -> {output_base}/{video_name}
        video_output_root = os.path.join(output_base, video_name)

    captures_dir = os.path.join(video_output_root, "captures")
    
    # 디렉토리 정리 (Standalone 모드일 때만 청소)
    if is_standalone and os.path.exists(captures_dir):
        shutil.rmtree(captures_dir)
        
    os.makedirs(captures_dir, exist_ok=True)
    
    # Standalone 모드일 때 로그 정리
    if is_standalone:
         for log_name in ["process_log_fast.txt", "process_log_ultimate.txt", "process_log_hybrid.txt", "capture_log.txt"]:
            log_path = os.path.join(video_output_root, log_name)
            if os.path.exists(log_path):
                os.remove(log_path)

    print(f"\n[Capture] Processing (Hybrid): {video_name}")
    
    # HybridSlideExtractor 초기화 및 실행
    extractor = HybridSlideExtractor(
        video_path,
        output_dir=captures_dir,
        sensitivity_diff=scene_threshold,
        sensitivity_sim=sensitivity_sim,
        min_interval=min_interval
    )
    
    start_time = time.time()
    slides = extractor.process(video_name=video_name)
    elapsed = time.time() - start_time
    
    # 비디오 메타데이터 (Duration)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0
        cap.release()
    else:
        duration_ms = 0
    
    # Manifest 생성 (Interval 변환 및 파일명 변경)
    refined_slides = []
    for i, slide in enumerate(slides):
        start_ms = slide.pop("timestamp_ms")
        
        # Calculate end_ms
        if i < len(slides) - 1:
            end_ms = slides[i+1]["timestamp_ms"]
        else:
            end_ms = duration_ms
            
        # Ensure valid interval
        if end_ms < start_ms:
            end_ms = start_ms 

        # 파일명 변경: sample1_001_{start_ms}_{end_ms}.jpg
        old_filename = slide["file_name"]
        new_filename = f"{video_name}_{i+1:03d}_{start_ms}_{end_ms}.jpg"
        
        old_path = os.path.join(captures_dir, old_filename)
        new_path = os.path.join(captures_dir, new_filename)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

        refined_slides.append({
            "file_name": new_filename,
            "start_ms": start_ms,
            "end_ms": end_ms
        })

    manifest_path = os.path.join(video_output_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(refined_slides, f, ensure_ascii=False, indent=2)
    
    if is_standalone:
        metrics = {
            "Total Time": f"{elapsed:.2f}s",
            "Mode": "Hybrid Single-Pass",
            "Total Slides": len(slides),
            "Output Path": captures_dir,
            "Manifest": manifest_path
        }
        print_summary_table(f"Hybrid Result: {video_name}", metrics)
    else:
        print(f"[Capture] Completed: {len(refined_slides)} slides captured")

    return refined_slides


# ============================================================
# run_video_pipeline.py 호환 인터페이스
# ============================================================
def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: float = 3.0,
    dedupe_threshold: float = 3.0,
    min_interval: float = 0.5
) -> list:
    """
    [역할] run_video_pipeline.py에서 호출되는 Wrapper
    """
    return _process_video_core(
        video_path=video_path,
        output_base=output_base,
        scene_threshold=scene_threshold,
        sensitivity_sim=SENSITIVITY_SIM, # Use global default for consistency or add arg if needed
        min_interval=min_interval,
        is_standalone=False
    )


def process_single_video_v2(video_path: str, output_root: str) -> None:
    """
    [역할] 독립 실행(Standalone) 시 호출되는 Wrapper
    """
    _process_video_core(
        video_path=video_path,
        output_base=output_root,
        scene_threshold=SENSITIVITY_DIFF,
        sensitivity_sim=SENSITIVITY_SIM,
        min_interval=MIN_INTERVAL,
        is_standalone=True
    )


def main() -> None:
    """
    [역할]
    스크립트의 진입점(Entry Point)입니다.
    선택적인 오디오 파일 경로 인자를 받아 처리하거나, 
    인자가 없는 경우 설정된 입력 폴더(src/data/input) 내의 모든 MP4 파일을 검색하여
    순차적으로 캡처 프로세스를 실행합니다.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Slide Capture Pipeline")
    parser.add_argument("--video", help="특정 비디오 파일 하나만 처리할 경우 경로 지정", default=None)
    
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    video_files = []
    
    if args.video:
        # 특정 비디오 지정 모드
        target_path = os.path.abspath(args.video)
        if not os.path.exists(target_path):
            print(f"Error: Video file not found: {target_path}")
            return
        video_files.append(target_path)
    else:
        # 배치 처리 모드 (기본 동작)
        # 입력 폴더에서 MP4 파일 검색
        video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))

    if not video_files:
        print("Warning: No video files to process.")
        return
        
    print(f"============================================================")
    print(f"Capture Pipeline Started (Target: {len(video_files)} files)")
    print(f"============================================================")
    
    # 각 비디오 파일 처리
    for video_path in video_files:
        process_single_video_v2(video_path, OUTPUT_DIR)
        
    print(f"============================================================")
    print(f"Capture Pipeline Completed")
    print(f"============================================================")

if __name__ == "__main__":
    main()
