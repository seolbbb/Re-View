"""
하이브리드 캡처 파이프라인의 실행 진입점.

프로젝트 루트 경로를 계산해 import 경로로 추가하고,
입력/출력 경로와 캡처 임계값을 이 모듈에서 정의한다.

상세 설명은 src/capture/README.md를 참고한다.
"""

import os
import sys
import glob
import time
import shutil
import json
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.capture.tools.hybrid_extractor import HybridSlideExtractor

INPUT_DIR = os.path.join(src_dir, "data", "input")
OUTPUT_DIR = os.path.join(src_dir, "data", "output")

SENSITIVITY_DIFF = 3.0
SENSITIVITY_SIM = 0.8
MIN_INTERVAL = 0.5


def print_summary_table(title: str, metrics: dict) -> None:
    """
    처리 결과를 정렬된 테이블 형태로 출력합니다.
    
    Args:
        title (str): 테이블 제목
        metrics (dict): 출력할 메트릭 딕셔너리 {키: 값}
    """
    print("\n" + "="*60)
    print(f"Result: {title}")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"   {key:<20}: {value}")
    print("="*60)


def _process_video_core(
    video_path: str,
    output_base: str,
    scene_threshold: float,
    sensitivity_sim: float,
    min_interval: float,
    is_standalone: bool = False
) -> list:
    """
    V2 캡처 처리 핵심 로직.

    지연 저장 방식으로 슬라이드 저장 시점을 조정하고,
    결과를 manifest.json으로 정리한다.

    standalone 모드일 때는 캡처/로그 폴더를 정리한다.

    video_path: 입력 비디오 파일 경로.
    output_base: 출력 기본 디렉터리.
    scene_threshold: 장면 전환 감지 임계값(픽셀 차이).
    sensitivity_sim: ORB 유사도 임계값.
    min_interval: 최소 캡처 간격(초).
    is_standalone: 독립 실행 모드 여부.

    반환: [{"file_name": str, "start_ms": int, "end_ms": int}, ...]
    """
    video_name = Path(video_path).stem
    
    video_output_root = os.path.join(output_base, video_name)
    captures_dir = os.path.join(video_output_root, "captures")
    
    if is_standalone and os.path.exists(captures_dir):
        shutil.rmtree(captures_dir)
        
    os.makedirs(captures_dir, exist_ok=True)
    
    if is_standalone:
        for log_name in ["capture_log.txt"]:
            log_path = os.path.join(video_output_root, log_name)
            if os.path.exists(log_path):
                os.remove(log_path)

    print(f"\n[Capture V2] Processing: {video_name}")
    
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
    
    manifest_path = os.path.join(video_output_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, ensure_ascii=False, indent=2)
    
    if is_standalone:
        metrics = {
            "Total Time": f"{elapsed:.2f}s",
            "Mode": "Hybrid V2 (Delayed Save)",
            "Total Slides": len(slides),
            "Output Path": captures_dir,
            "Manifest": manifest_path
        }
        print_summary_table(f"Capture Result: {video_name}", metrics)
    else:
        print(f"[Capture V2] Completed: {len(slides)} slides in {elapsed:.2f}s")

    return slides


def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: float = 3.0,
    dedupe_threshold: float = 3.0,
    min_interval: float = 0.5
) -> list:
    """
    run_video_pipeline.py에서 호출되는 캡처 인터페이스.

    V2 최적화 로직으로 슬라이드를 추출한다.

    video_path: 처리할 비디오 파일 경로.
    output_base: 출력 기본 디렉터리.
    scene_threshold: 장면 전환 감지 임계값.
    dedupe_threshold: 미사용 파라미터(호환 유지용).
    min_interval: 최소 캡처 간격(초).

    반환: 추출된 슬라이드 메타데이터 리스트.
    """
    return _process_video_core(
        video_path=video_path,
        output_base=output_base,
        scene_threshold=scene_threshold,
        sensitivity_sim=SENSITIVITY_SIM,
        min_interval=min_interval,
        is_standalone=False
    )


def process_single_video_standalone(video_path: str, output_root: str) -> None:
    """
    독립 실행용 래퍼 함수.

    커맨드 라인에서 직접 실행할 때 사용하며,
    결과 요약 테이블을 출력한다.
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
    스크립트의 진입점.

    --video가 있으면 해당 파일만 처리하고,
    없으면 입력 폴더의 MP4를 모두 처리한다.

    Usage:
        python src/capture/process_content.py                      # 모든 비디오 처리
        python src/capture/process_content.py --video sample1.mp4  # 특정 비디오만 처리
    """
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Slide Capture Pipeline (V2)")
    parser.add_argument("--video", help="특정 비디오 파일 하나만 처리할 경우 경로 지정", default=None)
    
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    video_files = []
    
    if args.video:
        target_path = os.path.abspath(args.video)
        if not os.path.exists(target_path):
            print(f"Error: Video file not found: {target_path}")
            return
        video_files.append(target_path)
    else:
        video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))

    if not video_files:
        print("Warning: No video files to process.")
        return
        
    print(f"============================================================")
    print(f"Capture Pipeline V2 Started (Target: {len(video_files)} files)")
    print(f"============================================================")
    
    for video_path in video_files:
        process_single_video_standalone(video_path, OUTPUT_DIR)
        
    print(f"============================================================")
    print(f"Capture Pipeline V2 Completed")
    print(f"============================================================")


if __name__ == "__main__":
    main()
