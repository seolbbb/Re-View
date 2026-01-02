"""
[메인 파이프라인 - process_content.py]

강의 영상 처리 파이프라인을 관리합니다.
1. STT JSON 파싱: ClovaSpeech 결과를 텍스트로 변환
2. 비디오 분석: 장면 전환 감지 + 1차/2차 정제 통합 + 시각화

Note: 1차 정제(장면 전환 감지)와 2차 정제(중복 제거)는 video_processor.py에서 한 번에 처리됩니다.
"""

import os
import sys
import glob
import json

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/capture
src_dir = os.path.dirname(current_dir)  # src
project_root = os.path.dirname(src_dir)  # project root

if project_root not in sys.path:
    sys.path.append(project_root)

# 내부 모듈 임포트
from src.data.json_parser import JsonParser
from src.capture.video_processor import VideoProcessor
from src.capture.scene_visualizer import SceneVisualizer


def process_single_video_capture(video_path, output_dir, scene_threshold=3, dedupe_threshold=3, min_interval=0.5):
    """
    단일 비디오에 대해 키프레임 추출, 메타데이터 생성, 분석 그래프 생성을 수행합니다.
    run_video_pipeline.py에서도 이 함수를 호출하여 동일한 로직을 공유합니다.
    """
    video_processor = VideoProcessor()
    scene_visualizer = SceneVisualizer()
    
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]
    
    # [구조화] 비디오별 폴더 생성 (표준 구조)
    video_root = os.path.join(output_dir, video_name)
    capture_output_dir = os.path.join(video_root, "captures")
    os.makedirs(capture_output_dir, exist_ok=True)
    
    print(f"\n🎬 분석 중: {filename}")
    
    # 키프레임 추출 (1차+2차 정제 통합)
    keyframes_metadata, diff_scores, fps = video_processor.extract_keyframes(
        video_path,
        output_dir=capture_output_dir,
        threshold=scene_threshold,
        min_interval=min_interval,
        verbose=True,
        video_name=video_name,
        return_analysis_data=True,
        dedupe_threshold=dedupe_threshold
    )

    if keyframes_metadata:
        # 메타데이터 JSON 저장 (manifest.json으로 정규화)
        metadata_path = os.path.join(video_root, "manifest.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(keyframes_metadata, f, indent=4, ensure_ascii=False)
        print(f"   📋 메타데이터 저장: {os.path.basename(metadata_path)}")
        
        # Scene Change Analysis 그래프 생성
        if diff_scores:
            graph_path = os.path.join(video_root, f"{video_name}_scene_analysis.png")
            scene_visualizer.create_scene_change_graph(
                diff_scores=diff_scores,
                keyframes_metadata=keyframes_metadata,
                threshold=scene_threshold,
                fps=fps,
                video_name=video_name,
                output_path=graph_path,
                dedupe_threshold=dedupe_threshold
            )
            print(f"   📊 그래프 저장: {os.path.basename(graph_path)}")
    
    return keyframes_metadata


def main():
    """
    [메인 오케스트레이터]
    전체 강의 영상 처리 파이프라인을 관리합니다.
    """
    
    # ============================================================
    # 경로 설정
    # ============================================================
    input_dir = os.path.join(src_dir, 'data', 'input')
    output_dir = os.path.join(src_dir, 'data', 'output')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 입력 경로: {input_dir}")
    print(f"📂 결과 경로: {output_dir}")

    # ============================================================
    # Step 1: ClovaSpeech STT JSON → 텍스트 변환
    # ============================================================
    print("\n" + "="*60)
    print("[1/2] STT JSON 파일 처리")
    print("="*60)
    
    json_parser = JsonParser(input_dir, output_dir)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print("⚠ 처리할 JSON 파일이 없습니다.")
    else:
        for json_path in json_files:
            filename = os.path.basename(json_path)
            print(f"   📄 {filename}")
            json_parser.parse_clova_speech(filename)

    # ============================================================
    # Step 2: 비디오 분석 (1차+2차 정제 통합)
    # ============================================================
    print("\n" + "="*60)
    print("[2/2] 비디오 분석 및 키프레임 추출")
    print("="*60)
    
    # 파이프라인 설정값
    SCENE_THRESHOLD = 3
    DEDUPE_THRESHOLD = 3
    MIN_INTERVAL = 0.5
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_files:
        print("⚠ 처리할 비디오 파일이 없습니다.")
    
    for video_path in video_files:
        process_single_video_capture(
            video_path, 
            output_dir, 
            scene_threshold=SCENE_THRESHOLD, 
            dedupe_threshold=DEDUPE_THRESHOLD, 
            min_interval=MIN_INTERVAL
        )

    print("\n" + "="*60)
    print("✅ 파이프라인 완료")
    print("="*60)


if __name__ == "__main__":
    main()
