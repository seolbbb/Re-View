import os
import sys
import glob

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data.json_parser import JsonParser
from src.capture.video_processor import VideoProcessor

def main():
    """
    강의 영상 자동 처리 메인 함수
    1. JSON 파싱: ClovaSpeech 결과 → 텍스트 변환
    2. 비디오 처리: 영상 → 슬라이드 이미지 추출
    """
    
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')

    # 폴더 생성
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")

    # ========================================
    # Step 1: JSON 처리
    # ========================================
    print("\n[1/2] Processing JSON Files...")
    json_parser = JsonParser(output_dir, output_dir)
    
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        print("⚠ No JSON files found.")
    
    for json_path in json_files:
        filename = os.path.basename(json_path)
        print(f"   - {filename}")
        json_parser.parse_clova_speech(filename)

    # ========================================
    # Step 2: 비디오 처리
    # ========================================
    print("\n[2/2] Processing Video Files...")
    video_processor = VideoProcessor()
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    if not video_files:
        print("⚠ No video files found.")
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"   - {filename}")
        
        # 출력 폴더 생성
        video_name = os.path.splitext(filename)[0]
        capture_output_dir = os.path.join(output_dir, f"{video_name}_frames")
        
        # 키프레임 추출
        video_processor.extract_keyframes(
            video_path,
            output_dir=capture_output_dir,
            threshold=8,
            min_interval=0.5,
            verbose=True
        )

    print("\n✅ All processing complete.")

if __name__ == "__main__":
    main()

