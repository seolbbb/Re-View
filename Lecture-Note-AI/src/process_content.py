import os
import sys
import glob

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# [모듈 임포트]
# - JsonParser: ClovaSpeech JSON 데이터를 텍스트로 변환하는 클래스
# - VideoProcessor: 영상에서 키프레임을 추출하고 사람을 지우는 클래스
from src.data.json_parser import JsonParser
from src.capture.video_processor import VideoProcessor

def main():
    """
    [워크플로우 메인 함수]
    1. 입력(Input) 및 출력(Output) 디렉토리 설정
    2. JSON 처리: ClovaSpeech 결과 파일(*.json)을 파싱하여 가독성 있는 텍스트로 변환
    3. 비디오 처리: 강의 영상(*.mp4)을 분석하여 장면 전환 시점의 슬라이드 캡처
    """
    
    # [설정] 경로 설정
    # - base_dir: 현재 스크립트가 위치한 경로
    # - input_dir: 처리할 원본 영상(*.mp4)이 위치해야 하는 폴더
    # - output_dir: ClovaSpeech 결과(*.json)가 위치하고, 결과물(텍스트, 이미지)이 저장될 폴더
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')

    # 폴더가 없으면 생성 (안전장치)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"📂 Input Directory: {input_dir}")
    print(f"📂 Output Directory: {output_dir}")

    # ---------------------------------------------------------
    # [Step 1] JSON 파일 처리 (Text Processing)
    # ---------------------------------------------------------
    # - 입력 데이터: data/output/*.json (ClovaSpeech STT 결과)
    # - 적용 함수: json_parser.parse_clova_speech()
    # - 결과 데이터: data/output/*_readable.txt
    print("\n[1/2] Processing JSON Files...")
    json_parser = JsonParser(output_dir, output_dir) # 입/출력 경로 설정
    
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        print("⚠ No JSON files found in output directory.")
    
    for json_path in json_files:
        filename = os.path.basename(json_path)
        print(f"   - Parsing: {filename}")
        
        # [함수 호출] JSON 파싱 및 텍스트 변환 실행
        json_parser.parse_clova_speech(filename)

    # ---------------------------------------------------------
    # [Step 2] 비디오 파일 처리 (Vision Processing)
    # ---------------------------------------------------------
    # - 입력 데이터: data/input/*.mp4 (강의 영상)
    # - 적용 함수: video_processor.extract_keyframes()
    # - 결과 데이터: data/output/{video_name}_frames/ (캡처된 이미지들)
    print("\n[2/2] Processing Video Files...")
    video_processor = VideoProcessor()
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    if not video_files:
        print("⚠ No MP4 video files found in input directory.")
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        print(f"   - Processing Video: {filename}")
        
        # 캡처된 이미지를 저장할 폴더 생성 (영상 이름 기반)
        video_name = os.path.splitext(filename)[0]
        capture_output_dir = os.path.join(output_dir, f"{video_name}_frames")
        
        # [함수 호출] 키프레임 추출 실행
        # - threshold=30: 장면 전환 감지 민감도 (픽셀 차이 평균)
        # - min_interval=2.0: 최소 2초 간격으로만 캡처 (중복 방지)
        # - capture_duration=3.0: 사람 제거(Inpainting)를 위해 3초간의 프레임을 수집
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
