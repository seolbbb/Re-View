import os
import sys
import glob
import json

# 프로젝트 루트 경로를 시스템 경로에 추가하여 src 패키지를 인식할 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 내부 모듈 임포트
from src.data.json_parser import JsonParser
from src.capture.video_processor import VideoProcessor

def main():
    """
    [메인 오케스트레이터]
    전체 강의 영상 처리 파이프라인을 관리합니다.
    1. STT JSON 파싱: ClovaSpeech 결과를 사람이 읽기 쉬운 텍스트로 변환
    2. 비디오 분석: 장면 전환을 감지하여 슬라이드 이미지를 추출하고 분석 메타데이터 생성
    """
    
    # 기본 경로 설정: 스크립트 위치를 기준으로 데이터 입출력 폴더 지정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data', 'input')
    output_dir = os.path.join(base_dir, 'data', 'output')

    # 필요한 폴더가 없으면 자동으로 생성
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 입력 경로 (Input): {input_dir}")
    print(f"📂 결과 경로 (Output): {output_dir}")

    # ============================================================
    # Step 1: ClovaSpeech STT JSON 처리
    # ============================================================
    print("\n[1/2] STT JSON 파일 처리 중...")
    # JsonParser 객체 생성: 입력 폴더와 출력 폴더를 동일하게 설정하거나 분리 가능
    json_parser = JsonParser(input_dir, output_dir)
    
    # input 폴더 내의 모든 .json 파일을 찾아 변환 작업 수행
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        print("⚠ 처리할 JSON 파일이 없습니다.")
    
    for json_path in json_files:
        filename = os.path.basename(json_path)
        print(f"   - 처리 파일: {filename}")
        # parse_clova_speech 함수가 가독성 있는 .txt 파일을 생성함
        json_parser.parse_clova_speech(filename)

    # ============================================================
    # Step 2: 비디오 키프레임 추출 및 메타데이터 생성
    # ============================================================
    print("\n[2/2] 비디오 파일 분석 및 키프레임 추출 중...")
    video_processor = VideoProcessor()
    
    # input 폴더 내의 모든 .mp4 파일을 찾아 분석 작업 수행
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    if not video_files:
        print("⚠ 처리할 비디오 파일이 없습니다.")
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        video_name = os.path.splitext(filename)[0]
        print(f"   - 분석 파일: {filename}")
        
        # 영상별 결과 저장 폴더 생성 (예: video_name_frames)
        capture_output_dir = os.path.join(output_dir, f"{video_name}_frames")
        
        # [핵심 로직] 키프레임 추출 및 상세 분석 데이터 획득
        # - threshold: 장면 전환 감도 (작을수록 예민함)
        # - min_interval: 캡처 간 최소 유효 시간
        keyframes_metadata = video_processor.extract_keyframes(
            video_path,
            output_dir=capture_output_dir,
            threshold=8,
            min_interval=0.5,
            verbose=True
        )

        # [팀 공유용] 추출 결과를 JSON 메타데이터 파일로 저장
        if keyframes_metadata:
            metadata_path = os.path.join(output_dir, f"{video_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # 인간이 읽기 쉽도록 4칸 들여쓰기 적용
                json.dump(keyframes_metadata, f, indent=4, ensure_ascii=False)
            print(f"   ✅ 분석 메타데이터 저장 완료: {os.path.basename(metadata_path)}")

    print("\n✅ 모든 파이프라인 처리가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()

