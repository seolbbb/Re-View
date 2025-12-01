import json
import os
from datetime import timedelta

class JsonParser:
    """
    ClovaSpeech API 결과(JSON)를 파싱하여 가독성 있는 텍스트 파일로 변환하는 클래스
    """
    def __init__(self, input_dir, output_dir):
        # input_dir: JSON 파일이 위치한 경로
        # output_dir: 변환된 텍스트 파일이 저장될 경로
        self.input_dir = input_dir
        self.output_dir = output_dir

    def parse_clova_speech(self, json_filename):
        """
        [핵심 기능] JSON 파일을 읽어 문장별로 정리된 텍스트 파일로 저장합니다.
        
        Args:
            json_filename (str): 파싱할 JSON 파일의 이름 (예: 'lecture_result.json')
            
        Returns:
            str: 생성된 텍스트 파일의 경로 (실패 시 None)
        """
        json_path = os.path.join(self.input_dir, json_filename)
        
        # 1. 파일 존재 여부 확인
        if not os.path.exists(json_path):
            print(f"❌ JSON file not found: {json_path}")
            return None

        # 2. JSON 로드
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode JSON: {e}")
            return None

        # 3. 데이터 추출 (segments 키 확인)
        # - ClovaSpeech JSON 구조: { "segments": [ { "start": 0, "text": "..." }, ... ] }
        segments = data.get('segments', [])
        if not segments:
            print("⚠ No segments found in JSON.")
            return None

        # 4. 출력 파일명 생성 (확장자만 .txt로 변경)
        base_name = os.path.splitext(json_filename)[0]
        output_filename = f"{base_name}_readable.txt"
        output_path = os.path.join(self.output_dir, output_filename)

        # 5. 라인별 포맷팅
        lines = []
        for segment in segments:
            # - start: 시작 시간 (밀리초 단위)
            # - text: 인식된 텍스트
            start_ms = segment.get('start', 0)
            text = segment.get('text', '')
            
            # 밀리초 -> 시:분:초 형식 변환 (예: 12345ms -> 0:00:12)
            timestamp = str(timedelta(milliseconds=start_ms)).split('.')[0] 
            
            # 최종 포맷: [0:00:12] 안녕하세요 반갑습니다.
            formatted_line = f"[{timestamp}] {text}"
            lines.append(formatted_line)

        # 6. 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"✅ Parsed JSON saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    # 테스트용 코드 (직접 실행 시 사용)
    parser = JsonParser("src/data/output", "src/data/output")
    # parser.parse_clova_speech("test.json")
