# 🛠 Lecture Note AI Workflow Explanation

이 문서는 `Lecture-Note-AI` 시스템의 데이터 처리 흐름과 각 코드 모듈의 역할을 상세히 설명합니다.

## 1. Overall Workflow (전체 흐름)

시스템은 크게 **텍스트 처리(Text Processing)**와 **비전 처리(Vision Processing)** 두 가지 트랙으로 나뉘어 동작하며, `process_content.py`가 이를 조율합니다.

```mermaid
graph TD
    Start[Start] --> Init[Initialize Directories]
    Init --> Track1[Track 1: Text Processing]
    Init --> Track2[Track 2: Vision Processing]
    
    subgraph "Track 1: Text Processing"
        InputJSON[Input: data/output/*.json] --> Parser[src/data/json_parser.py]
        Parser --> OutputText[Output: *_readable.txt]
    end
    
    subgraph "Track 2: Vision Processing"
        InputVideo[Input: data/input/*.mp4] --> Processor[src/capture/video_processor.py]
        Processor --> SceneDetect{Scene Change?}
        SceneDetect -- Yes --> Capture[Capture Frame]
        Capture --> Inpaint[Remove Human (MediaPipe)]
        Inpaint --> SaveImg[Output: *_frames/*.jpg]
    end
    
    OutputText --> End[Complete]
    SaveImg --> End
```

---

## 2. Detailed Module Explanation (모듈별 상세 설명)

### 🅰 Orchestrator: `src/process_content.py`

전체 파이프라인의 **지휘자**입니다.

- **입력 데이터**:
  - `data/input/` 폴더의 모든 `.mp4` 파일
  - `data/output/` 폴더의 모든 `.json` 파일
- **핵심 로직**:
    1. `glob` 라이브러리를 사용하여 처리할 파일 목록을 스캔합니다.
    2. JSON 파일이 발견되면 `JsonParser` 클래스를 호출합니다.
    3. 비디오 파일이 발견되면 `VideoProcessor` 클래스를 호출합니다.
    4. 각 모듈의 실행 결과를 콘솔에 출력합니다.

### 🅱 Text Processing: `src/data/json_parser.py`

ClovaSpeech의 STT 결과를 사람이 읽기 편한 형태로 변환합니다.

- **입력 데이터**: ClovaSpeech 결과 JSON 파일
  - 구조: `{ "segments": [ { "start": 1234, "text": "안녕하세요" }, ... ] }`
- **적용 함수**: `parse_clova_speech(json_filename)`
  - **`json.load()`**: JSON 파일을 파이썬 딕셔너리로 로드합니다.
  - **`timedelta`**: 밀리초(ms) 단위의 시간을 `HH:MM:SS` 형식으로 변환합니다.
  - **Formatting**: `[시간] 텍스트` 형식으로 문자열을 조합합니다.
- **출력 데이터**: `*_readable.txt` (텍스트 파일)

### 🆎 Vision Processing: `src/capture/video_processor.py`

강의 영상에서 중요한 슬라이드 화면만 깨끗하게 추출합니다.

- **입력 데이터**: 강의 영상 파일 (`.mp4`)
- **적용 함수**: `extract_keyframes(video_path)`
  - **`cv2.VideoCapture`**: 영상을 프레임 단위로 읽어옵니다.
  - **`cv2.absdiff`**: 이전 프레임과 현재 프레임의 픽셀 차이를 계산하여 장면 전환을 감지합니다. (`threshold` 파라미터로 민감도 조절)
  - **`_collect_and_reconstruct`**: 장면 전환이 감지되면, 전후 3초간의 프레임을 모아 **Temporal Median** 필터를 적용합니다. 이는 움직이는 물체(사람)를 지우고 배경(슬라이드)만 남기는 효과가 있습니다.
  - **`_remove_human`**: **MediaPipe Selfie Segmentation** 모델을 사용하여 남은 사람의 잔상을 마스킹하고, **Inpainting** 기술로 지웁니다.
  - **`_remove_duplicates`**: **dHash** 알고리즘을 사용하여 연속으로 캡처된 유사한 중복 이미지를 제거합니다.
- **출력 데이터**: `*_frames/` 폴더 내의 `.jpg` 이미지 파일들

---

## 3. Directory Structure (폴더 구조)

```text
Lecture-Note-AI/
├── src/
│   ├── process_content.py       # [메인] 전체 실행 스크립트
│   ├── data/
│   │   ├── json_parser.py       # [모듈] JSON 파서
│   │   ├── input/               # [데이터] 원본 영상 위치 (*.mp4)
│   │   └── output/              # [데이터] JSON 결과 및 최종 산출물 (*.json, *.txt, images)
│   └── capture/
│       └── video_processor.py   # [모듈] 비디오 처리 및 캡처
```
