# Audio 모듈 안내

## 구성
- 파일 트리
```
src/audio/
├── __init__.py
├── clova_stt.py
├── extract_audio.py
├── stt_router.py
├── whisper_stt.py
```

- `clova_stt.py`: Clova Speech STT 클라이언트. `stt.json` 스키마(v1)로 저장하며, 세그먼트마다 `confidence`를 포함할 수 있음.
- `stt_router.py`: STT 제공자 라우터. `clova`/`whisper` 지원, `STT_PROVIDER`로 선택. (영상 입력 시 `transcribe_media()` 사용)
- `extract_audio.py`: ffmpeg로 영상에서 오디오 추출(기본 16k/mono WAV).
- `whisper_stt.py`: Whisper 로컬 STT 클라이언트(오디오 입력 전용).
- `__init__.py`: 패키지 초기화.

## 실행 방법

## 설치/의존성
- 공통(Python): `pip install -r requirements.txt`
- 오디오 추출: `ffmpeg` 설치 필요 (예: `apt-get install ffmpeg`)
- 토큰 계산(선택): `pip install tiktoken`
- Whisper(선택): `pip install -U openai-whisper torch` + `ffmpeg`
- 현재 audio 테스트에서 설치한 패키지: `requests`, `pydantic`, `python-dotenv`, `tiktoken`, `openai-whisper`, `torch`
- 환경변수: `CLOVA_SPEECH_URL`, `CLOVA_SPEECH_API_KEY` (또는 `CLOVA_SPEECH_SECRET`)
- 설치 패키지 버전 (py310 기준):
  - `requests==2.32.5`
  - `pydantic==2.12.5`
  - `python-dotenv==1.2.1`
  - `tiktoken==0.12.0`
  - `openai-whisper==20250625`
  - `torch==2.9.1`
  - 참고: torch 설치 시 CUDA 관련 패키지가 함께 설치될 수 있음(GPU 빌드).

### 1) Clova STT (스키마 v1 출력)
```bash
python src/audio/clova_stt.py --media-path src/data/input/sample.mp4
```

기본 출력:
- `src/data/output/<입력파일명>/stt.json`
- 형식: `{ "segments": [{ "start_ms", "end_ms", "text", "confidence?" }] }`

옵션 요약:
- `include_confidence`: 세그먼트별 confidence 포함(기본 True)
- `include_raw_response`: 원본 응답(raw_response) 포함
- `word_alignment`: 단어 단위 타임스탬프 요청(raw_response에 words 추가)
- `full_text`: 전체 텍스트 필드 요청(raw_response에 fullText 추가)
- `completion`: `sync`/`async` (async는 폴링 구현 필요)

환경변수:
- `CLOVA_SPEECH_URL`
- `CLOVA_SPEECH_API_KEY` (또는 `CLOVA_SPEECH_SECRET`)
- `.env`는 `/data/ephemeral/home/Screentime-MVP/.env` 우선 로드

### 2) STT 라우터 사용
```python
from src.audio.stt_router import STTRouter

router = STTRouter()  # STT_PROVIDER 없으면 clova
# 오디오 파일이 이미 있을 때
router.transcribe("src/data/input/sample.wav", provider="whisper")

# 영상 → 오디오 추출 → STT
router.transcribe_media("src/data/input/sample.mp4", provider="clova", mono_method="auto")
```

CLI 예시:
```bash
# 영상 입력 + clova
python src/audio/stt_router.py --media-path src/data/input/sample.mp4 --provider clova

# 오디오 입력 + whisper (추출 스킵)
python src/audio/stt_router.py --media-path src/data/input/sample.wav --provider whisper --no-extract --model-size base
```

### 3) 오디오 추출
```bash
python src/audio/extract_audio.py --media-path src/data/input/sample.mp4
```

기본 출력:
- `src/data/input/<입력파일명>.wav`
모노 옵션:
- `--mono-method downmix|left|right|phase-fix|auto` (기본: auto, 60초 구간 기준으로 자동 선택)

### 4) Whisper STT (오디오 입력)
```bash
python src/audio/whisper_stt.py --audio-path src/data/input/sample.wav --model-size base
```
