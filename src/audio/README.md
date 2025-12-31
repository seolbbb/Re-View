# Audio 모듈 안내

## 구성
- 파일 트리
```
src/audio/
├── __init__.py
├── audio_processor.py
├── clova_stt.py
├── extract_audio.py
├── stt_router.py
└── test.py
```

- `clova_stt.py`: Clova Speech STT 클라이언트. `stt.json` 스키마(v1)로 저장하며, 세그먼트마다 `confidence`를 포함할 수 있음.
- `stt_router.py`: STT 제공자 라우터. 현재는 `clova`만 지원하며, `STT_PROVIDER` 환경변수로 선택.
- `extract_audio.py`: ffmpeg로 영상에서 오디오 추출(기본 16k/mono WAV).
- `test.py`: Clova REST 테스트용 스크립트. **원본 응답(raw)** 저장 목적.
- `audio_processor.py`: Whisper 기반 로컬 STT 파이프라인(추출+전사) 실험용.
- `__init__.py`: 패키지 초기화.

## 실행 방법

## 설치/의존성
- 공통(Python): `pip install -r requirements.txt`
- 오디오 추출: `ffmpeg` 설치 필요 (예: `apt-get install ffmpeg`)
- 토큰 계산(선택): `pip install tiktoken`
- Whisper(선택): `pip install -U openai-whisper torch` + `ffmpeg`
- 현재 audio 테스트에서 설치한 패키지: `requests`, `pydantic`, `python-dotenv`, `tiktoken`
- 환경변수: `CLOVA_SPEECH_URL`, `CLOVA_SPEECH_API_KEY` (또는 `CLOVA_SPEECH_SECRET`)

### 1) Clova STT (스키마 v1 출력)
```bash
python src/audio/clova_stt.py --media-path src/data/input/sample.mp4
```

기본 출력:
- `src/data/output/<입력파일명>/stt.json`
- 형식: `{ "schema_version": 1, "segments": [{ "start_ms", "end_ms", "text", "confidence?" }] }`

환경변수:
- `CLOVA_SPEECH_URL`
- `CLOVA_SPEECH_API_KEY` (또는 `CLOVA_SPEECH_SECRET`)
- `.env`는 `/data/ephemeral/home/Screentime-MVP/.env` 우선 로드

### 2) STT 라우터 사용
```python
from src.audio.stt_router import STTRouter

router = STTRouter()  # STT_PROVIDER 없으면 clova
router.transcribe("src/data/input/sample.mp4")
```

### 3) 오디오 추출
```bash
python src/audio/extract_audio.py --media-path src/data/input/sample.mp4
```

기본 출력:
- `src/data/output/<입력파일명>/audio.wav`

### 4) Clova 원본 응답 테스트
```bash
python src/audio/test.py --media-path src/data/input/sample.mp4
```

`test.py`는 **원본 응답 저장용**이라 `clova_stt.py` 출력 포맷과 다릅니다.
