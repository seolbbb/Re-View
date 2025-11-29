# Screentime MVP (Python Modules)

이 디렉토리는 Screentime 서비스의 핵심 기능을 담당하는 Python 모듈들을 포함하고 있습니다.

## 📂 Modules

- **`video_processor.py`**: 비디오에서 슬라이드 키프레임을 추출하고, 사람(강사)을 제거하는 전처리 모듈.
- **`audio_processor.py`**: 비디오에서 오디오를 추출하고 Whisper 모델을 사용하여 STT(Speech-to-Text)를 수행하는 모듈.
- **`mask_video.py`**: 비디오의 특정 영역을 마스킹하는 유틸리티.

## 🔍 Structure Review

현재 코드 구조에 대한 상세한 분석과 리팩토링 제안은 [STRUCTURE_REVIEW.md](STRUCTURE_REVIEW.md)를 참고하세요.

## 🚀 Usage

각 파일은 독립적으로 실행 가능합니다.

```bash
python video_processor.py
python audio_processor.py
```

*주의: 내부 테스트 코드의 파일 경로를 본인의 환경에 맞게 수정해야 합니다.*
