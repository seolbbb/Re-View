# 🏗️ Screentime MVP Structure Review

## 1. Current Status

`screentime_mvp` 폴더는 비디오 처리(`video_processor.py`), 오디오 처리(`audio_processor.py`), 그리고 유틸리티(`mask_video.py`)로 구성된 **스크립트 모음(Collection of Scripts)** 형태입니다.

### ✅ Pros (장점)

- **Modularity**: 각 파일이 명확한 역할(비디오/오디오/마스킹)을 가지고 클래스/함수로 분리되어 있습니다.
- **Runnability**: 모든 파일에 `if __name__ == "__main__":` 블록이 있어 개별 테스트가 용이합니다.
- **Robustness**: `try-except` 블록을 통해 `MediaPipe`나 `FFmpeg` 같은 외부 의존성 부재 시에도 안내 메시지를 출력하도록 처리되어 있습니다.

### ⚠ Cons (단점 & 이슈)

- **Hardcoded Paths**: 테스트 코드(`__main__`) 내에 `C:\Users\irubw\...`와 같은 절대 경로가 하드코딩되어 있어, 다른 환경에서 즉시 실행이 불가능합니다.
- **Missing Requirements**: `requirements.txt`가 없어 필요한 라이브러리 버전을 알기 어렵습니다.
- **No Unified Entry**: 전체 파이프라인(비디오+오디오 처리)을 한 번에 실행하는 메인 스크립트가 없습니다.

---

## 2. Issues & Fixes

### 2.1. Hardcoded Paths

**Issue**:

```python
video_file = r"C:\Users\irubw\geminiProject\screentime_mvp\screentime_MVP\dirty_ex2_masked.mp4"
```

**Fix**: 상대 경로를 사용하거나, `argparse`를 통해 명령줄 인자로 받도록 수정해야 합니다.

### 2.2. Dependencies

다음 라이브러리들이 필요합니다:

- `opencv-python`
- `numpy`
- `mediapipe` (Optional but recommended)
- `openai-whisper`
- `torch`
- `ffmpeg` (System dependency)

---

## 3. Refactoring Proposal

### 3.1. Recommended Directory Structure

```text
screentime_mvp/
├── main.py                 # [NEW] Unified Entry Point
├── requirements.txt        # [NEW] Dependency List
├── src/                    # [NEW] Source Code Directory
│   ├── __init__.py
│   ├── video_processor.py  # Moved
│   ├── audio_processor.py  # Moved
│   └── utils.py            # (mask_video.py renamed)
└── data/                   # [NEW] Data Directory
    ├── input/
    └── output/
```

### 3.2. Action Items

1. **`requirements.txt` 생성**: 의존성 명시.
2. **`main.py` 작성**: `VideoProcessor`와 `AudioProcessor`를 통합하여 실행하는 오케스트레이터 생성.
3. **Path Handling**: `os.path` 또는 `pathlib`을 사용하여 경로를 유연하게 처리.

---

## 4. Conclusion

현재 상태로도 **개별 모듈의 기능 검증(Unit Testing)**은 가능하지만, **통합 서비스(Integrated Service)**로 동작하기 위해서는 리팩토링이 필요합니다.
특히 하드코딩된 경로는 즉시 수정이 권장됩니다.
