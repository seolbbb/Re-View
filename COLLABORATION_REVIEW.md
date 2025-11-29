# 🤝 Collaboration Readiness Review

## 1. Status Check

`capture_test` 폴더의 코드들은 이제 **상대 경로(Relative Paths)**를 사용하여 어떤 환경에서도 실행 가능하도록 수정되었습니다.

### ✅ Fixes Applied

- **Hardcoded Paths Removed**: `C:\Users\irubw\...`와 같은 절대 경로를 모두 `data/input/`, `data/output/`과 같은 상대 경로로 변경했습니다.
- **Directory Safety**: 출력 폴더가 없을 경우 자동으로 생성하거나, 입력 파일이 없을 경우 안내 메시지를 출력하도록 개선했습니다.

---

## 2. Integration Plan (Next Steps)

현재 `capture_test`에 있는 파일들은 `Lecture-Note-AI` 프로젝트의 정식 모듈로 통합되어야 합니다.

### 📂 Recommended Structure

```text
Lecture-Note-AI/
├── src/
│   ├── capture/
│   │   └── video_processor.py  <-- (Move from capture_test)
│   ├── audio/
│   │   └── audio_processor.py  <-- (Move from capture_test)
│   └── utils/
│       └── mask_video.py       <-- (Move from capture_test)
```

### 🚀 Action Items

1. **Move Files**: 위 구조대로 파일을 이동시키십시오.
2. **Update Imports**: 이동 후 `import` 경로가 깨질 수 있으므로, `src.capture.video_processor`와 같이 패키지 경로를 수정해야 합니다.
3. **Merge Dependencies**: `capture_test/requirements.txt`의 내용을 `Lecture-Note-AI/requirements.txt`에 병합하십시오.

---

## 3. How to Test (Locally)

테스트를 위해서는 프로젝트 루트에 `data` 폴더를 만들고 영상을 넣어야 합니다.

```bash
mkdir -p data/input
# 테스트 영상(dirty_ex2.mp4 등)을 data/input에 복사
python capture_test/video_processor.py
```
