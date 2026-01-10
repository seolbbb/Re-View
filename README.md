# 📚 Re:View - 강의 영상 자동 요약 시스템

강의 영상에서 슬라이드를 자동으로 추출하고, 음성을 텍스트로 변환하여 체계적인 강의 노트를 생성하는 AI 파이프라인입니다.

## ✨ 주요 기능

- **🎬 스마트 슬라이드 캡처**: dHash + ORB + RANSAC 기반 장면 전환 감지 및 중복 제거
- **🎤 음성 텍스트 변환**: Clova Speech / Whisper STT 지원
- **👁️ 시각 정보 추출**: VLM(Qwen3-VL)으로 슬라이드 내용 분석
- **📝 AI 요약 생성**: Gemini 기반 독립형 강의 노트 생성 (프롬프트 v1.5)
- **✅ 품질 검증**: Judge Agent를 통한 자동 품질 평가 (Groundedness, Note Quality, Spec Compliance)

---

## 🚀 Quick Start

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
GOOGLE_API_KEY=...          # Gemini (ADK, Summarize, Judge)
OPENROUTER_API_KEY=...      # VLM (Qwen)
CLOVA_SPEECH_URL=...        # STT (Clova)
CLOVA_SPEECH_API_KEY=...
```

### 3. 실행

#### Option A: End-to-End 파이프라인 (벤치마크 포함) ⭐ 추천

```bash
python src/run_video_pipeline.py --video "data/inputs/lecture.mp4"
```

**출력물:**

- `data/outputs/{video_name}/stt.json` - STT 결과
- `data/outputs/{video_name}/captures/*.jpg` - 캡처 이미지
- `data/outputs/{video_name}/manifest.json` - 캡처 메타데이터
- `data/outputs/{video_name}/vlm.json` - VLM 결과
- `data/outputs/{video_name}/fusion/segment_summaries.jsonl` - 구간별 요약
- `data/outputs/{video_name}/fusion/outputs/final_summary_*.md` - 최종 요약
- `data/outputs/{video_name}/benchmark_report.md` - 벤치마크 리포트

#### Option B: ADK 파이프라인 (대화형)

```bash
# Step 1: Pre-ADK (STT + Capture)
python src/pre_adk_pipeline.py --video "lecture.mp4"

# Step 2: ADK Web UI
adk web src/adk_pipeline
# 브라우저에서 http://localhost:8000 접속
```

---

## 📂 프로젝트 구조

```
Re:View/
├── data/
│   ├── inputs/                     # 입력 비디오 (.mp4)
│   └── outputs/                    # 출력 (비디오별 폴더)
│       └── {video_name}/
│           ├── stt.json            # STT 결과
│           ├── manifest.json       # 캡처 메타데이터
│           ├── captures/           # 캡처 이미지
│           ├── vlm_raw.json        # VLM 원시 결과
│           ├── vlm.json            # VLM 정제 결과
│           ├── config.yaml         # Fusion 설정
│           └── fusion/
│               ├── segments.jsonl
│               ├── segments_units.jsonl
│               ├── segment_summaries.jsonl
│               ├── segment_summaries.md
│               ├── judge.json
│               └── outputs/
│                   ├── final_summary_timeline.md
│                   └── final_summary_tldr_timeline.md
│
├── src/
│   ├── run_video_pipeline.py       # End-to-End CLI (벤치마크 포함)
│   ├── pre_adk_pipeline.py         # Pre-ADK CLI
│   │
│   ├── adk_pipeline/               # ADK 멀티에이전트
│   │   ├── agent.py                # Agent 정의 (Root + Sub-agents)
│   │   ├── store.py                # VideoStore (파일시스템 추상화)
│   │   ├── paths.py                # 경로 유틸리티
│   │   └── tools/
│   │       ├── root_tools.py       # list_available_videos, set_pipeline_config
│   │       ├── preprocessing_tools.py  # load_data, run_vlm, run_sync
│   │       ├── summarize_tools.py  # run_summarizer, render_md, write_final_summary
│   │       ├── judge_tools.py      # evaluate_summary
│   │       └── internal/           # 내부 구현 모듈
│   │
│   ├── audio/                      # STT 모듈
│   │   ├── stt_router.py           # STT 라우터 (Clova/Whisper)
│   │   ├── clova_stt.py            # Clova Speech 클라이언트
│   │   ├── whisper_stt.py          # Whisper 클라이언트
│   │   └── extract_audio.py        # ffmpeg 오디오 추출
│   │
│   ├── capture/                    # 슬라이드 캡처 모듈
│   │   ├── process_content.py      # 캡처 진입점
│   │   └── tools/
│   │       ├── hybrid_extractor.py # HybridSlideExtractor (메인 엔진)
│   │       ├── video_processor.py  # VideoProcessor (레거시)
│   │       └── scene_visualizer.py # 디버깅용 시각화
│   │
│   ├── vlm/                        # Vision-Language Model
│   │   ├── vlm_engine.py           # VLM 엔진 (OpenRouter)
│   │   ├── vlm_fusion.py           # VLM 결과 변환
│   │   └── qwen3_detect.py         # Qwen3 객체 탐지
│   │
│   ├── fusion/                     # 동기화, 요약, 렌더링
│   │   ├── sync_engine.py          # STT + VLM 동기화
│   │   ├── summarizer.py           # Gemini 요약 (프롬프트 v1.5)
│   │   ├── renderer.py             # Markdown 렌더링
│   │   ├── final_summary_composer.py  # 최종 요약 생성
│   │   ├── config.py               # 설정 로드
│   │   └── io_utils.py             # I/O 유틸리티
│   │
│   ├── judge/                      # 품질 평가
│   │   └── judge.py                # LLM Judge (Gemini 기반)
│   │
│   ├── common/                     # 공통 스키마
│   │   └── schemas.py              # Pydantic 모델
│   │
│   └── utils/                      # 유틸리티
│       ├── token_counter.py        # 토큰 카운터
│       └── postgres_ingest.py      # DB 인제스트
│
└── docs/
    ├── DEVELOPER_GUIDE.md          # 상세 개발 가이드
    ├── PRD.md                      # 제품 요구사항
    └── PROJECT_DIRECTION.md        # 프로젝트 방향성
```

---

## 🏗️ 아키텍처

```
[Video Input]
      │
      ├─── STT (Clova/Whisper) ──→ stt.json
      │
      └─── Capture (HybridSlideExtractor) ──→ manifest.json + captures/
             │
             │  [dHash 장면 감지 → 2.5초 안정화 → ORB+RANSAC 중복 제거]
             │
             ▼
      ┌─────────────────────────────────────────────────┐
      │              ADK Multi-Agent Pipeline           │
      │                                                 │
      │   ┌─────────────────────────────────────────┐   │
      │   │          Root Agent                     │   │
      │   │  (screentime_pipeline)                  │   │
      │   └─────────────────────────────────────────┘   │
      │          │           │            │             │
      │          ▼           ▼            ▼             │
      │   Preprocessing  Summarize     Judge            │
      │   (VLM+Sync)    (Gemini)    (Quality)           │
      │          │           │            │             │
      │          │           │◀── FAIL ───┘             │
      └─────────────────────────────────────────────────┘
                    │
                    ▼
          [final_summary_*.md]
```

---

## 🔧 CLI 옵션

### run_video_pipeline.py (End-to-End)

| 옵션                         | 기본값  | 설명                       |
| ---------------------------- | ------- | -------------------------- |
| `--video`                    | (필수)  | 입력 비디오 경로           |
| `--stt-backend`              | `clova` | STT 백엔드 (clova/whisper) |
| `--capture-threshold`        | `3.0`   | 장면 전환 감지 임계값      |
| `--capture-dedupe-threshold` | `3.0`   | 중복 제거 임계값           |
| `--vlm-batch-size`           | `1`     | VLM 배치 크기              |
| `--vlm-concurrency`          | `4`     | VLM 동시 요청 수           |
| `--parallel`                 | `True`  | STT+Capture 병렬 실행      |

### pre_adk_pipeline.py (Pre-ADK)

| 옵션            | 기본값  | 설명                       |
| --------------- | ------- | -------------------------- |
| `--video`       | (필수)  | 입력 비디오 경로           |
| `--stt-backend` | `clova` | STT 백엔드 (clova/whisper) |
| `--parallel`    | `True`  | STT+Capture 병렬 실행      |

---

## 📖 문서

| 문서                                                     | 설명                                    |
| -------------------------------------------------------- | --------------------------------------- |
| [AGENTS.md](./AGENTS.md)                                 | 코딩 에이전트 가이드라인, 코드 스타일   |
| [docs/DEVELOPER_GUIDE.md](./docs/DEVELOPER_GUIDE.md)     | 상세 개발 가이드, ADK 구조, 확장 포인트 |
| [docs/PRD.md](./docs/PRD.md)                             | 제품 요구사항 문서                      |
| [docs/PROJECT_DIRECTION.md](./docs/PROJECT_DIRECTION.md) | 프로젝트 방향성, 최적화 계획            |

---

## 📊 성능 지표

- **처리 속도**: 6분 영상 기준 약 3분 (End-to-End, 병렬 처리 시)
- **슬라이드 감지 정확도**: 약 95% (HybridSlideExtractor)
- **마우스/노이즈 제거율**: 약 95% (Temporal Median + 2.5초 안정화)

---

## 🤝 기여

- 코드 스타일: `ruff format`, `ruff check`, `mypy --strict`
- 커밋 메시지: 한글 작성, `type(scope): 제목` 형식
- PR: 한글 작성, 하나의 목적당 하나의 PR
