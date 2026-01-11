# Re:View 개발자 가이드

이 문서는 Re:View 프로젝트의 아키텍처, 디렉터리 구조, 상세 파이프라인 흐름 및 개발 방법을 설명합니다.

## 📚 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [상세 파이프라인 흐름](#3-상세-파이프라인-흐름)
4. [ADK 멀티에이전트 구조](#4-adk-멀티에이전트-구조)
5. [CLI 옵션 및 실행 방법](#5-cli-옵션-및-실행-방법)
6. [개발 영역별 가이드](#6-개발-영역별-가이드)
7. [기여 가이드](#7-기여-가이드)

---

## 1. 전체 아키텍처

### 1.1 데이터 흐름도

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
      │   (VLM+Sync)    (Gemini-2.5)(Gemini-2.5)        │
      │          │           │            │             │
      │          │           │◀── FAIL ───┘             │
      └─────────────────────────────────────────────────┘
                    │
                    ▼
          [final_summary_*.md]
```

### 1.2 핵심 컴포넌트

| 컴포넌트      | 역할                                                | 주요 파일                                               |
| ------------- | --------------------------------------------------- | ------------------------------------------------------- |
| **Pre-ADK**   | 오디오 추출, STT, 슬라이드 캡처 (병렬 실행)         | `src/pre_adk_pipeline.py`, `src/capture/`               |
| **ADK Agent** | 파이프라인 오케스트레이션, 에러 핸들링, 재실행 로직 | `src/adk_pipeline/agent.py`                             |
| **Fusion**    | STT/VLM 시간 동기화, 마크다운 요약 생성             | `src/fusion/sync_engine.py`, `src/fusion/summarizer.py` |
| **Judge**     | 요약 품질(환각, 누락) 평가 및 피드백                | `src/judge/judge.py`                                    |

---

## 2. 디렉터리 구조

```
Re:View/
├── data/
│   ├── inputs/                     # 입력 비디오 (.mp4)
│   └── outputs/                    # 출력 (비디오별 폴더)
│       └── {video_name}/
│           ├── stt.json            # STT 결과 (Whisper/Clova)
│           ├── manifest.json       # 캡처 메타데이터
│           ├── captures/           # 캡처 이미지 파일
│           ├── vlm.json            # VLM 분석 결과
│           ├── config.yaml         # 파이프라인 설정
│           └── fusion/
│               ├── segments_units.jsonl      # 동기화된 단위 데이터
│               ├── segment_summaries.jsonl   # 생성된 구간 요약
│               ├── judge.json                # 품질 평가 결과
│               └── outputs/
│                   └── final_summary_*.md    # 최종 결과물
├── src/
│   ├── run_video_pipeline.py       # End-to-End CLI (벤치마크 용)
│   ├── pre_adk_pipeline.py         # Pre-ADK CLI (STT+Capture)
│   │
│   ├── adk_pipeline/               # Google ADK 기반 에이전트
│   │   ├── agent.py                # Agent 정의 (Root, Preprocessing, Summarize, Judge)
│   │   ├── tools/                  # 에이전트 도구 (Tools)
│   │   └── store.py                # 데이터 접근 추상화 (VideoStore)
│   │
│   ├── audio/                      # 오디오 처리 및 STT
│   │   ├── stt_router.py           # STT 백엔드 라우터
│   │   └── extract_audio.py        # fps, sample rate 조정
│   │
│   ├── capture/                    # 슬라이드 캡처
│   │   └── tools/hybrid_extractor.py # dHash+ORB+RANSAC 엔진
│   │
│   ├── vlm/                        # Vision Language Model
│   │   └── vlm_engine.py           # OpenRouter(Qwen) 클라이언트
│   │
│   ├── fusion/                     # 데이터 병합 및 요약
│   │   ├── sync_engine.py          # Time-based Aligning
│   │   └── summarizer.py           # Gemini 요약 엔진
│   │
│   ├── judge/                      # 품질 평가
│   │   └── judge.py                # LLM Judge 구현체
│   │
│   └── common/                     # 공통 스키마 및 유틸
└── docs/                           # 문서화 리소스
```

---

## 3. 상세 파이프라인 흐름

### 3.1 1단계: Pre-ADK (데이터 확보)

- **입력**: MP4 비디오 파일
- **오디오 추출**: ffmpeg를 사용하여 16kHz mono WAV 변환
- **STT**: Clova Speech API 또는 Whisper(Local/API)를 사용하여 텍스트 변환
- **캡처**: `HybridSlideExtractor` 실행
  1. **dHash**: 인접 프레임 간 해시 차이로 장면 전환 감지
  2. **Stabilization**: 전환 후 2.5초 대기하여 깨끗한 화면 확보
  3. **Deduplication**: ORB 특징점 매칭 + RANSAC으로 중복 제거

### 3.2 2단계: ADK (지능형 처리)

- **Preprocessing Agent**:
  - `run_vlm`: 추출된 슬라이드 이미지를 VLM(Qwen)에 전송하여 텍스트/수식 추출
  - `run_sync`: STT 타임스탬프와 캡처 타임스탬프를 기준으로 세그먼트 매핑
- **Summarize Agent**:
  - `run_summarizer`: Gemini 3 Flash를 사용하여 세그먼트별 독립형 노트 생성
  - 프롬프트 전략: "초학자 튜터" 페르소나, 시각적 지시어 배제
- **Judge Agent**:
  - `evaluate_summary`: 생성된 요약의 **Groundedness(근거 기반)**, **Completeness(완전성)** 평가
  - 점수 미달 시 `FAIL` 리턴 → Summarize Agent가 재작성 수행

---

## 4. ADK 멀티에이전트 구조

### 4.1 에이전트 정의 (`src/adk_pipeline/agent.py`)

- **Root Agent**: 파이프라인의 진입점. 하위 에이전트로 작업을 위임하고 상태를 관리합니다.
- **Preprocessing Agent**: 데이터 로드, VLM 분석, Sync 작업을 수행합니다.
- **Summarize Agent**: 요약 생성, 마크다운 렌더링을 수행합니다. 실패 시 재시도 로직을 포함합니다.
- **Judge Agent**: 최종 산출물을 평가합니다.

### 4.2 상태 관리 (State)

ADK의 `ToolContext`를 통해 세션 상태를 공유합니다.

- `video_name`: 현재 처리 중인 비디오 식별자
- `current_rerun`: 재시도 횟수 카운터
- `judge_feedback`: Judge가 남긴 피드백 (재생성 시 반영)

### 4.3 도구 (Tools) 구현 (`src/adk_pipeline/tools/`)

도구는 에이전트가 실행하는 실제 함수들입니다.

- **Bridge 패턴**: `tools/*.py`는 ADK 인터페이스만 맞추고, 실제 로직은 `tools/internal/` 또는 `src/fusion/` 등의 Core 모듈을 호출합니다.

---

## 5. CLI 옵션 및 실행 방법

### 5.1 End-to-End 실행 (`run_video_pipeline.py`)

가장 권장되는 실행 방식입니다. 벤치마크 리포트까지 자동으로 생성합니다.

| 옵션                         | 기본값  | 설명                              |
| ---------------------------- | ------- | --------------------------------- |
| `--video`                    | (필수)  | 입력 비디오 경로                  |
| `--stt-backend`              | `clova` | STT 백엔드 (clova/whisper)        |
| `--capture-threshold`        | `3.0`   | 장면 전환 감지 임계값 (dHash)     |
| `--capture-dedupe-threshold` | `3.0`   | 중복 제거 임계값 (ORB)            |
| `--vlm-batch-size`           | `1`     | VLM 배치 크기                     |
| `--vlm-concurrency`          | `4`     | VLM 동시 요청 수 (AsyncIO)        |
| `--parallel`                 | `True`  | STT와 캡처를 병렬로 실행할지 여부 |

### 5.2 Pre-ADK 실행 (`pre_adk_pipeline.py`)

STT와 캡처만 수행하고 싶을 때 사용합니다.

```bash
python src/pre_adk_pipeline.py --video "data/inputs/sample.mp4" --parallel
```

### 5.3 ADK Web UI 실행

Pre-ADK가 완료된 상태에서, 에이전트와 대화하며 파이프라인을 진행합니다.

```bash
adk web src
```

---

## 6. 개발 영역별 가이드

### 6.1 Judge 로직 수정 (`src/judge/judge.py`)

현재 Judge는 Gemini를 사용하여 다음 항목을 평가합니다:

- **Groundedness**: 원본(STT/VLM)에 없는 내용을 지어냈는가?
- **Note Quality**: 강의 노트로서의 가독성과 구조가 적절한가?
- **Spec Compliance**: JSON 포맷 및 금지어(지시대명사 등) 규칙 준수 여부

평가 프롬프트를 수정하려면 `_build_prompt` 함수를, 점수 산정 로직을 수정하려면 `_compute_final_score` 함수를 참고하세요.

### 6.2 캡처 알고리즘 튜닝 (`src/capture/tools/hybrid_extractor.py`)

- **민감도 조절**: `sensitivity_diff` 값을 낮추면 더 작은 변화도 감지합니다. (기본 3.0)
- **중복 제거 강도**: `sensitivity_sim` 값을 높이면 더 엄격하게 중복을 제거합니다. (기본 0.8)

### 6.3 DB 연동 (Future Work)

현재 `src/adk_pipeline/store.py`의 `VideoStore` 클래스는 파일 시스템 기반입니다. 이를 DB로 마이그레이션하려면 해당 클래스의 메서드를 DB 쿼리로 교체하면 됩니다. (비즈니스 로직 수정 불필요)

---

## 7. 기여 가이드

### 7.1 코드 스타일

이 프로젝트는 `ruff`와 `mypy`를 사용합니다.

```bash
# 포맷팅
ruff format .

# 린트 검사
ruff check .

# 타입 검사
mypy src
```

### 7.2 커밋 메시지 컨벤션

- `type(scope): subject` 형식을 따릅니다.
- 예: `feat(capture): ORB 알고리즘 최적화`, `fix(judge): 점수 파싱 에러 수정`

### 7.3 PR 프로세스

1. Issue 생성 및 논의
2. 브랜치 생성 (`feat/기능명`, `fix/버그명`)
3. 작업 및 테스트
4. PR 생성 (상세한 설명 포함)
5. 리뷰 및 병합
