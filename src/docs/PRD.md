# Re:View PRD v2.0: 강의/발표 영상 자동 요약 파이프라인 (Google ADK Orchestration)

> ⚠️ **문서 상태**: v1.0(2025-12-30) 초기 요구사항을 2026-01-11 현재 구현에 맞게 업데이트함.

작성일: 2025-12-30 (Asia/Seoul) → 업데이트: 2026-01-11  
대상: Boostcamp CV-02 팀 내부 개발/운영 기준 문서  
입력: 로컬 영상 파일  
출력: 구간별 요약 + 최종 요약 노트(Markdown), 중간 산출물(JSON/JSONL), 실행 로그/메타데이터

---

## 1. 문제 정의 및 목표

### 1.1 문제

- 영상에는 음성(설명) + 시각(슬라이드/데모/화이트보드/코드/그래프) 정보가 함께 존재한다.
- STT만으로는 시각 근거가 누락되고, 프레임 기반 추출만으로는 설명 맥락이 누락된다.
- 긴 영상 처리 시 비용/시간이 커 재시도, 재개(resume), 중간 산출물 저장이 필수다.

### 1.2 목표

- 영상에서 "구간(segment)" 단위로 STT(한국어)와 VLM(프레임 요약)을 타임라인으로 동기화해, 근거 기반 구간 요약과 전체 요약 노트를 생성한다.
- 단계별 산출물과 실행 상태를 표준화된 포맷으로 저장해 부분 재실행이 가능하게 한다.
- LLM Judge를 통해 환각/비약을 탐지하고, 해당 구간만 재생성하는 검증 루프를 제공한다.

### 1.3 비목표

- 실시간(라이브) 처리
- 완전한 UI(웹 편집기) 제공
- 그래프/도표 수치의 완전 추출(초기에는 요약 중심)

---

## 2. 사용자/이해관계자 및 성공 기준

### 2.1 주요 사용자

- 수강생/학습자: 강의 내용을 빠르게 복습
- 팀/조직: 세미나·회의 녹화 요약, 액션아이템 정리
- 제작자: 노트 자동 생성 후 배포

### 2.2 성공 기준(정량/정성)

| 지표      | 목표                                                 | 현재 상태               |
| --------- | ---------------------------------------------------- | ----------------------- |
| 커버리지  | 주요 슬라이드 변화 지점이 요약에 반영                | ✅ 달성                 |
| 정합성    | 구간 요약이 해당 구간 STT/VLM 근거에서 벗어나지 않음 | ✅ Judge 평가 기준 포함 |
| 재현성    | 실패 시 전체 재실행 없이 실패 단계부터 재개 가능     | ✅ 단계별 산출물 저장   |
| 비용 통제 | 캡처/LLM 호출량이 설정으로 제어 가능                 | ✅ config.yaml          |

---

## 3. 시스템 개요(End-to-End)

### 3.1 파이프라인 단계

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User (mp4 upload)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Preprocess Pipeline (CLI)                            │
│            python src/run_preprocess_pipeline.py --video "xxx.mp4"          │
│                       또는                                                   │
│            python src/run_video_pipeline.py --video "xxx.mp4"               │
│                                                                             │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │   STT (Clova)   │   Parallel   │     Capture     │                      │
│   │   -> stt.json   │   Execution  │ -> manifest.json│                      │
│   │                 │              │ -> captures/*.jpg│                     │
│   └─────────────────┘              └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    data/outputs/{video_name}/                               │
│   Preprocess Outputs: stt.json, manifest.json, captures/                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ADK Pipeline (Interactive)                            │
│                       adk web src/adk_pipeline                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Root Agent (screentime_pipeline)                 │   │
│   │                                                                     │   │
│   │   Tools: list_available_videos, set_pipeline_config, get_status     │   │
│   │   Sub-Agents: preprocessing, summarize, judge                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│          │                         │                         │              │
│          ▼                         ▼                         ▼              │
│   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│   │Preprocessing│  ──────▶  │  Summarize  │  ──────▶  │    Judge    │       │
│   │   Agent     │           │   Agent     │           │   Agent     │       │
│   │             │           │             │           │             │       │
│   │ VLM + Sync  │           │ Sum + Render│           │ Quality Eval│       │
│   └─────────────┘           └─────────────┘           └─────────────┘       │
│                                    │                         │              │
│                                    │◀── FAIL + can_rerun ────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    data/outputs/{video_name}/fusion/                        │
│   ADK Outputs: segment_summaries.jsonl, final_summary_*.md, judge.json      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 아티팩트 저장

- `data/outputs/{video_name}/` 하위에 단계별 디렉터리(멱등성 확보)
- 모든 단계는 "입력/출력 계약"을 가진다.

---

## 4. 오케스트레이션 설계(Google ADK)

### 4.1 ADK 사용 방식(현재 구현)

- 워크플로는 예측 가능한 순차 파이프라인이므로 기본은 **Sequential workflow**로 구성.
- ADK는 워크플로 에이전트(Sequential/Parallel/Loop)로 파이프라인을 정의할 수 있다.
- 단계별 실패 처리, 재시도, 체크포인트는 **Agent Transfer** 및 **도구 내부 로직**에서 구현.

### 4.2 상태 모델(State Machine)

각 Step은 아래 상태를 가진다.

- `PENDING` → `RUNNING` → `SUCCEEDED`
- `RUNNING` → `FAILED` → (`RETRYING` → `RUNNING` …) → `FAILED`(terminal)
- `SKIPPED` (조건부 스킵: 예, 오디오 없음)

### 4.3 재시도 정책

| 항목          | 설정                                        |
| ------------- | ------------------------------------------- |
| 대상          | 외부 API 호출(STT/VLM/LLM), 일시적 I/O 오류 |
| 최대 재시도   | 기본 3회 (설정 가능)                        |
| 백오프        | 2^k 초 (상한 설정)                          |
| 비재시도 오류 | 입력 포맷 오류, 인증키 오류 등 "영구 오류"  |

### 4.4 실행 메타데이터 (`pipeline_run.json`, `benchmark_report.md`)

```json
{
  "started_at": "2026-01-10T15:30:00Z",
  "ended_at": "2026-01-10T15:33:00Z",
  "input_video_path": "data/inputs/lecture.mp4",
  "video_info": {
    "duration_sec": 360,
    "resolution": "1920x1080",
    "fps": 30
  },
  "stages": {
    "stt": { "elapsed_sec": 45.2, "status": "success" },
    "capture": { "elapsed_sec": 12.3, "status": "success", "count": 24 },
    "vlm": { "elapsed_sec": 89.5, "status": "success" },
    "sync": { "elapsed_sec": 1.2, "status": "success" },
    "summarize": { "elapsed_sec": 67.8, "status": "success" },
    "judge": { "elapsed_sec": 23.4, "status": "success", "score": 8.5 }
  },
  "total_elapsed_sec": 180.5
}
```

---

## 5. 단계별 상세 요구사항

### 5.1 영상 확보/전처리

**입력**: `local_video_path`

**기능**:

- 파일 검증: 존재/확장자/코덱 정보 추출
- 오디오 추출: STT 입력용 오디오 생성 (ffmpeg: 16kHz, mono WAV)

**출력 (아티팩트)**:

- `{video_name}.wav` (오디오)
- 비디오 메타데이터 (duration, resolution, fps)

**에러/예외**:

- 오디오 트랙 없음: STT step SKIPPED, 이후 sync에서 transcript_text 비어있을 수 있음

---

### 5.2 변화 감지 + 캡처 (HybridSlideExtractor)

**입력**: video (원본)

**엔진**: `src/capture/tools/hybrid_extractor.py`

**알고리즘 (Single-Pass)**:

```
입력 비디오
     │
     ▼
┌─────────────────────────────────────┐
│  1. 프레임 읽기 + 스킵 최적화            │
│     - IDLE 상태에서 빠른 스킵       │
│     - 변화 감지 중에는 상세 분석    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  2. dHash 장면 전환 감지            │
│     - sensitivity_diff (기본 3.0)   │
│     - 픽셀 차이 기반                │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  3. 2.5초 스마트 버퍼링             │
│     - SAFE_DURATION 동안 대기       │
│     - Temporal Median 노이즈 제거   │
│     - 마우스/트랜지션 안정화        │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  4. ORB + RANSAC 중복 제거          │
│     - sensitivity_sim (기본 0.8)    │
│     - 기하학적 일관성 검사          │
└─────────────────────────────────────┘
     │
     ▼
출력: captures/*.jpg + manifest.json
```

**핵심 파라미터**:

| 파라미터           | 기본값 | 설명                                        |
| ------------------ | ------ | ------------------------------------------- |
| `sensitivity_diff` | 3.0    | dHash 픽셀 차이 민감도 (낮을수록 민감)      |
| `sensitivity_sim`  | 0.8    | ORB 구조 유사도 (높을수록 엄격한 중복 제거) |
| `SAFE_DURATION`    | 2.5초  | 장면 전환 후 안정화 대기 시간               |
| `min_interval`     | 1.0초  | 연속 캡처 최소 간격                         |

**출력**:

- `captures/*.jpg` (또는 .png)
- `manifest.json`:
  ```json
  [
    {
      "timestamp_ms": 1000,
      "frame_index": 30,
      "file_name": "capture_0001.jpg",
      "diff_score": 5.2
    }
  ]
  ```

---

### 5.3 STT (NAVER Clova Speech / Whisper)

**입력**: 영상 또는 오디오 파일

**STT 제공자**:

- **Clova Speech** (기본): HTTP Post 요청 (Sync Mode)
- **Whisper** (대체): 로컬 모델 또는 API

**구현**: `src/audio/stt_router.py`

**요구사항**:

- 언어: 한국어 고정
- 타임스탬프 포함 세그먼트 단위 반환: `start_ms`, `end_ms`, `text`
- `confidence` 포함 가능

**출력 (`stt.json`)**:

```json
{
  "segments": [
    { "start_ms": 0, "end_ms": 5000, "text": "안녕하세요", "confidence": 0.95 }
  ]
}
```

---

### 5.4 VLM (Visual Language Model)

**입력**: `captures/manifest.json` + 이미지 파일

**엔진**: OpenRouter API (`qwen/qwen3-vl-32b-instruct`)

**구현**: `src/vlm/vlm_engine.py`

**요구사항**:

- VLM 호출 단위: 이미지 1장 = 1회 (배치 지원)
- 출력 필드: `timestamp_ms`, `extracted_text`
- 출력 형식: Markdown (테이블, LaTeX 수식 지원)

**프롬프트**:

- System: "Output only Markdown. Use Markdown tables... Use LaTeX for equations..."
- User: "이미지 내 텍스트/수식을 마크다운으로 추출하라..."

**출력 (`vlm.json`)**:

```json
{
  "items": [
    {
      "timestamp_ms": 1000,
      "extracted_text": "## Diffusion Model\n- Forward process: q(x_t|x_{t-1})\n- Reverse process: p_θ(x_{t-1}|x_t)"
    }
  ]
}
```

---

### 5.5 정렬/병합 (Sync Engine)

**입력**:

- `stt.json` (segments)
- `vlm.json` (items)

**엔진**: `src/fusion/sync_engine.py`

**처리 규칙**:

- 캡처 변화 시점 기반 세그먼트화 (슬라이드 단위)
- 최소/최대 세그먼트 길이 제한 (`min_segment_sec`, `max_segment_sec`)
- 너무 긴 구간은 재귀적으로 분할 (`_split_segment_recursive`)

**출력**:

- `segments.jsonl`: 기본 세그먼트
- `segments_units.jsonl`: 상세 유닛 정보 포함

**JSONL 스키마 (`segments_units.jsonl`)**:

```json
{
  "segment_id": 1,
  "start_ms": 0,
  "end_ms": 60000,
  "transcript_text": "...",
  "transcript_units": [
    { "unit_id": "t1", "text": "...", "start_ms": 0, "end_ms": 3000 }
  ],
  "visual_text": "...",
  "visual_units": [{ "unit_id": "v1", "text": "...", "timestamp_ms": 1000 }]
}
```

---

### 5.6 LLM 요약 (Gemini Summarizer)

**입력**: `segments_units.jsonl`

**엔진**: `src/fusion/summarizer.py`

**프롬프트 버전**: `sum_v1.5`

**요약 철학**: "요약가"가 아니라 "초학자 튜터(독립형 강의 노트 작성자)"

- 사용자가 영상/슬라이드를 보지 않아도 이해할 수 있는 "독립형 노트" 생성
- 금지 표현: "슬라이드", "화면", "그림을 보면", "위/아래", "여기" 등

**출력 구조** (`segment_summaries.jsonl`):

```json
{
  "segment_id": 1,
  "summary": {
    "bullets": [
      {
        "bullet_id": "1-1",
        "claim": "ELBO는 log p(x)의 하한으로, 직접 계산 불가능한 marginal likelihood를 간접 최적화할 수 있게 한다.",
        "source_type": "inferred",
        "evidence_refs": ["t1", "v1"],
        "confidence": "high",
        "notes": ""
      }
    ],
    "definitions": [
      {
        "term": "ELBO (Evidence Lower Bound)",
        "definition": "log p(x)의 하한(lower bound). 직접 계산 불가능한 marginal likelihood를 최적화하기 위한 대리 목표.",
        "source_type": "direct",
        "evidence_refs": ["v1"],
        "confidence": "high",
        "notes": ""
      }
    ],
    "explanations": [...],
    "open_questions": [...]
  }
}
```

**최종 요약 포맷** (`src/fusion/renderer.py`):

- `final_summary_timeline.md`: 시간 순 타임라인 노트
- `final_summary_tldr_timeline.md`: TL;DR + 시간 순 하이브리드

---

### 5.7 검증 루프 (LLM Judge)

**입력**:

- `segments_units.jsonl` (원본 데이터)
- `segment_summaries.jsonl` (생성된 요약)

**엔진**: `src/judge/judge.py`

**프롬프트 버전**: `judge_v4`

**평가 기준** (0-10점 척도):

| 항목                | 가중치 | 설명                                                             |
| ------------------- | ------ | ---------------------------------------------------------------- |
| **Groundedness**    | 45%    | 근거 없는 내용(Hallucination) 여부, 증거(evidence_refs)의 정확성 |
| **Note Quality**    | 35%    | 노트 자체의 완성도 (강의 영상 없이도 이해 가능한지, 명확한 흐름) |
| **Spec Compliance** | 20%    | JSON 형식, 금지어(지시 대명사 등) 사용 여부, 필수 필드 준수      |

**동작**:

1. 세그먼트별 병렬 평가 (batch_size, workers 설정 가능)
2. 최종 평균 점수 계산
3. 기준 미달 시 `FAIL` 판정 (기본 threshold: 7.0)
4. `FAIL + can_rerun=True`: Summarize Agent부터 재실행

**출력 (`judge.json`)**:

```json
{
  "pass": true,
  "final_score": 8.5,
  "threshold": 7.0,
  "scores": {
    "groundedness": 8.2,
    "note_quality": 8.8,
    "spec_compliance": 9.0
  },
  "segment_reports": [...]
}
```

---

## 6. 인터페이스/모듈 경계

### 6.1 모듈 구조

```
src/
├── run_video_pipeline.py       # End-to-End CLI (벤치마크 포함)
├── run_preprocess_pipeline.py  # Preprocess CLI
│
├── adk_pipeline/               # ADK 멀티에이전트
│   ├── agent.py                # Agent 정의 (Root + Sub-agents)
│   ├── store.py                # VideoStore (파일시스템 추상화)
│   ├── paths.py                # 경로 유틸리티
│   └── tools/
│       ├── root_tools.py
│       ├── preprocessing_tools.py
│       ├── summarize_tools.py
│       ├── judge_tools.py
│       └── internal/           # (9개 내부 모듈)
│
├── audio/                      # STT (4개 파일)
├── capture/                    # Capture (tools/ 포함)
├── vlm/                        # VLM (3개 파일)
├── fusion/                     # Fusion (7개 파일)
├── judge/                      # Judge (1개 파일)
├── common/                     # 공통 스키마
└── utils/                      # 유틸리티
```

### 6.2 모듈 간 계약

모듈 간 계약은 **파일 기반**:

- 이유: 장애/재시도/재개가 단순해지고, 팀이 서로의 런타임 환경 차이를 흡수하기 쉬움
- 향후: DB 도입 시 `VideoStore` 클래스를 통해 추상화 (`src/adk_pipeline/store.py`)

---

## 7. 산출물 구조

```
data/outputs/{video_name}/
├── stt.json                    # STT 결과 (schema v1)
├── {video_name}.wav            # 추출된 오디오
├── manifest.json               # 캡처 메타데이터
├── captures/                   # 캡처 이미지
│   └── capture_XXXX.jpg
├── vlm_raw.json                # VLM 원시 결과
├── vlm.json                    # VLM 정제 결과
├── config.yaml                 # Fusion 설정
├── pipeline_run.json           # 실행 메타데이터
├── benchmark_report.md         # 벤치마크 리포트
└── fusion/
    ├── segments.jsonl
    ├── segments_units.jsonl
    ├── sync.json
    ├── trace_map.json
    ├── segment_summaries.jsonl
    ├── segment_summaries.md
    ├── judge.json
    ├── judge_segments.json     # 세그먼트별 평가
    ├── attempts/               # 재실행 아카이브
    │   └── attempt_01/
    └── outputs/
        ├── final_summary_timeline.md
        └── final_summary_tldr_timeline.md
```

---

## 8. 향후 개선 과제

자세한 내용은 `docs/PROJECT_DIRECTION.md` 및 `TODO.md` 참조.

### 8.1 성능 최적화 (우선순위 높음)

| 항목       | 현재 상태                                          | 개선 방향                      |
| ---------- | -------------------------------------------------- | ------------------------------ |
| VLM 호출   | Sequential/Batch                                   | AsyncIO 병렬화 (`AsyncOpenAI`) |
| Summarizer | Single Call (모든 세그먼트 한 번에)                | Map-Reduce (세그먼트별 병렬)   |
| Judge      | 세그먼트별 배치                                    | 더 세분화된 병렬 처리          |
| 입력 토큰  | 중복 포함 (`transcript_text` + `transcript_units`) | 경량화                         |

### 8.2 UX 개선 (중간 우선순위)

- **스트리밍 파이프라인**: 세그먼트 2~3개 단위로 점진적 표시
- **Fast Summary**: STT 결과만으로 빠른 초안 생성
- **Chatbot Mode**: 특정 구간 심층 분석 요청

### 8.3 인프라 (낮은 우선순위)

- **DB 도입**: 로컬 파일시스템 → Database (`VideoStore` 추상화 활용)
- **Video RAG 피벗**: Latency 한계 시 On-demand 검색 방식 전환 (`colqwen3`)
