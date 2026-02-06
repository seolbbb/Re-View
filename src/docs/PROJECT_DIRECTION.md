# Project Direction & Meeting Notes

이 파일은 프로젝트의 현재 방향성과 회의 내용을 정리하는 곳입니다.
Git에 올라가지 않으며, 에이전트(Antigravity)가 작업을 진행할 때 참고합니다.

## 1. 프로젝트 개요 (Context)

이 프로젝트는 **"강의 영상을 입력받아, 화면 속 슬라이드 내용(텍스트, 수식, 코드)을 캡처하고 설명하여 최종 요약 노트를 생성하는 파이프라인"**입니다.
현재 MVP 단계에서 기능 구현은 완료되었으나, 서비스 운영 관점에서 **실행 속도(Latency)**와 **확장성(Scalability)**에 치명적인 한계가 발견되어 이에 대한 근본적인 최적화가 필요합니다.

## 2. 현재 파이프라인 상세 분석 (AS-IS Architecture)

현재 시스템은 Python 기반의 **Sequential & Blocking (직렬 대기)** 구조로 설계되어 있습니다.

### Phase 1: 비디오 캡처 & 전처리 (Video Extraction)

- **Component**: `src/capture` (Video) & `src/audio` (Audio)
- **Engine**:
  - **Video**: `HybridSlideExtractor` (`src/capture/tools/hybrid_extractor.py`, `src/capture/process_content.py`)
  - **Audio**: `STTRouter` wrapping `ClovaSpeechClient` or `WhisperSTTClient` (`src/audio/stt_router.py`)
- **Process Details**:
  1. **Audio Extraction**: `ffmpeg`를 사용해 영상에서 오디오 추출 (Mono, 16kHz).
  2. **STT (Speech-to-Text)**:
     - `ClovaSpeechClient` (Default): HTTP Post 요청 (Sync Mode). 타임스탬프, 신뢰도(Confidence), 단어 정렬(Word Alignment) 데이터 수신.
     - `WhisperSTTClient` (Fallback): 로컬/API Whisper 모델 지원.
  3. **Hybrid Slide Extraction** (Single-Pass):
     - **Scene Detection**: dHash(Difference Hash) 기반. `SENSITIVITY_DIFF=3.0`.
     - **Stabilization**: 장면 전환 후 약 2.5초(`SAFE_DURATION`) 대기하여 화면 안정화(마우스/트랜지션 노이즈 제거).
     - **Deduplication**: `ORB` Feature Matching + `RANSAC`. `SENSITIVITY_SIM=0.8` (유사도 0.8 이상 시 중복 판정).
     - **IO Write**: 최종 선별된 이미지를 `{output_root}/{video}/captures/`에 저장하고 `manifest.json` 생성.
- **Performance Constraints**:
  - **Blocking I/O**: STT 요청과 비디오 프레임 순회(Capture)가 완료될 때까지 다음 단계가 시작되지 않음.
  - **Sequential Processing**: 긴 영상일수록 Phase 1 수행 시간이 선형적으로 증가.

### Phase 2: 시각 정보 추출 (VLM Processing)

- **Engine**: **Qwen (DashScope) API** (`qwen3-vl-32b-instruct`)
- **Implementation**: `src/vlm/vlm_engine.py`
- **Process**:
  1. Phase 1에서 생성된 이미지 경로 리스트를 받아옴.
  2. **Image Loading**: 로컬 디스크에서 이미지를 읽어 `Base64`로 인코딩 (Memory Overhead 발생 가능).
  3. **Request Mode**:
     - **Sequential (Batch Size=1)**: 이미지를 하나씩 순회하며 API 호출. `client.chat.completions.create`가 동기(blocking) 방식으로 호출됨.
     - **Batch (Batch Size>1)**: 여러 이미지를 하나의 프롬프트에 묶어서 전송. 하지만 내부적으로 Batch 단위로 순회(`for` loop)하므로 여전히 직렬 처리.
  4. **Prompt**:
     - System: "Output only Markdown. Use Markdown tables... Use LaTeX for equations..."
     - User: "이미지 내 텍스트/수식을 마크다운으로 추출하라..."
- **Performance Constraints**:
  - **Synchronous Blocking**: API 응답이 올 때까지 메인 스레드가 멈춤. Network I/O 대기 시간이 전체 실행 시간의 대부분을 차지.
  - **No Parallelism**: 비동기 처리가 없어 100장 처리 시 (1장 시간 \* 100) 만큼 소요.

### Phase 3: 융합 및 요약 (Fusion & Summarization)

- **Engine**: **Google Gemini (or Vertex AI)**
- **Implementation**: `src/fusion` (`sync_engine.py`, `summarizer.py`, `renderer.py`)
- **Process**:
  1. **Sync Engine** (`sync_engine.py`):
     - STT와 VLM 결과를 로드하고, 타임스탬프 기준으로 정렬 및 병합(Segmentation).
     - 너무 긴 구간은 재귀적으로 분할 (`_split_segment_recursive`).
  2. **Summarizer** (`summarizer.py`):
     - **Batch Context Construction**: 모든 세그먼트 데이터를 하나의 거대한 JSONL 문자열로 변환하여 프롬프트에 포함.
     - **Single Call**: Gemini API에 **단 한 번의 요청**으로 모든 세그먼트의 요약을 생성.
     - **Validation**: JSON Schema & Retry (`_repair_prompt`) 로직 포함.
  3. **Rendering** (`renderer.py`):
     - 구조화된 요약 데이터를 Markdown으로 변환하고, 근거(evidence)를 각주로 표기.
- **Performance Constraints**:
  - **Token Explosion (Critical)**: 영상이 길어지면 입력 프롬프트가 수십만 토큰까지 증가. 비용 급증 및 Latency 악화.
  - **Blocking Architecture**: Phase 1, 2가 완료되어야 Phase 3 시작 가능.
  - **Single Point of Failure**: 한 번의 거대 요청 실패 시 전체 요약 실패 위험.

### Phase 4: 품질 평가 (Judge & Evaluation)

- **Engine**: **Google Gemini** (via `Judge Agent`)
- **Implementation**: `src/judge/judge.py`, `src/adk_pipeline/tools/judge_tools.py`
- **Role**: AI 기반의 요약 품질 검증 (Quality Assurance).
- **Process**:
  1. **Evaluation**: 생성된 요약(`segment_summaries.jsonl`)과 원본 데이터(`segments_units.jsonl`)를 비교 분석.
  2. **Scoring Criteria** (0-10점 척도):
     - **Groundedness (45%)**: 근거 없는 내용(Hallucination) 여부, 증거(Usage Reference)의 정확성.
     - **Note Quality (35%)**: 노트 자체의 완성도 (강의 영상 없이도 이해 가능한지, 명확한 흐름).
     - **Spec Compliance (20%)**: JSON 형식, 금지어(지시 대명사 등) 사용 여부, 필수 필드 준수.
     - **Multimodal Use** (Ref Only): 시각 자료 활용도 (최종 점수에는 미반영).
  3. **Feedback Loop**:
     - 최종 평균 점수가 기준(Default 7.0) 미만일 경우 `FAIL` 판정.
     - `PASS`: 파이프라인 종료.
     - `FAIL`: **Summarize Agent**에게 제어권을 넘겨 요약 재생성 요청 (`max_reruns` 횟수 내에서). (Self-healing Pipeline)

---

## 3. 회의 내용 및 주요 이슈 (Meeting Notes 2026-01-09)

오늘 회의에서 논의된 주요 내용과 제기된 문제들을 기능/목적 단위로 정리했습니다.

### A. 성능 최적화 (Optimization) - 모듈별 과업

#### 1. VLM (Visual Language Model)

- **AsyncIO (비동기 병렬 처리)**: 기존 배치 처리가 직렬 방식이라 효과가 미미했음. `AsyncOpenAI`를 도입하여 네트워크 대기 시간을 활용, 진정한 동시성(Concurrency)을 확보해야 함. (최우선 과제)
- **Batch Size Tuning**: 비동기 환경에서 최적의 Batch Size 재측정.

#### 2. Judge

- **Map-Reduce (병렬 처리)**: 현재 6분 영상 요약에 1분 소요됨. 전체를 한 번에 넣는 통짜 구조(`Single Call`)를, **Segment 별 병렬 요청** 구조로 변경하여 Latency 단축 필요.
- **Parallel Judge**: Judge 또한 세그먼트별로 쪼개서 비동기 검증을 수행하도록 구조 개선.

#### 3. Summarization

- **Input Reduction (입력 경량화)**: 현재 `segments_units.jsonl`에 전문(`transcript_text`)과 분할 유닛(`transcript_units`)이 중복 포함됨. 이를 정리하여 Summarizer 입력 토큰을 획기적으로 줄이는 실험 필요.

- **Map-Reduce (병렬 처리)**: 현재 6분 영상 요약에 1분 30초 소요됨. 전체를 한 번에 넣는 통짜 구조(`Single Call`)를, **Segment 별 병렬 요청** 구조로 변경하여 Latency 단축 필요.

### B. 사용자 경험 및 파이프라인 구조 (UX & Architecture)

1.  **스트리밍 & 부분 결과 제공 (Streaming Pipeline)**
    - **아이디어**: 전체 처리가 끝날 때까지 기다리지 않고, ADK 파이프라인을 세그먼트 2~3개 단위로 쪼개서 반복 실행.
    - **효과**: 앞부분 처리가 완료되는 즉시 사용자에게 보여주어 체감 대기 시간 단축 (Netflix 스타일). 사용자가 앞부분을 읽는 동안 뒷부분 백그라운드 처리.
2.  **Chatbot / Interactive Mode**
    - **아이디어**: 사용자가 특정 부분에 대해 "더 자세히 설명해줘"라고 요청하면, 그 시점에 해당 구간을 심층 분석(VLM/LLM)하여 답변.
3.  **Fast Summary (초안 생성)**: STT 결과만으로 먼저 빠르게 요약본을 생성하여, VLM/LLM 분석이 진행되는 동안 사용자에게 즉시 결과를 보여주는 UX 개선.

### C. 인프라 및 피벗 전략 (Infra & Pivot)

1.  **DB 도입 (Infrastructure)**
    - **현황**: 현재 로컬 파일 시스템(JSON/JSONL)에 의존.
    - **Action**: 메타데이터, 캡처 정보, 요약 결과를 체계적으로 관리하기 위해 Database 도입 필요.
2.  **기능 피벗 가능성 (Pivot Idea: Video RAG)**
    - **시나리오**: 만약 Latency 최적화가 한계에 봉착하여 실시간성이 확보되지 않는다면?
    - **대안**: "전체 요약" 기능을 포기하고, **Video RAG** (예: `colqwen3` 모델) 서비스로 피벗. On-demand 검색 방식 전환.

### D. 유지/보수 (Maintenance)

1.  **Video Capture**: 현재 슬라이드 중심 영상에 대해 성능이 우수하므로, 큰 구조 변경보다는 점진적 개선(Incremental Improvement) 방향으로 유지.
