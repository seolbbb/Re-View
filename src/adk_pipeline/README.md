# ADK 파이프라인

Google ADK 기반 비디오 분석 파이프라인입니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               User Access                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Video Upload (mp4)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         ┌──────────────────┐            ┌──────────────────┐
         │ Audio Extraction │            │  Screen Capture  │
         │   stt.json       │            │  manifest.json   │
         │ (Clova Speech)   │            │  captures/*.png  │
         └──────────────────┘            └──────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────────┐
           │  pre_adk_pipeline.py (CLI)                         │
           │  python src/pre_adk_pipeline.py --video "xxx.mp4"  │
           └────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DB (data/outputs/{video_name})                           │
│   Pre-ADK Outputs: stt.json, manifest.json, captures/                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌────────────────────────────────────────────────────┐
           │  adk web src/adk_pipeline                          │
           │  (Interactive Pipeline Execution)                  │
           └────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Root Agent (LlmAgent)                             │
│                                                                             │
│   Tools:                                                                    │
│   - list_available_videos: List Available Videos                            │
│   - set_pipeline_config: Select & Configure Video                           │
│   - get_pipeline_status: Get Status                                         │
│                                                                             │
│   Sub-Agents:                                                               │
│   - preprocessing_agent → summarize_agent → judge_agent                     │
└─────────────────────────────────────────────────────────────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Preprocessing  │      │    Summarize    │      │      Judge      │
│     Agent       │      │     Agent       │      │     Agent       │
│                 │      │                 │      │                 │
│  Tools:         │      │  Tools:         │      │  Tools:         │
│  - load_data    │      │  - run_         │      │  - evaluate_    │
│  - run_vlm      │      │    summarizer   │      │    summary      │
│  - run_sync     │      │  - render_md    │      │                 │
│                 │      │  - write_final  │      │                 │
│  Output:        │      │    _summary     │      │  Output:        │
│  segments_      │      │                 │      │  judge.json     │
│  units.jsonl    │      │  segment_       │      │  (PASS/FAIL)    │
│                 │      │  summaries.*    │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                    │
                                    │ FAIL + can_rerun?
                                    ▼
                         ┌─────────────────┐
                         │  summarize_agent│
                         │  (Rerun from)   │
                         └─────────────────┘
```

## 실행 방법

### Step 1: Pre-ADK 실행 (STT + Capture)

```bash
# 기본 실행
python src/pre_adk_pipeline.py --video "my_video.mp4"

# 절대 경로로 비디오 지정
python src/pre_adk_pipeline.py --video "C:/path/to/video.mp4"

# Whisper STT 사용
python src/pre_adk_pipeline.py --video "my_video.mp4" --stt-backend whisper
```

**산출물:**
```
data/outputs/{video_name}/
├── stt.json          # STT 결과
├── manifest.json     # 캡처 메타데이터
└── captures/         # 캡처 이미지들
```

### Step 2: ADK 파이프라인 실행 (대화형)

```bash
# 프로젝트 루트에서 실행
adk web src/adk_pipeline
```

브라우저에서 `http://localhost:8000` 접속

## ADK Web 대화 예시

```
User: 처리 가능한 비디오 목록을 보여줘

Agent: [list_available_videos 호출]
       다음 비디오들이 처리 가능합니다:
       1. test3_Diffusion (Pre-ADK 완료)
       2. lecture_01 (Pre-ADK 완료)

User: test3_Diffusion으로 파이프라인 실행해줘

Agent: [set_pipeline_config 호출]
       비디오 'test3_Diffusion' 설정 완료.

       [preprocessing_agent로 transfer]
       Preprocessing 완료: segments_units.jsonl 생성됨

       [summarize_agent로 transfer]
       Summarize 완료: segment_summaries.jsonl, final_summary_*.md 생성됨

       [judge_agent로 transfer]
       Judge 결과: PASS

       파이프라인이 완료되었습니다.
       최종 결과: data/outputs/test3_Diffusion/fusion/outputs/
```

## 환경 변수 설정

`.env` 파일에 API 키 설정:

```bash
GOOGLE_API_KEY=your_google_api_key       # Gemini (ADK, Summarize)
OPENROUTER_API_KEY=your_openrouter_api_key  # VLM (Qwen)
CLOVA_API_KEY=your_clova_api_key         # STT (Clova 백엔드 사용 시)
```

## 디렉터리 구조

### 입출력 구조

```
data/
├── inputs/                      # 입력 비디오
│   └── my_video.mp4
└── outputs/                     # 출력 (DB 대체)
    └── my_video/
        ├── stt.json             # STT 결과 (Pre-ADK)
        ├── my_video.wav         # 추출된 오디오 (Pre-ADK)
        ├── manifest.json        # 캡처 메타데이터 (Pre-ADK)
        ├── captures/            # 캡처 이미지들 (Pre-ADK)
        │   ├── frame_0001.png
        │   └── ...
        ├── vlm_raw.json         # VLM 원시 결과
        ├── vlm.json             # VLM 정제 결과
        ├── config.yaml          # Fusion 설정
        └── fusion/
            ├── segments.jsonl
            ├── segments_units.jsonl
            ├── segment_summaries.jsonl
            ├── segment_summaries.md
            ├── judge.json
            ├── attempts/        # 재실행 아카이브
            │   ├── attempt_01/
            │   └── ...
            └── outputs/
                ├── final_summary_A.md
                ├── final_summary_B.md
                └── final_summary_C.md
```

### 코드 구조

```
src/adk_pipeline/
├── __init__.py          # 패키지 초기화, sys.path 설정
├── agent.py             # ADK Agent 정의 (root + sub-agents)
├── store.py             # VideoStore - 산출물 경로 규약
├── paths.py             # 경로 유틸리티
└── tools/
    ├── __init__.py
    ├── root_tools.py          # Root Agent 도구
    ├── preprocessing_tools.py # Preprocessing Agent 도구
    ├── summarize_tools.py     # Summarize Agent 도구
    ├── judge_tools.py         # Judge Agent 도구
    └── internal/              # 내부 구현 모듈 (Agent에 직접 노출 안됨)
        ├── __init__.py
        ├── vlm_openrouter.py  # VLM 실행
        ├── sync_data.py       # Sync Engine
        ├── summarize.py       # Gemini 요약
        ├── render_md.py       # MD 렌더링
        ├── final_summary.py   # 최종 요약 생성
        ├── judge_gemini.py    # Judge stub
        ├── attempts.py        # 재실행 아카이브
        ├── fusion_config.py   # Fusion 설정 생성
        └── pre_db.py          # Pre-ADK 실행 (CLI 전용)

src/pre_adk_pipeline.py  # Pre-ADK CLI 엔트리포인트
```

## Agent 구조

### Root Agent
- **역할**: 전체 파이프라인 조율, 사용자와 대화
- **도구**: `list_available_videos`, `set_pipeline_config`, `get_pipeline_status`
- **Sub-Agents**: preprocessing_agent, summarize_agent, judge_agent

### Preprocessing Agent
- **역할**: VLM + Sync 실행
- **도구**: `load_data`, `run_vlm`, `run_sync`
- **출력**: `vlm.json`, `segments_units.jsonl`

### Summarize Agent
- **역할**: 세그먼트 요약 생성
- **도구**: `run_summarizer`, `render_md`, `write_final_summary`
- **출력**: `segment_summaries.jsonl`, `segment_summaries.md`, `final_summary_*.md`

### Judge Agent
- **역할**: 요약 품질 평가
- **도구**: `evaluate_summary`
- **출력**: `judge.json` (PASS/FAIL)
- **FAIL 시**: `can_rerun=True`이면 summarize_agent부터 재실행

## Pre-ADK CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--video` | (필수) | 입력 비디오 경로 |
| `--input-dir` | `data/inputs` | 입력 디렉터리 |
| `--output-base` | `data/outputs` | 출력 디렉터리 |
| `--stt-backend` | `clova` | STT 백엔드 (clova/whisper) |
| `--parallel` | `True` | STT+Capture 병렬 실행 |
| `--capture-threshold` | `3.0` | 장면 전환 감지 임계값 |
| `--capture-dedupe-threshold` | `3.0` | 중복 제거 임계값 |
| `--capture-min-interval` | `0.5` | 캡처 최소 간격(초) |

## 확장 포인트

### DB 연결

현재는 파일시스템(`data/outputs/`)을 DB 대체로 사용합니다.
실제 DB 연결 시 다음 파일들을 수정하세요:

- `src/adk_pipeline/store.py`: VideoStore 클래스
- `src/adk_pipeline/tools/*_tools.py`: DB read/write 로직

### Judge 고도화

현재 Judge는 stub (항상 PASS)입니다.
실제 평가 로직 구현 시:

- `src/adk_pipeline/tools/internal/judge_gemini.py`: `judge_stub_gemini()` 함수
- 평가 기준: 요약 품질, 정보 포함도, 일관성 등

## 트러블슈팅

### ADK Web에서 agent가 보이지 않는 경우

1. 프로젝트 루트에서 `adk web src/adk_pipeline` 실행
2. `agent.py` 파일에 `root_agent` 변수가 있는지 확인
3. `GOOGLE_API_KEY` 환경 변수 설정 확인

### Import 오류

```bash
# 프로젝트 루트에서 실행
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 또는 Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

### VLM/STT API 오류

- `.env` 파일의 API 키 확인
- 네트워크 연결 확인
- API 할당량 확인
