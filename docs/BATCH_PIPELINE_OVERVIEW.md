# ADK 배치 파이프라인 변경 요약

## 변경 목적

VLM 이후 단계를 배치로 분할하여 앞부분 요약을 먼저 제공하고, 이전 배치의 요약 맥락을 유지하면서 점진적으로 결과를 생성한다.

## 처리 흐름 변경

### 기존 흐름

```
Pre-ADK: video → stt.json, manifest.json, captures/
    ↓
VLM (전체 캡처) → vlm.json
    ↓
Sync (전체) → segments_units.jsonl
    ↓
Summarize (전체) → segment_summaries.jsonl
    ↓
Judge (전체) → PASS/FAIL
```

### 배치 흐름

```
Pre-ADK: video → stt.json, manifest.json, captures/
    ↓
load_data: stt.json, manifest.json 로드
    ↓
[배치 루프] - 시간 범위 기반
  ┌──────────────────────────────────────────────┐
  │ Batch i (start_ms ~ end_ms)                  │
  │   VLM (해당 캡처만) → batches/batch_i/vlm.json│
  │   Sync (해당 범위만) → segments_units.jsonl   │
  │   Summarize (+ 이전 context) → summaries.jsonl│
  │   Judge → judge.json                          │
  └──────────────────────────────────────────────┘
    ↓
Merge: 모든 배치 파일 병합
    ↓
Final: 통합 요약 생성
```

## 배치 상태(state)와 파일 구조

### 배치 상태(state)

```python
state = {
    "batch_mode": True,
    "batch_duration_ms": 200000,
    "total_duration_ms": 600000,
    "total_batches": 3,
    "current_batch_index": 0,
    "completed_batches": [],
    "context_max_chars": 500,
    "previous_context": "",
}
```

### 파일 구조

```
data/outputs/{video_name}/
├── stt.json
├── manifest.json
├── captures/
├── batches/
│   ├── batch_0/
│   │   ├── vlm.json
│   │   ├── segments_units.jsonl
│   │   ├── segment_summaries.jsonl
│   │   └── judge.json
│   └── batch_1/ ...
├── vlm.json
└── fusion/
    ├── segments_units.jsonl
    ├── segment_summaries.jsonl
    └── outputs/
        └── final_summary_*.md
```

## 신규 도구 및 에이전트

### 신규 도구

- `src/adk_pipeline/tools/batch_tools.py`
  - 배치 모드 초기화, 배치 범위 조회, 배치 완료 처리, 이전 context 추출
- `src/adk_pipeline/tools/merge_tools.py`
  - 배치 산출물 병합, 최종 통합 요약 생성

### 신규 에이전트

- `batch_preprocessing_agent`: 배치 VLM + Sync
- `batch_summarize_agent`: 배치 요약 + 배치 MD 생성
- `merge_agent`: 배치 병합 + 최종 요약 생성

## 배치 처리 단계 (권장 루프)

1. `set_pipeline_config(batch_mode=True, batch_duration_ms=200000)`
2. `init_batch_mode`로 배치 상태 초기화
3. 배치 반복 처리
   - `batch_preprocessing_agent`로 VLM + Sync
   - `batch_summarize_agent`로 요약 생성
   - `judge_agent`로 배치 평가
   - `mark_batch_complete`로 다음 배치 이동
4. `merge_agent`로 병합 및 최종 요약 생성

## 컨텍스트 전달 방식

- 이전 배치의 요약에서 각 세그먼트 첫 번째 bullet의 claim을 추출
- `previous_context`에 누적하여 다음 배치 프롬프트에 포함

## 주요 변경 파일

- `src/adk_pipeline/agent.py`
- `src/adk_pipeline/store.py`
- `src/adk_pipeline/tools/batch_tools.py`
- `src/adk_pipeline/tools/merge_tools.py`
- `src/adk_pipeline/tools/preprocessing_tools.py`
- `src/adk_pipeline/tools/summarize_tools.py`
- `src/adk_pipeline/tools/judge_tools.py`
- `src/adk_pipeline/tools/internal/vlm_openrouter.py`
- `src/fusion/sync_engine.py`
- `src/fusion/summarizer.py`
