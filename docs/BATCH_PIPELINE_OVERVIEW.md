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
[배치 루프] - 캡처 개수 기반 (batch_size=4)
  ┌──────────────────────────────────────────────┐
  │ Batch i (start_idx ~ end_idx)                 │
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
    "batch_size": 4,              # 배치당 캡처 개수
    "total_captures": 12,         # 총 캡처 개수
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
│   │   ├── vlm_raw.json
│   │   ├── segments_units.jsonl
│   │   ├── segment_summaries.jsonl
│   │   ├── segment_summaries.md
│   │   └── judge.json
│   ├── batch_1/ ...
│   └── batch_context.json    # 누적 context 저장
├── vlm.json (병합)
└── fusion/
    ├── segments_units.jsonl (병합)
    ├── segment_summaries.jsonl (병합)
    ├── segment_summaries.md
    └── outputs/
        └── final_summary_*.md
```

## 에이전트 구조

### 에이전트 목록

| 에이전트              | 역할                       | 도구                                                                             |
| --------------------- | -------------------------- | -------------------------------------------------------------------------------- |
| `preprocessing_agent` | VLM + Sync 실행            | `load_data`, `init_batch_mode`, `run_vlm`, `run_sync`                            |
| `summarize_agent`     | 요약 생성 + 배치 MD 렌더링 | `run_summarizer`, `render_batch_md`                                              |
| `judge_agent`         | 품질 평가 (PASS/FAIL)      | `evaluate_summary`                                                               |
| `merge_agent`         | 배치 병합 + 최종 요약      | `merge_all_batches`, `render_md`, `generate_final_summary`, `merge_and_finalize` |
| `root_agent`          | 전체 파이프라인 조율       | 배치 관리 도구들                                                                 |

### 도구 파일

- `src/adk_pipeline/tools/batch_tools.py`
  - `init_batch_mode`: 배치 모드 초기화 (캡처 개수 기반)
  - `get_batch_info`: 현재 배치 상태 조회
  - `get_current_batch_time_range`: 현재 배치의 시간 범위/캡처 인덱스 반환
  - `mark_batch_complete`: 배치 완료 처리 후 다음 배치로 이동
  - `get_previous_context`: 이전 배치 요약 context 조회
  - `reset_batch_mode`: 배치 모드 초기화

- `src/adk_pipeline/tools/merge_tools.py`
  - `merge_all_batches`: 모든 배치 파일 병합
  - `generate_final_summary`: LLM으로 전체 통합 요약 생성
  - `merge_and_finalize`: 병합 + 렌더링 + 최종 요약 한번에 실행

## 배치 처리 단계 (권장 루프)

1. `set_pipeline_config(video_name, batch_mode=True, batch_size=4)`
2. `preprocessing_agent`로 transfer → `init_batch_mode`로 배치 상태 초기화
3. 배치 반복 처리
   - `preprocessing_agent`로 VLM + Sync
   - `summarize_agent`로 요약 생성 + 배치 MD 렌더링
   - `judge_agent`로 배치 평가
   - `mark_batch_complete`로 다음 배치 이동
4. `merge_agent`로 병합 및 최종 요약 생성

## 컨텍스트 전달 방식

- 이전 배치의 요약에서 각 세그먼트 첫 번째 bullet의 claim을 추출
- `batch_context.json`에 누적 저장
- `previous_context`로 다음 배치 프롬프트에 포함

## 주요 변경 파일

- `src/adk_pipeline/agent.py`
- `src/adk_pipeline/store.py`
- `src/adk_pipeline/tools/batch_tools.py`
- `src/adk_pipeline/tools/merge_tools.py`
- `src/adk_pipeline/tools/preprocessing_tools.py`
- `src/adk_pipeline/tools/summarize_tools.py`
- `src/adk_pipeline/tools/judge_tools.py`
- `src/adk_pipeline/tools/root_tools.py`
- `src/fusion/sync_engine.py`
- `src/fusion/summarizer.py`
- `src/vlm/vlm_engine.py`
