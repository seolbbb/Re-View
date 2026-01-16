# 프로젝트 가이드

## 제일 먼저 볼 파일

- `src/run_video_pipeline.py`
  - 현재 파이프라인의 메인 진입점.
  - STT → Capture → VLM → Fusion → Judge 순서가 한 파일에 보입니다.

## 실행 흐름 핵심

- `src/pipeline/stages.py`
  - 각 단계(STT/Capture/VLM/Fusion/Judge)를 실제로 호출하는 모듈.
  - 배치 모드 흐름도 여기에서 조립됩니다.

## 모듈별 핵심 파일

- STT: `src/audio/`
  - `src/audio/stt_router.py`: STT 백엔드 선택/라우팅
  - `src/audio/clova_stt.py`: Clova STT API 호출 및 응답 정규화
- Capture: `src/capture/`
  - `src/capture/process_content.py`: 캡쳐 실행 엔트리(슬라이드 캡쳐 + manifest 생성)
  - `src/capture/tools/hybrid_extractor.py`: 슬라이드 전환 감지/중복 제거 핵심 알고리즘
- VLM: `src/vlm/`
  - `src/vlm/vlm_engine.py`: OpenRouter VLM 호출 + 결과 수집
  - `src/vlm/vlm_fusion.py`: vlm_raw.json → vlm.json 변환
- Fusion: `src/fusion/`
  - `src/fusion/sync_engine.py`: STT/VLM 동기화 + 세그먼트 생성
  - `src/fusion/summarizer.py`: LLM 요약 생성 (segment_summaries.jsonl)
- Judge: `src/judge/judge.py`
  - 요약 품질 평가 로직(Gemini 기반)
- DB 동기화: `src/db/pipeline_sync.py`
  - 파이프라인 완료 후 Supabase 저장

## 설정 파일 위치

- 파이프라인 런타임: `config/pipeline/settings.yaml`
- 캡처 설정: `config/capture/settings.yaml`
- STT 설정: `config/audio/settings.yaml`
- VLM 설정: `config/vlm/settings.yaml`
- Fusion 설정: `config/fusion/settings.yaml`
- Judge 설정: `config/judge/settings.yaml`
- 프롬프트들은 `config/*/prompts.yaml`에 모여 있습니다. 요약, VLM 등의 prompt를 바꿔 보세요!

## 출력 구조 (실행 결과)

기본 출력 폴더: `data/outputs/<video_name>/`

- `stt.json` / `manifest.json` / `vlm.json`
- `fusion/segments_units.jsonl`
- `fusion/segment_summaries.jsonl`
- `fusion/judge.json`
- `benchmark_report.md`

배치 모드는 `data/outputs/<video_name>/batches/` 아래에 배치별 결과가 쌓입니다.


# JSON 플로우 (전체 단계)

아래는 파이프라인에서 JSON이 생성되고 다음 단계로 넘어가는 흐름입니다.

## 배치 모드 OFF

```
video.mp4
  ├─ STT → stt.json
  ├─ Capture → manifest.json (+ captures/*.jpg)
  ├─ VLM → vlm_raw.json → vlm.json
  ├─ Sync → fusion/segments_units.jsonl
  ├─ Summarize → fusion/segment_summaries.jsonl
  ├─ Judge → fusion/judge.json
  └─ Benchmark → benchmark_report.md
```

## 배치 모드 ON

```
video.mp4
  ├─ STT → stt.json
  ├─ Capture → manifest.json (+ captures/*.jpg)
  ├─ For each batch in batches/batch_N/
  │   ├─ VLM → vlm_raw.json → vlm.json
  │   ├─ Sync → segments_units.jsonl
  │   ├─ Summarize → segment_summaries.jsonl
  │   └─ Judge → judge.json
  ├─ Merge summaries → fusion/segment_summaries.jsonl
  ├─ Render → fusion/segment_summaries.md
  ├─ Judge report (final) → fusion/judge.json
  └─ Benchmark → benchmark_report.md
```

배치 모드에서는 배치별 결과를 만든 뒤, `fusion/`에 요약 결과를 누적하고
렌더링/최종 Judge는 `fusion/` 기준으로 다시 수행합니다.
