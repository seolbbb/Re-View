# 프로젝트 가이드

## 제일 먼저 볼 파일

- `src/run_preprocess_pipeline.py`
  - 전처리(STT + Capture) 전용 메인 진입점.
- `src/run_process_pipeline.py`
  - 처리(VLM + Fusion + Judge) 전용 메인 진입점.

## 실행 흐름 핵심

- `src/pipeline/stages.py`
  - 각 단계(STT/Capture/VLM/Fusion/Judge)를 실제로 호출하는 모듈.
  - 배치 모드 흐름도 여기에서 조립됩니다.
  - 실제 실행은 `run_preprocess_pipeline.py` → `run_process_pipeline.py` 두 단계 기준으로 이해하면 됩니다.

## 모듈별 핵심 파일

- STT: `src/audio/`
  - `src/audio/stt_router.py`: STT 백엔드 선택/라우팅 수행하는 핵심 파일
  - `src/audio/clova_stt.py`: Clova STT API 호출 및 응답 정규화
- Capture: `src/capture/`
  - `src/capture/process_content.py`: 캡쳐 실행 엔트리(슬라이드 캡쳐 + manifest 생성)
  - `src/capture/tools/hybrid_extractor.py`: 슬라이드 전환 감지/중복 제거 핵심 알고리즘
- VLM: `src/vlm/`
- `src/vlm/vlm_engine.py`: Qwen VLM 호출 + 결과 수집
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

- `stt.json` / `manifest.json` / `captures/*.jpg`
- `vlm.json` (배치 OFF만 생성, 배치 ON은 `batches/batch_N/vlm.json`)
- `pipeline_run.json` / `benchmark_report.md`
- `fusion/segments_units.jsonl` (배치 OFF만 생성, 배치 ON은 `batches/batch_N/segments_units.jsonl`)
- `fusion/segment_summaries.jsonl` / `fusion/segment_summaries.md`
- `results/final_summary_*.md`
- Judge 결과
  - 배치 OFF: `fusion/judge.json` + `fusion/judge_segment_reports.jsonl`
  - 배치 ON: `batches/batch_N/judge.json` + `batches/batch_N/judge_segment_reports.jsonl`

배치 모드는 `data/outputs/<video_name>/batches/` 아래에 배치별 결과가 쌓입니다.


# JSON 플로우 (전체 단계)

아래는 파이프라인에서 JSON이 생성되고 다음 단계로 넘어가는 흐름입니다.

## 배치 모드 OFF

```
video.mp4
  ├─ STT → stt.json
  ├─ Capture → manifest.json (+ captures/*.jpg)
  ├─ VLM → (temp) vlm_raw.json → vlm.json
  ├─ Sync → fusion/segments_units.jsonl
  ├─ Summarize → fusion/segment_summaries.jsonl
  ├─ Judge → fusion/judge.json + fusion/judge_segment_reports.jsonl
  ├─ Render → fusion/segment_summaries.md
  ├─ Final → results/final_summary_*.md
  └─ Benchmark → benchmark_report.md
```

## 배치 모드 ON

```
video.mp4
  ├─ STT → stt.json
  ├─ Capture → manifest.json (+ captures/*.jpg)
  ├─ For each batch in batches/batch_N/
  │   ├─ VLM → (temp) vlm_raw.json → vlm.json
  │   ├─ Sync → segments_units.jsonl
  │   ├─ Summarize → segment_summaries.jsonl
  │   └─ Judge → judge.json + judge_segment_reports.jsonl
  ├─ Merge summaries → fusion/segment_summaries.jsonl
  ├─ Render → fusion/segment_summaries.md
  ├─ Final → results/final_summary_*.md
  └─ Benchmark → benchmark_report.md
```

배치 모드에서는 배치별 결과를 만든 뒤, `fusion/`에 요약 결과를 누적하고
렌더링/최종 요약은 `fusion/` 기준으로 수행합니다. (최종 Judge는 배치별만 수행)



# JSON 출력 핵심 (batch / no batch 기준)


## 공통 핵심

- `stt.json`: STT 세그먼트(시간/텍스트)
- `manifest.json`: 캡처 이미지 목록(파일/시간)
- `fusion/segment_summaries.jsonl`: 세그먼트 요약 결과
- `results/final_summary_*.md`: 최종 요약 마크다운
- `pipeline_run.json`: 실행 메타(시간/옵션/통계)

## 배치 모드 OFF (`no batch`) 전용

- 위 파일들이 `data/outputs/<video>/`와 `fusion/`에 직접 생성됨
- `vlm.json`: 이미지별 VLM 텍스트
- `fusion/segments_units.jsonl`: STT+VLM 결합 세그먼트
- `fusion/judge.json`: PASS/FAIL + 평균 점수(`report.scores_avg`) + 요약 리포트
- `fusion/judge_segment_reports.jsonl`: 세그먼트별 Judge 상세
- `fusion/token_usage.json`: 토큰 로그

## 배치 모드 ON (`diffusion_batch`) 전용

- 배치별 결과는 `batches/batch_N/`에 생성됨
- 배치 요약은 마지막에 `fusion/segment_summaries.jsonl`로 합쳐짐
- `batches/batch_N/vlm.json`: 배치별 VLM 결과
- `batches/batch_N/segments_units.jsonl`: 배치별 동기화 결과
- `batches/batch_N/judge.json`: 배치별 PASS/FAIL + 평균 점수
- `batches/batch_N/judge_segment_reports.jsonl`: 배치별 Judge 상세
- `batches/batch_N/token_usage.json`: 배치 요약/저지 토큰 로그
- 배치 모드는 루트에 `vlm.json`을 만들지 않음


## 기타 JSON (보조/참고/디버깅용)
- `fusion/token_usage.json`: LLM 입력 토큰 로그
- `vlm_raw.json`: 변환 후 삭제되는 임시 파일(기본은 저장하지 않음)

## 기타 파일 (보조/참고/디버깅용)
- 필요한 경우에만 확인하면 되는 파일들, `data/outputs/<video>/` 기준
- `benchmark_report.md`: 시간 얼마 걸렸는지 확인할 때
- `config.yaml`: config 정보들 모아 놓은 yaml 파일

## .env 형식 (예시)

프로젝트 루트 `.env`를 읽습니다. 필요한 값만 설정하세요.

```env
# Qwen (VLM)
QWEN_API_KEYS=key1,key2
# 또는
QWEN_API_KEY_1=key1
QWEN_API_KEY_2=key2
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# Gemini Developer API (Summarizer/Judge)
GOOGLE_API_KEYS=key1,key2
# 또는
GOOGLE_API_KEY_1=key1
GOOGLE_API_KEY_2=key2
GEMINI_API_KEY=key1

# Supabase (DB sync)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_key

# Clova STT
CLOVA_SPEECH_URL=https://api.clova.ai/...
CLOVA_SPEECH_API_KEY=your_key
```
