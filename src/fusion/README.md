# Fusion 코어 모듈

STT/VLM 동기화 → 요약 → 렌더링을 담당하는 코어 모듈입니다.
`src/adk_pipeline`과 파이프라인 단계에서 호출됩니다.

## 역할

- STT/VLM 동기화로 세그먼트/유닛 생성
- 세그먼트 요약 생성
- 마크다운 렌더링 + 최종 요약 생성
- (옵션) judge 평가 리포트 생성

## 모듈 구조

```
src/fusion/
├── __init__.py              # 패키지 초기화
├── config.py                # 설정 로드 (load_config)
├── io_utils.py              # I/O 유틸리티
├── sync_engine.py           # STT/VLM 동기화 (run_sync_engine)
├── summarizer.py            # Gemini 요약 (run_summarizer)
├── renderer.py              # MD 렌더링 + 최종 요약 (render_segment_summaries_md, compose_final_summaries)
└── prompt_versions.md       # 프롬프트 버전 참고
```

## 실행 흐름

이 모듈은 직접 실행하지 않고 파이프라인이 호출합니다.

```bash
# Step 1: Preprocess (STT + Capture)
python src/run_preprocess_pipeline.py --video "my_video.mp4"

# Step 2: ADK 파이프라인 (VLM → Sync → Summarize → Judge)
adk web src/adk_pipeline
```

## 핵심 함수

### sync_engine.run_sync_engine(config, limit)
STT와 VLM 결과를 동기화하여 세그먼트를 생성합니다.

**출력:**
- `segments.jsonl`: 기본 세그먼트 정보
- `segments_units.jsonl`: 상세 유닛 정보 포함
- `sync.json`, `trace_map.json`: 동기화 메타데이터

### summarizer.run_summarizer(config, limit)
Gemini를 사용하여 세그먼트별 요약을 생성합니다.

**출력:**
- `segment_summaries.jsonl`: 구조화된 요약 (bullets, definitions, explanations 등)

### renderer.render_segment_summaries_md(...)
요약 JSONL을 마크다운으로 변환합니다.

**출력:**
- `segment_summaries.md`: 읽기 쉬운 마크다운 형식

### renderer.compose_final_summaries(...)
세그먼트 요약을 기반으로 최종 요약을 생성합니다.

**출력:**
- `final_summary_A.md`, `final_summary_B.md`, `final_summary_C.md`: 포맷별 최종 요약

## 설정 (config.yaml)

템플릿 설정 파일은 `config/fusion/settings.yaml`에 있으며, `paths`는 실행 시
생성된 config에서 자동으로 주입됩니다.

```yaml
sync_engine:
  min_segment_sec: 10
  max_segment_sec: 120
  max_transcript_chars: 2000
  # ...

summarizer:
  temperature: 0.1
  bullets_per_segment_min: 2
  bullets_per_segment_max: 5
  prompt_version: "sum_v1.6"
  # ...

llm_gemini:
  backend: "developer_api"
  model: "gemini-2.0-flash"
  # ...
```

## Summarizer 프롬프트 버전

- 프롬프트 버전은 `config/fusion/prompts.yaml`에서 관리합니다.
- 기본 버전은 `config/fusion/settings.yaml`의 `summarizer.prompt_version`으로 결정됩니다.
- 템플릿 치환 토큰 예시:
  - `{{CLAIM_MAX_CHARS}}`
  - `{{BULLETS_MIN}}`
  - `{{BULLETS_MAX}}`
  - `{{SEGMENTS_TEXT}}`

## Gemini 설정

### Developer API
```bash
# .env
GOOGLE_API_KEY=your_api_key
```

### Vertex AI (ADC)
```yaml
llm_gemini:
  backend: "vertex_ai"
  vertex_ai:
    auth_mode: "adc"
    project: "your-project-id"
    location: "us-central1"
```

## 산출물 구조

ADK 파이프라인 실행 시:
```
data/outputs/{video_name}/
├── stt.json              # Preprocess
├── manifest.json         # Preprocess
├── captures/             # Preprocess
├── vlm.json              # Preprocessing
├── config.yaml           # 동적 생성된 설정
└── fusion/
    ├── segments.jsonl
    ├── segments_units.jsonl
    ├── sync.json
    ├── trace_map.json
    ├── segment_summaries.jsonl
    ├── segment_summaries.md
    ├── judge/
    │   ├── judge_report.json
    │   └── judge_segment_reports.jsonl
    └── outputs/
        ├── final_summary_A.md
        ├── final_summary_B.md
        └── final_summary_C.md
```

## 주의사항

- 이 모듈은 라이브러리로 사용됩니다. 실행은 파이프라인에서 수행합니다.
- 모든 JSONL 입출력은 스트리밍 방식으로 처리됩니다.
