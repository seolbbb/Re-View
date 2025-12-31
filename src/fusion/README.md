# Fusion 파이프라인 (STT+VLM → 요약 → 최종 문서)

## 개요
이 모듈은 강의/발표 영상 요약 파이프라인의 핵심 단계를 파일 기반으로 재현 가능하게 수행합니다.

1) sync_engine: STT/VLM 동기화 및 세그먼트 생성  
2) summarizer: Gemini 기반 구간 요약(JSONL)  
3) renderer: segment_summaries.jsonl → Markdown  
4) final_summary_composer: 최종 요약 A/B/C 생성

## 산출물 위치
기본 output_root는 `src/fusion/outputs`이며, 실제 산출물은 아래 경로에 생성됩니다.

- `src/fusion/outputs/fusion/sync.json`
- `src/fusion/outputs/fusion/segments.jsonl`
- `src/fusion/outputs/fusion/segments_units.jsonl`
- `src/fusion/outputs/fusion/trace_map.json`
- `src/fusion/outputs/fusion/segment_summaries.jsonl`
- `src/fusion/outputs/fusion/segment_summaries.md`
- `src/fusion/outputs/fusion/outputs/final_summary_A.md`
- `src/fusion/outputs/fusion/outputs/final_summary_B.md`
- `src/fusion/outputs/fusion/outputs/final_summary_C.md`

## 설정 파일
`src/fusion/config.yaml`을 사용합니다. 입력 경로와 output_root는 반드시 설정해야 합니다.

```yaml
paths:
  stt_json: "src/data/demo/stt.json"
  vlm_json: "src/data/demo/vlm.json"
  captures_manifest_json: "src/data/demo/manifest.json"
  output_root: "src/fusion/outputs"
```

## Gemini 설정 (Developer API / Vertex AI)
### Developer API
환경 변수 우선순위: `GOOGLE_API_KEY` > `GEMINI_API_KEY`

```bash
export GOOGLE_API_KEY="YOUR_KEY"
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

### Vertex AI (express_api_key)
```yaml
llm_gemini:
  backend: "vertex_ai"
  vertex_ai:
    auth_mode: "express_api_key"
    project: "your-project-id"
    location: "us-central1"
    api_key_env: "GOOGLE_API_KEY"
```

## 데모 입력 생성
```bash
python src/fusion/make_demo_inputs.py --config src/fusion/config.yaml
```

## Sanity 테스트
```bash
# sync_engine
python src/fusion/run_sync_engine.py --config src/fusion/config.yaml --limit 2

# summarizer (dry run)
python src/fusion/run_summarizer.py --config src/fusion/config.yaml --limit 2 --dry_run

# summarizer (실행)
python src/fusion/run_summarizer.py --config src/fusion/config.yaml --limit 2

# renderer
python src/fusion/render_segment_summaries_md.py --config src/fusion/config.yaml --limit 2

# final summary
python src/fusion/run_final_summary.py --config src/fusion/config.yaml
```

- sync_engine/summarizer 실행 시 JSONL 상위 2줄이 stdout에 출력됩니다.

## 주의사항
- output_root는 현재 `src/fusion/outputs`로 고정되어 있습니다(추후 config 기반 변경 가능).
- Judge/재생성 루프는 이번 스코프에서 제외되었습니다.
- 최종 요약(A/B/C)은 `segment_summaries.jsonl`만 근거로 작성됩니다(새 사실 생성 금지).
- 모든 JSONL 입출력은 스트리밍 방식으로 처리합니다.
