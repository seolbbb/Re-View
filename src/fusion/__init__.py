"""STT/VLM Fusion 파이프라인 코어 모듈.

이 패키지는 ADK 파이프라인에서 사용하는 핵심 처리 로직을 제공합니다.

Modules:
    - config: 설정 로드 (load_config)
    - io_utils: I/O 유틸리티 (read_json, write_json, read_jsonl 등)
    - sync_engine: STT/VLM 동기화 및 세그먼트 생성 (run_sync_engine)
    - summarizer: Gemini 기반 세그먼트 요약 (run_summarizer)
    - renderer: 요약 마크다운/최종 요약 렌더링 (render_segment_summaries_md, compose_final_summaries)

Usage:
    ADK 파이프라인에서 사용됩니다.
    직접 CLI 실행은 adk_pipeline을 통해 수행합니다.

    # Pre-ADK (STT + Capture)
    python src/pre_adk_pipeline.py --video "my_video.mp4"

    # ADK 파이프라인 (VLM → Sync → Summarize → Judge)
    adk web src/adk_pipeline
"""
