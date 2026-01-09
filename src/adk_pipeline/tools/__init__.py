"""ADK 파이프라인 tools.

이 패키지는 ADK 파이프라인에서 사용하는 도구들을 제공합니다.

구조:
    tools/
    ├── root_tools.py          # Root Agent 도구
    ├── preprocessing_tools.py # Preprocessing Agent 도구
    ├── summarize_tools.py     # Summarize Agent 도구
    ├── judge_tools.py         # Judge Agent 도구
    └── internal/              # 내부 구현 모듈 (Agent에 직접 노출 안됨)

Agent별 도구:
    - root_tools: list_available_videos, set_pipeline_config, get_pipeline_status
    - preprocessing_tools: load_data, run_vlm, run_sync
    - summarize_tools: run_summarizer, render_md, write_final_summary
    - judge_tools: evaluate_summary
"""

from __future__ import annotations
