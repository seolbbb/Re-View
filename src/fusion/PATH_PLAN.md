PATH_PLAN (repo discovery)

Fusion 루트
- 확정: src/fusion

Config 위치
- 확정: src/fusion/config.yaml

CLI 스크립트 위치
- 확정: src/fusion/
  - run_sync_engine.py
  - run_summarizer.py
  - render_segment_summaries_md.py
  - run_final_summary.py
  - make_demo_inputs.py

Demo 입력
- 확정: src/data/demo/
  - src/data/demo/stt.json
  - src/data/demo/vlm.json
  - src/data/demo/manifest.json
- make_demo_inputs.py가 해당 파일을 생성/재생성

Output root
- 확정: src/fusion/outputs
- 실제 산출물 위치: output_root/fusion/*
- 최종 요약(timeline/tldr_timeline) 위치: output_root/fusion/outputs/*

README
- 확정: src/fusion/README.md
