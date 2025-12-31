"""summarizer 실행 CLI."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config
from src.fusion.renderer import render_segment_summaries_md
from src.fusion.summarizer import run_summarizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini 요약 실행")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="config YAML 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 segment 수 제한")
    parser.add_argument("--dry_run", action="store_true", help="LLM 미호출 모드")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_summarizer(config, limit=args.limit, dry_run=args.dry_run)
        if not args.dry_run:
            output_dir = config.paths.output_root / "fusion"
            render_segment_summaries_md(
                summaries_jsonl=output_dir / "segment_summaries.jsonl",
                output_md=output_dir / "segment_summaries.md",
                include_sources=config.raw.render.include_sources,
                sources_jsonl=output_dir / "segments_units.jsonl",
                md_wrap_width=config.raw.render.md_wrap_width,
                limit=args.limit,
            )
    except Exception as exc:
        print(f"[ERROR] summarizer 실패: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
