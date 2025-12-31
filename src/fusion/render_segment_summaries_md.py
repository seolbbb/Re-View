"""segment_summaries.jsonl -> MD 렌더링 CLI."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config
from src.fusion.io_utils import ensure_output_root
from src.fusion.renderer import render_segment_summaries_md


def main() -> None:
    parser = argparse.ArgumentParser(description="segment_summaries.jsonl -> Markdown 렌더")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="config YAML 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 segment 수 제한")
    parser.add_argument("--dry_run", action="store_true", help="출력 미생성 모드")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        ensure_output_root(config.paths.output_root)
        if args.dry_run:
            print("[DRY RUN] renderer 출력 미생성")
            return
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
        print(f"[ERROR] renderer 실패: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
