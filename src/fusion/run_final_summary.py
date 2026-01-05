"""최종 요약(timeline/tldr_timeline) 생성 CLI."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config
from src.fusion.final_summary_composer import compose_final_summaries
from src.fusion.io_utils import ensure_output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="최종 요약 timeline/tldr_timeline 생성"
    )
    parser.add_argument("--config", default="src/fusion/config.yaml", help="config YAML 경로")
    parser.add_argument("--limit", type=int, default=None, help="근거로 사용할 segment 수 제한")
    parser.add_argument("--dry_run", action="store_true", help="출력 미생성 모드")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        ensure_output_root(config.paths.output_root)
        output_dir = config.paths.output_root / "fusion"
        input_jsonl = output_dir / "segment_summaries.jsonl"
        if not input_jsonl.exists():
            raise FileNotFoundError(f"segment_summaries.jsonl이 없습니다: {input_jsonl}")

        summaries = compose_final_summaries(
            summaries_jsonl=input_jsonl,
            max_chars=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=args.limit,
        )

        if args.dry_run:
            print("[DRY RUN] final_summary 출력 미생성")
            return

        output_subdir = output_dir / "outputs"
        output_subdir.mkdir(parents=True, exist_ok=True)
        for fmt in config.raw.final_summary.generate_formats:
            if fmt not in summaries:
                continue
            output_subdir.joinpath(f"final_summary_{fmt}.md").write_text(
                summaries[fmt], encoding="utf-8"
            )
    except Exception as exc:
        print(f"[ERROR] final_summary 실패: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
