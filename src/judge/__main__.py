"""
Judge 모듈 CLI 엔트리포인트.

Usage:
    python -m src.judge \
        --segments-units <path> \
        --segment-summaries <path> \
        [--output-report <path>] \
        [--output-segments <path>] \
        [--config-file <path>]
"""

import argparse
import sys
from pathlib import Path

from src.fusion.config import load_config
from src.judge.judge import run_judge

def main():
    parser = argparse.ArgumentParser(description="Run Judge module evaluation.")
    
    # 필수 인자
    parser.add_argument(
        "--segments-units",
        type=Path,
        required=True,
        help="Path to segments_units.jsonl input file"
    )
    parser.add_argument(
        "--segment-summaries",
        type=Path,
        required=True,
        help="Path to segment_summaries.jsonl input file"
    )
    
    # 선택 인자 (출력 경로)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to output judge_report.json (default: same dir as input)"
    )
    parser.add_argument(
        "--output-segments",
        type=Path,
        default=None,
        help="Path to output judge_segments.jsonl (default: same dir as input)"
    )
    
    # 선택 인자 (설정/실행 옵션)
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/fusion/settings.yaml"),
        help="Path to fusion settings.yaml (default: 'config/fusion/settings.yaml')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of segments to evaluate (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs"
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write output files (dry-run)"
    )

    args = parser.parse_args()

    # 경로 검증
    if not args.segments_units.exists():
        print(f"Error: --segments-units file not found: {args.segments_units}", file=sys.stderr)
        sys.exit(1)
    if not args.segment_summaries.exists():
        print(f"Error: --segment-summaries file not found: {args.segment_summaries}", file=sys.stderr)
        sys.exit(1)

    # 출력 경로 기본값 설정
    output_report = args.output_report
    if output_report is None:
        output_report = args.segments_units.parent / "judge_report.json"
    
    output_segments = args.output_segments
    if output_segments is None:
        output_segments = args.segments_units.parent / "judge_segments.jsonl"

    # 설정 로드
    try:
        config_path = args.config_file
        if not config_path.exists():
             # 상대 경로일 수 있으므로 현재 디렉토리 기준 확인
             if not Path(config_path).exists():
                 print(f"Error: Config file not found: {config_path}", file=sys.stderr)
                 sys.exit(1)
        config = load_config(str(config_path))
    except Exception as e:
        print(f"Error loading config from {args.config_file}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Judge evaluation...")
    print(f"  - Units: {args.segments_units}")
    print(f"  - Summaries: {args.segment_summaries}")
    print(f"  - Output Report: {output_report}")
    print(f"  - Output Segments: {output_segments}")
    print(f"  - Config: {args.config_file}")

    try:
        result = run_judge(
            config=config,
            segments_units_path=args.segments_units,
            segment_summaries_path=args.segment_summaries,
            output_report_json=output_report,
            output_segments_jsonl=output_segments,
            batch_size=config.judge.batch_size,
            workers=config.judge.workers,
            json_repair_attempts=config.judge.json_repair_attempts,
            limit=args.limit,
            verbose=args.verbose,
            write_outputs=not args.no_write,
        )
        
        report = result["report"]
        scores = report["scores_avg"]
        print("\nEvaluation Complete!")
        print(f"  - Final Score: {scores['final']} / 10")
        print(f"  - Groundedness: {scores['groundedness']}")
        print(f"  - Note Quality: {scores['note_quality']}")
        print(f"  - Multimodal Use: {scores['multimodal_use']}")
        print(f"  - Total Tokens: {result['token_usage']['total_tokens']}")

    except Exception as e:
        print(f"\nExecution Failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
