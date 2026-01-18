# -*- coding: utf-8 -*-
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.summarizer import run_summarizer
from src.judge.judge import run_judge
from src.pipeline.benchmark import BenchmarkTimer
from src.dev.utils import DevContext

def main():
    parser = argparse.ArgumentParser(description="Run Fusion Summarizer + Judge Verification")
    parser.add_argument("--video", default="sample4", help="Video ID (default: sample4)")
    parser.add_argument("--limit", type=int, default=2, help="Limit segments (default: 2)")
    parser.add_argument("--prompt-version", default="v2", help="Summarizer prompt version override")
    args = parser.parse_args()

    ctx = DevContext(args.video)
    if not ctx.exists():
        print(f"Error: Video output not found at {ctx.video_dir}")
        return

    print(f"Loading config from: {ctx.paths['config']}")
    config_bundle = ctx.load_config()

    # Apply overrides
    config_bundle.raw.summarizer.prompt_version = args.prompt_version
    config_bundle.judge.prompt_version = "v2" # Judge is fixed to v2 for now
    config_bundle.judge.verbose = True

    print(f"Summarizer Prompt: {config_bundle.raw.summarizer.prompt_version}")
    print(f"Judge Prompt: {config_bundle.judge.prompt_version}")

    timer = BenchmarkTimer()
    timer.start_total()
    
    # 1. Summarizer
    print("\n[STEP 1] Running Summarizer...")
    t1 = time.perf_counter()
    run_summarizer(config_bundle, limit=args.limit)
    print(f"Summarizer done: {time.perf_counter() - t1:.2f}s")
    
    # 2. Judge
    print("\n[STEP 2] Running Judge...")
    t2 = time.perf_counter()
    
    judge_dir = ctx.fusion_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    result = run_judge(
        config=config_bundle,
        segments_units_path=ctx.paths["segments_units"],
        segment_summaries_path=ctx.paths["segment_summaries"],
        output_report_path=judge_dir / "judge_report.json",
        output_segments_path=judge_dir / "judge.jsonl",
        batch_size=config_bundle.judge.batch_size,
        workers=config_bundle.judge.workers,
        json_repair_attempts=config_bundle.judge.json_repair_attempts,
        limit=args.limit,
        verbose=True,
        write_outputs=True
    )
    print(f"Judge done: {time.perf_counter() - t2:.2f}s")
    
    report = result.get("report", {})
    scores = report.get("scores", {})
    print(f"\n[RESULT] Final Score: {scores.get('final')} (Groundedness: {scores.get('groundedness')})")
    
    meta = report.get("meta", {})
    print(f"[META] Prompt Version: {meta.get('prompt_version')}")

if __name__ == "__main__":
    main()
