# -*- coding: utf-8 -*-
"""
================================================================================
run_prompt_tuning.py - Prompt Tuning Pipeline for ReView
================================================================================

Runs batch summarizer + judge pipeline and logs experiment results to CSV.

Usage:
    python src/dev/run_prompt_tuning.py --video sample4 --limit 2 --note "test run"
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import ConfigBundle, load_config
from src.fusion.summarizer import run_batch_summarizer
from src.fusion.io_utils import read_jsonl
from src.judge.judge import run_judge
from src.dev.show_summary import convert_jsonl_to_md
from src.dev.utils import ExperimentLogger


def _get_input_files(video_dir: Path) -> Dict[str, Path]:
    """Locate required input files."""
    return {
        "stt": video_dir / "stt.json",
        "vlm": video_dir / "vlm.json",
        "manifest": video_dir / "capture.json",
        "config": video_dir / "config.yaml",
        "segments_units": video_dir / "fusion" / "segments_units.jsonl",
        "token_usage": video_dir / "fusion" / "token_usage.json",
        "pipeline_run": video_dir / "pipeline_run.json",
    }


def main():
    parser = argparse.ArgumentParser(description="Run Prompt Tuning Pipeline")
    parser.add_argument("--video", required=True, help="Video ID (e.g. sample4)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of segments")
    parser.add_argument("--note", default="", help="Experiment note")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers for Judge")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of segments per batch (default: 10)")
    
    args = parser.parse_args()
    
    # Capture command line
    command_str = f"python {' '.join(sys.argv)}"
    
    # 1. Setup
    output_root = ROOT / "data" / "outputs"
    video_dir = output_root / args.video
    
    fusion_dir = video_dir / "fusion"
    cleanup_summaries = fusion_dir / "segment_summaries.jsonl"
    if cleanup_summaries.exists():
        print(f"[INFO] Removing previous summaries: {cleanup_summaries}")
        cleanup_summaries.unlink()

    input_files = _get_input_files(video_dir)
    
    # Check config file exists
    if not input_files["config"].exists():
        raise FileNotFoundError(f"Config file not found: {input_files['config']}")
    
    config = load_config(str(input_files["config"]))
    prompt_version = config.raw.summarizer.prompt_version or "sum_v1.6"

    # 2. Load segments
    full_segments = list(read_jsonl(input_files["segments_units"]))
    
    if args.limit > 0:
        full_segments = full_segments[:args.limit]

    total_segments = len(full_segments)
    batch_size = args.batch_size
    total_batches = max(1, math.ceil(total_segments / batch_size))

    print(f"\n📦 배치 모드: {total_segments}개 세그먼트를 {total_batches}개 배치로 처리 (배치당 {batch_size}개)")

    batches_dir = video_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    
    # Accumulated summaries path
    accumulated_summaries_path = fusion_dir / "segment_summaries.jsonl"
    if accumulated_summaries_path.exists():
        accumulated_summaries_path.unlink()
        
    logger = ExperimentLogger()
    
    total_input_tokens = 0
    
    # Try to load previous timing stats if available (to mimic full pipeline log)
    pre_stats = {
        "stt_sec": 0.0,
        "capture_sec": 0.0,
        "vlm_sec": 0.0
    }
    if input_files["pipeline_run"].exists():
        try:
            run_meta = json.loads(input_files["pipeline_run"].read_text(encoding="utf-8"))
            durations = run_meta.get("durations_sec", {})
            pre_stats["stt_sec"] = durations.get("stt_sec", 0.0)
            pre_stats["capture_sec"] = durations.get("capture_sec", 0.0)
            pre_stats["vlm_sec"] = durations.get("vlm_sec", 0.0)
        except Exception as e:
            print(f"[WARN] Failed to read pipeline_run.json: {e}")

    previous_context = ""
    cumulative_segments = 0
    
    t0 = time.time()
    
    for batch_idx in range(total_batches):
        # Rate Limiting
        if batch_idx > 0:
            print(f"\n⏳ API rate limiting 방지를 위해 3초 대기...")
            time.sleep(3)
            
        print(f"\n{'='*50}")
        print(f"🚀 배치 {batch_idx + 1}/{total_batches} 처리 중...")
        print(f"{'='*50}")
        
        batch_dir = batches_dir / f"batch_{batch_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Slice segments for current batch
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total_segments)
        current_batch_segments = full_segments[start_i:end_i]
        
        if not current_batch_segments:
            continue

        # Write temp segments file for this batch
        batch_segments_path = batch_dir / "segments_units.jsonl"
        with batch_segments_path.open("w", encoding="utf-8") as f:
            for s in current_batch_segments:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                
        # 1. Run Summarizer for Batch
        batch_t0 = time.time()
        
        summarize_result = run_batch_summarizer(
            segments_units_jsonl=batch_segments_path,
            output_dir=batch_dir,
            config=config,
            previous_context=previous_context,
            limit=None
        )
        
        batch_elapsed = time.time() - batch_t0
        
        # Update Context
        new_context = summarize_result.get("context", "")
        if new_context:
            previous_context = new_context[:500]
            
        # Accumulate Results
        batch_summaries_path = batch_dir / "segment_summaries.jsonl"
        if batch_summaries_path.exists():
            content = batch_summaries_path.read_text(encoding="utf-8")
            with accumulated_summaries_path.open("a", encoding="utf-8") as f:
                f.write(content)

        # 2. Token Logging
        b_input = 0
        batch_usage_path = batch_dir / "token_usage.json"
        if batch_usage_path.exists():
            usage_data = json.loads(batch_usage_path.read_text(encoding="utf-8"))
            bs_data = usage_data.get("batch_summarizer", [])
            if bs_data:
                last_entry = bs_data[-1]
                b_input = last_entry.get("input_tokens", 0)
        
        total_input_tokens += b_input
        
        logger.log(
            prompt_version=prompt_version,
            scores={},
            input_tokens=b_input,
            elapsed_sec=batch_elapsed,
            note=args.note,
            model=config.raw.llm_gemini.model,
            temperature=config.raw.summarizer.temperature,
            segments_count=len(current_batch_segments),
            workers=1,
            segment_indices=",".join(str(s["segment_id"]) for s in current_batch_segments),
            command=command_str,
            component="batch_summarizer",
            summarizer_sec=batch_elapsed  # Map batch elapsed to summarizer_sec for component log
        )
        
        # 3. Run Judge for Batch
        batch_judge_path = batch_dir / "judge.json"
        
        run_judge(
            config=config,
            segments_units_path=batch_segments_path,
            segment_summaries_path=batch_summaries_path,
            output_report_path=batch_judge_path,
            output_segments_path=batch_dir / "judge_segments.jsonl",
            batch_size=3,
            workers=args.workers,
            json_repair_attempts=1,
            limit=None,
            write_outputs=True,
            verbose=False
        )
        
        if batch_judge_path.exists():
            j_res = json.loads(batch_judge_path.read_text(encoding="utf-8"))
            print(f"  ⚖️ Batch Judge Score: {j_res.get('final_score', 0)}")
            
        cumulative_segments += len(current_batch_segments)

    # 4. Final Aggregated Judge
    print(f"\n{'='*50}")
    print("⚖️ Final Aggregated Judge Running...")
    print(f"{'='*50}")
    
    judge_output_dir = fusion_dir / "judge"
    judge_output_dir.mkdir(parents=True, exist_ok=True)
    (judge_output_dir / "judge_report.json").unlink(missing_ok=True)
    
    judge_start = time.time()
    scores = run_judge(
        config=config,
        segments_units_path=input_files["segments_units"],
        segment_summaries_path=accumulated_summaries_path,
        output_report_path=judge_output_dir / "judge_report.json",
        output_segments_path=judge_output_dir / "judge_segment_reports.jsonl",
        write_outputs=True, 
        verbose=True,
        batch_size=3,
        workers=args.workers,
        json_repair_attempts=3,
        limit=None
    )
    judge_time = time.time() - judge_start
    
    # Read Judge tokens
    judge_input_tokens = 0
    judge_usage_path = judge_output_dir / "token_usage.json"
    if judge_usage_path.exists():
        try:
            usage_data = json.loads(judge_usage_path.read_text(encoding="utf-8"))
            judge_entries = usage_data.get("judge", [])
            for entry in judge_entries:
                judge_input_tokens += entry.get("input_tokens", 0)
        except Exception:
            pass
    
    # Log Judge Component
    report = scores.get("report", {})
    scores_dict = report.get("scores", {})
    
    logger.log(
        prompt_version=prompt_version,
        scores=scores_dict,
        input_tokens=judge_input_tokens,
        elapsed_sec=judge_time,
        note=args.note,
        model=config.raw.llm_gemini.model,
        temperature=config.raw.summarizer.temperature,
        segments_count=cumulative_segments,
        workers=args.workers,
        segment_indices=f"1-{cumulative_segments}",
        command=command_str,
        component="judge",
        judge_elapsed_sec=judge_time
    )
    
    # 5. Save versioned copy & MD
    if accumulated_summaries_path.exists():
        versioned_output = accumulated_summaries_path.parent / f"segment_summaries_{prompt_version}.jsonl"
        shutil.copy2(accumulated_summaries_path, versioned_output)
        print(f"[INFO] Saved versioned summaries to: {versioned_output}") 
        
        versioned_md = versioned_output.with_suffix('.md')
        convert_jsonl_to_md(versioned_output, versioned_md)     
    
    # 6. Log Final Pipeline Stats (Mimic run_video_pipeline format)
    total_elapsed = time.time() - t0
    
    # Construct standard note
    final_note = f"Full pipeline run: {args.video}"
    if args.note:
        final_note += f" | {args.note}"

    # Log as 'run_video_pipeline' to match main pipeline output format
    logger.log(
        prompt_version=prompt_version,
        scores=scores_dict,
        input_tokens=total_input_tokens, 
        elapsed_sec=total_elapsed + pre_stats["stt_sec"] + pre_stats["capture_sec"] + pre_stats["vlm_sec"], # Estimate total including pre-stages
        note=final_note,
        model=config.raw.llm_gemini.model,
        temperature=config.raw.summarizer.temperature,
        segments_count=cumulative_segments,
        workers=args.workers,
        segment_indices=f"1-{cumulative_segments}",
        command=f"run_video_pipeline --video {args.video}", # Mimic command
        component="run_video_pipeline",
        # Detailed timings
        stt_sec=pre_stats["stt_sec"],
        capture_sec=pre_stats["capture_sec"],
        vlm_sec=pre_stats["vlm_sec"],
        summarizer_sec=total_elapsed - judge_time, # Rough estimate of summarizer time
        judge_elapsed_sec=judge_time
    )

    print("\n" + "="*50)
    print(f"Batch Execution Logged: {logger.csv_path}")
    print(f"Final Score: {scores_dict.get('final', 0)}")
    print("="*50)

if __name__ == "__main__":
    main()
