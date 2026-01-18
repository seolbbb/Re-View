"""
Preprocessing pipeline entrypoint.

Runs STT and Capture only, then optionally uploads the artifacts to Supabase.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import argparse
from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from src.db import sync_pipeline_results_to_db
from src.pipeline.benchmark import BenchmarkTimer, format_duration, get_video_info, print_benchmark_report
from src.pipeline.stages import run_capture, run_stt


def _sanitize_video_name(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"


def run_preprocess_pipeline(
    *,
    video: str,
    output_base: str = "data/outputs",
    stt_backend: str = "clova",
    parallel: bool = True,
    capture_threshold: Optional[float] = None,
    capture_dedupe_threshold: Optional[float] = None,
    capture_min_interval: Optional[float] = None,
    capture_verbose: bool = False,
    limit: Optional[int] = None,
    sync_to_db: bool = True,
) -> None:
    """Run STT + Capture and stop after uploading inputs."""
    video_path = Path(video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("pipeline settings must be a mapping.")

    if capture_threshold is None:
        capture_threshold = float(settings.get("capture_threshold", 3.0))
    if capture_dedupe_threshold is None:
        capture_dedupe_threshold = float(settings.get("capture_dedupe_threshold", 3.0))
    if capture_min_interval is None:
        capture_min_interval = float(settings.get("capture_min_interval", 0.5))

    output_base_path = (ROOT / Path(output_base)).resolve()
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base_path / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    timer = BenchmarkTimer()
    video_info = get_video_info(video_path)

    run_meta_path = video_root / "pipeline_run.json"
    run_args = {
        "pipeline_type": "preprocess",
        "video": str(video_path),
        "output_base": str(output_base_path),
        "stt_backend": stt_backend,
        "parallel": parallel,
        "capture_threshold": capture_threshold,
        "capture_dedupe_threshold": capture_dedupe_threshold,
        "capture_min_interval": capture_min_interval,
        "capture_verbose": capture_verbose,
        "limit": limit,
    }
    run_meta: Dict[str, Any] = {
        "video_path": str(video_path),
        "video_name": video_name,
        "video_info": video_info,
        "output_base": str(output_base_path),
        "video_root": str(video_root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": run_args,
        "durations_sec": {},
        "benchmark": {},
        "status": "running",
    }
    run_meta_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    timer.start_total()
    capture_count = 0

    try:
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "capture.json"

        print(f"\nStarting preprocessing (parallel={parallel})...")
        print("-" * 50)

        stt_elapsed = 0.0
        capture_elapsed = 0.0

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                def run_stt_timed() -> float:
                    start = time.perf_counter()
                    run_stt(video_path, stt_json, backend=stt_backend)
                    return time.perf_counter() - start

                def run_capture_timed():
                    start = time.perf_counter()
                    result = run_capture(
                        video_path,
                        output_base_path,
                        threshold=capture_threshold,
                        dedupe_threshold=capture_dedupe_threshold,
                        min_interval=capture_min_interval,
                        verbose=capture_verbose,
                        video_name=video_name,
                    )
                    return result, time.perf_counter() - start

                stt_future = executor.submit(run_stt_timed)
                capture_future = executor.submit(run_capture_timed)

                stt_elapsed = stt_future.result()
                capture_result, capture_elapsed = capture_future.result()
                capture_count = len(capture_result) if capture_result else 0

            timer.record_stage("stt", stt_elapsed)
            timer.record_stage("capture", capture_elapsed)
            print(f"  STT done in {format_duration(stt_elapsed)} (parallel)")
            print(f"  Capture done in {format_duration(capture_elapsed)} (parallel)")
        else:
            _, stt_elapsed = timer.time_stage("stt", run_stt, video_path, stt_json, backend=stt_backend)
            capture_result, capture_elapsed = timer.time_stage(
                "capture",
                run_capture,
                video_path,
                output_base_path,
                threshold=capture_threshold,
                dedupe_threshold=capture_dedupe_threshold,
                min_interval=capture_min_interval,
                verbose=capture_verbose,
                video_name=video_name,
            )
            capture_count = len(capture_result) if capture_result else 0

        timer.end_total()

        md_report = print_benchmark_report(
            video_info=video_info,
            timer=timer,
            capture_count=capture_count,
            segment_count=0,
            video_path=video_path,
            output_root=video_root,
            parallel=parallel,
        )
        report_path = video_root / "benchmark_report.md"
        report_path.write_text(md_report, encoding="utf-8")

        run_meta["durations_sec"] = {
            "stt_sec": round(stt_elapsed, 6),
            "capture_sec": round(capture_elapsed, 6),
            "total_sec": round(timer.get_total_elapsed(), 6),
        }
        run_meta["benchmark"] = timer.get_report(video_info.get("duration_sec"))
        run_meta["processing_stats"] = {
            "capture_count": capture_count,
        }
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "ok"
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if sync_to_db:
            print("\nSyncing preprocessing artifacts to Supabase...")
            db_success = sync_pipeline_results_to_db(
                video_path=video_path,
                video_root=video_root,
                run_meta=run_meta,
                duration_sec=video_info.get("duration_sec"),
                provider=stt_backend,
            )
            if db_success:
                print("Database sync completed.")
            else:
                print("Database sync skipped or failed (check logs above).")

        print("\nPreprocessing completed.")
        print(f"Outputs: {video_root}")
        print(f"Benchmark: {report_path}")

    except Exception as exc:
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nPreprocessing failed: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess pipeline (STT + Capture only)")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--stt-backend", choices=["clova"], default="clova", help="STT backend")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run STT and Capture in parallel",
    )
    parser.add_argument("--capture-verbose", action="store_true", help="Enable capture logs")
    parser.add_argument("--no-db-sync", action="store_true", help="Skip Supabase sync")
    args = parser.parse_args()

    run_preprocess_pipeline(
        video=args.video,
        output_base=args.output_base,
        stt_backend=args.stt_backend,
        parallel=args.parallel,
        capture_verbose=args.capture_verbose,
        sync_to_db=not args.no_db_sync,
    )


if __name__ == "__main__":
    main()
