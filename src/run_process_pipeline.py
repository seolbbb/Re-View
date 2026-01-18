"""
Processing pipeline entrypoint.

Loads STT/Capture artifacts from DB (or local cache), then runs VLM + Fusion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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

from src.db import get_supabase_adapter, sync_pipeline_results_to_db
from src.pipeline.benchmark import BenchmarkTimer, print_benchmark_report
from src.pipeline.stages import (
    generate_fusion_config,
    run_batch_fusion_pipeline,
    run_fusion_pipeline,
    run_vlm_openrouter,
)


def _sanitize_video_name(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"


def run_processing_pipeline(
    *,
    video_name: Optional[str],
    video_id: Optional[str] = None,
    output_base: str = "data/outputs",
    batch_mode: bool = False,
    batch_size: Optional[int] = None,
    vlm_batch_size: Optional[int] = None,
    vlm_concurrency: Optional[int] = None,
    vlm_show_progress: Optional[bool] = None,
    limit: Optional[int] = None,
    sync_to_db: bool = False,
    force_db: bool = False,
) -> None:
    """Run VLM + Fusion using DB-backed inputs."""
    if not video_name and not video_id:
        raise ValueError("video_name or video_id is required.")

    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("pipeline settings must be a mapping.")

    if vlm_batch_size is None:
        vlm_batch_size = settings.get("vlm_batch_size", 2)
    if vlm_concurrency is None:
        vlm_concurrency = settings.get("vlm_concurrency", 3)
    if vlm_show_progress is None:
        vlm_show_progress = settings.get("vlm_show_progress", True)
    if batch_size is None:
        batch_size = settings.get("batch_size", 10)

    output_base_path = (ROOT / Path(output_base)).resolve()
    safe_video_name = _sanitize_video_name(video_name) if video_name else None
    video_root = output_base_path / safe_video_name if safe_video_name else None

    stt_json = video_root / "stt.json" if video_root else None
    captures_dir = video_root / "captures" if video_root else None
    manifest_json = video_root / "capture.json" if video_root else None

    local_ready = (
        stt_json
        and manifest_json
        and captures_dir
        and stt_json.exists()
        and manifest_json.exists()
        and captures_dir.exists()
    )

    db_duration = None
    if force_db or not local_ready:
        adapter = get_supabase_adapter()
        if not adapter:
            raise ValueError("Supabase adapter not configured. Check SUPABASE_URL/SUPABASE_KEY.")

        video_row = None
        if video_id:
            result = (
                adapter.client.table("videos")
                .select("id,name,original_filename,duration_sec")
                .eq("id", video_id)
                .limit(1)
                .execute()
            )
            rows = result.data or []
            video_row = rows[0] if rows else None
        else:
            for candidate in [video_name, safe_video_name]:
                if not candidate:
                    continue
                result = (
                    adapter.client.table("videos")
                    .select("id,name,original_filename,duration_sec")
                    .eq("name", candidate)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                rows = result.data or []
                if rows:
                    video_row = rows[0]
                    break

        if not video_row:
            target = video_id or video_name
            raise ValueError(f"Video not found in DB: {target}")

        video_id = video_row["id"]
        if not safe_video_name:
            safe_video_name = _sanitize_video_name(video_row.get("name") or video_id)
        if not safe_video_name:
            safe_video_name = "video"

        video_root = output_base_path / safe_video_name
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "capture.json"
        video_root.mkdir(parents=True, exist_ok=True)

        latest_run_id = None
        run_result = (
            adapter.client.table("pipeline_runs")
            .select("id,created_at")
            .eq("video_id", video_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        run_rows = run_result.data or []
        if run_rows:
            latest_run_id = run_rows[0].get("id")

        stt_query = adapter.client.table("stt_results").select("*").eq("video_id", video_id)
        if latest_run_id:
            stt_query = stt_query.eq("pipeline_run_id", latest_run_id)
        stt_rows = stt_query.execute().data or []
        if not stt_rows and latest_run_id:
            stt_rows = (
                adapter.client.table("stt_results")
                .select("*")
                .eq("video_id", video_id)
                .execute()
                .data
                or []
            )
        if not stt_rows:
            raise ValueError("stt_results not found in DB.")

        stt_segments: list = []
        first_segments = stt_rows[0].get("segments")
        if isinstance(first_segments, list):
            stt_segments = first_segments
        else:
            for row in stt_rows:
                stt_segments.append(
                    {
                        "start_ms": row.get("start_ms"),
                        "end_ms": row.get("end_ms"),
                        "text": row.get("text", ""),
                        "confidence": row.get("confidence"),
                    }
                )
            if stt_segments and stt_segments[0].get("start_ms") is not None:
                stt_segments.sort(key=lambda item: item.get("start_ms") or 0)
        stt_json.write_text(
            json.dumps({"segments": stt_segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        captures_query = (
            adapter.client.table("captures")
            .select("file_name,start_ms,end_ms,storage_path,pipeline_run_id")
            .eq("video_id", video_id)
        )
        if latest_run_id:
            captures_query = captures_query.eq("pipeline_run_id", latest_run_id)
        capture_rows = captures_query.execute().data or []
        if not capture_rows and latest_run_id:
            capture_rows = (
                adapter.client.table("captures")
                .select("file_name,start_ms,end_ms,storage_path")
                .eq("video_id", video_id)
                .execute()
                .data
                or []
            )
        if not capture_rows:
            raise ValueError("captures not found in DB.")

        captures_dir.mkdir(parents=True, exist_ok=True)
        manifest_payload = []
        for row in sorted(capture_rows, key=lambda item: item.get("start_ms") or 0):
            file_name = row.get("file_name")
            if not file_name:
                continue
            manifest_payload.append(
                {
                    "file_name": file_name,
                    "start_ms": row.get("start_ms"),
                    "end_ms": row.get("end_ms"),
                }
            )
            image_path = captures_dir / file_name
            if image_path.exists() and not force_db:
                continue
            storage_path = row.get("storage_path")
            if not storage_path:
                raise ValueError(f"Missing storage_path for capture: {file_name}")
            image_bytes = adapter.client.storage.from_("captures").download(storage_path)
            image_path.write_bytes(image_bytes)

        manifest_json.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        db_duration = video_row.get("duration_sec")

    if not (stt_json and manifest_json and captures_dir):
        raise ValueError("Input artifacts are not resolved.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    capture_count = len(manifest_payload) if isinstance(manifest_payload, list) else 0

    timer = BenchmarkTimer()
    run_meta_path = video_root / "pipeline_run.json"
    run_args = {
        "pipeline_type": "process",
        "video_name": video_name or safe_video_name,
        "video_id": video_id,
        "output_base": str(output_base_path),
        "batch_mode": batch_mode,
        "batch_size": batch_size,
        "vlm_batch_size": vlm_batch_size,
        "vlm_concurrency": vlm_concurrency,
        "vlm_show_progress": vlm_show_progress,
        "limit": limit,
    }
    run_meta: Dict[str, Any] = {
        "video_name": video_name or safe_video_name,
        "video_id": video_id,
        "video_root": str(video_root),
        "output_base": str(output_base_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": run_args,
        "durations_sec": {},
        "benchmark": {},
        "status": "running",
    }
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    timer.start_total()
    segment_count = 0

    try:
        print(f"\nStarting processing pipeline (batch_mode={batch_mode})...")
        print("-" * 50)

        if batch_mode:
            fusion_info = run_batch_fusion_pipeline(
                video_root=video_root,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                stt_json=stt_json,
                video_name=safe_video_name,
                batch_size=batch_size,
                timer=timer,
                vlm_batch_size=vlm_batch_size,
                vlm_concurrency=vlm_concurrency,
                vlm_show_progress=vlm_show_progress,
                limit=limit,
                repo_root=ROOT,
            )
            segment_count = fusion_info.get("segment_count", 0)
            vlm_image_count = capture_count
            vlm_elapsed = fusion_info["timings"].get("vlm_sec", 0.0)
        else:
            vlm_image_count, vlm_elapsed = timer.time_stage(
                "vlm",
                run_vlm_openrouter,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                video_name=safe_video_name,
                output_base=output_base_path,
                batch_size=vlm_batch_size,
                concurrency=vlm_concurrency,
                show_progress=vlm_show_progress,
            )

            template_config = ROOT / "config" / "fusion" / "settings.yaml"
            if not template_config.exists():
                raise FileNotFoundError(f"fusion settings template not found: {template_config}")

            fusion_config_path = video_root / "config.yaml"
            generate_fusion_config(
                template_config=template_config,
                output_config=fusion_config_path,
                repo_root=ROOT,
                stt_json=stt_json,
                vlm_json=video_root / "vlm.json",
                manifest_json=manifest_json,
                output_root=video_root,
            )

            fusion_info = run_fusion_pipeline(
                fusion_config_path,
                limit=limit,
                timer=timer,
            )
            segment_count = fusion_info.get("segment_count", 0)

        timer.end_total()

        md_report = print_benchmark_report(
            video_info={"duration_sec": db_duration, "width": None, "height": None, "fps": None, "codec": None, "file_size_mb": None},
            timer=timer,
            capture_count=capture_count,
            segment_count=segment_count,
            video_path=Path(video_name or safe_video_name or "video"),
            output_root=video_root,
            parallel=False,
        )
        report_path = video_root / "benchmark_report.md"
        report_path.write_text(md_report, encoding="utf-8")

        run_meta["durations_sec"] = {
            "vlm_sec": round(vlm_elapsed, 6),
            "total_sec": round(timer.get_total_elapsed(), 6),
            **{f"fusion.{k}": round(v, 6) for k, v in fusion_info.get("timings", {}).items()},
        }
        run_meta["benchmark"] = timer.get_report(db_duration)
        run_meta["processing_stats"] = {
            "capture_count": capture_count,
            "vlm_image_count": vlm_image_count,
            "segment_count": segment_count,
        }
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "ok"
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if sync_to_db:
            print("\nSyncing processing outputs to Supabase...")
            db_success = sync_pipeline_results_to_db(
                video_path=Path(video_name or safe_video_name or "video"),
                video_root=video_root,
                run_meta=run_meta,
                duration_sec=db_duration,
                provider="clova",
            )
            if db_success:
                print("Database sync completed.")
            else:
                print("Database sync skipped or failed (check logs above).")

        print("\nProcessing completed.")
        print(f"Outputs: {video_root}")
        print(f"Benchmark: {report_path}")

        # Log to experiment_summary.csv (Added for consistency with run_video_pipeline)
        try:
            from src.dev.utils import ExperimentLogger
            logger = ExperimentLogger()
            token_usage_path = video_root / "fusion" / "token_usage.json"
            judge_json_path = video_root / "fusion" / "judge.json"
            token_data = {}
            
            if token_usage_path.exists():
                with token_usage_path.open("r", encoding="utf-8") as f:
                    token_data = json.load(f)
            
            # Try to read the detailed judge report first
            judge_report_path = video_root / "fusion" / "judge" / "judge_report.json"
            judge_data = {}
            scores = {}
            
            if judge_report_path.exists():
                with judge_report_path.open("r", encoding="utf-8") as f:
                    report_content = json.load(f)
                    judge_data = report_content.get("meta", {})
                    # Add model from meta if present, or fallback
                    if "model" in report_content.get("meta", {}):
                        judge_data["model"] = report_content["meta"]["model"]
                    scores = report_content.get("scores", {})
            elif judge_json_path.exists():
                with judge_json_path.open("r", encoding="utf-8") as f:
                    judge_data = json.load(f)
                scores = {"final": judge_data.get("final_score", 0)}
            
            # token_usage stores arrays, get last entry
            summarizer_tokens = token_data.get("summarizer", [])
            last_summarizer = summarizer_tokens[-1] if summarizer_tokens else {}
            
            # Also try to read judge token usage from verify step
            judge_verify_tokens_path = video_root / "fusion" / "judge" / "token_usage.json"
            last_judge = {}
            if judge_verify_tokens_path.exists():
                try:
                    v_usage = json.loads(judge_verify_tokens_path.read_text(encoding="utf-8"))
                    v_judge = v_usage.get("judge", [])
                    if v_judge:
                       total_j_input = sum(x.get("input_tokens", 0) for x in v_judge)
                       total_j_output = sum(x.get("output_tokens", 0) for x in v_judge)
                       last_judge = {"input_tokens": total_j_input, "output_tokens": total_j_output}
                except:
                    pass

            # Try to read preprocess timings from pipeline_run.json
            run_meta_path = video_root / "pipeline_run.json"
            stt_sec = 0.0
            capture_sec = 0.0
            if run_meta_path.exists():
                try:
                    prev_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                    durations = prev_meta.get("durations_sec", {})
                    stt_sec = durations.get("stt_sec", 0.0)
                    capture_sec = durations.get("capture_sec", 0.0)
                except:
                    pass

            logger.log(
                prompt_version=judge_data.get("prompt_version", "v2"),
                scores=scores,
                input_tokens=last_summarizer.get("input_tokens", 0),
                elapsed_sec=timer.get_total_elapsed(),
                note=f"Process pipeline run: {safe_video_name}",
                model=judge_data.get("model", ""),
                temperature=0.2,
                segments_count=segment_count,
                workers=2,
                command=f"run_process_pipeline --video-name {safe_video_name}",
                component="run_process_pipeline",
                judged=bool(scores),
                # Module timings
                stt_sec=stt_sec,
                capture_sec=capture_sec,
                vlm_sec=vlm_elapsed,
                summarizer_sec=fusion_info.get("timings", {}).get("summarizer_sec", 0.0),
                judge_elapsed_sec=fusion_info.get("timings", {}).get("judge_sec", 0.0),
                judge_input_tokens=last_judge.get("input_tokens", 0),
                judge_output_tokens=last_judge.get("output_tokens", 0)
            )
            print("Logged experiment result to csv.")
        except Exception as e:
            print(f"Warning: Failed to log experiment summary: {e}")

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
        print(f"\nProcessing failed: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Processing pipeline (VLM + Fusion)")
    parser.add_argument("--video-name", default=None, help="Video name (videos.name)")
    parser.add_argument("--video-id", default=None, help="Video ID (videos.id)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--batch-mode", action="store_true", default=False, help="Enable batch mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit segments")
    parser.add_argument("--force-db", action="store_true", help="Force DB download even if local exists")
    parser.add_argument("--sync-to-db", action="store_true", help="Upload processing outputs to Supabase")
    args = parser.parse_args()

    run_processing_pipeline(
        video_name=args.video_name,
        video_id=args.video_id,
        output_base=args.output_base,
        batch_mode=args.batch_mode,
        limit=args.limit,
        sync_to_db=args.sync_to_db,
        force_db=args.force_db,
    )


if __name__ == "__main__":
    main()
