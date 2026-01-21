"""
STT + 캡처 전처리 파이프라인 엔트리포인트 (선택적 DB 동기화 포함).

영상 파일을 입력받아 STT 추출 결과와 프레임 캡처 이미지를 생성합니다.
로컬 실행(`--local-json`) 시 `data/outputs/{video_name}`에 결과물이 저장됩니다.

Usage:
    python src/run_preprocess_pipeline.py --video data/input/sample.mp4 [options]

Arguments:
    --video            (Required) 입력 비디오 파일 경로
    --output-base      (Optional) 출력 루트 디렉토리 (기본값: data/outputs)
    --parallel         (Optional) STT와 캡처 병렬 실행 여부 (기본값: True)
    --local-json       (Optional) 로컬에 JSON 아티팩트 저장 여부 (기본값: config 설정 따름)
    --db-sync          (Optional) Supabase DB 동기화 여부

Examples:
    # 기본 실행 (로컬 JSON 생성 + 병렬 처리)
    python src/run_preprocess_pipeline.py --video data/input/sample4.mp4 --local-json

    # DB 동기화 없이 로컬 생성만 수행
    python src/run_preprocess_pipeline.py --video data/input/sample4.mp4 --local-json --no-db-sync
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
from dotenv import load_dotenv
import yaml

# 스크립트 실행 시 로컬 import가 동작하도록 레포 루트를 설정한다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# API 키와 로컬 설정을 위해 환경 변수를 로드한다.
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from src.capture.settings import load_capture_settings
from src.db.pipeline_sync import (
    finalize_preprocess_db_sync,
    prepare_preprocess_db_sync,
    sync_preprocess_artifacts_to_db,
)
from src.pipeline.benchmark import BenchmarkTimer, format_duration, get_video_info, print_benchmark_report
from src.pipeline.stages import run_capture, run_stt


def _sanitize_video_name(stem: str) -> str:
    """파일명 stem을 안전한 출력 폴더명으로 정규화한다."""
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"

def _append_benchmark_report(path: Path, report_md: str, pipeline_label: str) -> None:
    """기존 리포트가 있으면 구분선+타임스탬프로 이어 붙인다."""
    timestamp = datetime.now(timezone.utc).isoformat()
    if path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n---\n")
            handle.write(f"Benchmark Append: {pipeline_label} | {timestamp}\n\n")
            handle.write(report_md)
    else:
        path.write_text(report_md, encoding="utf-8")


def run_preprocess_pipeline(
    *,
    video: Optional[str],
    output_base: str = "data/outputs",
    stt_backend: str = "clova",
    parallel: bool = True,
    capture_threshold: Optional[float] = None,
    capture_dedupe_threshold: Optional[float] = None,
    capture_min_interval: Optional[float] = None,
    capture_verbose: bool = False,
    limit: Optional[int] = None,
    write_local_json: Optional[bool] = None,
    sync_to_db: Optional[bool] = None,
) -> None:
    """STT + Capture를 실행하고 입력 산출물까지만 생성한다."""
    # CLI 인자로 덮어쓸 수 있는 기본 설정을 불러온다.
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("pipeline settings must be a mapping.")

    video_value = video
    if not video_value or not str(video_value).strip():
        video_settings = settings.get("video", {})
        if isinstance(video_settings, dict):
            video_value = video_settings.get("preprocess_path")
    if not video_value or not str(video_value).strip():
        raise ValueError("video path is required (CLI --video or config/pipeline/settings.yaml: video.preprocess_path)")

    # 부분 출력이 생기기 전에 입력 경로를 먼저 확정한다.
    video_path = Path(str(video_value)).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 캡처 설정은 config/capture/settings.yaml에서 기본값을 가져온다.
    capture_settings = load_capture_settings()
    if capture_threshold is None:
        capture_threshold = float(capture_settings.sensitivity_diff)
    if capture_dedupe_threshold is None:
        capture_dedupe_threshold = float(capture_settings.sensitivity_sim)
    if capture_min_interval is None:
        capture_min_interval = float(capture_settings.min_interval)

    db_settings = settings.get("db", {})
    if not isinstance(db_settings, dict):
        db_settings = {}
    preprocess_settings = settings.get("preprocess", {})
    if not isinstance(preprocess_settings, dict):
        preprocess_settings = {}
    if sync_to_db is None:
        sync_to_db = db_settings.get("sync_to_db_preprocess")
        if sync_to_db is None:
            sync_to_db = True
    if not isinstance(sync_to_db, bool):
        sync_to_db = True
    if write_local_json is None:
        write_local_json = preprocess_settings.get("write_local_json")
        if write_local_json is None:
            write_local_json = True
    if not isinstance(write_local_json, bool):
        write_local_json = True

    # 이번 비디오 실행에 대한 출력 루트를 만든다.
    output_base_path = (ROOT / Path(output_base)).resolve()
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base_path / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    # 벤치마크와 로그용 메타데이터를 수집한다.
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
        "write_local_json": write_local_json,
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
    # 진행 중에도 확인할 수 있도록 메타데이터를 먼저 저장한다.
    run_meta_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    db_context: Optional[Tuple[Any, str, Optional[str]]] = None
    db_errors: List[str] = []
    printed_db_errors: set[str] = set()
    if sync_to_db:
        db_context = prepare_preprocess_db_sync(
            video_path=video_path,
            video_root=video_root,
            run_meta=run_meta,
            duration_sec=video_info.get("duration_sec"),
        )
        if db_context:
            _, _, pipeline_run_id = db_context
            if pipeline_run_id:
                run_meta["pipeline_run_id"] = pipeline_run_id
            print("\nSyncing preprocessing artifacts to Supabase as stages complete...")

    timer.start_total()
    capture_count = 0

    try:
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "manifest.json"

        print(f"\nStarting preprocessing (parallel={parallel})...")
        print("-" * 50)

        stt_elapsed = 0.0
        capture_elapsed = 0.0

        def _sync_stage(
            stage: str,
            *,
            stt_payload: Optional[Dict[str, Any]] = None,
            captures_payload: Optional[List[Dict[str, Any]]] = None,
        ) -> None:
            if not db_context:
                return
            adapter, video_id, pipeline_run_id = db_context
            stage_results = sync_preprocess_artifacts_to_db(
                adapter=adapter,
                video_id=video_id,
                video_root=video_root,
                provider=stt_backend,
                pipeline_run_id=pipeline_run_id,
                include_stt=stage == "stt",
                include_captures=stage == "capture",
                stt_payload=stt_payload if stage == "stt" else None,
                captures_payload=captures_payload if stage == "capture" else None,
            )
            saved = stage_results.get("saved", {})
            errors = stage_results.get("errors", [])
            if saved:
                summary = ", ".join(f"{key}={value}" for key, value in saved.items())
                print(f"[DB] {stage} upload: {summary}")
            if errors:
                print(f"[DB] {stage} upload had {len(errors)} errors.")
                for err in errors:
                    if err in printed_db_errors:
                        continue
                    printed_db_errors.add(err)
                    print(f"[DB] {stage} error: {err}")
                db_errors.extend(errors)

        if parallel:
            # STT와 캡처를 동시에 실행한다.
            with ThreadPoolExecutor(max_workers=2) as executor:
                def run_stt_timed() -> Tuple[Dict[str, Any], float]:
                    # 벤치마크를 위해 STT 실행 시간을 측정한다.
                    start = time.perf_counter()
                    stt_payload = run_stt(
                        video_path,
                        stt_json,
                        backend=stt_backend,
                        write_output=write_local_json,
                    )
                    return stt_payload, time.perf_counter() - start

                def run_capture_timed() -> Tuple[List[Dict[str, Any]], float]:
                    # 캡처 실행 시간을 측정하고 결과를 반환한다.
                    start = time.perf_counter()
                    result = run_capture(
                        video_path,
                        output_base_path,
                        threshold=capture_threshold,
                        dedupe_threshold=capture_dedupe_threshold,
                        min_interval=capture_min_interval,
                        verbose=capture_verbose,
                        video_name=video_name,
                        write_manifest=write_local_json,
                    )
                    return result, time.perf_counter() - start

                stt_future = executor.submit(run_stt_timed)
                capture_future = executor.submit(run_capture_timed)

                futures = {
                    stt_future: "stt",
                    capture_future: "capture",
                }
                for future in as_completed(futures):
                    stage = futures[future]
                    if stage == "stt":
                        stt_payload, stt_elapsed = future.result()
                        timer.record_stage("stt", stt_elapsed)
                        print(f"  STT done in {format_duration(stt_elapsed)} (parallel)")
                        _sync_stage("stt", stt_payload=stt_payload)
                    else:
                        capture_result, capture_elapsed = future.result()
                        capture_count = len(capture_result) if capture_result else 0
                        timer.record_stage("capture", capture_elapsed)
                        print(f"  Capture done in {format_duration(capture_elapsed)} (parallel)")
                        _sync_stage("capture", captures_payload=capture_result)
        else:
            # 동일한 타이밍 측정을 유지하며 순차 실행한다.
            stt_payload, stt_elapsed = timer.time_stage(
                "stt",
                run_stt,
                video_path,
                stt_json,
                backend=stt_backend,
                write_output=write_local_json,
            )
            _sync_stage("stt", stt_payload=stt_payload)
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
                write_manifest=write_local_json,
            )
            capture_count = len(capture_result) if capture_result else 0
            _sync_stage("capture", captures_payload=capture_result)

        timer.end_total()

        # 사람이 읽을 수 있는 벤치마크 리포트를 함께 저장한다.
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
        _append_benchmark_report(report_path, md_report, "Preprocess")

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
        # 선택적 DB 동기화 전에 최종 메타데이터를 저장한다.
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if sync_to_db:
            if db_context:
                try:
                    adapter, video_id, pipeline_run_id = db_context
                    finalize_preprocess_db_sync(
                        adapter=adapter,
                        video_id=video_id,
                        pipeline_run_id=pipeline_run_id,
                        run_meta=run_meta,
                        errors=db_errors,
                    )
                except Exception as exc:
                    db_errors.append(f"finalize: {exc}")

                if db_errors:
                    print(f"Database sync completed with {len(db_errors)} errors.")
                    for err in db_errors:
                        if err in printed_db_errors:
                            continue
                        printed_db_errors.add(err)
                        print(f"[DB] error: {err}")
                else:
                    print("Database sync completed.")
            else:
                print("Database sync skipped or failed (check logs above).")

        print("\nPreprocessing completed.")
        print(f"Outputs: {video_root}")
        print(f"Benchmark: {report_path}")

    except Exception as exc:
        # 재발생 전에 실패 메타데이터를 항상 기록한다.
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if sync_to_db and db_context:
            try:
                adapter, video_id, pipeline_run_id = db_context
                db_errors.append(f"pipeline: {exc}")
                finalize_preprocess_db_sync(
                    adapter=adapter,
                    video_id=video_id,
                    pipeline_run_id=pipeline_run_id,
                    run_meta=run_meta,
                    errors=db_errors,
                )
            except Exception:
                pass
        print(f"\nPreprocessing failed: {exc}")
        raise


def main() -> None:
    """CLI 인자를 파싱하고 전처리 파이프라인을 실행한다."""
    parser = argparse.ArgumentParser(description="Preprocess pipeline (STT + Capture only)")
    parser.add_argument("--video", default=None, help="Input video file path (or config/pipeline/settings.yaml)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--stt-backend", choices=["clova"], default="clova", help="STT backend")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run STT and Capture in parallel",
    )
    parser.add_argument("--capture-verbose", action="store_true", help="Enable capture logs")
    parser.add_argument(
        "--local-json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write preprocess JSON artifacts to disk",
    )
    parser.add_argument("--db-sync", dest="db_sync", action="store_true", help="Enable Supabase sync")
    parser.add_argument("--no-db-sync", dest="db_sync", action="store_false", help="Skip Supabase sync")
    parser.set_defaults(db_sync=None)
    args = parser.parse_args()

    run_preprocess_pipeline(
        video=args.video,
        output_base=args.output_base,
        stt_backend=args.stt_backend,
        parallel=args.parallel,
        capture_verbose=args.capture_verbose,
        write_local_json=args.local_json,
        sync_to_db=args.db_sync,
    )


if __name__ == "__main__":
    main()
