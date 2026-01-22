"""VLM + Fusion 처리 파이프라인 엔트리포인트 (DB/로컬 입력 사용)."""

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

from src.db import get_supabase_adapter, sync_processing_results_to_db
from src.pipeline.benchmark import BenchmarkTimer, print_benchmark_report
from src.pipeline.stages import (
    generate_fusion_config,
    run_batch_fusion_pipeline,
    run_fusion_pipeline,
    run_vlm_openrouter,
)


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


def run_processing_pipeline(
    *,
    video_name: Optional[str],
    video_id: Optional[str] = None,
    output_base: str = "data/outputs",
    batch_mode: Optional[bool] = None,
    batch_size: Optional[int] = None,
    vlm_batch_size: Optional[int] = None,
    vlm_concurrency: Optional[int] = None,
    vlm_show_progress: Optional[bool] = None,
    limit: Optional[int] = None,
    sync_to_db: Optional[bool] = None,
    force_db: Optional[bool] = None,
    use_db: Optional[bool] = None,
) -> None:
    """DB 또는 로컬 입력을 사용해 VLM + Fusion을 실행한다."""
    # 파이프라인 기본 설정을 읽어 CLI 인자에 적용한다.
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("pipeline settings must be a mapping.")

    video_settings = settings.get("video", {})
    if not isinstance(video_settings, dict):
        video_settings = {}
    if not video_name or not str(video_name).strip():
        candidate = video_settings.get("process_name")
        if candidate and str(candidate).strip():
            video_name = str(candidate).strip()

    if not video_name and not video_id:
        raise ValueError("video_name or video_id is required (CLI or config/pipeline/settings.yaml).")

    # 명시적으로 전달되지 않은 값만 기본값을 채운다.
    if batch_mode is None:
        batch_mode = settings.get("batch_mode", False)
    if batch_size is None:
        batch_size = settings.get("batch_size", 10)
    if vlm_batch_size is None:
        vlm_batch_size = settings.get("vlm_batch_size", 2)
    if vlm_concurrency is None:
        vlm_concurrency = settings.get("vlm_concurrency", 3)
    if vlm_show_progress is None:
        vlm_show_progress = settings.get("vlm_show_progress", True)

    db_settings = settings.get("db", {})
    if not isinstance(db_settings, dict):
        db_settings = {}
    if use_db is None:
        use_db = db_settings.get("use_db", True)
    if force_db is None:
        force_db = db_settings.get("force_db", False)
    if sync_to_db is None:
        sync_to_db = db_settings.get("sync_to_db_process", db_settings.get("sync_to_db"))
        if sync_to_db is None:
            sync_to_db = False
    include_preprocess_in_process_sync = db_settings.get(
        "include_preprocess_in_process_sync", False
    )
    if not isinstance(use_db, bool):
        use_db = True
    if not isinstance(force_db, bool):
        force_db = False
    if not isinstance(sync_to_db, bool):
        sync_to_db = False
    if not isinstance(include_preprocess_in_process_sync, bool):
        include_preprocess_in_process_sync = False

    # 출력 경로와 안전한 영상 이름을 계산한다.
    output_base_path = (ROOT / Path(output_base)).resolve()
    safe_video_name = _sanitize_video_name(video_name) if video_name else None
    video_root = output_base_path / safe_video_name if safe_video_name else None

    # 로컬 전처리 산출물 경로를 준비한다.
    stt_json = video_root / "stt.json" if video_root else None
    captures_dir = video_root / "captures" if video_root else None
    manifest_json = video_root / "manifest.json" if video_root else None

    # 로컬에 필요한 입력이 모두 있는지 확인한다.
    local_ready = (
        stt_json
        and manifest_json
        and captures_dir
        and stt_json.exists()
        and manifest_json.exists()
        and captures_dir.exists()
    )

    input_source = "local"
    input_reason = "local artifacts present"
    if force_db:
        input_source = "db"
        input_reason = "forced"
    elif not local_ready:
        input_source = "db"
        input_reason = "local artifacts missing"

    # DB에서 가져온 경우 duration 정보를 보존한다.
    db_duration = None
    db_captures_data = None  # DB에서 가져온 captures 데이터 (sync_engine에서 직접 사용)
    if force_db:
        use_db = True
    if force_db or not local_ready:
        if not use_db:
            raise ValueError("DB usage is disabled and local artifacts are missing.")
        print(f"[Input] Using Supabase artifacts ({input_reason}).")
        # Supabase 설정이 없으면 DB 모드를 사용할 수 없다.
        adapter = get_supabase_adapter()
        if not adapter:
            raise ValueError("Supabase adapter not configured. Check SUPABASE_URL/SUPABASE_KEY.")

        # video_id가 없으면 name으로 최신 레코드를 찾는다.
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

        # DB에 레코드가 없으면 진행할 수 없다.
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
        manifest_json = video_root / "manifest.json"
        video_root.mkdir(parents=True, exist_ok=True)

        # 최신 전처리 작업 ID가 있으면 그 결과를 우선 사용한다.
        latest_preprocess_job_id = None
        preprocess_result = (
            adapter.client.table("preprocessing_jobs")
            .select("id,created_at")
            .eq("video_id", video_id)
            .eq("status", "DONE")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        preprocess_rows = preprocess_result.data or []
        if preprocess_rows:
            latest_preprocess_job_id = preprocess_rows[0].get("id")

        # STT 결과를 우선 최신 전처리 작업 기준으로 가져온다.
        stt_query = adapter.client.table("stt_results").select("*").eq("video_id", video_id)
        if latest_preprocess_job_id:
            stt_query = stt_query.eq("preprocess_job_id", latest_preprocess_job_id)
        stt_rows = stt_query.execute().data or []
        if not stt_rows and latest_preprocess_job_id:
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

        # DB 스키마 형태에 맞춰 STT 세그먼트 리스트를 구성한다.
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
            .select("file_name,start_ms,end_ms,storage_path,preprocess_job_id")
            .eq("video_id", video_id)
        )
        if latest_preprocess_job_id:
            captures_query = captures_query.eq("preprocess_job_id", latest_preprocess_job_id)
        capture_rows = captures_query.execute().data or []
        if not capture_rows and latest_preprocess_job_id:
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
        
        # DB에서 가져온 captures 데이터 저장 (sync_engine에서 직접 사용)
        db_captures_data = capture_rows

        # 캡처 파일과 manifest를 로컬에 재구성한다.
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
            # 스토리지에서 캡처 이미지를 다운로드한다.
            image_bytes = adapter.client.storage.from_("captures").download(storage_path)
            image_path.write_bytes(image_bytes)

        manifest_json.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        db_duration = video_row.get("duration_sec")
    else:
        print("[Input] Using local artifacts.")

    # 입력 아티팩트 경로가 모두 준비되었는지 확인한다.
    if not (stt_json and manifest_json and captures_dir):
        raise ValueError("Input artifacts are not resolved.")

    # 캡처 개수를 로딩해 리포트에 반영한다.
    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    capture_count = len(manifest_payload) if isinstance(manifest_payload, list) else 0

    # processing_jobs 레코드 생성 (DB 사용 시)
    processing_job_id = None
    adapter_for_job = None
    if sync_to_db or use_db:
        adapter_for_job = get_supabase_adapter()
        if adapter_for_job and video_id:
            try:
                job = adapter_for_job.create_processing_job(
                    video_id,
                    triggered_by="MANUAL",
                )
                processing_job_id = job.get("id")
                if processing_job_id:
                    print(f"[DB] Created processing_job: {processing_job_id}")
            except Exception as e:
                print(f"[DB] Warning: Failed to create processing_job: {e}")

    # 메타데이터와 타이머를 초기화한다.
    timer = BenchmarkTimer()
    run_meta_path = video_root / "pipeline_run.json"
    run_args = {
        "pipeline_type": "process",
        "video_name": video_name or safe_video_name,
        "video_id": video_id,
        "processing_job_id": processing_job_id,
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
        "processing_job_id": processing_job_id,
        "video_root": str(video_root),
        "output_base": str(output_base_path),
        "input_source": input_source,
        "input_reason": input_reason,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": run_args,
        "durations_sec": {},
        "benchmark": {},
        "status": "running",
    }
    # 진행 중 상태를 먼저 기록해 상태 조회가 가능하도록 한다.
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # processing_job 상태를 VLM_RUNNING으로 업데이트
    if processing_job_id and adapter_for_job:
        try:
            adapter_for_job.update_processing_job_status(processing_job_id, "VLM_RUNNING")
            adapter_for_job.update_processing_job_progress(processing_job_id, 0, capture_count)
        except Exception as e:
            print(f"[DB] Warning: Failed to update processing_job status: {e}")

    timer.start_total()
    segment_count = 0

    try:
        print(f"\nStarting processing pipeline (batch_mode={batch_mode})...")
        print("-" * 50)

        if batch_mode:
            # Status 업데이트용 콜백 정의
            def status_callback(status: str, current: int, total: int) -> None:
                if processing_job_id and adapter_for_job:
                    try:
                        adapter_for_job.update_processing_job_status(processing_job_id, status)
                    except Exception as e:
                        print(f"[DB] Warning: Failed to update status to {status}: {e}")
            
            # 배치 모드에서는 VLM+Fusion을 배치 단위로 처리한다.
            fusion_info = run_batch_fusion_pipeline(
                video_root=video_root,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                captures_data=db_captures_data,  # DB에서 가져온 captures 직접 전달
                stt_json=stt_json,
                video_name=safe_video_name,
                batch_size=batch_size,
                timer=timer,
                vlm_batch_size=vlm_batch_size,
                vlm_concurrency=vlm_concurrency,
                vlm_show_progress=vlm_show_progress,
                limit=limit,
                repo_root=ROOT,
                status_callback=status_callback if processing_job_id else None,
            )
            segment_count = fusion_info.get("segment_count", 0)
            vlm_image_count = capture_count
            vlm_elapsed = fusion_info["timings"].get("vlm_sec", 0.0)
        else:
            # 단일 모드에서는 VLM 실행 후 Fusion으로 넘어간다.
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

            # Fusion 설정 파일을 생성해 파이프라인에 전달한다.
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

            # Fusion 단계에서 요약 및 결과물을 생성한다.
            fusion_info = run_fusion_pipeline(
                fusion_config_path,
                limit=limit,
                timer=timer,
            )
            segment_count = fusion_info.get("segment_count", 0)

        timer.end_total()

        # 처리 결과를 벤치마크 리포트로 저장한다.
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
        _append_benchmark_report(report_path, md_report, "Process")

        # 실행 통계와 상태를 기록한다.
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
        # 최종 메타데이터를 저장한다.
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if sync_to_db and processing_job_id:
            # 선택적으로 결과물을 DB에 업로드한다.
            print("\nSyncing processing outputs to Supabase (Processing Jobs)...")
            db_results = sync_processing_results_to_db(
                video_root=video_root,
                video_id=video_id,
                processing_job_id=processing_job_id,
            )
            saved = db_results.get("saved", {})
            errors = db_results.get("errors", [])
            
            print(f"[DB] Upload Summary (Processing Job {processing_job_id}):")
            for k, v in saved.items():
                print(f"  - {k}: {v} records")
            
            if errors:
                print(f"[DB] ⚠️ Completed with {len(errors)} errors:")
                for e in errors:
                    print(f"  - {e}")
            else:
                print("[DB] ✅ Processing artifacts uploaded successfully.")

        # processing_job 상태를 DONE으로 업데이트
        if processing_job_id and adapter_for_job:
            try:
                adapter_for_job.update_processing_job_status(processing_job_id, "DONE")
                adapter_for_job.update_processing_job_progress(processing_job_id, capture_count, capture_count)
                print(f"[DB] processing_job {processing_job_id} marked as DONE")
            except Exception as e:
                print(f"[DB] Warning: Failed to update processing_job status: {e}")

        print("\nProcessing completed.")
        print(f"Outputs: {video_root}")
        print(f"Benchmark: {report_path}")

    except Exception as exc:
        # 에러 발생 시 상태를 기록한 뒤 예외를 다시 올린다.
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        
        # processing_job 상태를 FAILED로 업데이트
        if processing_job_id and adapter_for_job:
            try:
                adapter_for_job.update_processing_job_status(
                    processing_job_id, "FAILED", error_message=str(exc)
                )
                print(f"[DB] processing_job {processing_job_id} marked as FAILED")
            except Exception:
                pass
        print(f"\nProcessing failed: {exc}")
        raise


def main() -> None:
    """CLI 인자를 파싱하고 처리 파이프라인을 실행한다."""
    parser = argparse.ArgumentParser(description="Processing pipeline (VLM + Fusion)")
    parser.add_argument("--video-name", default=None, help="Video name (videos.name)")
    parser.add_argument("--video-id", default=None, help="Video ID (videos.id)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--batch-mode", dest="batch_mode", action="store_true", help="Enable batch mode")
    parser.add_argument("--no-batch-mode", dest="batch_mode", action="store_false", help="Disable batch mode")
    parser.set_defaults(batch_mode=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit segments")
    parser.add_argument("--force-db", dest="force_db", action="store_true", help="Force DB download even if local exists")
    parser.add_argument("--no-force-db", dest="force_db", action="store_false", help="Disable DB download")
    parser.set_defaults(force_db=None)
    parser.add_argument("--sync-to-db", dest="sync_to_db", action="store_true", help="Upload processing outputs to Supabase")
    parser.add_argument("--no-sync-to-db", dest="sync_to_db", action="store_false", help="Skip Supabase upload")
    parser.set_defaults(sync_to_db=None)
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
