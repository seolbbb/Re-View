"""
VLM + Fusion 처리 파이프라인 엔트리포인트 (Qwen 전용).

Purpose:
    전처리된 아티팩트(STT, 캡처)를 입력으로 받아 VLM 분석과 Fusion(요약+평가)을 수행합니다.
    로컬 아티팩트가 있으면 우선 사용하고, 없거나 force_db가 True이면 스토리지에서 다운로드합니다.

Storage Integration (R2 Priority):
    - 캡처 이미지 다운로드: adapter.s3_client 있으면 R2, 없으면 Supabase Storage
    - 다운로드 경로: {video_id}/{r2_prefix_captures}/{filename}
    - VLM 처리 시 R2 presigned URL 사용 (get_signed_url)

Qwen 환경 변수:
    QWEN_API_KEY_1    (Required, round-robin 지원)
    QWEN_API_KEY_2    (Optional)
    QWEN_API_KEYS     (Optional, comma-separated)
    QWEN_BASE_URL     (Optional, default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1)
    QWEN_MODEL_NAME   (Optional, default: qwen3-vl-32b-instruct)

R2 환경 변수:
    R2_ENDPOINT_URL       (Required for R2)
    R2_ACCESS_KEY_ID      (Required for R2)
    R2_SECRET_ACCESS_KEY  (Required for R2)
    R2_BUCKET_CAPTURES    (Optional, default: review-storage)
    R2_PREFIX_CAPTURES    (Optional, default: captures)

Usage:
    python src/run_process_pipeline.py --video-name sample_video [options]

Arguments:
    --video-name       (Required) 실행할 비디오 폴더명 (data/outputs/{video_name}) 또는 DB의 name
    --video-id         (Optional) DB의 video_id (video_name 대신 사용 가능)
    --batch-mode       (Optional) 배치 처리 모드 활성화 (기본값: False)
    --limit            (Optional) 처리할 최대 세그먼트 수 (테스트용)
    --db-sync          (Optional) 처리 결과를 DB에 업로드
    --force-db         (Optional) 로컬 아티팩트 무시하고 스토리지에서 다운로드

Examples:
    # 기본 실행 (로컬 아티팩트 사용, 단일 모드)
    python src/run_process_pipeline.py --video-name sample4

    # 배치 모드로 실행 (VLM 병렬 처리 최적화)
    python src/run_process_pipeline.py --video-name sample4 --batch-mode

    # DB 강제 다운로드 + 결과 DB 업로드
    python src/run_process_pipeline.py --video-name sample4 --force-db --db-sync

API Connections:
    - Called by: process_api.py (_run_full_pipeline_from_storage)
    - Supabase Tables: videos, captures, stt_results, segments, summaries
    - R2 Storage: captures bucket for image download
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
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

from src.db import get_supabase_adapter, sync_processing_results_to_db, upsert_final_summary_results
from src.db.adapters import compute_config_hash
from src.pipeline.benchmark import BenchmarkTimer, print_benchmark_report
from src.pipeline.cancel import PipelineCanceled, raise_if_cancel_requested
from src.pipeline.stages import (
    generate_fusion_config,
    run_batch_fusion_pipeline,
    run_fusion_pipeline,
    run_vlm_qwen,
    _get_sort_key_timestamp,
)

QWEN_BASE_URL_DEFAULT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_DEFAULT = "qwen3-vl-32b-instruct"


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


def _get_timestamp() -> str:
    """[YYYY-MM-DD | HH:MM:SS.mmm] 형식의 타임스탬프를 반환한다."""
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


def _apply_qwen_vlm_overrides(repo_root: Path) -> Path:
    """Qwen 전용 환경/설정 override를 적용한다."""
    key_candidates = [
        os.getenv("QWEN_API_KEYS", ""),
        os.getenv("QWEN_API_KEY_1", ""),
        os.getenv("QWEN_API_KEY_2", ""),
        os.getenv("QWEN_API_KEY", ""),
    ]
    if not any(candidate.strip() for candidate in key_candidates):
        raise ValueError("QWEN_API_KEY_1 (or QWEN_API_KEYS/QWEN_API_KEY) 환경변수가 설정되지 않았습니다.")

    base_url = os.getenv("QWEN_BASE_URL", QWEN_BASE_URL_DEFAULT)
    model_name = os.getenv("QWEN_MODEL_NAME", QWEN_MODEL_DEFAULT)

    os.environ["QWEN_BASE_URL"] = base_url

    settings_path = repo_root / "config" / "vlm" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"VLM settings file not found: {settings_path}")

    payload = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid VLM settings format (must be a map).")

    payload["model_name"] = model_name

    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"vlm_settings_qwen_{os.getpid()}.yaml"
    temp_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    from src.vlm import vlm_engine as _vlm_engine

    _vlm_engine.SETTINGS_CONFIG_PATH = temp_path
    return temp_path


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
    db_table_name: str = "captures",
    continuous: bool = False,
    poll_interval: int = 10,
) -> None:
    """DB 또는 로컬 입력을 사용해 VLM + Fusion을 실행한다."""
    # 파이프라인 기본 설정을 읽어 CLI 인자에 적용한다.
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("pipeline settings must be a mapping.")

    _apply_qwen_vlm_overrides(ROOT)

    video_settings = settings.get("video", {})
    if not isinstance(video_settings, dict):
        video_settings = {}
    if not video_name or not str(video_name).strip():
        # video_id가 있는 경우 DB에서 이름을 가져옴
        if video_id:
            temp_adapter = get_supabase_adapter()
            if not temp_adapter:
                raise ValueError("Supabase adapter not configured. Check SUPABASE_URL/SUPABASE_KEY.")
            v_data = temp_adapter.client.table("videos").select("name").eq("id", video_id).execute()
            if v_data.data:
                video_name = v_data.data[0].get("name")
                print(f"{_get_timestamp()} [DB] Resolved video_name '{video_name}' for ID {video_id}")
        
        # 여전히 없으면 설정 파일의 process_name 사용
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
        
        adapter = get_supabase_adapter()
        if not adapter:
            raise ValueError("Supabase adapter not configured. Check SUPABASE_URL/SUPABASE_KEY.")

        print(f"{_get_timestamp()} [Input] Using Supabase artifacts ({input_reason}).")

        video_row = None
        # Continuous 모드일 경우 비디오 레코드가 생성될 때까지 대기한다.
        # 전처리가 오래 걸릴 수 있으므로 타임아웃을 넉넉하게 설정 (1시간)
        # 10s * 360 = 3600s = 1h
        max_retries = 360 if continuous else 1
        retry_count = 0
        
        while retry_count < max_retries:
            # video_id가 없으면 name으로 최신 레코드를 찾는다.
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
            
            if video_row:
                break
                
            if continuous:
                if retry_count == 0:
                    print(f"{_get_timestamp()} [Input] Video '{video_name}' not found in DB. Waiting for Preprocessor to create record...")
                time.sleep(poll_interval)
                retry_count += 1
            else:
                break

        # DB에 레코드가 없으면 진행할 수 없다.
        if not video_row:
            target = video_id or video_name
            raise ValueError(f"Video not found in DB (after {retry_count} checks): {target}")

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

        # STT 결과가 생성될 때까지 대기한다 (Continuous 모드일 경우)
        stt_rows = []
        retry_count_stt = 0
        while retry_count_stt < max_retries:
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
            
            if stt_rows:
                break
                
            if continuous:
                if retry_count_stt == 0:
                    print(f"{_get_timestamp()} [Input] STT results not found in DB. Waiting for Preprocessor to finish STT...")
                time.sleep(poll_interval)
                retry_count_stt += 1
            else:
                break

        if not stt_rows:
            raise ValueError(f"stt_results not found in DB (after {retry_count_stt} checks).")

        # DB 스키마 형태에 맞춰 STT 세그먼트 리스트를 구성한다.
        stt_segments: list = []
        first_segments = stt_rows[0].get("segments")
        if isinstance(first_segments, list):
            stt_segments = first_segments
        else:
            for row in stt_rows:
                stt_segments.append(
                    {
                        "id": row.get("stt_id") or row.get("id"),
                        "start_ms": row.get("start_ms"),
                        "end_ms": row.get("end_ms"),
                        "text": row.get("transcript", ""),
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
            adapter.client.table(db_table_name)
            .select("id,cap_id,video_id,file_name,time_ranges,storage_path,preprocess_job_id")
            .eq("video_id", video_id)
        )
        if latest_preprocess_job_id:
            captures_query = captures_query.eq("preprocess_job_id", latest_preprocess_job_id)
        capture_rows = captures_query.execute().data or []
        if not capture_rows and latest_preprocess_job_id:
            capture_rows = (
                adapter.client.table(db_table_name)
                .select("id,cap_id,video_id,file_name,time_ranges,storage_path")
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
        
        # 정렬: time_ranges의 첫 번째 start_ms 기준
        def _get_start_ms(item):
            ranges = item.get("time_ranges")
            if ranges and isinstance(ranges, list) and len(ranges) > 0:
                return ranges[0].get("start_ms") or 0
            return 0

        for row in sorted(capture_rows, key=_get_start_ms):
            file_name = row.get("file_name")
            if not file_name:
                continue
            
            # Manifest 항목 재구성 (cap_id가 있으면 우선 사용)
            manifest_item = {
                "id": row.get("cap_id") or row.get("id"),
                "file_name": file_name,
                "time_ranges": row.get("time_ranges") or [],
            }
            # 호환성을 위해 최상위 start_ms/end_ms도 채워줌 (첫 번째 구간 기준)
            ranges = manifest_item["time_ranges"]
            if ranges:
                manifest_item["start_ms"] = ranges[0].get("start_ms")
                manifest_item["end_ms"] = ranges[0].get("end_ms")
            
            image_path = captures_dir / file_name
            
            # 로컬 파일이 이미 있고 force_db가 아니면 다운로드 스킵 (Manifest에는 추가)
            if image_path.exists() and not force_db:
                manifest_payload.append(manifest_item)
                continue
                
            storage_path = row.get("storage_path")
            if not storage_path:
                print(f"{_get_timestamp()} [Warning] Skipping {file_name}: Missing storage_path in DB")
                continue
            
            try:
                # R2 우선, Supabase fallback
                if adapter.s3_client:
                    import io
                    buffer = io.BytesIO()
                    adapter.s3_client.download_fileobj(adapter.r2_bucket, storage_path, buffer)
                    image_bytes = buffer.getvalue()
                else:
                    if getattr(adapter, "r2_only", False):
                        raise RuntimeError("R2 storage is required (check R2_* env vars)")
                    image_bytes = adapter.client.storage.from_("captures").download(storage_path)
                image_path.write_bytes(image_bytes)
                manifest_payload.append(manifest_item)
            except Exception as e:
                print(f"{_get_timestamp()} [Warning] Skipping {file_name}: Download failed ({e})")

        manifest_json.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        db_duration = video_row.get("duration_sec")
    else:
        print(f"{_get_timestamp()} [Input] Using local artifacts.")

    # 입력 아티팩트 경로가 모두 준비되었는지 확인한다.
    if not (stt_json and manifest_json and captures_dir):
        raise ValueError("Input artifacts are not resolved.")

    # DB Sync가 켜져 있는데 video_id가 없으면, video_name으로 DB에서 조회한다.
    if (sync_to_db or use_db) and not video_id:
        adapter_lookup = get_supabase_adapter()
        if adapter_lookup:
            for candidate in [video_name, safe_video_name]:
                if not candidate:
                    continue
                v_res = (
                    adapter_lookup.client.table("videos")
                    .select("id,name")
                    .eq("name", candidate)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if v_res.data:
                    video_id = v_res.data[0]["id"]
                    print(f"{_get_timestamp()} [DB] Resolved video_id {video_id} for name '{candidate}'")
                    break
            if not video_id:
                print(f"{_get_timestamp()} [DB] Warning: Could not resolve video_id for '{video_name}'. DB sync might fail.")

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
                # Processing 관련 config 파일들의 해시 계산
                config_hash = compute_config_hash([
                    ROOT / "config" / "fusion" / "settings.yaml",
                    ROOT / "config" / "judge" / "settings.yaml",
                    ROOT / "config" / "pipeline" / "settings.yaml",
                    ROOT / "config" / "vlm" / "settings.yaml",
                ])

                job = adapter_for_job.create_processing_job(
                    video_id,
                    triggered_by="MANUAL",
                    config_hash=config_hash,
                )
                processing_job_id = job.get("id")
                if processing_job_id:
                    print(f"{_get_timestamp()} [DB] Created processing_job: {processing_job_id} (config_hash: {config_hash})")
            except Exception as e:
                print(f"{_get_timestamp()} [DB] Warning: Failed to create processing_job: {e}")

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
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # processing_job 상태를 VLM_RUNNING으로 업데이트
    if processing_job_id and adapter_for_job:
        try:
            adapter_for_job.update_processing_job_status(processing_job_id, "VLM_RUNNING")
            # Note: 초기 배치 진행률(current_batch, total_batch)은 run_batch_fusion_pipeline에서
            # 실제 배치 수 계산 후 설정됨. 여기서 설정하지 않음.
        except Exception as e:
            print(f"{_get_timestamp()} [DB] Warning: Failed to update processing_job status: {e}")

    timer.start_total()
    segment_count = 0

    try:
        print(f"\n{_get_timestamp()} Starting processing pipeline (batch_mode={batch_mode})...")
        print("-" * 50)

        # Early cancel (e.g., user deleted the video while a BackgroundTask is running).
        raise_if_cancel_requested(adapter_for_job, video_id)

        if batch_mode:
            current_adapter = adapter_for_job
            if continuous and not current_adapter:
                current_adapter = get_supabase_adapter()
                if not current_adapter:
                    raise ValueError("Supabase adapter is required for continuous monitoring.")

            # Status 업데이트용 콜백
            def status_callback(status: str, current: int, total: int) -> None:
                if processing_job_id and current_adapter:
                    try:
                        current_adapter.update_processing_job_status(processing_job_id, status)
                    except Exception:
                        pass

            # 초기 데이터 로드
            pending_captures = []
            if db_captures_data:
                pending_captures = sorted(db_captures_data, key=lambda x: _get_sort_key_timestamp(x))
            elif manifest_json and manifest_json.exists():
                try:
                    loaded = json.loads(manifest_json.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        pending_captures = sorted(loaded, key=lambda x: _get_sort_key_timestamp(x))
                except Exception:
                    pass
            
            # 처리된 ID 추적 (중복 방지)
            processed_ids = set()
            for c in pending_captures:
                cid = str(c.get("id") or c.get("cap_id") or "")
                if cid: processed_ids.add(cid)

            next_batch_idx = 0
            total_segments_acc = 0
            vlm_elapsed_acc = 0.0

            # 전체 배치 수 미리 계산하여 DB에 설정 (프론트엔드 진행률 표시용)
            total_captures_count = len(pending_captures)
            if total_captures_count > 0 and batch_size > 0:
                import math
                total_batches_estimated = math.ceil(total_captures_count / batch_size)
            else:
                total_batches_estimated = 1

            # DB에 total_batch 초기 설정
            if current_adapter and processing_job_id:
                try:
                    current_adapter.update_processing_job_progress(processing_job_id, 0, total_batches_estimated)
                    print(f"{_get_timestamp()} [DB] Initialized total_batch: 0/{total_batches_estimated}")
                except Exception as e:
                    print(f"{_get_timestamp()} [DB] Warning: Failed to initialize total_batch: {e}")

            print(f"{_get_timestamp()} [{'Continuous' if continuous else 'Batch'} Mode] Started processing loop (captures={total_captures_count}, batches={total_batches_estimated})")

            while True:
                raise_if_cancel_requested(current_adapter, video_id)

                # 1. 전처리 작업 상태 확인 (Continuous 모드일 때만)
                preprocess_done = False
                if continuous and current_adapter and video_id:
                    try:
                        # 최신 전처리 작업 상태 조회
                        pp_res = (
                             current_adapter.client.table("preprocessing_jobs")
                            .select("status")
                            .eq("video_id", video_id)
                            .order("created_at", desc=True)
                            .limit(1)
                            .execute()
                        )
                        # DONE 또는 FAILED일 때도 전처리가 끝난 것으로 간주 (남은 데이터 처리 후 종료)
                        if pp_res.data and pp_res.data[0]["status"] in ("DONE", "FAILED"):
                            preprocess_done = True
                            if pp_res.data[0]["status"] == "FAILED":
                                print(f"{_get_timestamp()} [Pipeline] Preprocessing FAILED (treating as done for remaining items).")
                    except Exception as e:
                        print(f"{_get_timestamp()} [Pipeline] Warning: Check preprocess status failed: {e}")

                # 2. 대기 중인 캡처 처리
                while True:
                    raise_if_cancel_requested(current_adapter, video_id)

                    chunk_size = batch_size
                    # 처리 가능 여부 판단
                    can_process = False
                    
                    if len(pending_captures) >= batch_size:
                        can_process = True
                    elif pending_captures:
                        # 자투리 데이터 처리 조건:
                        # - Continuous 모드가 아님 (무조건 처리)
                        # - Continuous 모드인데 전처리가 끝났음 (마지막 자투리)
                        if not continuous or preprocess_done:
                            chunk_size = min(len(pending_captures), batch_size) # 남은 것 중 배치 크기만큼
                            can_process = True
                    
                    if not can_process:
                        break
                        
                    # 청크 추출 및 실행
                    chunk = pending_captures[:chunk_size]
                    pending_captures = pending_captures[chunk_size:]
                     
                    print(f"\n{_get_timestamp()} [Pipeline] Processing batch {next_batch_idx + 1} (Size: {len(chunk)})")
                    raise_if_cancel_requested(current_adapter, video_id)
                     
                    # 다음 청크가 있으면 그 시작 시간을 강제 종료 시간으로 설정 (Clamping)
                    # 이를 통해 청크 단위로 실행되더라도 시간 범위가 겹치지 않게 함
                    forced_end_ms = None
                    if pending_captures:
                         next_start_item = pending_captures[0]
                         forced_end_ms = _get_sort_key_timestamp(next_start_item)

                    fusion_info = run_batch_fusion_pipeline(
                        video_root=video_root,
                        captures_dir=captures_dir,
                        captures_data=chunk,
                        stt_json=stt_json,
                        video_name=safe_video_name,
                        batch_size=batch_size, # run_batch 내부 로직용
                        timer=timer,
                        vlm_batch_size=vlm_batch_size,
                        vlm_concurrency=vlm_concurrency,
                        vlm_show_progress=vlm_show_progress,
                        limit=limit,
                        repo_root=ROOT,
                        status_callback=status_callback if processing_job_id else None,
                        processing_job_id=processing_job_id,
                        video_id=video_id,
                        sync_to_db=sync_to_db,
                        adapter=current_adapter,
                        start_batch_index=next_batch_idx,
                        preserve_files=True,
                        forced_batch_end_ms=forced_end_ms,
                    )
                    
                    next_batch_idx += 1
                    total_segments_acc += fusion_info.get("segment_count", 0)
                    vlm_elapsed_acc += fusion_info.get("timings", {}).get("vlm_sec", 0.0)

                # 3. 종료 조건 확인 및 폴링
                if not continuous:
                    break
                
                if preprocess_done and not pending_captures:
                    print(f"{_get_timestamp()} [Pipeline] Preprocessing finished and no pending captures. Exiting loop.")
                    break
                
                # 4. 신규 데이터 폴링
                if current_adapter and video_id:
                    print(f"{_get_timestamp()} [Pipeline] Waiting for input... (Pending: {len(pending_captures)} < Batch: {batch_size}, Preprocess: {'RUNNING' if not preprocess_done else 'DONE'})")
                    # print(f"           Sleeping {poll_interval}s...") # 간결한 로그를 위해 생략 혹은 필요한 경우 활성화
                    raise_if_cancel_requested(current_adapter, video_id)
                    time.sleep(poll_interval)
                    try:
                        # captures 테이블에서 video_id로 조회
                        # 최적화: id > last_checked_id 같은 걸 쓰면 좋지만, 여기선 단순히 전체 가져와서 ID로 필터링
                        # (데이터가 아주 많지 않다고 가정)
                        # 또는 created_at 기준으로 가져올 수도 있음
                        caps_res = (
                            current_adapter.client.table(db_table_name)
                            .select("id,cap_id,file_name,time_ranges,storage_path")
                            .eq("video_id", video_id)
                            .execute()
                        )
                        new_items = []
                        if caps_res.data:
                            sorted_rows = sorted(caps_res.data, key=lambda x: _get_sort_key_timestamp(x))
                            for row in sorted_rows:
                                cid = str(row.get("id") or row.get("cap_id") or "")
                                if cid and cid not in processed_ids:
                                    # 새 항목 발견 -> 다운로드 및 추가
                                    file_name = row.get("file_name")
                                    storage_path = row.get("storage_path")
                                    if file_name and storage_path:
                                        img_path = captures_dir / file_name
                                        if not img_path.exists():
                                            # R2 우선, Supabase fallback
                                            if current_adapter.s3_client:
                                                import io
                                                buffer = io.BytesIO()
                                                current_adapter.s3_client.download_fileobj(current_adapter.r2_bucket, storage_path, buffer)
                                                img_bytes = buffer.getvalue()
                                            else:
                                                if getattr(current_adapter, "r2_only", False):
                                                    raise RuntimeError("R2 storage is required (check R2_* env vars)")
                                                img_bytes = current_adapter.client.storage.from_("captures").download(storage_path)
                                            img_path.write_bytes(img_bytes)
                                    
                                    # Manifest Item 구성
                                    m_item = {
                                        "id": row.get("cap_id") or row.get("id"),
                                        "file_name": file_name,
                                        "time_ranges": row.get("time_ranges") or [],
                                        # 호환성 필드
                                        "timestamp_ms": _get_sort_key_timestamp(row),
                                        "start_ms": _get_sort_key_timestamp(row)
                                    }
                                    new_items.append(m_item)
                                    processed_ids.add(cid)
                        
                        if new_items:
                            print(f"{_get_timestamp()} [Loop] Found {len(new_items)} new captures.")
                            pending_captures.extend(new_items)
                            # 다시 정렬
                            pending_captures.sort(key=lambda x: _get_sort_key_timestamp(x))
                            
                    except Exception as e:
                        print(f"{_get_timestamp()} [Loop] Warning: Fetch new captures failed: {e}")

            # 루프 종료 후 통계 정리
            segment_count = total_segments_acc
            vlm_image_count = len(processed_ids)
            vlm_elapsed = vlm_elapsed_acc
            fusion_info = {} # 마지막 fusion_info 정보는 의미가 퇴색되므로 초기화 혹은 마지막 값 사용
        else:
            # 단일 모드에서는 VLM 실행 후 Fusion으로 넘어간다.
            raise_if_cancel_requested(adapter_for_job, video_id)

            vlm_image_count, vlm_elapsed = timer.time_stage(
                "vlm",
                run_vlm_qwen,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                video_name=safe_video_name,
                output_base=output_base_path,
                batch_size=vlm_batch_size,
                concurrency=vlm_concurrency,
                show_progress=vlm_show_progress,
            )
            raise_if_cancel_requested(adapter_for_job, video_id)

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
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Don't upload or mark DONE after delete is requested.
        raise_if_cancel_requested(adapter_for_job, video_id)

        if sync_to_db and processing_job_id and adapter_for_job:
            # Final summary_results UPSERT (timeline, tldr 포맷 저장)
            print(f"\n{_get_timestamp()} [DB] Upserting final summary results...")
            results_dir = video_root / "results"
            summaries_path = video_root / "fusion" / "segment_summaries.jsonl"
            try:
                upsert_result = upsert_final_summary_results(
                    adapter_for_job,
                    video_id,
                    processing_job_id,
                    summaries_path,
                    results_dir,
                )
                if upsert_result.get("saved"):
                    for fmt, result_id in upsert_result["saved"].items():
                        print(f"{_get_timestamp()}   [DB] summary_results ({fmt}): {result_id}")
                if upsert_result.get("errors"):
                    for err in upsert_result["errors"]:
                        print(f"{_get_timestamp()}   [DB] Warning: {err}")
            except Exception as e:
                print(f"{_get_timestamp()} [DB] Warning: Failed to upsert final summary results: {e}")

            # 배치별 업로드가 이미 완료되었으므로 기존 sync는 스킵 가능
            # 하지만 단일 모드나 fallback을 위해 유지
            if not batch_mode:
                print(f"\n{_get_timestamp()} Syncing processing outputs to Supabase (Processing Jobs)...")
                db_results = sync_processing_results_to_db(
                    video_root=video_root,
                    video_id=video_id,
                    processing_job_id=processing_job_id,
                )
                saved = db_results.get("saved", {})
                errors = db_results.get("errors", [])

                print(f"{_get_timestamp()} [DB] Upload Summary (Processing Job {processing_job_id}):")
                for k, v in saved.items():
                    print(f"{_get_timestamp()}   - {k}: {v} records")

                if errors:
                    print(f"{_get_timestamp()} [DB] Completed with {len(errors)} errors:")
                    for e in errors:
                        print(f"{_get_timestamp()}   - {e}")
                else:
                    print(f"{_get_timestamp()} [DB] Processing artifacts uploaded successfully.")
            else:
                print(f"{_get_timestamp()} [DB] Batch mode: artifacts already uploaded during pipeline execution.")

        # processing_job 상태를 DONE으로 업데이트
        if processing_job_id and adapter_for_job:
            try:
                adapter_for_job.update_processing_job_status(processing_job_id, "DONE")
                # 배치 모드일 경우 total_batch 사용, 아니면 1로 설정
                total_batches = fusion_info.get("batch_count", 1) if batch_mode else 1
                adapter_for_job.update_processing_job_progress(processing_job_id, total_batches, total_batches)
                print(f"{_get_timestamp()} [DB] processing_job {processing_job_id} marked as DONE")
            except Exception as e:
                print(f"{_get_timestamp()} [DB] Warning: Failed to update processing_job status: {e}")

        print(f"\n{_get_timestamp()} Processing completed.")
        # [User Request] 로컬 경로를 상대 경로로 표시
        rel_video_root = os.path.relpath(video_root, ROOT)
        rel_report_path = os.path.relpath(report_path, ROOT)
        print(f"{_get_timestamp()} Outputs: {rel_video_root}")
        print(f"{_get_timestamp()} Benchmark: {rel_report_path}")

    except PipelineCanceled as exc:
        # Canceled by user deletion request; exit quietly (no re-raise).
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "canceled"
        run_meta["error"] = str(exc)
        run_meta.setdefault("durations_sec", {})
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if processing_job_id and adapter_for_job:
            try:
                adapter_for_job.update_processing_job_status(
                    processing_job_id,
                    "FAILED",
                    error_message=f"canceled: {exc}",
                )
            except Exception:
                pass

        print(f"\n{_get_timestamp()} Processing canceled: {exc}")
        return

    except Exception as exc:
        # 에러 발생 시 상태를 기록한 뒤 예외를 다시 올린다.
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        
        # processing_job 상태를 FAILED로 업데이트
        if processing_job_id and adapter_for_job:
            try:
                adapter_for_job.update_processing_job_status(
                    processing_job_id, "FAILED", error_message=str(exc)
                )
                print(f"{_get_timestamp()} [DB] processing_job {processing_job_id} marked as FAILED")
            except Exception:
                pass
        print(f"\n{_get_timestamp()} Processing failed: {exc}")
        raise


def get_parser() -> argparse.ArgumentParser:
    """처리 파이프라인용 ArgumentParser를 생성해 반환한다."""
    parser = argparse.ArgumentParser(description="Processing pipeline (DashScope VLM + Fusion)")
    parser.add_argument("--video", default=None, help="Input video file path (for unifying with preprocess CLI)")
    parser.add_argument("--video-name", default=None, help="Video name (videos.name)")
    parser.add_argument("--video-id", default=None, help="Video ID (videos.id)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--batch-mode", dest="batch_mode", action="store_true", help="Enable batch mode")
    parser.add_argument("--no-batch-mode", dest="batch_mode", action="store_false", help="Disable batch mode")
    parser.set_defaults(batch_mode=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Processing size per batch (default: 10)")
    parser.add_argument("--limit", type=int, default=None, help="Limit segments")
    parser.add_argument("--force-db", dest="force_db", action="store_true", help="Force DB download even if local exists")
    parser.add_argument("--no-force-db", dest="force_db", action="store_false", help="Disable DB download")
    parser.set_defaults(force_db=None)
    parser.add_argument("--db-sync", dest="db_sync", action="store_true", help="Upload processing outputs to Supabase")
    parser.add_argument("--no-db-sync", dest="db_sync", action="store_false", help="Skip Supabase upload")
    parser.set_defaults(db_sync=None)
    parser.add_argument("--db-table", default="captures", help="DB table name for captures (default: captures)")
    parser.add_argument("--continuous", dest="continuous", action="store_true", help="Enable continuous monitoring mode")
    parser.add_argument("--poll-interval", type=int, default=10, help="Poll interval in seconds for continuous mode")
    return parser


def main() -> None:
    """CLI 인자를 파싱하고 처리 파이프라인을 실행한다."""
    parser = get_parser()
    args = parser.parse_args()

    # --video가 있으면 video_name을 추출하여 사용
    video_name = args.video_name
    if args.video and not video_name:
        video_name = Path(args.video).stem

    run_processing_pipeline(
        video_name=video_name,
        video_id=args.video_id,
        output_base=args.output_base,
        batch_mode=args.batch_mode,
        batch_size=args.batch_size,
        limit=args.limit,
        sync_to_db=args.db_sync,
        force_db=args.force_db,
        db_table_name=args.db_table,
        continuous=args.continuous,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
