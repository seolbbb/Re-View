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
import os
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
from src.db.adapters import compute_config_hash
from src.pipeline.benchmark import BenchmarkTimer, format_duration, get_video_info, print_benchmark_report
from src.pipeline.cancel import PipelineCanceled, is_local_cancel_requested, raise_if_cancel_requested
from src.pipeline.logger import pipeline_logger
from src.pipeline.stages import run_capture, run_stt, run_stt_only, run_stt_from_storage
from src.audio.extract_audio import extract_audio


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
    capture_dedup_enabled: bool = True,
    capture_verbose: bool = False,
    limit: Optional[int] = None,
    write_local_json: Optional[bool] = None,
    sync_to_db: Optional[bool] = None,
    db_table_name: str = "captures",
    existing_video_id: Optional[str] = None,
) -> Optional[str]:
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
        # 입력된 경로에 없으면 data/inputs 폴더를 찾아본다.
        fallback_inputs = ROOT / "data" / "inputs" / video_value
        if fallback_inputs.exists():
            video_path = fallback_inputs.resolve()
            # [User Request] 로컬 절대 경로 제거
            rel_video_path = os.path.relpath(video_path, ROOT)
            print(f"[Info] Resolved video path from data/inputs: {rel_video_path}")
        else:
             print(f"\n[Error] Video file not found: {video_value}")
             print(f"Please check if the file exists in 'data/inputs/' directory.")
             # [User Request] 로컬 절대 경로 제거
             rel_input_dir = os.path.relpath(ROOT / 'data' / 'inputs', ROOT)
             print(f"Standard Input Directory: {rel_input_dir}")
             raise FileNotFoundError(f"Video file not found: {video_value} (checked data/inputs/)")

    # 캡처 설정은 config/capture/settings.yaml에서 기본값을 가져온다.
    capture_settings = load_capture_settings()
    if capture_threshold is None:
        capture_threshold = float(capture_settings.persistence_drop_ratio)
    if capture_dedupe_threshold is None:
        capture_dedupe_threshold = float(capture_settings.dedup_sim_threshold)
    if capture_min_interval is None:
        capture_min_interval = float(capture_settings.min_interval)

    # 캡처 모드 로깅
    mode_str = "DEDUPLICATION" if capture_dedup_enabled else "ALL SLIDES"
    print(f"[Info] Running Capture Pipeline in '{mode_str}' mode.")
    if capture_dedup_enabled:
        print("   - Strategy: pHash + ORB deduplication")
    else:
        print("   - Strategy: Save ALL slides (No deduplication)")

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
    # Preprocessing 관련 config 파일들의 해시 계산
    preprocess_config_hash = compute_config_hash([
        ROOT / "config" / "audio" / "settings.yaml",
        ROOT / "config" / "capture" / "settings.yaml",
    ])

    run_args = {
        "pipeline_type": "preprocess",
        "video": str(video_path),
        "output_base": str(output_base_path),
        "stt_backend": stt_backend,
        "config_hash": preprocess_config_hash,
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
        json.dumps(run_meta, ensure_ascii=False, indent=2),
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
            stt_backend=stt_backend,
            table_name=db_table_name,
            existing_video_id=existing_video_id,
        )
        if db_context:
            _, _, preprocess_job_id = db_context
            if preprocess_job_id:
                run_meta["preprocess_job_id"] = preprocess_job_id
            print("\nSyncing preprocessing artifacts to Supabase as stages complete...")

    timer.start_total()
    capture_count = 0

    try:
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "manifest.json"

        # Early cancel (e.g., user deleted the video while a BackgroundTask is running).
        if db_context:
            adapter, video_id, _preprocess_job_id = db_context
            raise_if_cancel_requested(adapter, video_id)
        else:
            # No DB context: still honor same-process cancellation.
            raise_if_cancel_requested(None, existing_video_id)

        print(f"\nStarting preprocessing (parallel={parallel})...")
        print("-" * 50)

        stt_elapsed = 0.0
        capture_elapsed = 0.0
        audio_storage_key: Optional[str] = None
        stt_payload = None
        capture_result = []

        def _finalize_stage(
            stage: str,
            proc_elapsed: float,
            **kwargs: Any,
        ) -> None:
            """통합 로깅 및 DB 동기화 헬퍼."""
            comp_map = {"audio": "Audio", "capture": "Capture", "stt": "STT", "video": "Video"}
            comp_name = comp_map.get(stage, "System")

            db_elapsed = 0.0
            nonlocal audio_storage_key
            if db_context:
                db_start = time.perf_counter()
                adapter, video_id, preprocess_job_id = db_context

                # Don't upload artifacts after delete is requested.
                raise_if_cancel_requested(adapter, video_id)
                 
                # [Fix] Capture 중복 업로드 방지
                # 스트리밍으로 이미 업로드된 경우(skip_db_capture=True), 배치 업로드는 수행하지 않음
                do_capture_sync = (stage == "capture" and not kwargs.get("skip_db_capture", False))
                
                # [Fix] Video 중복 업로드 방지
                # existing_video_id가 있는 경우 프론트엔드에서 이미 presigned URL로 업로드했으므로 재업로드 불필요
                do_video_sync = (stage == "video" and existing_video_id is None)
                
                stage_results = sync_preprocess_artifacts_to_db(
                    adapter=adapter,
                    video_id=video_id,
                    video_root=video_root,
                    provider=stt_backend,
                    preprocess_job_id=preprocess_job_id,
                    include_stt=(stage == "stt"),
                    include_captures=do_capture_sync,
                    include_audio=False,  # [Fix] R2 오디오 업로드 비활성화
                    include_video=do_video_sync,
                    video_path=video_path,
                    **kwargs,
                    table_name=db_table_name,
                )
                db_elapsed = time.perf_counter() - db_start
                if stage_results.get("errors"):
                    db_errors.extend(stage_results["errors"])
                # audio 업로드 후 storage_key 저장
                if stage == "audio" and stage_results.get("saved", {}).get("audio"):
                    audio_storage_key = stage_results.get("audio_storage_key")

            # 통합 로깅 포맷 적용
            res_info = ""
            if stage == "stt" and "stt_payload" in kwargs:
                payload = kwargs["stt_payload"]
                res_info = f" | seg: {len(payload.get('segments', []))}" if payload else ""
            elif stage == "capture" and "captures_payload" in kwargs:
                res_info = f" | cap: {len(kwargs['captures_payload'])}"

            msg = f"DONE (Task: {proc_elapsed:.1f}s | DB: {db_elapsed:.1f}s{res_info})"
            pipeline_logger.log(comp_name, msg)

        # 병렬 또는 순차 실행을 위한 내부 함수들
        def on_capture_event(event_type: str, slide_data: Dict[str, Any]) -> None:
            """캡처 이벤트(신규/업데이트) 발생 시 DB에 즉시 반영한다."""
            if not db_context:
                return
            adapter, video_id, preprocess_job_id = db_context

            # Avoid frequent DB reads here; rely on local cancel marker for responsiveness.
            if is_local_cancel_requested(video_id):
                raise PipelineCanceled("local_cancel")

            try:
                if event_type == "new":
                    # 단일 항목 리스트로 감싸서 업로드 재활용
                    payload = [slide_data]
                    res = adapter.save_captures_with_upload_payload(
                        video_id=video_id,
                        captures=payload,
                        captures_dir=captures_dir,
                        preprocess_job_id=preprocess_job_id,
                        table_name=db_table_name
                    )
                    if res.get("errors"):
                        pipeline_logger.log("DB", f"Error streaming capture: {res['errors']}")
                    else:
                        # 너무 빈번한 로그 방지
                        # pipeline_logger.log("DB", f"Streamed capture: {slide_data.get('file_name')}")
                        pass

                elif event_type == "update":
                    # 기존 캡처의 time_ranges 업데이트
                    cap_id = slide_data.get("id")
                    time_ranges = slide_data.get("time_ranges")
                    
                    if cap_id and time_ranges:
                        # Adapter를 통하지 않고 직접 update (Mixin에 update 메서드가 없을 경우)
                        adapter.client.table(db_table_name).update({
                            "time_ranges": time_ranges
                        }).eq("video_id", video_id).eq("cap_id", cap_id).execute()
                        pipeline_logger.log("DB", f"Updated capture times: {cap_id}")
            except Exception as e:
                pipeline_logger.log("DB", f"Streaming error: {e}")

        def handle_audio_stt_chain() -> Dict[str, Any]:
            """Audio 추출 → STT를 체인으로 실행."""
            nonlocal stt_elapsed
            if db_context:
                adapter, video_id, _ = db_context
                raise_if_cancel_requested(adapter, video_id)
            else:
                raise_if_cancel_requested(None, existing_video_id)
            from src.audio.stt_router import load_audio_settings
            audio_settings = load_audio_settings()
            extract_settings = audio_settings.get("extract", {})
            audio_codec = extract_settings.get("codec", "libmp3lame")
            codec_ext_map = {"pcm_s16le": ".wav", "flac": ".flac", "libmp3lame": ".mp3"}
            audio_ext = codec_ext_map.get(audio_codec, ".wav")

            # (1) Audio Extract
            audio_path = video_root / f"{video_name}{audio_ext}"
            pipeline_logger.log("Audio", "Extracting...")
            start = time.perf_counter()
            extract_audio(
                video_path,
                output_path=audio_path,
                sample_rate=extract_settings.get("sample_rate", 16000),
                channels=extract_settings.get("channels", 1),
                codec=audio_codec,
                mp3_bitrate=extract_settings.get("mp3_bitrate", "128k"),
                mono_method=extract_settings.get("mono_method", "auto"),
            )
            elapsed_audio = time.perf_counter() - start
            timer.record_stage("audio", elapsed_audio)
            _finalize_stage("audio", elapsed_audio, audio_path=audio_path)

            # (2) STT Analysis
            pipeline_logger.log("STT", "Analyzing...")
            start = time.perf_counter()
            payload = run_stt_only(audio_path, stt_json, backend=stt_backend, write_output=write_local_json)
            elapsed_stt = time.perf_counter() - start
            stt_elapsed = elapsed_stt
            timer.record_stage("stt", elapsed_stt)
            _finalize_stage("stt", elapsed_stt, stt_payload=payload)
            return payload

        def handle_capture() -> List[Dict[str, Any]]:
            """캡처를 실행."""
            nonlocal capture_elapsed
            if db_context:
                adapter, video_id, _ = db_context
                raise_if_cancel_requested(adapter, video_id)
            else:
                raise_if_cancel_requested(None, existing_video_id)
            pipeline_logger.log("Capture", "Extracting...")
            start = time.perf_counter()
            results = run_capture(
                video_path,
                output_base_path,
                threshold=capture_threshold,
                dedupe_threshold=capture_dedupe_threshold,
                min_interval=capture_min_interval,
                verbose=capture_verbose,
                video_name=video_name,
                write_manifest=write_local_json,
                callback=on_capture_event,
            )
            elapsed = time.perf_counter() - start
            capture_elapsed = elapsed
            timer.record_stage("capture", elapsed)
            
            # [Fix] 스트리밍으로 이미 업로드했으므로 최종 단계에서는 로컬 파일 동기화만 하거나 생략한다.
            # 중복 저장을 막기 위해 skip_db_capture=True 전달
            _finalize_stage("capture", elapsed, captures_payload=results, skip_db_capture=True)
            return results

        pipeline_logger.log("System", f"Starting Preprocessing (Parallel={parallel})")
        
        # [User Request] Video 원본 업로드 (video_storage_key 채우기)
        _finalize_stage("video", 0.0)
 
        if parallel:
            import concurrent.futures
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Future -> Task Name Mapping
                future_to_task = {
                    executor.submit(handle_audio_stt_chain): "stt",
                    executor.submit(handle_capture): "capture"
                }
                
                # as_completed를 사용하여 먼저 끝나는 작업부터 처리
                for future in concurrent.futures.as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        res = future.result()
                        if task_name == "stt":
                            stt_payload = res
                            # handle_audio_stt_chain 내부에서 _finalize_stage("stt")가 호출되므로
                            # 이미 DB 업로드는 완료된 상태임.
                            pipeline_logger.log("Pipeline", "STT stage finalized immediately.")
                        elif task_name == "capture":
                            capture_result = res
                            # handle_capture 내부에서 _finalize_stage("capture")가 호출되므로
                            # 이미 DB 업로드는 완료된 상태임.
                            pipeline_logger.log("Pipeline", "Capture stage finalized immediately.")
                    except Exception as exc:
                        pipeline_logger.log("Pipeline", f"{task_name} generated an exception: {exc}")
                        # 예외 발생 시 플래그 처리 등을 할 수 있음
                        pass
        else:
            stt_payload = handle_audio_stt_chain()
            capture_result = handle_capture()

        capture_count = len(capture_result) if capture_result else 0
        segment_count = len(stt_payload.get("segments", [])) if stt_payload else 0

        timer.end_total()

        # 사람이 읽을 수 있는 벤치마크 리포트를 함께 저장한다.
        md_report = print_benchmark_report(
            video_info=video_info,
            timer=timer,
            capture_count=capture_count,
            segment_count=segment_count,
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
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if sync_to_db:
            if db_context:
                try:
                    adapter, video_id, preprocess_job_id = db_context
                    finalize_preprocess_db_sync(
                        adapter=adapter,
                        video_id=video_id,
                        preprocess_job_id=preprocess_job_id,
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
        # [User Request] 로컬 경로를 상대 경로로 표시
        rel_video_root = os.path.relpath(video_root, ROOT)
        rel_report_path = os.path.relpath(report_path, ROOT)
        print(f"Outputs: {rel_video_root}")
        print(f"Benchmark: {rel_report_path}")
        
        if sync_to_db and db_context:
            try:
                _, video_id, _ = db_context
                return video_id
            except Exception:
                pass
        return None

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

        if sync_to_db and db_context:
            try:
                adapter, video_id, preprocess_job_id = db_context
                finalize_preprocess_db_sync(
                    adapter=adapter,
                    video_id=video_id,
                    preprocess_job_id=preprocess_job_id,
                    run_meta=run_meta,
                    errors=[f"canceled: {exc}"],
                )
            except Exception:
                pass

        print(f"\nPreprocessing canceled: {exc}")
        if sync_to_db and db_context:
            try:
                _, video_id, _ = db_context
                return video_id
            except Exception:
                pass
        return None

    except Exception as exc:
        # 재발생 전에 실패 메타데이터를 항상 기록한다.
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        run_meta_path.write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if sync_to_db and db_context:
            try:
                adapter, video_id, preprocess_job_id = db_context
                db_errors.append(f"pipeline: {exc}")
                finalize_preprocess_db_sync(
                    adapter=adapter,
                    video_id=video_id,
                    preprocess_job_id=preprocess_job_id,
                    run_meta=run_meta,
                    errors=db_errors,
                )
            except Exception:
                pass
        print(f"\nPreprocessing failed: {exc}")
        raise


def get_parser() -> argparse.ArgumentParser:
    """전처리 파이프라인용 ArgumentParser를 생성해 반환한다."""
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
    parser.add_argument(
        "--capture-mode",
        choices=["all", "dedup"],
        default="dedup",
        help="Capture mode: 'all' (save all slides), 'dedup' (deduplicate similar slides)",
    )
    parser.add_argument(
        "--db-table",
        default="captures",
        help="Target Supabase table for captures (default: 'captures')",
    )
    parser.add_argument("--db-sync", dest="db_sync", action="store_true", help="Enable Supabase sync")
    parser.add_argument("--no-db-sync", dest="db_sync", action="store_false", help="Skip Supabase sync")
    parser.add_argument("--video-id", dest="existing_video_id", help="Existing video UUID (to avoid re-creating)")
    parser.set_defaults(db_sync=None)
    return parser


def main() -> None:
    """CLI 인자를 파싱하고 전처리 파이프라인을 실행한다."""
    parser = get_parser()
    args = parser.parse_args()

    run_preprocess_pipeline(
        video=args.video,
        output_base=args.output_base,
        stt_backend=args.stt_backend,
        parallel=args.parallel,
        capture_dedup_enabled=(args.capture_mode == "dedup"),
        capture_verbose=args.capture_verbose,
        write_local_json=args.local_json,
        sync_to_db=args.db_sync,
        db_table_name=args.db_table,
    )


if __name__ == "__main__":
    main()
