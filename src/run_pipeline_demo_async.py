"""
Queue/Event 기반 비동기 데모 파이프라인 엔트리포인트.

실행 흐름:
1) Producer: Audio/STT + Capture 이벤트 생성
2) VLM Worker: Capture chunk를 VLM 처리
3) Fusion Worker: STT gate 이후 Fusion/Summary/Judge 처리

세 워커를 `asyncio.gather()`로 동시에 실행한다.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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

from src.capture.settings import load_capture_settings
from src.pipeline.contracts import (
    EndOfStreamEvent,
    FusionDoneEvent,
    JudgeDoneEvent,
    PipelineContext,
    PipelineErrorEvent,
    SummaryDoneEvent,
)
from src.pipeline.fusion_worker_async import (
    AsyncFusionSummaryJudgeWorker,
    FusionWorkerConfig,
)
from src.pipeline.orchestrator_async import AsyncPipelineOrchestrator
from src.pipeline.producers_async import (
    AsyncCaptureSttProducer,
    CaptureSttProducerConfig,
)
from src.pipeline.vlm_worker_async import AsyncVlmWorker, VlmWorkerConfig
from src.pipeline.cancel import PipelineCanceled, raise_if_cancel_requested

QWEN_BASE_URL_DEFAULT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_DEFAULT = "qwen3-vl-32b-instruct"


def _ts() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _sanitize_video_name(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"


def _load_pipeline_defaults() -> Dict[str, Any]:
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        return {}
    payload = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_or(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _load_yaml_runtime_defaults() -> Dict[str, Any]:
    """config/*.yaml 값을 모아 async demo 기본값으로 사용한다(파일 수정 없음)."""
    pipeline = _load_pipeline_defaults()

    audio_path = ROOT / "config" / "audio" / "settings.yaml"
    capture_path = ROOT / "config" / "capture" / "settings.yaml"

    audio_payload = {}
    capture_payload = {}
    if audio_path.exists():
        raw = yaml.safe_load(audio_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            audio_payload = raw
    if capture_path.exists():
        raw = yaml.safe_load(capture_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            capture_payload = raw

    stt_block = audio_payload.get("stt", {})
    if not isinstance(stt_block, dict):
        stt_block = {}

    vlm_concurrency = _int_or(pipeline.get("vlm_concurrency"), 3)
    if vlm_concurrency < 1:
        vlm_concurrency = 1

    return {
        "output_base": str(capture_payload.get("output_dir", "data/outputs")),
        "stt_backend": str(stt_block.get("default_provider", "clova")),
        "capture_batch_size": max(1, _int_or(pipeline.get("batch_size"), 6)),
        "vlm_parallelism": vlm_concurrency,
        "vlm_inner_concurrency": vlm_concurrency,
        "vlm_batch_size": max(1, _int_or(pipeline.get("vlm_batch_size"), 6)),
        "vlm_show_progress": _bool_or(pipeline.get("vlm_show_progress"), True),
    }


def _resolve_video_path(video: str) -> Path:
    raw = Path(video).expanduser()
    candidate = raw if raw.is_absolute() else (ROOT / raw)
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate

    fallback = (ROOT / "data" / "inputs" / video).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Video not found: {video}")


def _apply_qwen_vlm_overrides(repo_root: Path) -> Path:
    """기존 run_process_pipeline.py와 동일한 Qwen VLM override를 적용한다."""
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

    temp_path = Path(tempfile.gettempdir()) / f"vlm_settings_qwen_async_{os.getpid()}.yaml"
    temp_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    from src.vlm import vlm_engine as _vlm_engine

    _vlm_engine.SETTINGS_CONFIG_PATH = temp_path
    return temp_path


@dataclass(frozen=True)
class FusionQueueStats:
    fusion_count: int = 0
    summary_count: int = 0
    judge_count: int = 0


@dataclass(frozen=True)
class AsyncDemoResult:
    run_id: str
    video_root: Path
    capture_event_count: int
    capture_item_count: int
    stt_segment_count: int
    vlm_chunk_count: int
    fusion_batch_count: int
    summary_batch_count: int
    judge_batch_count: int
    error_count: int


async def _consume_fusion_events(orchestrator: AsyncPipelineOrchestrator) -> FusionQueueStats:
    fusion_count = 0
    summary_count = 0
    judge_count = 0

    while True:
        event = await orchestrator.fusion_q.get()
        try:
            if isinstance(event, EndOfStreamEvent):
                break
            if isinstance(event, FusionDoneEvent):
                fusion_count += 1
                _log(
                    f"[Fusion] batch={event.batch_index} segments={event.segment_count} "
                    f"elapsed={event.elapsed_sec:.2f}s"
                    if event.elapsed_sec is not None
                    else f"[Fusion] batch={event.batch_index} segments={event.segment_count}"
                )
            elif isinstance(event, SummaryDoneEvent):
                summary_count += 1
                _log(f"[Summary] batch={event.batch_index} done")
            elif isinstance(event, JudgeDoneEvent):
                judge_count += 1
                _log(f"[Judge] batch={event.batch_index} score={event.score:.2f}")
        finally:
            orchestrator.fusion_q.task_done()

    return FusionQueueStats(
        fusion_count=fusion_count,
        summary_count=summary_count,
        judge_count=judge_count,
    )


async def _consume_error_events(
    orchestrator: AsyncPipelineOrchestrator,
) -> List[PipelineErrorEvent]:
    errors: List[PipelineErrorEvent] = []
    while True:
        event = await orchestrator.error_q.get()
        try:
            if isinstance(event, EndOfStreamEvent):
                break
            if isinstance(event, PipelineErrorEvent):
                errors.append(event)
                _log(
                    f"[Error] stage={event.stage} retriable={event.retriable} "
                    f"message={event.message}"
                )
        finally:
            orchestrator.error_q.task_done()
    return errors


async def run_async_demo(
    *,
    video_path: Path,
    output_base: Path,
    run_id: str,
    stt_backend: str,
    capture_batch_size: int,
    vlm_parallelism: int,
    vlm_inner_concurrency: int,
    vlm_batch_size: int | None,
    vlm_show_progress: bool,
    max_inflight_chunks: int,
    queue_maxsize: int,
    strict_batch_order: bool,
    sync_to_db: bool,
    upload_video_to_r2: bool,
    upload_audio_to_r2: bool,
    existing_video_id: str | None = None,
) -> AsyncDemoResult:
    output_base.mkdir(parents=True, exist_ok=True)
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    capture_settings = load_capture_settings()

    video_id = None
    preprocess_job_id = None
    processing_job_id = None
    adapter_for_cancel = None
    if sync_to_db:
        from src.db import get_supabase_adapter
        from src.db.adapters import compute_config_hash

        adapter = get_supabase_adapter()
        if not adapter:
            raise RuntimeError("Supabase adapter not configured; set SUPABASE_URL/SUPABASE_KEY.")
        adapter_for_cancel = adapter

        if existing_video_id:
            video_id = existing_video_id
        else:
            existing = adapter.get_video_by_filename(None, video_path.name)
            if existing:
                video_id = existing.get("id")
            else:
                video = adapter.create_video(
                    name=video_name,
                    original_filename=video_path.name,
                )
                video_id = video.get("id")

        if not video_id:
            raise RuntimeError("Failed to resolve video_id for DB sync.")

        if upload_video_to_r2:
            try:
                existing = adapter.get_video(video_id)
                storage_key = existing.get("video_storage_key") if existing else None
                if not storage_key:
                    adapter.upload_video(video_id, video_path)
            except Exception as exc:
                raise RuntimeError(f"Video upload failed: {exc}") from exc

        # Preprocess 관련 config 파일들의 해시 계산
        preprocess_config_hash = compute_config_hash([
            ROOT / "config" / "audio" / "settings.yaml",
            ROOT / "config" / "capture" / "settings.yaml",
        ])

        preprocess_job = adapter.create_preprocessing_job(
            video_id,
            source="SERVER",
            stt_backend=stt_backend,
            config_hash=preprocess_config_hash,
        )
        preprocess_job_id = preprocess_job.get("id")
        if preprocess_job_id:
            adapter.update_preprocessing_job_status(preprocess_job_id, "RUNNING")

        # Processing 관련 config 파일들의 해시 계산
        processing_config_hash = compute_config_hash([
            ROOT / "config" / "fusion" / "settings.yaml",
            ROOT / "config" / "judge" / "settings.yaml",
            ROOT / "config" / "pipeline" / "settings.yaml",
            ROOT / "config" / "vlm" / "settings.yaml",
        ])

        job = adapter.create_processing_job(
            video_id,
            triggered_by="MANUAL",
            config_hash=processing_config_hash,
        )
        processing_job_id = job.get("id")
        if not processing_job_id:
            raise RuntimeError("Failed to create processing_job for DB sync.")

        # 시작 상태 설정
        adapter.update_processing_job_status(processing_job_id, "VLM_RUNNING")
        adapter.update_video_status(video_id, "PROCESSING")

    context = PipelineContext(
        run_id=run_id,
        video_name=video_name,
        output_root=video_root,
        video_id=video_id,
        preprocess_job_id=preprocess_job_id,
        processing_job_id=processing_job_id,
        sync_to_db=sync_to_db,
        upload_video_to_r2=upload_video_to_r2,
        upload_audio_to_r2=upload_audio_to_r2,
        batch_size=capture_batch_size,
        vlm_batch_size=vlm_batch_size,
        vlm_concurrency=vlm_parallelism,
    )
    orchestrator = AsyncPipelineOrchestrator(
        context=context,
        queue_maxsize=queue_maxsize,
        vlm_parallelism=vlm_parallelism,
    )

    producer = AsyncCaptureSttProducer(
        context=context,
        orchestrator=orchestrator,
        config=CaptureSttProducerConfig(
            video_path=video_path,
            video_root=video_root,
            video_name=video_name,
            capture_output_base=output_base,
            stt_backend=stt_backend,
            write_local_json=True,
            capture_batch_size=capture_batch_size,
            capture_threshold=float(capture_settings.persistence_drop_ratio),
            capture_dedupe_threshold=float(capture_settings.dedup_sim_threshold),
            capture_min_interval=float(capture_settings.min_interval),
            capture_verbose=False,
        ),
    )
    vlm_worker = AsyncVlmWorker(
        context=context,
        orchestrator=orchestrator,
        config=VlmWorkerConfig(
            video_root=video_root,
            captures_dir=video_root / "captures",
            video_name=video_name,
            video_id=video_id,
            vlm_batch_size=vlm_batch_size,
            vlm_inner_concurrency=vlm_inner_concurrency,
            vlm_show_progress=vlm_show_progress,
            max_inflight_chunks=max_inflight_chunks,
        ),
    )
    fusion_worker = AsyncFusionSummaryJudgeWorker(
        context=context,
        orchestrator=orchestrator,
        config=FusionWorkerConfig(
            video_root=video_root,
            video_name=video_name,
            captures_dir=video_root / "captures",
            repo_root=ROOT,
            batch_size=capture_batch_size,
            vlm_batch_size=vlm_batch_size,
            vlm_concurrency=vlm_inner_concurrency,
            vlm_show_progress=False,
            strict_batch_order=strict_batch_order,
        ),
    )

    await orchestrator.start()

    fusion_consumer_task = asyncio.create_task(
        _consume_fusion_events(orchestrator), name="fusion_consumer"
    )
    error_consumer_task = asyncio.create_task(
        _consume_error_events(orchestrator), name="error_consumer"
    )

    producer_task = asyncio.create_task(producer.run(), name="producer")
    vlm_task = asyncio.create_task(vlm_worker.run(), name="vlm_worker")
    fusion_task = asyncio.create_task(fusion_worker.run(), name="fusion_worker")
    core_tasks = [producer_task, vlm_task, fusion_task]

    async def _watch_delete_requested() -> None:
        # Block forever until a delete/cancel signal is observed.
        # The caller cancels this task on normal completion/failure.
        if not video_id:
            while True:
                await asyncio.sleep(3600)
        while True:
            raise_if_cancel_requested(adapter_for_cancel, video_id)
            await asyncio.sleep(1)

    cancel_watch_task = asyncio.create_task(_watch_delete_requested(), name="cancel_watcher")
    # asyncio.gather() returns a Future (not a coroutine), so don't wrap it with create_task().
    core_gather = asyncio.gather(*core_tasks)

    try:
        done, _pending = await asyncio.wait(
            {core_gather, cancel_watch_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_watch_task in done and not cancel_watch_task.cancelled():
            cancel_exc: PipelineCanceled = PipelineCanceled("delete_requested")
            try:
                await cancel_watch_task
            except PipelineCanceled as exc:
                cancel_exc = exc
            except Exception as exc:
                cancel_exc = PipelineCanceled(str(exc))

            await orchestrator.stop(reason="delete_requested")
            for task in core_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*core_tasks, return_exceptions=True)
            core_gather.cancel()
            await asyncio.gather(core_gather, return_exceptions=True)
            await fusion_consumer_task
            await error_consumer_task

            # 취소 상태 업데이트 (best-effort; the row may already be deleted).
            if sync_to_db and processing_job_id:
                try:
                    from src.db import get_supabase_adapter
                    adapter = get_supabase_adapter()
                    if adapter:
                        adapter.update_processing_job_status(
                            processing_job_id, "FAILED", error_message=f"canceled: {cancel_exc}"
                        )
                        adapter.update_video_status(video_id, "FAILED", error=f"canceled: {cancel_exc}")
                except Exception:
                    pass

            raise cancel_exc

        # Core tasks finished first (success or failure).
        cancel_watch_task.cancel()
        await asyncio.gather(cancel_watch_task, return_exceptions=True)
        producer_result, vlm_chunk_count, _fusion_processed = await core_gather

    except Exception as exc:
        cancel_watch_task.cancel()
        await asyncio.gather(cancel_watch_task, return_exceptions=True)

        for task in core_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*core_tasks, return_exceptions=True)
        await orchestrator.stop(reason="pipeline_failed")
        await fusion_consumer_task
        await error_consumer_task

        # 실패 상태 업데이트
        if sync_to_db and processing_job_id:
            try:
                from src.db import get_supabase_adapter
                adapter = get_supabase_adapter()
                if adapter:
                    adapter.update_processing_job_status(
                        processing_job_id, "FAILED", error_message=str(exc)
                    )
                    adapter.update_video_status(video_id, "FAILED", error=str(exc))
            except Exception:
                pass
        raise
    else:
        await orchestrator.stop(reason="pipeline_completed")
        fusion_stats = await fusion_consumer_task
        errors = await error_consumer_task

        # 완료 상태 업데이트
        if sync_to_db and processing_job_id:
            try:
                from src.db import get_supabase_adapter, upsert_final_summary_results
                adapter = get_supabase_adapter()
                if adapter:
                    # total_batch를 최종 값으로 설정
                    adapter.update_processing_job_progress(
                        processing_job_id,
                        fusion_stats.fusion_count,
                        fusion_stats.fusion_count,  # total_batch 설정
                    )
                    adapter.update_processing_job_status(processing_job_id, "DONE")
                    adapter.update_video_status(video_id, "DONE")

                    # summary_results UPSERT (timeline, tldr 포맷 저장)
                    try:
                        fusion_dir = video_root / "fusion"
                        summaries_path = fusion_dir / "segment_summaries.jsonl"
                        upsert_final_summary_results(
                            adapter=adapter,
                            video_id=video_id,
                            processing_job_id=processing_job_id,
                            summaries_path=summaries_path,
                            results_dir=video_root / "results",  # final_summary_*.md 파일 위치
                        )
                    except Exception as summary_exc:
                        print(f"[DB] Warning: summary_results upsert failed: {summary_exc}")
            except Exception:
                pass
    finally:
        if not cancel_watch_task.done():
            cancel_watch_task.cancel()
            await asyncio.gather(cancel_watch_task, return_exceptions=True)

    return AsyncDemoResult(
        run_id=run_id,
        video_root=video_root,
        capture_event_count=producer_result.capture_event_count,
        capture_item_count=producer_result.capture_item_count,
        stt_segment_count=producer_result.stt_segment_count,
        vlm_chunk_count=vlm_chunk_count,
        fusion_batch_count=fusion_stats.fusion_count,
        summary_batch_count=fusion_stats.summary_count,
        judge_batch_count=fusion_stats.judge_count,
        error_count=len(errors),
    )


def _build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run async pipeline demo (Producer -> VLM -> Fusion) with asyncio.gather()."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--output-base",
        default=str(defaults.get("output_base", "data/outputs")),
        help="Output base directory",
    )
    parser.add_argument("--run-id", default="", help="Optional fixed run_id")
    parser.add_argument(
        "--stt-backend",
        default=str(defaults.get("stt_backend", "clova")),
        help="STT backend (default from config/audio/settings.yaml)",
    )

    parser.add_argument(
        "--capture-batch-size",
        type=int,
        default=int(defaults.get("capture_batch_size", 6)),
        help="Capture chunk size (also used as batch size in fusion)",
    )
    parser.add_argument(
        "--vlm-parallelism",
        type=int,
        default=int(defaults.get("vlm_parallelism", 3)),
        help="Pipeline-level VLM parallelism (orchestrator semaphore)",
    )
    parser.add_argument(
        "--vlm-inner-concurrency",
        type=int,
        default=int(defaults.get("vlm_inner_concurrency", 1)),
        help="Inner concurrency passed to run_vlm_for_batch",
    )
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=int(defaults.get("vlm_batch_size", 6)),
        help="Inner VLM request batch size",
    )
    parser.add_argument(
        "--vlm-show-progress",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("vlm_show_progress", True)),
        help="Show VLM progress logs",
    )
    parser.add_argument(
        "--max-inflight-chunks",
        type=int,
        default=8,
        help="Max inflight chunk tasks inside VLM worker",
    )
    parser.add_argument(
        "--queue-maxsize",
        type=int,
        default=0,
        help="Queue max size (0 means unbounded)",
    )
    parser.add_argument(
        "--strict-batch-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep strict batch index order in fusion worker",
    )
    parser.add_argument(
        "--sync-to-db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Supabase DB sync (requires SUPABASE_URL/SUPABASE_KEY)",
    )
    parser.add_argument(
        "--upload-video-to-r2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload source video to R2 when sync_to_db is enabled",
    )
    parser.add_argument(
        "--upload-audio-to-r2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload extracted audio to R2 when sync_to_db is enabled",
    )
    return parser


def main() -> None:
    defaults = _load_yaml_runtime_defaults()
    parser = _build_parser(defaults)
    args = parser.parse_args()

    video_path = _resolve_video_path(args.video)
    output_base = Path(args.output_base)
    if not output_base.is_absolute():
        output_base = (ROOT / output_base).resolve()
    else:
        output_base = output_base.resolve()

    run_id = args.run_id.strip()
    if not run_id:
        video_name = _sanitize_video_name(video_path.stem)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"async_{video_name}_{stamp}"

    _log("=" * 72)
    _log("Re:View Async Pipeline Demo")
    _log(f"video={video_path}")
    _log(f"output_base={output_base}")
    _log(f"run_id={run_id}")
    _log(
        f"capture_batch_size={args.capture_batch_size} "
        f"vlm_parallelism={args.vlm_parallelism} "
        f"vlm_inner_concurrency={args.vlm_inner_concurrency}"
    )
    _log(
        f"sync_to_db={args.sync_to_db} "
        f"upload_video_to_r2={args.upload_video_to_r2} "
        f"upload_audio_to_r2={args.upload_audio_to_r2}"
    )
    _log("=" * 72)

    # 기존 run_process_pipeline.py와 동일하게 Qwen 모델/엔드포인트 override 적용
    _apply_qwen_vlm_overrides(ROOT)

    try:
        result = asyncio.run(
            run_async_demo(
                video_path=video_path,
                output_base=output_base,
                run_id=run_id,
                stt_backend=args.stt_backend,
                capture_batch_size=args.capture_batch_size,
                vlm_parallelism=args.vlm_parallelism,
                vlm_inner_concurrency=args.vlm_inner_concurrency,
                vlm_batch_size=args.vlm_batch_size,
                vlm_show_progress=args.vlm_show_progress,
                max_inflight_chunks=args.max_inflight_chunks,
                queue_maxsize=args.queue_maxsize,
                strict_batch_order=args.strict_batch_order,
                sync_to_db=args.sync_to_db,
                upload_video_to_r2=args.upload_video_to_r2,
                upload_audio_to_r2=args.upload_audio_to_r2,
            )
        )
    except KeyboardInterrupt:
        _log("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        _log(f"FAILED: {exc}")
        raise SystemExit(1)

    _log("Pipeline finished.")
    _log(f"video_root={result.video_root}")
    _log(
        f"capture_events={result.capture_event_count} capture_items={result.capture_item_count} "
        f"stt_segments={result.stt_segment_count}"
    )
    _log(
        f"vlm_chunks={result.vlm_chunk_count} fusion_batches={result.fusion_batch_count} "
        f"summary_batches={result.summary_batch_count} judge_batches={result.judge_batch_count} "
        f"errors={result.error_count}"
    )


if __name__ == "__main__":
    main()
