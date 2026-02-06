"""비동기 Capture/STT producer 연동 모듈.

현재 단계 목표:
- 기존 blocking 함수(`run_capture`, `extract_audio`, `run_stt_only`)를 `asyncio.to_thread`로 실행
- Capture 결과를 chunk 단위 이벤트로 오케스트레이터 큐에 게시
- STT 완료 시 오케스트레이터 STT 게이트를 열기
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.audio.extract_audio import CODEC_TO_EXT, extract_audio
from src.audio.stt_router import load_audio_settings
from src.pipeline.logger import pipeline_logger
from src.pipeline.stages import run_capture, run_stt_only

from .contracts import (
    CaptureChunkReadyEvent,
    CaptureUnit,
    EndOfStreamEvent,
    PipelineContext,
    SttDoneEvent,
    TimeWindow,
)
from .orchestrator_async import AsyncPipelineOrchestrator

logger = logging.getLogger(__name__)


def _int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _capture_start_ms(item: Mapping[str, Any]) -> int:
    if "timestamp_ms" in item:
        return max(0, _int_or(item.get("timestamp_ms"), 0))
    ranges = item.get("time_ranges")
    if isinstance(ranges, list) and ranges and isinstance(ranges[0], Mapping):
        return max(0, _int_or(ranges[0].get("start_ms"), 0))
    return max(0, _int_or(item.get("start_ms"), 0))


def _capture_end_ms(item: Mapping[str, Any]) -> int:
    ranges = item.get("time_ranges")
    if isinstance(ranges, list) and ranges:
        max_end = 0
        for rng in ranges:
            if isinstance(rng, Mapping):
                max_end = max(max_end, _int_or(rng.get("end_ms"), 0))
        if max_end > 0:
            return max_end
    if "end_ms" in item:
        return max(0, _int_or(item.get("end_ms"), 0))
    start_ms = _capture_start_ms(item)
    return start_ms + 1000


@dataclass(frozen=True)
class CaptureSttProducerConfig:
    """Capture/STT producer 실행 설정."""

    video_path: Path
    video_root: Path
    video_name: str
    capture_output_base: Optional[Path] = None
    stt_backend: str = "clova"
    write_local_json: bool = True
    capture_batch_size: int = 6
    capture_threshold: float = 30.0
    capture_dedupe_threshold: float = 0.92
    capture_min_interval: float = 1.0
    capture_verbose: bool = False

    def __post_init__(self) -> None:
        if not self.video_name.strip():
            raise ValueError("video_name is required")
        if self.capture_batch_size < 1:
            raise ValueError("capture_batch_size must be >= 1")


@dataclass(frozen=True)
class CaptureSttProducerResult:
    """Capture/STT producer 실행 결과."""

    capture_event_count: int
    capture_item_count: int
    stt_segment_count: int
    stt_json_path: Path


class AsyncCaptureSttProducer:
    """Capture/STT producer를 비동기로 실행해 오케스트레이터에 이벤트를 게시한다."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        orchestrator: AsyncPipelineOrchestrator,
        config: CaptureSttProducerConfig,
    ) -> None:
        self.context = context
        self.orchestrator = orchestrator
        self.config = config
        self.captures_dir = config.video_root / "captures"
        self.stt_json_path = config.video_root / "stt.json"
        self.capture_output_base = config.capture_output_base or config.video_root.parent
        self._db_adapter = None

        self._capture_event_count = 0
        self._capture_item_count = 0

    async def run(self) -> CaptureSttProducerResult:
        """Capture/STT producer를 병렬 실행한다."""
        self.config.video_root.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        stt_task = asyncio.create_task(self._run_stt_chain(), name="stt_producer")
        capture_task = asyncio.create_task(self._run_capture_loop(), name="capture_producer")

        try:
            stt_event, _ = await asyncio.gather(stt_task, capture_task)
        except Exception:
            for task in (stt_task, capture_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stt_task, capture_task, return_exceptions=True)
            if self.context.sync_to_db:
                self._mark_preprocess_failed("producer_failed")
            raise
        else:
            if self.context.sync_to_db:
                self._mark_preprocess_done()

        return CaptureSttProducerResult(
            capture_event_count=self._capture_event_count,
            capture_item_count=self._capture_item_count,
            stt_segment_count=stt_event.segment_count,
            stt_json_path=stt_event.stt_json_path,
        )

    def _ensure_db_adapter(self):
        if self._db_adapter is None:
            from src.db import get_supabase_adapter

            self._db_adapter = get_supabase_adapter()
        return self._db_adapter

    def _mark_preprocess_done(self) -> None:
        adapter = self._ensure_db_adapter()
        if not adapter or not self.context.preprocess_job_id:
            return
        try:
            adapter.update_preprocessing_job_status(self.context.preprocess_job_id, "DONE")
        except Exception as exc:
            pipeline_logger.log("DB", f"preprocess status update failed: {exc}")

    def _mark_preprocess_failed(self, message: str) -> None:
        adapter = self._ensure_db_adapter()
        if not adapter or not self.context.preprocess_job_id:
            return
        try:
            adapter.update_preprocessing_job_status(
                self.context.preprocess_job_id,
                "FAILED",
                error_message=message,
            )
        except Exception as exc:
            pipeline_logger.log("DB", f"preprocess failed update error: {exc}")

    async def _run_stt_chain(self) -> SttDoneEvent:
        """Audio 추출 후 STT를 실행하고 STT 완료 이벤트를 게시한다."""
        try:
            audio_settings = load_audio_settings()
            extract_settings = audio_settings.get("extract", {})
            if not isinstance(extract_settings, dict):
                extract_settings = {}

            audio_codec = str(extract_settings.get("codec", "libmp3lame"))
            audio_ext = CODEC_TO_EXT.get(audio_codec, ".wav")
            audio_path = self.config.video_root / f"{self.config.video_name}{audio_ext}"

            pipeline_logger.log("Audio", "Extracting...")
            t_audio = time.perf_counter()
            await asyncio.to_thread(
                extract_audio,
                self.config.video_path,
                audio_path,
                sample_rate=int(extract_settings.get("sample_rate", 16000)),
                channels=int(extract_settings.get("channels", 1)),
                codec=audio_codec,
                mp3_bitrate=str(extract_settings.get("mp3_bitrate", "128k")),
                mono_method=str(extract_settings.get("mono_method", "auto")),
            )
            elapsed_audio = time.perf_counter() - t_audio
            pipeline_logger.log("Audio", f"DONE ({elapsed_audio:.1f}s)")

            if self.context.sync_to_db and self.context.video_id and self.context.upload_audio_to_r2:
                await asyncio.to_thread(self._upload_audio_sync, audio_path)

            pipeline_logger.log("STT", "Analyzing...")
            t0 = time.perf_counter()
            stt_payload = await asyncio.to_thread(
                run_stt_only,
                audio_path,
                self.stt_json_path,
                backend=self.config.stt_backend,
                write_output=self.config.write_local_json,
            )
            elapsed = time.perf_counter() - t0
            pipeline_logger.log("STT", f"DONE ({elapsed:.1f}s)")

            segment_count = 0
            if isinstance(stt_payload, dict):
                segments = stt_payload.get("segments")
                if isinstance(segments, list):
                    segment_count = len(segments)

            if self.context.sync_to_db and self.context.video_id:
                await asyncio.to_thread(self._upload_stt_sync, stt_payload)

            event = SttDoneEvent(
                run_id=self.context.run_id,
                stt_json_path=self.stt_json_path,
                segment_count=segment_count,
                elapsed_sec=elapsed,
            )
            await self.orchestrator.mark_stt_done(event)
            return event
        except Exception as exc:
            pipeline_logger.log("STT", "FAILED")
            await self.orchestrator.publish_error(
                stage="stt_producer",
                message=str(exc),
                retriable=False,
            )
            raise

    def _upload_audio_sync(self, audio_path: Path) -> None:
        adapter = self._ensure_db_adapter()
        if not adapter or not self.context.video_id:
            return
        try:
            upload_result = adapter.upload_audio(self.context.video_id, audio_path)
            audio_storage_key = upload_result.get("storage_path") if isinstance(upload_result, dict) else None
            if audio_storage_key and self.context.preprocess_job_id:
                adapter.client.table("preprocessing_jobs").update(
                    {"audio_storage_key": audio_storage_key}
                ).eq("id", self.context.preprocess_job_id).execute()
        except Exception as exc:
            pipeline_logger.log("DB", f"audio upload failed: {exc}")

    def _upload_stt_sync(self, stt_payload: Any) -> None:
        adapter = self._ensure_db_adapter()
        if not adapter or not self.context.video_id:
            return
        segments = []
        if isinstance(stt_payload, dict):
            segments = stt_payload.get("segments", [])
            if isinstance(segments, dict):
                segments = segments.get("segments", [])
        elif isinstance(stt_payload, list):
            segments = stt_payload
        if not isinstance(segments, list):
            segments = []
        try:
            adapter.save_stt_result(
                self.context.video_id,
                segments,
                preprocess_job_id=self.context.preprocess_job_id,
                provider=self.config.stt_backend,
            )
        except Exception as exc:
            pipeline_logger.log("DB", f"stt upload failed: {exc}")

    async def _run_capture_loop(self) -> Tuple[int, int]:
        """Capture를 실행하고 chunk 단위로 capture 이벤트를 게시한다."""
        batch_size = self.config.capture_batch_size
        run_id = self.context.run_id
        loop = asyncio.get_running_loop()
        buffered_rows: List[Dict[str, Any]] = []
        next_batch_index = 1

        async def emit_chunk(rows: List[Dict[str, Any]], batch_index: int) -> None:
            if not rows:
                return

            enriched_rows = rows
            if self.context.sync_to_db and self.context.video_id:
                enriched_rows = await asyncio.to_thread(self._upload_captures_sync, rows)

            captures = tuple(CaptureUnit.from_mapping(row) for row in enriched_rows)
            start_ms = min(c.timestamp_ms for c in captures)
            end_ms = max(_capture_end_ms(row) for row in enriched_rows)
            if end_ms < start_ms:
                end_ms = start_ms
            time_window = TimeWindow(start_ms=start_ms, end_ms=end_ms)

            event = CaptureChunkReadyEvent(
                run_id=run_id,
                batch_index=batch_index,
                chunk_id=f"capture_chunk_{batch_index:04d}",
                captures=captures,
                time_window=time_window,
            )
            await self.orchestrator.publish_capture_chunk(event)
            self._capture_event_count += 1
            self._capture_item_count += len(captures)

        def emit_chunk_threadsafe(rows: List[Dict[str, Any]], batch_index: int) -> None:
            future = asyncio.run_coroutine_threadsafe(emit_chunk(rows, batch_index), loop)
            future.result()

        def on_capture_event(event_type: str, slide_data: Dict[str, Any]) -> None:
            nonlocal next_batch_index
            if event_type != "new":
                return

            try:
                # 유효성 검증 겸 정규화 가능 여부를 먼저 확인한다.
                CaptureUnit.from_mapping(slide_data)
                buffered_rows.append(dict(slide_data))
                if len(buffered_rows) >= batch_size:
                    rows = buffered_rows[:batch_size]
                    del buffered_rows[:batch_size]
                    curr = next_batch_index
                    next_batch_index += 1
                    emit_chunk_threadsafe(rows, curr)
            except Exception as exc:
                err_future = asyncio.run_coroutine_threadsafe(
                    self.orchestrator.publish_error(
                        stage="capture_producer.callback",
                        message=str(exc),
                        retriable=False,
                    ),
                    loop,
                )
                err_future.result()

        try:
            pipeline_logger.log("Capture", "Extracting...")
            t_capture = time.perf_counter()
            metadata = await asyncio.to_thread(
                run_capture,
                self.config.video_path,
                self.capture_output_base,
                threshold=self.config.capture_threshold,
                dedupe_threshold=self.config.capture_dedupe_threshold,
                min_interval=self.config.capture_min_interval,
                verbose=self.config.capture_verbose,
                video_name=self.config.video_name,
                write_manifest=self.config.write_local_json,
                callback=on_capture_event,
            )
            elapsed_capture = time.perf_counter() - t_capture
            pipeline_logger.log("Capture", f"DONE ({elapsed_capture:.1f}s)")

            # 콜백에서 남은 자투리 청크를 종료 전에 게시한다.
            while buffered_rows:
                rows = buffered_rows[:batch_size]
                del buffered_rows[:batch_size]
                curr = next_batch_index
                next_batch_index += 1
                await emit_chunk(rows, curr)

            # 안전장치: 콜백이 동작하지 않는 환경이면 metadata로 chunk 생성.
            if self._capture_event_count == 0 and isinstance(metadata, list) and metadata:
                for i in range(0, len(metadata), batch_size):
                    chunk = [row for row in metadata[i : i + batch_size] if isinstance(row, dict)]
                    if not chunk:
                        continue
                    curr = next_batch_index
                    next_batch_index += 1
                    await emit_chunk(chunk, curr)

            await self.orchestrator.capture_q.put(
                EndOfStreamEvent(run_id=run_id, source="capture_producer")
            )
            return self._capture_event_count, self._capture_item_count
        except Exception as exc:
            pipeline_logger.log("Capture", "FAILED")
            await self.orchestrator.publish_error(
                stage="capture_producer",
                message=str(exc),
                retriable=False,
            )
            raise

    def _upload_captures_sync(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        adapter = self._ensure_db_adapter()
        if not adapter or not self.context.video_id:
            return rows

        enriched: List[Dict[str, Any]] = []
        for row in rows:
            row_copy = dict(row)
            file_name = row_copy.get("file_name")
            storage_path = None
            if file_name:
                image_path = self.captures_dir / file_name
                if image_path.exists():
                    try:
                        upload_result = adapter.upload_capture_image(self.context.video_id, image_path)
                        storage_path = upload_result.get("storage_path") if isinstance(upload_result, dict) else None
                    except Exception as exc:
                        pipeline_logger.log("DB", f"capture upload failed ({file_name}): {exc}")
            if storage_path:
                row_copy["storage_path"] = storage_path
            enriched.append(row_copy)

        try:
            adapter.save_captures(
                self.context.video_id,
                enriched,
                preprocess_job_id=self.context.preprocess_job_id,
            )
        except Exception as exc:
            pipeline_logger.log("DB", f"capture DB save failed: {exc}")

        return enriched


__all__ = [
    "AsyncCaptureSttProducer",
    "CaptureSttProducerConfig",
    "CaptureSttProducerResult",
]
