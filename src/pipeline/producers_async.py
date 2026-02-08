"""
[Intent]
비동기 파이프라인에서 Capture(슬라이드 추출)와 STT(음성→텍스트) 두 Producer를
asyncio 기반으로 병렬 실행하고, 결과를 오케스트레이터 큐에 이벤트로 게시하는 모듈입니다.
기존 blocking 함수를 asyncio.to_thread로 감싸 non-blocking으로 실행합니다.

[Usage]
- src/run_pipeline_demo_async.py: run_async_demo() 내에서 AsyncCaptureSttProducer 인스턴스 생성 및 실행
- src/pipeline/orchestrator_async.py: capture_q, stt_gate 등 큐/이벤트와 연동

[Usage Method]
- CaptureSttProducerConfig에 config/capture/settings.yaml의 12개 파라미터를 주입
- AsyncCaptureSttProducer.run()을 await하면 내부에서:
  1) _run_stt_chain(): extract_audio → run_stt_only → SttDoneEvent 게시
  2) _run_capture_loop(): run_capture(blocking) → 콜백으로 chunk 단위 CaptureChunkReadyEvent 게시
  두 태스크가 asyncio.gather()로 병렬 실행됨
- sync_to_db=True 시 Supabase API를 통해 캡처 이미지/STT 결과를 업로드
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
    """
    [Usage File]
    - src/run_pipeline_demo_async.py: run_async_demo() 내에서 인스턴스 생성

    [Purpose]
    Capture/STT producer 실행에 필요한 모든 설정을 담는 불변 데이터 클래스입니다.
    config/capture/settings.yaml의 12개 캡처 파라미터를 1:1로 매핑합니다.

    [Connection]
    - config/capture/settings.yaml → CaptureSettings → 이 Config로 전달
    - _run_capture_loop()에서 capture_threshold, capture_sample_interval이 run_capture()에 전달
    - 나머지 캡처 파라미터는 process_content.py → HybridSlideExtractor에서 settings.yaml 직접 로드로 적용
    """

    video_path: Path                # 입력 비디오 파일 절대 경로
    video_root: Path                # 비디오별 출력 루트 ({output_base}/{video_name})
    video_name: str                 # 비디오 식별용 정규화된 이름
    capture_output_base: Optional[Path] = None  # 캡처 출력 최상위 디렉토리 (None이면 video_root.parent)
    stt_backend: str = "clova"      # STT 백엔드 ("clova", "whisper" 등)
    write_local_json: bool = True   # STT/캡처 결과를 로컬 JSON으로 저장할지 여부
    capture_batch_size: int = 6     # VLM/Fusion에 전달할 캡처 청크 크기

    # --- config/capture/settings.yaml 1:1 매핑 (기본값 = settings.yaml 기본값) ---
    capture_threshold: float = 0.15             # persistence_drop_ratio: 슬라이드 종료 판정 비율 (0.0~1.0)
    capture_sample_interval: float = 1.0        # sample_interval_sec: 프레임 샘플링 간격 (초)
    capture_persistence_threshold: int = 6      # persistence_threshold: 특징점 지속 프레임 수
    capture_min_orb_features: int = 50          # min_orb_features: 최소 특징점 개수
    capture_dedup_phash_threshold: int = 12     # dedup_phash_threshold: pHash 해밍 거리 임계값
    capture_dedup_orb_distance: int = 60        # dedup_orb_distance: ORB 매칭 거리 임계값
    capture_dedup_sim_threshold: float = 0.7    # dedup_sim_threshold: 유사도 비율 임계값
    capture_enable_roi_detection: bool = True   # enable_roi_detection: ROI 자동 감지 활성화
    capture_roi_padding: int = 5                # roi_padding: ROI 영역 여백 (픽셀)
    capture_enable_smart_roi: bool = True       # enable_smart_roi: Smart ROI Median Lock 활성화
    capture_roi_warmup_frames: int = 30         # roi_warmup_frames: Smart ROI 수집 프레임 수
    capture_enable_adaptive_resize: bool = True # enable_adaptive_resize: 계층형 리사이징 활성화

    capture_verbose: bool = False   # 캡처 상세 로그 출력 여부

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
                min_interval=self.config.capture_sample_interval,
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
        missing_count = 0
        missing_examples: List[str] = []
        for row in rows:
            row_copy = dict(row)
            file_name = row_copy.get("file_name")
            storage_path = None
            if file_name:
                image_path = self.captures_dir / file_name
                if not image_path.exists():
                    missing_count += 1
                    if len(missing_examples) < 3:
                        missing_examples.append(str(file_name))
                else:
                    try:
                        upload_result = adapter.upload_capture_image(self.context.video_id, image_path)
                        storage_path = upload_result.get("storage_path") if isinstance(upload_result, dict) else None
                    except Exception as exc:
                        pipeline_logger.log("DB", f"capture upload failed ({file_name}): {exc}")
            if storage_path:
                row_copy["storage_path"] = storage_path
            enriched.append(row_copy)

        if missing_count:
            examples = ", ".join(missing_examples)
            extra = f" (e.g. {examples})" if examples else ""
            pipeline_logger.log(
                "DB",
                f"capture upload skipped: {missing_count} missing file(s) under {self.captures_dir}{extra}",
            )

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
