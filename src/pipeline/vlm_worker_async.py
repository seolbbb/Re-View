"""비동기 VLM worker 연동 모듈.

현재 단계 목표:
- capture 큐 이벤트를 소비해 VLM 실행
- 오케스트레이터 semaphore로 동시성 제한
- 처리 완료 시 `VlmDoneEvent`를 게시
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.pipeline.stages import run_vlm_for_batch

from .contracts import (
    CaptureChunkReadyEvent,
    CaptureUnit,
    EndOfStreamEvent,
    PipelineContext,
    TimeWindow,
    VlmDoneEvent,
)
from .orchestrator_async import AsyncPipelineOrchestrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VlmWorkerConfig:
    """VLM worker 실행 설정."""

    video_root: Path
    captures_dir: Optional[Path]
    video_name: str
    video_id: Optional[str] = None
    vlm_batch_size: Optional[int] = None
    vlm_inner_concurrency: int = 1
    vlm_show_progress: bool = True
    max_inflight_chunks: int = 8
    write_local_json: bool = True  # vlm.json 로컬 저장 여부

    def __post_init__(self) -> None:
        if not self.video_name.strip():
            raise ValueError("video_name is required")
        if self.vlm_inner_concurrency < 1:
            raise ValueError("vlm_inner_concurrency must be >= 1")
        if self.max_inflight_chunks < 1:
            raise ValueError("max_inflight_chunks must be >= 1")


class AsyncVlmWorker:
    """capture 큐를 소비해 VLM을 실행하고 결과 이벤트를 게시한다."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        orchestrator: AsyncPipelineOrchestrator,
        config: VlmWorkerConfig,
    ) -> None:
        self.context = context
        self.orchestrator = orchestrator
        self.config = config
        self.captures_dir = config.captures_dir or (config.video_root / "captures")

    async def run(self) -> int:
        """capture 큐를 끝(EOS)까지 소비하고 처리 개수를 반환한다."""
        inflight: Set[asyncio.Task[int]] = set()
        processed_chunks = 0

        while True:
            event = await self.orchestrator.capture_q.get()
            try:
                if isinstance(event, EndOfStreamEvent):
                    break
                if not isinstance(event, CaptureChunkReadyEvent):
                    continue

                task = asyncio.create_task(self._process_chunk(event))
                inflight.add(task)

                # inflight가 커지면 최소 1개 완료를 기다려 메모리/지연 폭주를 막는다.
                if len(inflight) >= self.config.max_inflight_chunks:
                    done, pending = await asyncio.wait(
                        inflight,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    inflight = set(pending)
                    for done_task in done:
                        processed_chunks += done_task.result()
            finally:
                self.orchestrator.capture_q.task_done()

        if inflight:
            results = await asyncio.gather(*inflight, return_exceptions=False)
            processed_chunks += sum(results)

        # VLM 단계 종료를 downstream에 알린다.
        await self.orchestrator.vlm_q.put(
            EndOfStreamEvent(run_id=self.context.run_id, source="vlm_worker")
        )
        return processed_chunks

    async def _process_chunk(self, event: CaptureChunkReadyEvent) -> int:
        """단일 capture chunk를 VLM에 전달하고 완료 이벤트를 게시한다."""
        await self.orchestrator.acquire_vlm_slot()
        try:
            batch_dir = self.config.video_root / "batches" / f"batch_{event.batch_index}"
            t0 = time.perf_counter()

            # run_vlm_for_batch가 기대하는 manifest row 형태로 변환
            batch_manifest = [self._capture_to_manifest_row(unit) for unit in event.captures]

            vlm_info = await asyncio.to_thread(
                run_vlm_for_batch,
                captures_dir=self.captures_dir,
                manifest_json=None,
                video_name=self.config.video_name,
                output_dir=batch_dir,
                start_idx=None,
                end_idx=None,
                batch_manifest=batch_manifest,
                batch_size=self.config.vlm_batch_size,
                concurrency=self.config.vlm_inner_concurrency,
                show_progress=self.config.vlm_show_progress,
                start_ms=None,
                end_ms=None,
                video_id=self.config.video_id,
                write_local_json=self.config.write_local_json,
            )
            elapsed = time.perf_counter() - t0

            # 결과 처리
            image_count = 0
            vlm_json_path = None
            
            if isinstance(vlm_info, dict):
                image_count = int(vlm_info.get("image_count", len(event.captures)))
                vlm_json_path_raw = vlm_info.get("vlm_json")
                if vlm_json_path_raw:
                    vlm_json_path = Path(vlm_json_path_raw)
            else:
                 image_count = len(event.captures)
            
            if not vlm_json_path and self.config.write_local_json:
                vlm_json_path = batch_dir / "vlm.json"

            # Phase 1: VLM 완료 즉시 DB 업로드 (로컬 파일 유무와 무관하게 동작)
            if self.context.sync_to_db and self.context.video_id and self.context.processing_job_id:
                try:
                    from src.db import get_supabase_adapter, upload_vlm_results_for_batch

                    adapter = get_supabase_adapter()
                    
                    # 업로드 데이터 준비: 로컬 파일 경로 또는 메모리 상의 결과
                    upload_target = None
                    if vlm_json_path and vlm_json_path.exists():
                        upload_target = vlm_json_path
                    elif isinstance(vlm_info, dict):
                        # write_local_json=False인 경우, run_vlm_for_batch가 반환한 메모리 데이터를 사용
                        raw_results = vlm_info.get("results", [])
                        manifest_items = vlm_info.get("manifest_items", [])
                        if raw_results:
                            # raw results를 fusion 포맷으로 변환 (메모리에서 수행)
                            from src.pipeline.stages import convert_vlm_raw_to_fusion_vlm_in_memory
                            upload_target = convert_vlm_raw_to_fusion_vlm_in_memory(
                                manifest_items, raw_results
                            )
                    
                    if adapter and upload_target:
                        upload_vlm_results_for_batch(
                            adapter,
                            self.context.video_id,
                            self.context.processing_job_id,
                            upload_target,
                        )
                        logger.info(
                            "[VLM Worker] Batch %d: DB upload done (%d images)",
                            event.batch_index,
                            image_count,
                        )
                except Exception as db_exc:
                    logger.warning(
                        "[VLM Worker] Batch %d: DB upload failed: %s",
                        event.batch_index,
                        db_exc,
                    )

            vlm_done_event = VlmDoneEvent(
                run_id=self.context.run_id,
                batch_index=event.batch_index,
                chunk_id=event.chunk_id,
                vlm_json_path=vlm_json_path,
                image_count=max(0, image_count),
                captures=event.captures,
                elapsed_sec=elapsed,
                time_window=event.time_window,
            )
            await self.orchestrator.publish_vlm_done(vlm_done_event)
            return 1
        except Exception as exc:
            await self.orchestrator.publish_error(
                stage=f"vlm_worker.batch_{event.batch_index}",
                message=str(exc),
                retriable=True,
            )
            raise
        finally:
            self.orchestrator.release_vlm_slot()

    @staticmethod
    def _capture_to_manifest_row(unit: CaptureUnit) -> Dict[str, object]:
        """CaptureUnit을 `run_vlm_for_batch` 입력 row로 변환한다."""
        if unit.time_ranges:
            ranges = [
                {"start_ms": tr.start_ms, "end_ms": tr.end_ms}
                for tr in unit.time_ranges
            ]
            timestamp_ms = ranges[0]["start_ms"]
        else:
            ranges = [{"start_ms": unit.timestamp_ms, "end_ms": unit.timestamp_ms + 1000}]
            timestamp_ms = unit.timestamp_ms

        row: Dict[str, object] = {
            "id": unit.capture_id,
            "cap_id": unit.capture_id,
            "file_name": unit.file_name,
            "time_ranges": ranges,
            "timestamp_ms": timestamp_ms,
            "start_ms": timestamp_ms,
        }
        if unit.storage_path:
            row["storage_path"] = unit.storage_path
        return row


__all__ = [
    "AsyncVlmWorker",
    "VlmWorkerConfig",
]
