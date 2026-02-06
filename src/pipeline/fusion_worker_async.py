"""비동기 Fusion/Summary/Judge worker 연동 모듈.

현재 단계 목표:
- VLM 완료 이벤트를 소비해 Fusion -> Summary -> Judge를 실행
- STT 완료 이벤트(`stt_done_event`)를 게이트로 사용
- batch_index 순서를 보장해 결과 이벤트를 게시
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.pipeline.benchmark import BenchmarkTimer
from src.pipeline.stages import run_batch_fusion_pipeline

from .contracts import (
    CaptureUnit,
    EndOfStreamEvent,
    FusionDoneEvent,
    JudgeDoneEvent,
    PipelineContext,
    SummaryDoneEvent,
    VlmDoneEvent,
)
from .orchestrator_async import AsyncPipelineOrchestrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionWorkerConfig:
    """Fusion/Summary/Judge worker 실행 설정."""

    video_root: Path
    video_name: str
    captures_dir: Optional[Path] = None
    repo_root: Optional[Path] = None
    batch_size: Optional[int] = None
    vlm_batch_size: Optional[int] = None
    vlm_concurrency: int = 1
    vlm_show_progress: bool = False
    limit: Optional[int] = None
    strict_batch_order: bool = True

    def __post_init__(self) -> None:
        if not self.video_name.strip():
            raise ValueError("video_name is required")
        if self.vlm_concurrency < 1:
            raise ValueError("vlm_concurrency must be >= 1")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.vlm_batch_size is not None and self.vlm_batch_size < 1:
            raise ValueError("vlm_batch_size must be >= 1")


class AsyncFusionSummaryJudgeWorker:
    """VLM 결과를 받아 Fusion/Summary/Judge를 실행하고 완료 이벤트를 게시한다."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        orchestrator: AsyncPipelineOrchestrator,
        config: FusionWorkerConfig,
    ) -> None:
        self.context = context
        self.orchestrator = orchestrator
        self.config = config

        self.captures_dir = config.captures_dir or (config.video_root / "captures")
        self.repo_root = config.repo_root or Path(__file__).resolve().parents[2]
        self._db_adapter = None

    async def run(self) -> int:
        """VLM 큐를 EOS까지 소비하고 처리한 배치 수를 반환한다."""
        pending: Dict[int, VlmDoneEvent] = {}
        next_batch_index = 1
        eos_received = False
        processed_batches = 0

        while True:
            event = await self.orchestrator.vlm_q.get()
            try:
                if isinstance(event, EndOfStreamEvent):
                    eos_received = True
                elif isinstance(event, VlmDoneEvent):
                    if event.batch_index in pending:
                        logger.warning(
                            "[FusionWorker] duplicate batch=%d; replacing previous event",
                            event.batch_index,
                        )
                    pending[event.batch_index] = event
                else:
                    continue

                while True:
                    current = pending.pop(next_batch_index, None)
                    if current is not None:
                        await self._process_batch(current)
                        processed_batches += 1
                        next_batch_index += 1
                        continue

                    if eos_received:
                        if not pending:
                            await self.orchestrator.fusion_q.put(
                                EndOfStreamEvent(
                                    run_id=self.context.run_id,
                                    source="fusion_worker",
                                )
                            )
                            return processed_batches

                        fallback_index = min(pending.keys())
                        if self.config.strict_batch_order and fallback_index > next_batch_index:
                            logger.warning(
                                "[FusionWorker] missing batch events before %d; "
                                "skip to next available batch=%d",
                                next_batch_index,
                                fallback_index,
                            )
                        next_batch_index = fallback_index
                        continue
                    break
            finally:
                self.orchestrator.vlm_q.task_done()

    async def _process_batch(self, event: VlmDoneEvent) -> None:
        """단일 배치를 Fusion/Summary/Judge로 처리하고 이벤트를 발행한다."""
        stt_event = await self.orchestrator.wait_stt_ready()

        batch_manifest = [self._capture_to_manifest_row(unit) for unit in event.captures]
        batch_size = self.config.batch_size or len(batch_manifest) or 1
        forced_batch_end_ms = event.time_window.end_ms if event.time_window else None
        timer = BenchmarkTimer()
        t0 = time.perf_counter()

        try:
            adapter = None
            if self.context.sync_to_db:
                if self._db_adapter is None:
                    from src.db import get_supabase_adapter

                    self._db_adapter = get_supabase_adapter()
                if not self._db_adapter:
                    raise RuntimeError(
                        "Supabase adapter not configured; DB sync requires SUPABASE_URL/SUPABASE_KEY."
                    )
                adapter = self._db_adapter

                # 배치 시작 시점에 current_batch 업데이트
                if self.context.processing_job_id:
                    try:
                        adapter.update_processing_job_progress(
                            self.context.processing_job_id,
                            event.batch_index,
                            None,  # total_batch는 마지막에 설정
                        )
                    except Exception as progress_exc:
                        logger.warning(
                            "[FusionWorker] Failed to update progress: %s", progress_exc
                        )

            await asyncio.to_thread(
                run_batch_fusion_pipeline,
                video_root=self.config.video_root,
                captures_dir=self.captures_dir,
                manifest_json=None,
                captures_data=batch_manifest,
                stt_json=stt_event.stt_json_path,
                video_name=self.config.video_name,
                batch_size=batch_size,
                timer=timer,
                vlm_batch_size=self.config.vlm_batch_size,
                vlm_concurrency=self.config.vlm_concurrency,
                vlm_show_progress=self.config.vlm_show_progress,
                limit=self.config.limit,
                repo_root=self.repo_root,
                skip_vlm=True,
                start_batch_index=event.batch_index - 1,
                preserve_files=True,
                forced_batch_end_ms=forced_batch_end_ms,
                processing_job_id=self.context.processing_job_id,
                video_id=self.context.video_id,
                sync_to_db=self.context.sync_to_db,
                adapter=adapter,
            )

            batch_dir = self.config.video_root / "batches" / f"batch_{event.batch_index}"
            fusion_dir = batch_dir / "fusion"

            segments_units_path = fusion_dir / "segments_units.jsonl"
            segment_summaries_path = fusion_dir / "segment_summaries.jsonl"
            judge_json_path = batch_dir / "judge.json"

            segment_count = self._count_jsonl_lines(segments_units_path)
            elapsed = time.perf_counter() - t0

            await self.orchestrator.publish_fusion_done(
                FusionDoneEvent(
                    run_id=self.context.run_id,
                    batch_index=event.batch_index,
                    segments_units_jsonl_path=segments_units_path,
                    segment_summaries_jsonl_path=segment_summaries_path,
                    segment_count=segment_count,
                    elapsed_sec=elapsed,
                )
            )
            await self.orchestrator.publish_summary_done(
                SummaryDoneEvent(
                    run_id=self.context.run_id,
                    batch_index=event.batch_index,
                    output_path=segment_summaries_path,
                )
            )
            await self.orchestrator.publish_judge_done(
                JudgeDoneEvent(
                    run_id=self.context.run_id,
                    batch_index=event.batch_index,
                    score=self._read_judge_score(judge_json_path),
                    output_path=judge_json_path,
                )
            )


        except Exception as exc:
            await self.orchestrator.publish_error(
                stage=f"fusion_worker.batch_{event.batch_index}",
                message=str(exc),
                retriable=True,
            )
            raise

    @staticmethod
    def _count_jsonl_lines(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return sum(1 for _ in path.open("rb"))
        except Exception:
            return 0

    @staticmethod
    def _read_judge_score(path: Path) -> float:
        if not path.exists():
            return 0.0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            score = float(payload.get("final_score", 0.0))
            return max(0.0, score)
        except Exception:
            return 0.0

    @staticmethod
    def _capture_to_manifest_row(unit: CaptureUnit) -> Dict[str, object]:
        """CaptureUnit을 batch fusion 입력 row로 변환한다."""
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
    "AsyncFusionSummaryJudgeWorker",
    "FusionWorkerConfig",
]

