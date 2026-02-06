"""비동기 파이프라인 오케스트레이터 뼈대.

이 모듈은 현재 단계에서 Queue/Event/Semaphore 기반 제어면만 정의한다.
실제 Capture/STT/VLM/Fusion 실행 로직은 이후 단계에서 연결한다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Union

from .contracts import (
    CaptureChunkReadyEvent,
    EndOfStreamEvent,
    FusionDoneEvent,
    JudgeDoneEvent,
    PipelineContext,
    PipelineErrorEvent,
    SttDoneEvent,
    SummaryDoneEvent,
    VlmDoneEvent,
)

logger = logging.getLogger(__name__)


# Queue별 이벤트 타입(현재 스켈레톤 기준)
CaptureQueueEvent = Union[CaptureChunkReadyEvent, EndOfStreamEvent]
VlmQueueEvent = Union[VlmDoneEvent, EndOfStreamEvent]
FusionQueueEvent = Union[FusionDoneEvent, SummaryDoneEvent, JudgeDoneEvent, EndOfStreamEvent]
ErrorQueueEvent = Union[PipelineErrorEvent, EndOfStreamEvent]


@dataclass(frozen=True)
class OrchestratorSnapshot:
    """오케스트레이터 상태 스냅샷."""

    started: bool
    stopped: bool
    stt_ready: bool
    capture_q_size: int
    vlm_q_size: int
    fusion_q_size: int
    error_q_size: int


class AsyncPipelineOrchestrator:
    """Queue/Event/Semaphore를 보유하는 파이프라인 제어 객체."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        queue_maxsize: int = 0,
        vlm_parallelism: Optional[int] = None,
    ) -> None:
        if queue_maxsize < 0:
            raise ValueError("queue_maxsize must be >= 0")

        self.context = context

        # 단계 간 이벤트 전달 큐
        self.capture_q: asyncio.Queue[CaptureQueueEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self.vlm_q: asyncio.Queue[VlmQueueEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self.fusion_q: asyncio.Queue[FusionQueueEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self.error_q: asyncio.Queue[ErrorQueueEvent] = asyncio.Queue(maxsize=queue_maxsize)

        # STT 완료 게이트
        self.stt_done_event = asyncio.Event()
        self.stt_done_payload: Optional[SttDoneEvent] = None

        # VLM 동시성 제한
        parallelism = vlm_parallelism if vlm_parallelism is not None else context.vlm_concurrency
        if parallelism < 1:
            raise ValueError("vlm_parallelism must be >= 1")
        self.vlm_semaphore = asyncio.Semaphore(parallelism)

        # 수명 주기 상태
        self._started = False
        self._stopped = False
        self._stop_event = asyncio.Event()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def stopped(self) -> bool:
        return self._stopped

    def snapshot(self) -> OrchestratorSnapshot:
        """현재 오케스트레이터 상태를 반환한다."""
        return OrchestratorSnapshot(
            started=self._started,
            stopped=self._stopped,
            stt_ready=self.stt_done_event.is_set(),
            capture_q_size=self.capture_q.qsize(),
            vlm_q_size=self.vlm_q.qsize(),
            fusion_q_size=self.fusion_q.qsize(),
            error_q_size=self.error_q.qsize(),
        )

    async def start(self) -> None:
        """오케스트레이터를 시작 상태로 전환한다."""
        if self._started:
            logger.debug("[Orchestrator] already started: run_id=%s", self.context.run_id)
            return

        self._started = True
        self._stopped = False
        logger.info(
            "[Orchestrator] start run_id=%s video=%s batch_size=%d vlm_concurrency=%d",
            self.context.run_id,
            self.context.video_name,
            self.context.batch_size,
            self.context.vlm_concurrency,
        )

    async def stop(self, *, reason: str = "normal_shutdown") -> None:
        """오케스트레이터를 종료 상태로 전환하고 EOS를 전파한다."""
        if self._stopped:
            return

        eos = EndOfStreamEvent(run_id=self.context.run_id, source=reason)
        await self.capture_q.put(eos)
        await self.vlm_q.put(eos)
        await self.fusion_q.put(eos)
        await self.error_q.put(eos)

        self._stop_event.set()
        self._stopped = True
        logger.info("[Orchestrator] stop run_id=%s reason=%s", self.context.run_id, reason)

    async def wait_until_stopped(self) -> None:
        """외부 종료 시그널까지 대기한다."""
        await self._stop_event.wait()

    async def mark_stt_done(self, event: SttDoneEvent) -> None:
        """STT 완료 게이트를 열고 payload를 저장한다."""
        self.stt_done_payload = event
        self.stt_done_event.set()
        logger.info(
            "[Orchestrator] stt_done run_id=%s segments=%d elapsed=%s",
            event.run_id,
            event.segment_count,
            event.elapsed_sec,
        )

    async def wait_stt_ready(self) -> SttDoneEvent:
        """STT 완료 이벤트까지 대기 후 payload를 반환한다."""
        await self.stt_done_event.wait()
        if self.stt_done_payload is None:
            raise RuntimeError("stt_done_event set but payload is missing")
        return self.stt_done_payload

    async def publish_capture_chunk(self, event: CaptureChunkReadyEvent) -> None:
        """Capture 청크 이벤트를 큐에 게시한다."""
        await self.capture_q.put(event)
        logger.debug(
            "[Orchestrator] capture_chunk queued run_id=%s batch=%d chunk=%s qsize=%d",
            event.run_id,
            event.batch_index,
            event.chunk_id,
            self.capture_q.qsize(),
        )

    async def publish_vlm_done(self, event: VlmDoneEvent) -> None:
        """VLM 완료 이벤트를 큐에 게시한다."""
        await self.vlm_q.put(event)
        logger.debug(
            "[Orchestrator] vlm_done queued run_id=%s batch=%d chunk=%s qsize=%d",
            event.run_id,
            event.batch_index,
            event.chunk_id,
            self.vlm_q.qsize(),
        )

    async def publish_fusion_done(self, event: FusionDoneEvent) -> None:
        """Fusion 완료 이벤트를 큐에 게시한다."""
        await self.fusion_q.put(event)
        logger.debug(
            "[Orchestrator] fusion_done queued run_id=%s batch=%d segments=%d qsize=%d",
            event.run_id,
            event.batch_index,
            event.segment_count,
            self.fusion_q.qsize(),
        )

    async def publish_summary_done(self, event: SummaryDoneEvent) -> None:
        """Summary 완료 이벤트를 큐에 게시한다."""
        await self.fusion_q.put(event)
        logger.debug(
            "[Orchestrator] summary_done queued run_id=%s batch=%d qsize=%d",
            event.run_id,
            event.batch_index,
            self.fusion_q.qsize(),
        )

    async def publish_judge_done(self, event: JudgeDoneEvent) -> None:
        """Judge 완료 이벤트를 큐에 게시한다."""
        await self.fusion_q.put(event)
        logger.debug(
            "[Orchestrator] judge_done queued run_id=%s batch=%d score=%.2f qsize=%d",
            event.run_id,
            event.batch_index,
            event.score,
            self.fusion_q.qsize(),
        )

    async def publish_error(
        self,
        *,
        stage: str,
        message: str,
        retriable: bool = False,
    ) -> None:
        """오류 이벤트를 에러 큐에 게시한다."""
        event = PipelineErrorEvent(
            run_id=self.context.run_id,
            stage=stage,
            message=message,
            retriable=retriable,
        )
        await self.error_q.put(event)
        logger.error(
            "[Orchestrator] error run_id=%s stage=%s retriable=%s msg=%s",
            self.context.run_id,
            stage,
            retriable,
            message,
        )

    async def acquire_vlm_slot(self) -> None:
        """VLM 병렬 슬롯을 획득한다."""
        await self.vlm_semaphore.acquire()

    def release_vlm_slot(self) -> None:
        """VLM 병렬 슬롯을 반환한다."""
        self.vlm_semaphore.release()


__all__ = [
    "AsyncPipelineOrchestrator",
    "CaptureQueueEvent",
    "ErrorQueueEvent",
    "FusionQueueEvent",
    "OrchestratorSnapshot",
    "VlmQueueEvent",
]
