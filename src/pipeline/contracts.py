"""비동기 파이프라인 이벤트 계약(contracts) 정의 모듈.

Queue/Event 기반 오케스트레이션에서 주고받는 payload를
강한 타입(dataclass)으로 통일해 사용하기 위한 파일이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Union


def _utc_now_iso() -> str:
    """현재 UTC 시간을 ISO-8601 문자열로 반환한다."""
    return datetime.now(timezone.utc).isoformat()


def _to_int(value: Any, default: int = 0) -> int:
    """정수 변환 헬퍼.

    변환 실패 시 예외를 올리지 않고 기본값을 반환한다.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TimeWindow:
    """밀리초 단위 구간(start/end) 표현."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("start_ms must be >= 0")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TimeWindow":
        """사전 형태 입력을 안전하게 TimeWindow로 변환한다."""
        start_ms = _to_int(payload.get("start_ms"), default=0)
        end_ms = _to_int(payload.get("end_ms"), default=start_ms)
        if end_ms < start_ms:
            end_ms = start_ms
        return cls(start_ms=start_ms, end_ms=end_ms)


@dataclass(frozen=True)
class CaptureUnit:
    """단일 캡처(이미지) 메타데이터 단위."""

    capture_id: str
    file_name: str
    timestamp_ms: int
    storage_path: Optional[str] = None
    time_ranges: tuple[TimeWindow, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CaptureUnit":
        """DB/manifest row를 CaptureUnit으로 정규화한다."""
        capture_id = str(payload.get("cap_id") or payload.get("id") or "").strip()
        if not capture_id:
            raise ValueError("capture_id is required")

        file_name = str(payload.get("file_name") or "").strip()
        if not file_name:
            raise ValueError("file_name is required")

        raw_ranges = payload.get("time_ranges")
        ranges: tuple[TimeWindow, ...] = ()
        if isinstance(raw_ranges, list):
            ranges = tuple(
                TimeWindow.from_mapping(rng) for rng in raw_ranges if isinstance(rng, Mapping)
            )

        if "timestamp_ms" in payload:
            timestamp_ms = _to_int(payload.get("timestamp_ms"), default=0)
        elif ranges:
            # time_ranges가 있으면 첫 구간 시작 시간을 대표 timestamp로 사용
            timestamp_ms = ranges[0].start_ms
        else:
            timestamp_ms = _to_int(payload.get("start_ms"), default=0)

        storage_path_raw = payload.get("storage_path")
        storage_path = str(storage_path_raw).strip() if storage_path_raw else None

        return cls(
            capture_id=capture_id,
            file_name=file_name,
            timestamp_ms=max(0, timestamp_ms),
            storage_path=storage_path,
            time_ranges=ranges,
        )


@dataclass(frozen=True)
class PipelineContext:
    """파이프라인 실행 단위(run) 공통 컨텍스트."""

    run_id: str
    video_name: str
    output_root: Path
    video_id: Optional[str] = None
    processing_job_id: Optional[str] = None
    batch_size: int = 1
    vlm_batch_size: Optional[int] = None
    vlm_concurrency: int = 1
    created_at_utc: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not self.video_name.strip():
            raise ValueError("video_name is required")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.vlm_batch_size is not None and self.vlm_batch_size < 1:
            raise ValueError("vlm_batch_size must be >= 1")
        if self.vlm_concurrency < 1:
            raise ValueError("vlm_concurrency must be >= 1")


class PipelineEventType(str, Enum):
    """파이프라인 이벤트 타입 집합."""

    CAPTURE_CHUNK_READY = "capture_chunk_ready"
    STT_DONE = "stt_done"
    VLM_DONE = "vlm_done"
    FUSION_DONE = "fusion_done"
    SUMMARY_DONE = "summary_done"
    JUDGE_DONE = "judge_done"
    PIPELINE_ERROR = "pipeline_error"
    END_OF_STREAM = "end_of_stream"


@dataclass(frozen=True)
class PipelineEvent:
    """모든 이벤트의 공통 베이스."""

    run_id: str
    event_type: PipelineEventType
    created_at_utc: str = field(default_factory=_utc_now_iso, kw_only=True)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")


@dataclass(frozen=True)
class CaptureChunkReadyEvent(PipelineEvent):
    """캡처 청크가 준비되었음을 알리는 이벤트."""

    batch_index: int
    chunk_id: str
    captures: tuple[CaptureUnit, ...]
    time_window: Optional[TimeWindow] = None
    event_type: PipelineEventType = field(
        default=PipelineEventType.CAPTURE_CHUNK_READY,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_index < 1:
            raise ValueError("batch_index must be >= 1")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id is required")
        if not self.captures:
            raise ValueError("captures must not be empty")


@dataclass(frozen=True)
class SttDoneEvent(PipelineEvent):
    """STT 완료 이벤트."""

    stt_json_path: Path
    segment_count: int
    elapsed_sec: Optional[float] = None
    event_type: PipelineEventType = field(default=PipelineEventType.STT_DONE, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.segment_count < 0:
            raise ValueError("segment_count must be >= 0")
        if self.elapsed_sec is not None and self.elapsed_sec < 0:
            raise ValueError("elapsed_sec must be >= 0")


@dataclass(frozen=True)
class VlmDoneEvent(PipelineEvent):
    """VLM 처리 완료 이벤트."""

    batch_index: int
    chunk_id: str
    vlm_json_path: Path
    image_count: int
    captures: tuple[CaptureUnit, ...] = ()
    elapsed_sec: Optional[float] = None
    time_window: Optional[TimeWindow] = None
    event_type: PipelineEventType = field(default=PipelineEventType.VLM_DONE, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_index < 1:
            raise ValueError("batch_index must be >= 1")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id is required")
        if self.image_count < 0:
            raise ValueError("image_count must be >= 0")
        if self.elapsed_sec is not None and self.elapsed_sec < 0:
            raise ValueError("elapsed_sec must be >= 0")


@dataclass(frozen=True)
class FusionDoneEvent(PipelineEvent):
    """Fusion 단계 완료 이벤트."""

    batch_index: int
    segments_units_jsonl_path: Path
    segment_summaries_jsonl_path: Path
    segment_count: int
    elapsed_sec: Optional[float] = None
    event_type: PipelineEventType = field(default=PipelineEventType.FUSION_DONE, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_index < 1:
            raise ValueError("batch_index must be >= 1")
        if self.segment_count < 0:
            raise ValueError("segment_count must be >= 0")
        if self.elapsed_sec is not None and self.elapsed_sec < 0:
            raise ValueError("elapsed_sec must be >= 0")


@dataclass(frozen=True)
class SummaryDoneEvent(PipelineEvent):
    """요약 단계 완료 이벤트."""

    batch_index: int
    output_path: Path
    event_type: PipelineEventType = field(default=PipelineEventType.SUMMARY_DONE, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_index < 1:
            raise ValueError("batch_index must be >= 1")


@dataclass(frozen=True)
class JudgeDoneEvent(PipelineEvent):
    """Judge 단계 완료 이벤트."""

    batch_index: int
    score: float
    output_path: Path
    event_type: PipelineEventType = field(default=PipelineEventType.JUDGE_DONE, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_index < 1:
            raise ValueError("batch_index must be >= 1")
        if self.score < 0:
            raise ValueError("score must be >= 0")


@dataclass(frozen=True)
class PipelineErrorEvent(PipelineEvent):
    """파이프라인 오류 이벤트."""

    stage: str
    message: str
    retriable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    event_type: PipelineEventType = field(default=PipelineEventType.PIPELINE_ERROR, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.stage.strip():
            raise ValueError("stage is required")
        if not self.message.strip():
            raise ValueError("message is required")


@dataclass(frozen=True)
class EndOfStreamEvent(PipelineEvent):
    """소스 스트림 종료(더 이상 입력 없음) 이벤트."""

    source: str
    event_type: PipelineEventType = field(default=PipelineEventType.END_OF_STREAM, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.source.strip():
            raise ValueError("source is required")

# 파이프라인 큐에서 전달 가능한 이벤트 payload 유니온 타입.
PipelineEventPayload = Union[
    CaptureChunkReadyEvent,
    SttDoneEvent,
    VlmDoneEvent,
    FusionDoneEvent,
    SummaryDoneEvent,
    JudgeDoneEvent,
    PipelineErrorEvent,
    EndOfStreamEvent,
]


__all__ = [
    "CaptureChunkReadyEvent",
    "CaptureUnit",
    "EndOfStreamEvent",
    "FusionDoneEvent",
    "JudgeDoneEvent",
    "PipelineContext",
    "PipelineErrorEvent",
    "PipelineEvent",
    "PipelineEventPayload",
    "PipelineEventType",
    "SttDoneEvent",
    "SummaryDoneEvent",
    "TimeWindow",
    "VlmDoneEvent",
]
