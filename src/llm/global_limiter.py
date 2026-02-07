"""Google LLM 호출의 프로세스-전역 동시성 제한 유틸리티."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _coerce_limit(value: Optional[int], default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


_DEFAULT_LIMIT = _coerce_limit(os.getenv("GOOGLE_LLM_MAX_CONCURRENT"), 8)
_STATE_LOCK = threading.Lock()
_SEMAPHORE = threading.Semaphore(_DEFAULT_LIMIT)
_LIMIT = _DEFAULT_LIMIT
_IN_FLIGHT = 0
_WAITING = 0


@dataclass(frozen=True)
class LimiterSnapshot:
    limit: int
    in_flight: int
    waiting: int


@dataclass(frozen=True)
class AcquireTicket:
    waited_sec: float
    snapshot: LimiterSnapshot


def configure_google_global_limit(limit: Optional[int], *, source: str = "unknown") -> int:
    """전역 동시성 제한 값을 구성한다.

    활성 요청이 없을 때만 값 변경을 허용한다.
    """
    global _LIMIT, _SEMAPHORE
    target = _coerce_limit(limit, _LIMIT)
    with _STATE_LOCK:
        if target == _LIMIT:
            return _LIMIT
        if _IN_FLIGHT > 0 or _WAITING > 0:
            logger.warning(
                "[LLM-Limiter] Ignore reconfigure while active: source=%s requested=%d current=%d in_flight=%d waiting=%d",
                source,
                target,
                _LIMIT,
                _IN_FLIGHT,
                _WAITING,
            )
            return _LIMIT
        _LIMIT = target
        _SEMAPHORE = threading.Semaphore(_LIMIT)
        logger.info("[LLM-Limiter] configured max_concurrent=%d source=%s", _LIMIT, source)
        return _LIMIT


def get_google_limiter_snapshot() -> LimiterSnapshot:
    with _STATE_LOCK:
        return LimiterSnapshot(limit=_LIMIT, in_flight=_IN_FLIGHT, waiting=_WAITING)


@contextmanager
def acquire_google_slot() -> Iterator[AcquireTicket]:
    """Google LLM 호출 슬롯을 획득/반납한다."""
    global _IN_FLIGHT, _WAITING
    queued_at = time.monotonic()
    with _STATE_LOCK:
        _WAITING += 1
    _SEMAPHORE.acquire()
    waited = time.monotonic() - queued_at

    with _STATE_LOCK:
        _WAITING = max(0, _WAITING - 1)
        _IN_FLIGHT += 1
        snapshot = LimiterSnapshot(limit=_LIMIT, in_flight=_IN_FLIGHT, waiting=_WAITING)

    try:
        yield AcquireTicket(waited_sec=waited, snapshot=snapshot)
    finally:
        with _STATE_LOCK:
            _IN_FLIGHT = max(0, _IN_FLIGHT - 1)
        _SEMAPHORE.release()

