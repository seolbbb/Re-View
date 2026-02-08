"""Pipeline cancellation helpers.

We support two cancellation signals:
1) In-memory (same-process): fast signal for FastAPI BackgroundTasks.
2) DB marker (cross-instance): videos.delete_requested_at set.

The DB marker is the source of truth for Cloud Run / multi-worker deployments.
The in-memory marker is best-effort to stop work quickly inside a single process.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


class PipelineCanceled(RuntimeError):
    """Raised when a running pipeline should stop early (video deleted or delete requested)."""


@dataclass(frozen=True)
class CancelCheckResult:
    canceled: bool
    reason: Optional[str] = None


_LOCAL_CANCELLED_VIDEO_IDS: set[str] = set()


def request_local_cancel(video_id: str) -> None:
    if video_id:
        _LOCAL_CANCELLED_VIDEO_IDS.add(str(video_id))


def clear_local_cancel(video_id: str) -> None:
    if video_id:
        _LOCAL_CANCELLED_VIDEO_IDS.discard(str(video_id))


def is_local_cancel_requested(video_id: str) -> bool:
    return bool(video_id) and str(video_id) in _LOCAL_CANCELLED_VIDEO_IDS


def _coerce_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def check_cancel_requested(adapter: Any, video_id: Optional[str]) -> CancelCheckResult:
    """Return cancellation decision for a video_id.

    - If local cancel is requested: cancel.
    - If adapter is provided, treat missing video row as cancel.
    - If videos.delete_requested_at is set: cancel.
    """
    if not video_id:
        return CancelCheckResult(False, None)

    if is_local_cancel_requested(video_id):
        return CancelCheckResult(True, "local_cancel")

    if not adapter:
        return CancelCheckResult(False, None)

    try:
        video = adapter.get_video(video_id)
    except Exception:
        # If we can't check, don't assume cancel.
        return CancelCheckResult(False, None)

    if not video:
        return CancelCheckResult(True, "video_deleted")

    dt = _coerce_dt(video.get("delete_requested_at"))
    if dt is not None:
        return CancelCheckResult(True, "delete_requested")

    return CancelCheckResult(False, None)


def raise_if_cancel_requested(adapter: Any, video_id: Optional[str]) -> None:
    decision = check_cancel_requested(adapter, video_id)
    if decision.canceled:
        raise PipelineCanceled(decision.reason or "canceled")

