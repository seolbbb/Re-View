"""Role-aware Google API key ordering and temporary cooldown helpers."""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

_ROLE_START_OFFSETS = {
    "summarizer": 0,  # 1 -> 2 -> 3
    "judge": 1,       # 2 -> 3 -> 1
    "chat": 2,        # 3 -> 1 -> 2
}

_LOCK = threading.Lock()
_KEY_COOLDOWN_UNTIL: Dict[str, float] = {}


def normalize_google_role(role: Optional[str]) -> str:
    value = (role or "").strip().lower()
    return value if value in _ROLE_START_OFFSETS else "summarizer"


def make_google_key_id(raw_key: Optional[str], *, fallback_label: str = "default") -> str:
    if not raw_key:
        return f"fallback:{fallback_label}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"key:{digest}"


def _build_ordered_indices(total: int, role: str) -> List[int]:
    if total <= 0:
        return []
    start = _ROLE_START_OFFSETS.get(normalize_google_role(role), 0) % total
    return [((start + offset) % total) for offset in range(total)]


def order_google_key_indices(
    key_ids: Sequence[str],
    role: Optional[str],
) -> Tuple[List[int], Dict[int, float]]:
    """Return key indices in role-priority order with cooling keys pushed to the tail.

    Returns:
        ordered_indices: Full index order (usable immediately). If all keys are cooling
            down, their original role order is returned to avoid dead-end behavior.
        cooling_remaining: Mapping index -> remaining cooldown seconds (>0).
    """
    ordered = _build_ordered_indices(len(key_ids), normalize_google_role(role))
    now = time.monotonic()
    active: List[int] = []
    cooling: List[int] = []
    cooling_remaining: Dict[int, float] = {}

    with _LOCK:
        for idx in ordered:
            key_id = key_ids[idx]
            remaining = _KEY_COOLDOWN_UNTIL.get(key_id, 0.0) - now
            if remaining > 0:
                cooling.append(idx)
                cooling_remaining[idx] = remaining
            else:
                active.append(idx)

    if active:
        return active + cooling, cooling_remaining
    return ordered, cooling_remaining


def mark_google_key_cooldown(key_id: str, cooldown_sec: float) -> None:
    duration = max(0.0, float(cooldown_sec))
    if duration <= 0:
        return
    until = time.monotonic() + duration
    with _LOCK:
        current = _KEY_COOLDOWN_UNTIL.get(key_id, 0.0)
        if until > current:
            _KEY_COOLDOWN_UNTIL[key_id] = until

