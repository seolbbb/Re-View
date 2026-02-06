from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .langgraph_session import LangGraphSession
from .summary_backend import ProcessApiBackend


@dataclass
class SessionEntry:
    session: LangGraphSession
    last_accessed: float


class ChatSessionStore:
    def __init__(self, ttl: int = 3600) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionEntry] = {}

    def get(self, session_id: str) -> Optional[LangGraphSession]:
        now = time.time()
        with self._lock:
            entry = self._sessions.get(session_id)
            if not entry:
                return None
            if now - entry.last_accessed > self._ttl:
                self._sessions.pop(session_id, None)
                self._close_session(entry.session)
                return None
            entry.last_accessed = now
            return entry.session

    def create(
        self,
        video_id: str,
        video_name: str,
        *,
        process_api_url: Optional[str] = None,
        initial_state: Optional[Dict[str, object]] = None,
        app_name: str = "screentime_process_api",
        user_id: str = "api",
    ) -> LangGraphSession:
        state: Dict[str, object] = {
            "video_id": video_id,
            "video_name": video_name,
        }
        if initial_state:
            state.update(initial_state)
        backend = ProcessApiBackend(process_api_url=process_api_url)
        session = LangGraphSession(
            app_name=app_name,
            user_id=user_id,
            backend=backend,
            initial_state=state,
        )
        with self._lock:
            self._sessions[session.session_id] = SessionEntry(
                session=session,
                last_accessed=time.time(),
            )
        return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry:
            self._close_session(entry.session)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired: Dict[str, SessionEntry] = {}
        with self._lock:
            for session_id, entry in list(self._sessions.items()):
                if now - entry.last_accessed > self._ttl:
                    expired[session_id] = entry
                    del self._sessions[session_id]
        for entry in expired.values():
            self._close_session(entry.session)
        return len(expired)

    @staticmethod
    def _close_session(session: LangGraphSession) -> None:
        try:
            session.close()
        except Exception:
            pass
