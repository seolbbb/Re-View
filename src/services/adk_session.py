from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdkMessage:
    author: str
    text: str
    is_final: bool


def _extract_event_text(event: Event) -> str:
    if event.content and event.content.parts:
        parts = [part.text or "" for part in event.content.parts]
        return "".join(parts).strip()
    if event.error_message:
        return f"Error: {event.error_message}"
    return ""


class AdkSession:
    def __init__(
        self,
        *,
        root_agent: BaseAgent,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._app_name = app_name
        self._user_id = user_id
        self._runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        state = initial_state or {}

        session_service = self._runner.session_service
        if hasattr(session_service, "create_session_sync"):
            session = session_service.create_session_sync(
                app_name=app_name,
                user_id=user_id,
                state=state,
                session_id=session_id,
            )
        else:
            session = asyncio.run(
                session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    state=state,
                    session_id=session_id,
                )
            )

        self._session_id = session.id

    @property
    def app_name(self) -> str:
        return self._app_name

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def send_message(self, text: str) -> List[AdkMessage]:
        content = types.Content(role="user", parts=[types.Part(text=text)])
        messages: List[AdkMessage] = []
        for event in self._runner.run(
            user_id=self._user_id,
            session_id=self._session_id,
            new_message=content,
        ):
            if event.author == "user":
                continue
            if event.partial:
                continue
            message_text = _extract_event_text(event)
            if not message_text:
                continue
            messages.append(
                AdkMessage(
                    author=event.author,
                    text=message_text,
                    is_final=event.is_final_response(),
                )
            )
        return messages

    def close(self) -> None:
        try:
            asyncio.run(self._runner.close())
        except RuntimeError as exc:
            logger.info("Runner close skipped: %s", exc)
