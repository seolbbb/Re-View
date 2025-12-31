# 사용:
# from src.audio.stt_router import STTRouter
#
# router = STTRouter()  # STT_PROVIDER 환경변수 없으면 기본 clova
# router.transcribe("src/data/input/screentime-mvp-video.mp4")
#
# 현재 지원: clova (향후 google/whisper 추가 예정)
"""STT router that dispatches to provider-specific clients."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from src.audio.clova_stt import ClovaSpeechClient

DEFAULT_PROVIDER = "clova"


class STTRouter:
    """Routes STT calls to the configured provider."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("STT_PROVIDER", DEFAULT_PROVIDER)).lower()

    def transcribe(
        self,
        media_path: str | Path,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_name = (provider or self.provider).lower()

        if provider_name == "clova":
            return ClovaSpeechClient().transcribe(media_path, **kwargs)

        raise ValueError(f"Unsupported STT provider: {provider_name}")
