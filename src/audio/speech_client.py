"""
Clova Speech API client for speech-to-text.

Inputs:
    - video_path: Path to a media file with an audio track.
Outputs:
    - List[AudioSegment]: Timecoded transcript segments.
"""

import json
import os
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv

from src.common.schemas import AudioSegment


class ClovaSpeechClient:
    """Lightweight wrapper around the Naver Clova Speech API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_url = api_url or os.getenv("CLOVA_SPEECH_URL")
        self.api_key = api_key or os.getenv("CLOVA_SPEECH_API_KEY")
        self.api_secret = api_secret or os.getenv("CLOVA_SPEECH_SECRET")

        if not all([self.api_url, self.api_key, self.api_secret]):
            raise ValueError("Clova Speech credentials are not fully configured in .env")

    def _build_headers(self) -> dict:
        return {
            "X-CLOVASPEECH-API-KEY": self.api_key,
            "X-CLOVASPEECH-SECRET": self.api_secret,
        }

    def transcribe(self, video_path: str) -> List[AudioSegment]:
        """
        Run speech-to-text on a media file.

        Returns:
            List[AudioSegment]: Ordered transcript segments.
        """
        media_path = Path(video_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {video_path}")

        files = {"media": media_path.open("rb")}
        payload = {
            "language": "ko-KR",
            "completion": "sync",
            "timestamps": True,
        }

        response = requests.post(
            self.api_url,
            headers=self._build_headers(),
            data={"params": json.dumps(payload)},
            files=files,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        segments: List[AudioSegment] = []
        for segment in result.get("segments", []):
            segments.append(
                AudioSegment(
                    start=float(segment.get("start", 0)),
                    end=float(segment.get("end", 0)),
                    text=segment.get("text", "").strip(),
                )
            )

        return segments
