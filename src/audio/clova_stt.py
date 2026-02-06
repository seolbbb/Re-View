from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _extract_segments(raw: Dict[str, Any], *, include_confidence: bool) -> List[Dict[str, Any]]:
    """Clova 응답에서 세그먼트 목록을 추출해 표준 형태로 정리한다."""
    segments_out: List[Dict[str, Any]] = []
    segment_index = 0
    for segment in raw.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = segment.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        segment_index += 1
        try:
            start_ms = int(round(float(segment.get("start"))))
        except (TypeError, ValueError):
            start_ms = 0
        try:
            end_ms = int(round(float(segment.get("end"))))
        except (TypeError, ValueError):
            end_ms = 0
        item = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "id": f"stt_{segment_index:03d}",
        }
        if include_confidence:
            try:
                item["confidence"] = float(segment.get("confidence"))
            except (TypeError, ValueError):
                pass
        segments_out.append(item)
    return segments_out


class ClovaSpeechClient:
    """Clova Speech API에 대한 간단한 클라이언트 래퍼."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """API URL/KEY를 준비하고 .env를 우선 로드한다."""
        if ENV_PATH.exists():
            load_dotenv(ENV_PATH)
        else:
            load_dotenv()
        self.api_url = api_url or os.getenv("CLOVA_SPEECH_URL")
        self.api_key = api_key or os.getenv("CLOVA_SPEECH_API_KEY") or os.getenv("CLOVA_SPEECH_SECRET")

        if not self.api_url or not self.api_key:
            raise ValueError("CLOVA_SPEECH_URL and CLOVA_SPEECH_API_KEY must be set.")

    def transcribe(
        self,
        media_path: str | Path,
        output_path: str | Path | None = None,
        *,
        write_output: bool = True,
        include_confidence: bool = True,
        include_raw_response: bool = False,
        word_alignment: bool = False,
        full_text: bool = False,
        completion: str = "sync",
        language: str = "ko-KR",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """로컬 미디어 파일을 STT로 처리해 stt.json 구조를 만든다."""

        media_path = Path(media_path).expanduser()
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        if write_output:
            if output_path is None:
                output_path = Path("data/outputs") / media_path.stem / "stt.json"
            else:
                output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "language": language,
            "completion": completion,
            "timestamps": True,
        }
        if word_alignment:
            payload["wordAlignment"] = True
        if full_text:
            payload["fullText"] = True

        headers = {"Accept": "application/json;UTF-8", "X-CLOVASPEECH-API-KEY": self.api_key}
        base_url = self.api_url.rstrip("/")
        if base_url.endswith("/recognizer/upload"):
            url = base_url
        elif base_url.endswith("/recognizer"):
            url = f"{base_url}/upload"
        else:
            url = f"{base_url}/recognizer/upload"
        params_payload = json.dumps(payload, ensure_ascii=False)

        with media_path.open("rb") as media_file:
            files = {
                "media": (media_path.name, media_file, "application/octet-stream"),
                "params": (None, params_payload, "application/json"),
            }
            response = requests.post(url, headers=headers, files=files, timeout=timeout)
        response.raise_for_status()
        raw = response.json()

        stt_data: Dict[str, Any] = {
            "segments": _extract_segments(raw, include_confidence=include_confidence),
        }

        if include_raw_response:
            stt_data["raw_response"] = raw

        if write_output and output_path is not None:
            output_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return stt_data
