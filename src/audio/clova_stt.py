from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from src.audio.settings import load_audio_settings

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _extract_segments(raw: Dict[str, Any], *, include_confidence: bool) -> List[Dict[str, Any]]:
    """Clova 응답에서 세그먼트 목록을 추출해 표준 형태로 정리한다."""
    segments_out: List[Dict[str, Any]] = []
    for segment in raw.get("segments", []):
        if not isinstance(segment, dict):
            continue
        text = segment.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
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
        """API URL/키를 준비하고 .env를 우선 로드한다."""
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
        include_confidence: Optional[bool] = None,
        include_raw_response: Optional[bool] = None,
        word_alignment: Optional[bool] = None,
        full_text: Optional[bool] = None,
        completion: Optional[str] = None,
        language: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """로컬 미디어 파일을 STT로 처리해 stt.json 구조를 만든다."""
        settings = load_audio_settings()
        stt_settings = settings.get("stt", {})
        if not isinstance(stt_settings, dict):
            raise ValueError("stt 설정 형식이 올바르지 않습니다(맵이어야 함).")
        clova_defaults = stt_settings.get("clova", {})
        if not isinstance(clova_defaults, dict):
            raise ValueError("stt.clova 설정 형식이 올바르지 않습니다(맵이어야 함).")
        if include_confidence is None:
            include_confidence = bool(clova_defaults.get("include_confidence", True))
        if include_raw_response is None:
            include_raw_response = bool(clova_defaults.get("include_raw_response", False))
        if word_alignment is None:
            word_alignment = bool(clova_defaults.get("word_alignment", False))
        if full_text is None:
            full_text = bool(clova_defaults.get("full_text", False))
        if completion is None:
            completion = str(clova_defaults.get("completion", "sync"))
        if language is None:
            language = str(clova_defaults.get("language", "ko-KR"))
        if timeout is None:
            timeout = int(clova_defaults.get("timeout", 60))

        media_path = Path(media_path).expanduser()
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        if output_path is None:
            output_path = Path("src/data/output") / media_path.stem / "stt.json"
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

        output_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return stt_data


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    settings = load_audio_settings()
    stt_settings = settings.get("stt", {})
    if not isinstance(stt_settings, dict):
        raise ValueError("stt 설정 형식이 올바르지 않습니다(맵이어야 함).")
    clova_defaults = stt_settings.get("clova", {})
    if not isinstance(clova_defaults, dict):
        raise ValueError("stt.clova 설정 형식이 올바르지 않습니다(맵이어야 함).")

    parser = argparse.ArgumentParser(description="Clova Speech STT client.")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument("--output-path", help="Override default stt.json output path.")
    parser.add_argument(
        "--confidence",
        dest="include_confidence",
        action=argparse.BooleanOptionalAction,
        default=bool(clova_defaults.get("include_confidence", True)),
        help="Include average confidence field.",
    )
    parser.add_argument(
        "--include-raw-response",
        action=argparse.BooleanOptionalAction,
        default=bool(clova_defaults.get("include_raw_response", False)),
        help="Attach raw provider response.",
    )
    parser.add_argument(
        "--word-alignment",
        action=argparse.BooleanOptionalAction,
        default=bool(clova_defaults.get("word_alignment", False)),
        help="Request word-level timestamps.",
    )
    parser.add_argument(
        "--full-text",
        action=argparse.BooleanOptionalAction,
        default=bool(clova_defaults.get("full_text", False)),
        help="Request fullText output if supported.",
    )
    parser.add_argument(
        "--completion",
        default=str(clova_defaults.get("completion", "sync")),
        help="sync or async (async not polled here).",
    )
    parser.add_argument(
        "--language",
        default=str(clova_defaults.get("language", "ko-KR")),
        help="Language code (e.g., ko-KR, en-US, enko).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(clova_defaults.get("timeout", 60)),
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 진입점."""
    args = parse_args()
    client = ClovaSpeechClient()
    client.transcribe(
        args.media_path,
        output_path=args.output_path,
        include_confidence=args.include_confidence,
        include_raw_response=args.include_raw_response,
        word_alignment=args.word_alignment,
        full_text=args.full_text,
        completion=args.completion,
        language=args.language,
        timeout=args.timeout,
    )

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        media_path = Path(args.media_path).expanduser()
        out_path = Path("src/data/output") / media_path.stem / "stt.json"
    print(f"[OK] stt.json saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
