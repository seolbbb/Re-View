# 사용:
# from src.audio.clova_stt import ClovaSpeechClient
#
# client = ClovaSpeechClient()
# client.transcribe(
#     "src/data/input/screentime-mvp-video.mp4",
#     include_confidence=True,
#     include_raw_response=True,
#     word_alignment=True,
#     full_text=True,
#     completion="sync", "sync" 기본. "async"는 결과 폴링 로직이 없어 segments가 비어 저장될 수 있음.
#     language="ko-KR", 예: ko-KR, en-US, enko, ja-JP, zh-CN, zh-TW (Clova 문서 기준)
#     timeout=120, 지정 초 내 응답 없으면 요청이 Timeout 예외로 종료됨 (긴 파일은 늘리거나 async 권장)
#     output_path="src/data/output/screentime-mvp-video/stt.json",
# )
#
# CLI사용: python src/audio/clova_stt.py --media-path src/data/input/screentime-mvp-video.mp4
# 출력: {schema_version: 1, segments: [{start_ms, end_ms, text, confidence?}], ...(옵션)}
# 옵션: include_confidence(기본 True, 세그먼트별), include_raw_response, word_alignment, full_text, completion, language, timeout, output_path
# .env: Screentime-MVP/.env 우선 로드, 없으면 기본 load_dotenv().
"""
Clova Speech API client (recognizer/upload).

Inputs:
    - media_path: Path to a local audio/video file.
Outputs:
    - dict matching stt.json schema.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


def build_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/recognizer/upload"):
        return base
    if base.endswith("/recognizer"):
        return f"{base}/upload"
    return f"{base}/recognizer/upload"


def _coerce_ms(value: object) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _extract_segments(raw: Dict[str, Any], *, include_confidence: bool) -> List[Dict[str, Any]]:
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
        item = {
            "start_ms": _coerce_ms(segment.get("start")),
            "end_ms": _coerce_ms(segment.get("end")),
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
    """Lightweight wrapper around the Naver Clova Speech API."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        load_env()
        self.api_url = api_url or os.getenv("CLOVA_SPEECH_URL")
        self.api_key = api_key or os.getenv("CLOVA_SPEECH_API_KEY") or os.getenv("CLOVA_SPEECH_SECRET")

        if not self.api_url or not self.api_key:
            raise ValueError("CLOVA_SPEECH_URL and CLOVA_SPEECH_API_KEY must be set.")

    def transcribe(
        self,
        media_path: str | Path,
        output_path: str | Path | None = None,
        *,
        include_confidence: bool = True,
        include_raw_response: bool = False,
        word_alignment: bool = False,
        full_text: bool = False,
        completion: str = "sync",
        language: str = "ko-KR",
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Run speech-to-text on a media file and emit stt.json schema.
        """
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
        url = build_url(self.api_url)
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
            "schema_version": 1,
            "segments": _extract_segments(raw, include_confidence=include_confidence),
        }

        if include_raw_response:
            stt_data["raw_response"] = raw

        output_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return stt_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clova Speech STT client.")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument("--output-path", help="Override default stt.json output path.")
    parser.add_argument(
        "--confidence",
        dest="include_confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include average confidence field.",
    )
    parser.add_argument("--include-raw-response", action="store_true", help="Attach raw provider response.")
    parser.add_argument("--word-alignment", action="store_true", help="Request word-level timestamps.")
    parser.add_argument("--full-text", action="store_true", help="Request fullText output if supported.")
    parser.add_argument("--completion", default="sync", help="sync or async (async not polled here).")
    parser.add_argument("--language", default="ko-KR", help="Language code (e.g., ko-KR, en-US, enko).")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds.")
    return parser.parse_args()


def main() -> None:
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
