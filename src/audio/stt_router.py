# 사용:
# from src.audio.stt_router import STTRouter
#
# router = STTRouter()  # STT_PROVIDER 환경변수 없으면 기본 clova
# router.transcribe("src/data/input/screentime-mvp-video.mp4")
# router.transcribe_media("src/data/input/screentime-mvp-video.mp4")
#
# CLI:
# python src/audio/stt_router.py --media-path src/data/input/test2.mp4 --provider clova
# python src/audio/stt_router.py --media-path src/data/input/test2.mp4 --provider whisper --model-size base
#
# 현재 지원: clova, whisper (향후 google 추가 예정)
"""STT router that dispatches to provider-specific clients."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from src.audio.clova_stt import ClovaSpeechClient
from src.audio.extract_audio import extract_audio
from src.audio.whisper_stt import WhisperSTTClient

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
        if provider_name == "whisper":
            model_size = kwargs.pop("model_size", "base")
            device = kwargs.pop("device", None)
            return WhisperSTTClient(model_size=model_size, device=device).transcribe(
                media_path, **kwargs
            )

        raise ValueError(f"Unsupported STT provider: {provider_name}")

    def transcribe_media(
        self,
        media_path: str | Path,
        *,
        provider: str | None = None,
        audio_output_path: str | Path | None = None,
        mono_method: str = "auto",
        sample_rate: int = 16000,
        channels: int = 1,
        codec: str = "pcm_s16le",
        **stt_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract audio first, then run STT using the selected provider.
        """
        audio_path = extract_audio(
            media_path,
            output_path=audio_output_path,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec,
            mono_method=mono_method,
        )
        return self.transcribe(audio_path, provider=provider, **stt_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT router (clova/whisper).")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=("clova", "whisper"),
        help="STT provider.",
    )
    parser.add_argument("--output-path", help="Output stt.json path.")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip audio extraction and call provider directly.",
    )
    parser.add_argument("--audio-output-path", help="Output audio path when extracting.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for extraction (Hz).")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels for extraction.")
    parser.add_argument("--codec", default="pcm_s16le", help="Audio codec for extraction.")
    parser.add_argument(
        "--mono-method",
        default="auto",
        choices=("downmix", "left", "right", "phase-fix", "auto"),
        help="Mono creation method for extraction.",
    )
    parser.add_argument("--language", help="Language code (clova: ko-KR, whisper: ko).")
    parser.add_argument(
        "--include-confidence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include confidence per segment when supported.",
    )
    parser.add_argument(
        "--include-raw-response",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Attach raw provider response when supported.",
    )
    parser.add_argument("--completion", choices=("sync", "async"), help="Clova completion mode.")
    parser.add_argument("--timeout", type=int, help="Clova request timeout (seconds).")
    parser.add_argument("--word-alignment", action="store_true", help="Clova word-level timestamps.")
    parser.add_argument("--full-text", action="store_true", help="Clova fullText output.")
    parser.add_argument("--model-size", help="Whisper model size (tiny/base/small/medium/large).")
    parser.add_argument("--device", help="Whisper device override (cuda/cpu).")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        help="Whisper task.",
    )
    parser.add_argument("--temperature", type=float, help="Whisper decoding temperature.")
    return parser.parse_args()


def _build_stt_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    stt_kwargs: Dict[str, Any] = {}

    if args.output_path:
        stt_kwargs["output_path"] = args.output_path
    if args.include_confidence is not None:
        stt_kwargs["include_confidence"] = args.include_confidence
    if args.include_raw_response is not None:
        stt_kwargs["include_raw_response"] = args.include_raw_response
    if args.language:
        stt_kwargs["language"] = args.language

    if args.provider == "clova":
        if args.word_alignment:
            stt_kwargs["word_alignment"] = True
        if args.full_text:
            stt_kwargs["full_text"] = True
        if args.completion:
            stt_kwargs["completion"] = args.completion
        if args.timeout is not None:
            stt_kwargs["timeout"] = args.timeout
    elif args.provider == "whisper":
        if args.model_size:
            stt_kwargs["model_size"] = args.model_size
        if args.device:
            stt_kwargs["device"] = args.device
        if args.task:
            stt_kwargs["task"] = args.task
        if args.temperature is not None:
            stt_kwargs["temperature"] = args.temperature

    return stt_kwargs


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if args.no_extract:
        stem = Path(args.media_path).expanduser().stem
    else:
        if args.audio_output_path:
            stem = Path(args.audio_output_path).expanduser().stem
        else:
            stem = Path(args.media_path).expanduser().stem
    return Path("src/data/output") / stem / "stt.json"


def main() -> None:
    args = parse_args()
    router = STTRouter(provider=args.provider)
    stt_kwargs = _build_stt_kwargs(args)

    if args.no_extract:
        router.transcribe(args.media_path, provider=args.provider, **stt_kwargs)
    else:
        router.transcribe_media(
            args.media_path,
            provider=args.provider,
            audio_output_path=args.audio_output_path,
            mono_method=args.mono_method,
            sample_rate=args.sample_rate,
            channels=args.channels,
            codec=args.codec,
            **stt_kwargs,
        )

    out_path = _resolve_output_path(args)
    print(f"[OK] stt.json saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
