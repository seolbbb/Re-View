# 사용:
# from src.audio.stt_router import STTRouter
#
# router = STTRouter(provider="clova")  # 기본값은 clova, provider로 clova/whisper 선택 가능
# router.transcribe_media(
#     "src/data/input/screentime-mvp-video.mp4",
#     provider="clova",
#     include_confidence=True,  # 문장 단위 신뢰도 포함
#     include_raw_response=True,  # segments 외에 전체 응답 포함
#     word_alignment=True,  # 추가적으로 단어 단위 타임스탬프 포함
#     full_text=True,  # include_raw_response가 True일 경우에만 의미 있음
#     completion="sync",  # "sync" 기본. "async"는 폴링 미구현으로 결과 비어 있을 수 있음
#     language="ko-KR",  # 예: ko-KR, en-US, enko, ja-JP, zh-CN, zh-TW
#     timeout=120,  # 지정 초 내 응답 없으면 Timeout 예외
#     mono_method="auto",  # 오디오 추출 모드 (downmix|left|right|phase-fix|auto)
#     output_path="src/data/output/screentime-mvp-video/stt.json",
# )
# router.transcribe(
#     "src/data/input/sample.wav",
#     provider="whisper",
#     model_size="base",
#     language="ko",
#     task="transcribe",
#     temperature=0.0,
# )
#
# CLI:
# python src/audio/stt_router.py --media-path src/data/input/sample.mp4 --provider clova
# python src/audio/stt_router.py --media-path src/data/input/sample.wav --provider whisper --no-extract --model-size base
#
# 현재 지원: clova, whisper (향후 google 추가 예정)
"""STT router that dispatches to provider-specific clients."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from src.audio.clova_stt import ClovaSpeechClient
from src.audio.extract_audio import extract_audio
from src.audio.settings import load_audio_settings
from src.audio.whisper_stt import WhisperSTTClient

DEFAULT_PROVIDER = "clova"


def _merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _coerce_mapping(value: Any, label: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} 설정 형식이 올바르지 않습니다(맵이어야 함).")
    return value


class STTRouter:
    """Routes STT calls to the configured provider."""

    def __init__(self, provider: str | None = None, *, settings_path: Optional[Path] = None) -> None:
        settings = load_audio_settings(settings_path=settings_path)
        stt_settings = _coerce_mapping(settings.get("stt"), "stt")
        extract_settings = _coerce_mapping(settings.get("extract"), "extract")

        default_provider = stt_settings.get("default_provider", DEFAULT_PROVIDER)
        if not isinstance(default_provider, str) or not default_provider.strip():
            default_provider = DEFAULT_PROVIDER
        env_provider = os.getenv("STT_PROVIDER")
        self.provider = (provider or env_provider or default_provider).lower()
        self._stt_settings = stt_settings
        self._extract_settings = extract_settings

    def transcribe(
        self,
        media_path: str | Path,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        provider_name = (provider or self.provider).lower()
        provider_defaults = _coerce_mapping(
            self._stt_settings.get(provider_name),
            f"stt.{provider_name}",
        )
        kwargs = _merge_defaults(provider_defaults, kwargs)

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
        mono_method: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        codec: Optional[str] = None,
        **stt_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract audio first, then run STT using the selected provider.
        """
        extract_defaults = self._extract_settings
        if sample_rate is None:
            sample_rate = int(extract_defaults.get("sample_rate", 16000))
        if channels is None:
            channels = int(extract_defaults.get("channels", 1))
        if codec is None:
            codec = str(extract_defaults.get("codec", "pcm_s16le"))
        if mono_method is None:
            mono_method = str(extract_defaults.get("mono_method", "auto"))
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
    settings = load_audio_settings()
    stt_settings = _coerce_mapping(settings.get("stt"), "stt")
    extract_settings = _coerce_mapping(settings.get("extract"), "extract")
    default_provider = stt_settings.get("default_provider", DEFAULT_PROVIDER)
    if not isinstance(default_provider, str) or not default_provider.strip():
        default_provider = DEFAULT_PROVIDER

    parser = argparse.ArgumentParser(description="STT router (clova/whisper).")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument(
        "--provider",
        default=default_provider,
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
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(extract_settings.get("sample_rate", 16000)),
        help="Sample rate for extraction (Hz).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=int(extract_settings.get("channels", 1)),
        help="Audio channels for extraction.",
    )
    parser.add_argument(
        "--codec",
        default=str(extract_settings.get("codec", "pcm_s16le")),
        help="Audio codec for extraction.",
    )
    parser.add_argument(
        "--mono-method",
        default=str(extract_settings.get("mono_method", "auto")),
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
