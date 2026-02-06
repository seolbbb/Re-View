"""STT 요청을 제공자(clova/whisper)로 분기하는 라우터."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.audio.clova_stt import ClovaSpeechClient
from src.audio.extract_audio import extract_audio
from src.audio.whisper_stt import WhisperSTTClient

DEFAULT_PROVIDER = "clova"
SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "audio" / "settings.yaml"


def load_audio_settings(*, settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """오디오 설정 파일을 로드해 딕셔너리로 반환한다."""
    path = settings_path or SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"audio settings file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audio settings must be a mapping.")
    return payload


def _merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """None이 아닌 값만 기본값 위에 덮어쓴다."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _coerce_mapping(value: Any, label: str) -> Dict[str, Any]:
    """설정 값이 매핑인지 확인하고, 없으면 빈 딕셔너리를 돌려준다."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} 설정 형식이 올바르지 않습니다(맵이어야 함).")
    return value


class STTRouter:
    """STT 호출을 설정된 제공자에 맞게 라우팅한다."""

    def __init__(self, provider: str | None = None, *, settings_path: Optional[Path] = None) -> None:
        """설정 파일을 로드하고 기본 제공자를 결정한다."""
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
        output_path: str | Path | None = None,
        write_output: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """지정된 제공자로 STT를 실행한다."""
        provider_name = (provider or self.provider).lower()
        provider_defaults = _coerce_mapping(
            self._stt_settings.get(provider_name),
            f"stt.{provider_name}",
        )
        kwargs = _merge_defaults(provider_defaults, kwargs)

        if provider_name == "clova":
            return ClovaSpeechClient().transcribe(
                media_path,
                output_path=output_path,
                write_output=write_output,
                **kwargs,
            )
        if provider_name == "whisper":
            model_size = kwargs.pop("model_size", "base")
            device = kwargs.pop("device", None)
            return WhisperSTTClient(model_size=model_size, device=device).transcribe(
                media_path,
                output_path=output_path,
                write_output=write_output,
                **kwargs,
            )

        raise ValueError(f"Unsupported STT provider: {provider_name}")

    def transcribe_media(
        self,
        media_path: str | Path,
        *,
        provider: str | None = None,
        audio_output_path: str | Path | None = None,
        output_path: str | Path | None = None,
        mono_method: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        codec: Optional[str] = None,
        write_output: bool = True,
        **stt_kwargs: Any,
    ) -> Dict[str, Any]:
        """오디오를 추출한 뒤 선택된 제공자로 STT를 실행한다."""
        extract_defaults = self._extract_settings
        keep_audio = extract_defaults.get("keep_audio", True)
        if not isinstance(keep_audio, bool):
            keep_audio = True
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
        try:
            return self.transcribe(
                audio_path,
                provider=provider,
                output_path=output_path,
                write_output=write_output,
                **stt_kwargs,
            )
        finally:
            if not keep_audio:
                Path(audio_path).unlink(missing_ok=True)
