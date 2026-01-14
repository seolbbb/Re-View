# 사용:
# from src.audio.whisper_stt import WhisperSTTClient
#
# client = WhisperSTTClient(model_size="base")
# client.transcribe(
#     "src/data/input/sample.wav",
#     output_path="src/data/output/sample/stt.json",
#     include_confidence=True,
#     include_raw_response=False,
#     language="ko",
#     task="transcribe",
#     temperature=0.0,
# )
#
# CLI:
# python src/audio/whisper_stt.py --audio-path src/data/input/test2.wav --model-size base
#
# 출력: {segments: [{start_ms, end_ms, text, confidence?}], ...(옵션)}
# 옵션: include_confidence, include_raw_response, model_size, device, language, task, temperature, output_path
# 참고: pip install -U openai-whisper torch
# 주의: confidence는 avg_logprob 기반으로 계산한 근사값(정확한 확률 아님)
"""Local Whisper STT client (audio input only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.audio.settings import load_audio_settings


def _segment_confidence(segment: Dict[str, Any]) -> float | None:
    avg_logprob = segment.get("avg_logprob")
    if avg_logprob is None:
        return None
    try:
        value = math.exp(float(avg_logprob))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, min(1.0, value))


class WhisperSTTClient:
    """Lightweight wrapper around openai-whisper."""

    def __init__(self, model_size: str = "base", device: str | None = None) -> None:
        try:
            import torch
            import whisper
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Whisper dependencies missing. Install with: pip install -U openai-whisper torch"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
        *,
        include_confidence: Optional[bool] = None,
        include_raw_response: Optional[bool] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        settings = load_audio_settings()
        stt_settings = settings.get("stt", {})
        if not isinstance(stt_settings, dict):
            raise ValueError("stt 설정 형식이 올바르지 않습니다(맵이어야 함).")
        whisper_defaults = stt_settings.get("whisper", {})
        if not isinstance(whisper_defaults, dict):
            raise ValueError("stt.whisper 설정 형식이 올바르지 않습니다(맵이어야 함).")
        if include_confidence is None:
            include_confidence = bool(whisper_defaults.get("include_confidence", False))
        if include_raw_response is None:
            include_raw_response = bool(whisper_defaults.get("include_raw_response", False))
        if language is None:
            language = str(whisper_defaults.get("language", "ko"))
        if task is None:
            task = str(whisper_defaults.get("task", "transcribe"))
        if temperature is None:
            temperature = float(whisper_defaults.get("temperature", 0.0))

        audio_path = Path(audio_path).expanduser()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            temperature=temperature,
        )

        segments_out: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            item: Dict[str, Any] = {
                "start_ms": int(round(float(segment.get("start", 0.0)) * 1000)),
                "end_ms": int(round(float(segment.get("end", 0.0)) * 1000)),
                "text": text,
            }
            if include_confidence:
                confidence = _segment_confidence(segment)
                if confidence is not None:
                    item["confidence"] = confidence
            segments_out.append(item)

        stt_data: Dict[str, Any] = {
            "segments": segments_out,
        }

        if include_raw_response:
            stt_data["raw_response"] = result

        if output_path is None:
            output_path = Path("src/data/output") / audio_path.stem / "stt.json"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return stt_data


def parse_args() -> argparse.Namespace:
    settings = load_audio_settings()
    stt_settings = settings.get("stt", {})
    if not isinstance(stt_settings, dict):
        raise ValueError("stt 설정 형식이 올바르지 않습니다(맵이어야 함).")
    whisper_defaults = stt_settings.get("whisper", {})
    if not isinstance(whisper_defaults, dict):
        raise ValueError("stt.whisper 설정 형식이 올바르지 않습니다(맵이어야 함).")

    parser = argparse.ArgumentParser(description="Whisper STT (audio input).")
    parser.add_argument("--audio-path", required=True, help="Path to local audio file.")
    parser.add_argument("--output-path", help="Output stt.json path.")
    parser.add_argument(
        "--model-size",
        default=str(whisper_defaults.get("model_size", "base")),
        help="Whisper model size (e.g., tiny/base/small).",
    )
    parser.add_argument("--device", default=whisper_defaults.get("device"), help="Override device (cuda/cpu).")
    parser.add_argument(
        "--language",
        default=str(whisper_defaults.get("language", "ko")),
        help="Language code (e.g., ko, en).",
    )
    parser.add_argument(
        "--task",
        default=str(whisper_defaults.get("task", "transcribe")),
        choices=("transcribe", "translate"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(whisper_defaults.get("temperature", 0.0)),
    )
    parser.add_argument(
        "--include-confidence",
        action=argparse.BooleanOptionalAction,
        default=bool(whisper_defaults.get("include_confidence", False)),
        help="Add confidence per segment.",
    )
    parser.add_argument(
        "--include-raw-response",
        action=argparse.BooleanOptionalAction,
        default=bool(whisper_defaults.get("include_raw_response", False)),
        help="Attach raw provider response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = WhisperSTTClient(model_size=args.model_size, device=args.device)
    client.transcribe(
        args.audio_path,
        output_path=args.output_path,
        include_confidence=args.include_confidence,
        include_raw_response=args.include_raw_response,
        language=args.language,
        task=args.task,
        temperature=args.temperature,
    )
    out_path = Path(args.output_path) if args.output_path else Path("src/data/output") / Path(
        args.audio_path
    ).stem / "stt.json"
    print(f"[OK] stt.json saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
