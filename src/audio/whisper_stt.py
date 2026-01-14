"""Whisper 기반 로컬 STT 클라이언트(오디오 입력 전용)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _segment_confidence(segment: Dict[str, Any]) -> float | None:
    """avg_logprob을 0~1 범위 신뢰도 값으로 변환한다."""
    avg_logprob = segment.get("avg_logprob")
    if avg_logprob is None:
        return None
    try:
        value = math.exp(float(avg_logprob))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, min(1.0, value))


class WhisperSTTClient:
    """openai-whisper 실행을 감싸는 래퍼."""

    def __init__(self, model_size: str = "base", device: str | None = None) -> None:
        """모델을 로딩하고 실행 장치를 결정한다."""
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
        include_confidence: bool = False,
        include_raw_response: bool = False,
        language: str = "ko",
        task: str = "transcribe",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """오디오 파일을 STT로 처리해 stt.json 구조를 만든다."""
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
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Whisper STT (audio input).")
    parser.add_argument("--audio-path", required=True, help="Path to local audio file.")
    parser.add_argument("--output-path", help="Output stt.json path.")
    parser.add_argument(
        "--model-size",
        default="base",
        help="Whisper model size (e.g., tiny/base/small).",
    )
    parser.add_argument("--device", help="Override device (cuda/cpu).")
    parser.add_argument(
        "--language",
        default="ko",
        help="Language code (e.g., ko, en).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=("transcribe", "translate"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--include-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add confidence per segment.",
    )
    parser.add_argument(
        "--include-raw-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach raw provider response.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 진입점."""
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
