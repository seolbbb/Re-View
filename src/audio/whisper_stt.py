"""Whisper 기반 로컬 STT 클라이언트(오디오 입력 전용)."""

from __future__ import annotations

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
        write_output: bool = True,
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
        segment_index = 0
        for segment in result.get("segments", []):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            segment_index += 1
            item: Dict[str, Any] = {
                "start_ms": int(round(float(segment.get("start", 0.0)) * 1000)),
                "end_ms": int(round(float(segment.get("end", 0.0)) * 1000)),
                "text": text,
                "id": f"stt_{segment_index:03d}",
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

        if write_output:
            if output_path is None:
                output_path = Path("data/outputs") / audio_path.stem / "stt.json"
            else:
                output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(stt_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return stt_data
