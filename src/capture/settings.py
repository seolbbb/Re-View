"""캡처 단계 설정을 로드하는 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "capture" / "settings.yaml"


@dataclass(frozen=True)
class CaptureSettings:
    """캡처 단계에서 사용하는 설정 값 묶음."""

    input_dir: Path
    output_dir: Path
    sensitivity_diff: float
    sensitivity_sim: float
    min_interval: float
    sample_interval_sec: float
    buffer_duration_sec: float
    transition_timeout_sec: float


def _coerce_str(settings: Dict[str, Any], key: str, default: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"capture 설정의 {key} 형식이 올바르지 않습니다(문자열이어야 함).")
    return value


def _coerce_float(settings: Dict[str, Any], key: str, default: float) -> float:
    value = settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture 설정의 {key} 값이 올바르지 않습니다: {value}") from exc


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_capture_settings(*, settings_path: Optional[Path] = None) -> CaptureSettings:
    """config/capture/settings.yaml을 읽어 CaptureSettings로 변환한다."""
    path = settings_path or SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"capture 설정 파일을 찾을 수 없습니다: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capture 설정 형식이 올바르지 않습니다(맵이어야 함).")

    input_dir = _resolve_path(_coerce_str(payload, "input_dir", "src/data/input"))
    output_dir = _resolve_path(_coerce_str(payload, "output_dir", "src/data/output"))

    return CaptureSettings(
        input_dir=input_dir,
        output_dir=output_dir,
        sensitivity_diff=_coerce_float(payload, "sensitivity_diff", 3.0),
        sensitivity_sim=_coerce_float(payload, "sensitivity_sim", 0.8),
        min_interval=_coerce_float(payload, "min_interval", 0.5),
        sample_interval_sec=_coerce_float(payload, "sample_interval_sec", 0.5),
        buffer_duration_sec=_coerce_float(payload, "buffer_duration_sec", 2.5),
        transition_timeout_sec=_coerce_float(payload, "transition_timeout_sec", 2.5),
    )


@lru_cache(maxsize=1)
def get_capture_settings() -> CaptureSettings:
    """캡처 설정을 한 번만 로드해 캐시한다."""
    return load_capture_settings()
