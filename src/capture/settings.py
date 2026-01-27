"""
[Intent]
캡처 단계에 필요한 각종 설정값(임계값, 샘플링 주기 등)을 
YAML 파일로부터 로드하고 유효성을 검증하여 파이썬 객체로 제공하는 모듈입니다.

[Usage]
- run_preprocess_pipeline.py에서 캡처 설정을 초기화할 때 호출됩니다.
- process_content.py에서 실제 캡처 엔진에 설정을 전달하기 위해 사용됩니다.

[Usage Method]
- get_capture_settings()를 호출하여 캐시된 설정 객체를 가져옵니다.
- load_capture_settings()를 통해 특정 경로의 YAML 파일을 직접 로드할 수 있습니다.
- 모든 경로는 프로젝트 루트를 기준으로 자동 해결(Resolve)됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# 프로젝트 루트 경로 (src/capture/settings.py 기준 2단계 상위)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 기본 설정 파일 경로
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "capture" / "settings.yaml"


@dataclass(frozen=True)
class CaptureSettings:
    """
    [Class Purpose]
    캡처 엔진의 동작을 제어하는 파라미터들을 담는 불변(Immutable) 데이터 클래스입니다.
    """
    input_dir: Path              # 입력 비디오 디렉토리
    output_dir: Path             # 결과 저장 디렉토리
    
    # --- 특징점 기반 추출 핵심 설정 ---
    sample_interval_sec: float    # 분석 프레임 샘플링 간격 (초)
    persistence_threshold: int    # 특징점 유효 인정을 위한 최소 지속 샘플 수
    persistence_drop_ratio: float # 슬라이드 종료 판단을 위한 특징점 감소 비율
    min_orb_features: int         # 새 슬라이드 인식을 위한 최소 특징점 수

    # --- 하위 호환성용 필드 ---
    sensitivity_diff: float       # 구버전 전이 감지 임계값
    min_interval: float           # 최소 캡처 간격 (초)


def _coerce_str(settings: Dict[str, Any], key: str, default: str) -> str:
    """[Purpose] 설정 사전에서 값을 가져와 문자열로 강제 변환합니다."""
    value = settings.get(key, default)
    if not isinstance(value, str):
        # 숫자가 들어온 경우에도 문자열로 수용
        return str(value)
    return value


def _coerce_float(settings: Dict[str, Any], key: str, default: float) -> float:
    """[Purpose] 설정 사전에서 값을 가져와 소수점(float)으로 강제 변환합니다."""
    value = settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture 설정의 {key} 값이 올바르지 않습니다: {value}") from exc


def _coerce_int(settings: Dict[str, Any], key: str, default: int) -> int:
    """[Purpose] 설정 사전에서 값을 가져와 정수(int)로 강제 변환합니다."""
    value = settings.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture 설정의 {key} 값이 올바르지 않습니다: {value}") from exc


def _resolve_path(path_value: str) -> Path:
    """[Purpose] 상대 경로 문자열을 프로젝트 루트 기준의 절대 경로로 변환합니다."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_capture_settings(*, settings_path: Optional[Path] = None) -> CaptureSettings:
    """
    [Usage File] run_preprocess_pipeline.py
    [Purpose] 지정된 경로의 YAML 파일을 읽어 CaptureSettings 객체를 생성합니다.
    [Connection] 로컬 파일 시스템(YAML)
    
    [Args]
    - settings_path (Optional[Path]): 로드할 설정 파일 경로. 생략 시 기본 경로 사용.
    
    [Returns]
    - CaptureSettings: 유효성이 검증된 설정 객체
    """
    path = settings_path or DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"capture 설정 파일을 찾을 수 없습니다: {path}")
    
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"YAML 파싱 중 에러가 발생했습니다: {exc}")

    if not isinstance(payload, dict):
        raise ValueError("capture 설정 형식이 올바르지 않습니다(Map 구조여야 함).")

    # 경로 해결
    input_dir = _resolve_path(_coerce_str(payload, "input_dir", "data/inputs"))
    output_dir = _resolve_path(_coerce_str(payload, "output_dir", "data/outputs"))

    # [Security] 경로 범위 검증 (data/inputs, data/outputs 이외 금지)
    allowed_inputs = (PROJECT_ROOT / "data" / "inputs").resolve()
    allowed_outputs = (PROJECT_ROOT / "data" / "outputs").resolve()

    def _check_allowed(path_obj: Path, label: str):
        p = path_obj.resolve()
        is_valid = False
        if p == allowed_inputs or p == allowed_outputs:
            is_valid = True
        else:
            try:
                p.relative_to(allowed_inputs)
                is_valid = True
            except ValueError:
                try:
                    p.relative_to(allowed_outputs)
                    is_valid = True
                except ValueError:
                    pass
        if not is_valid:
            try:
                rel_path = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = p
            error_msg = f"[Path Error] {label}이 허용된 범위를 벗어났습니다: {rel_path}\n(허용 범위: data/inputs 또는 data/outputs)"
            print(error_msg)
            raise RuntimeError(error_msg)

    _check_allowed(input_dir, "Capture Input Dir")
    _check_allowed(output_dir, "Capture Output Dir")

    return CaptureSettings(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_interval_sec=_coerce_float(payload, "sample_interval_sec", 0.5),
        persistence_threshold=_coerce_int(payload, "persistence_threshold", 6),
        persistence_drop_ratio=_coerce_float(payload, "persistence_drop_ratio", 0.4),
        min_orb_features=_coerce_int(payload, "min_orb_features", 50),
        sensitivity_diff=_coerce_float(payload, "sensitivity_diff", 0.3),
        min_interval=_coerce_float(payload, "min_interval", 0.5),
    )


@lru_cache(maxsize=1)
def get_capture_settings() -> CaptureSettings:
    """
    [Usage File] process_content.py, run_preprocess_pipeline.py
    [Purpose] 시스템 전체에서 사용될 캡처 설정을 싱글톤 형태로 제공합니다 (LRU 캐시 활용).
    
    [Returns]
    - CaptureSettings: 캐시된 설정 객체 (최초 호출 시 로드)
    """
    return load_capture_settings()
