"""
[Intent]
캡처 파이프라인에서 사용되는 모든 설정값(임계값, 경로, 옵션 등)을 관리하는 모듈입니다.
YAML 설정 파일을 로드하고, 타입을 강제 변환하며, 유효성을 검증하여 안전한 CaptureSettings 객체를 제공합니다.

[Usage]
- run_preprocess_pipeline.py: 초기화 시 설정을 로드하기 위해 호출
- process_content.py: 캡처 엔진에 필요한 파라미터를 가져오기 위해 호출

[Usage Method]
- get_capture_settings(): 싱글톤 패턴으로 캐시된 설정 객체를 반환합니다.
- load_capture_settings(settings_path): 특정 경로의 YAML 파일에서 설정을 새로 로드합니다.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# 프로젝트 루트 경로 (src/capture/settings.py 기준 2단계 상위 -> src/capture -> src -> ReViewFeature)
# 실제 파일 위치: src/capture/settings.py
# parents[0]: src/capture
# parents[1]: src
# parents[2]: ReViewFeature (Root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 기본 설정 파일 경로
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "capture" / "settings.yaml"


@dataclass(frozen=True)
class CaptureSettings:
    """
    [Class Purpose]
    캡처 엔진 구동에 필요한 모든 제어 변수를 담는 불변 데이터 클래스입니다.
    """
    input_dir: Path              # 입력 비디오가 위치한 디렉토리
    output_dir: Path             # 캡처 결과가 저장될 디렉토리
    
    # --- 특징점 기반 추출 핵심 설정 ---
    sample_interval_sec: float    # 영상 분석 샘플링 간격 (초 단위)
    persistence_threshold: int    # 슬라이드 인식을 위한 특징점 지속 프레임 수
    persistence_drop_ratio: float # 슬라이드 전환(종료) 판단을 위한 특징점 감소 비율 (0.4 = 40%)
    min_orb_features: int         # 유효한 슬라이드로 간주할 최소 특징점 개수

    # --- 중복 제거 설정 ---
    dedup_phash_threshold: int    # Perceptual Hash 해밍 거리 임계값 (작을수록 엄격)
    dedup_orb_distance: int       # ORB Descriptor 매칭 거리 임계값
    dedup_sim_threshold: float    # 두 이미지 간 특징점 유사도 임계값 (0.0 ~ 1.0)

    # --- ROI 설정 ---
    enable_roi_detection: bool    # 레터박스(상하좌우 여백) 자동 감지 및 제거 활성화
    roi_padding: int              # 감지된 ROI 영역에 추가할 여백 (픽셀)

    # --- 하위 호환성용 필드 ---
    sensitivity_diff: float       # (구버전) 이미지 차이 민감도
    min_interval: float           # (구버전) 최소 캡처 간격


def _coerce_str(settings: Dict[str, Any], key: str, default: str) -> str:
    """
    [Usage File] Internal use (settings.py)
    [Purpose] 설정 딕셔너리에서 값을 안전하게 가져와 문자열로 변환합니다.
    [Args] settings(Dict), key(str), default(str)
    """
    value = settings.get(key, default)
    if not isinstance(value, str):
        return str(value)
    return value


def _coerce_float(settings: Dict[str, Any], key: str, default: float) -> float:
    """
    [Usage File] Internal use (settings.py)
    [Purpose] 설정 딕셔너리에서 값을 안전하게 가져와 실수형(float)으로 변환합니다.
    [Args] settings(Dict), key(str), default(float)
    """
    value = settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture 설정의 {key} 값이 올바르지 않습니다: {value}") from exc


def _coerce_int(settings: Dict[str, Any], key: str, default: int) -> int:
    """
    [Usage File] Internal use (settings.py)
    [Purpose] 설정 딕셔너리에서 값을 안전하게 가져와 정수형(int)으로 변환합니다.
    [Args] settings(Dict), key(str), default(int)
    """
    value = settings.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"capture 설정의 {key} 값이 올바르지 않습니다: {value}") from exc


def _resolve_path(path_value: str) -> Path:
    """
    [Usage File] Internal use (settings.py)
    [Purpose] 입력된 경로 문자열이 상대 경로일 경우 프로젝트 루트 기준 절대 경로로 변환합니다.
    [Args] path_value(str)
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_capture_settings(*, settings_path: Optional[Path] = None) -> CaptureSettings:
    """
    [Usage File]
    - run_preprocess_pipeline.py
    
    [Purpose]
    - 지정된 YAML 파일을 읽어 파싱하고, 유효성 검사를 거쳐 CaptureSettings 객체를 생성합니다.
    - 입력/출력 경로가 허용된 범위(data/inputs, data/outputs) 내에 있는지 보안 검사를 수행합니다.
    
    [Connection]
    - File System: YAML 설정 파일 읽기

    [Args]
    - settings_path (Optional[Path]): 로드할 설정 파일의 경로 (기본값: config/capture/settings.yaml)

    [Returns]
    - CaptureSettings: 초기화된 설정 객체
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

    # [Security] 경로 범위 검증
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
        
        # Deduplication params
        dedup_phash_threshold=_coerce_int(payload, "dedup_phash_threshold", 12),
        dedup_orb_distance=_coerce_int(payload, "dedup_orb_distance", 60),
        dedup_sim_threshold=_coerce_float(payload, "dedup_sim_threshold", 0.5),

        # ROI Detection
        enable_roi_detection=bool(payload.get("enable_roi_detection", True)),
        roi_padding=_coerce_int(payload, "roi_padding", 10),

        sensitivity_diff=_coerce_float(payload, "sensitivity_diff", 0.3),
        min_interval=_coerce_float(payload, "min_interval", 0.5),
    )


@lru_cache(maxsize=1)
def get_capture_settings() -> CaptureSettings:
    """
    [Usage File]
    - process_content.py, run_preprocess_pipeline.py

    [Purpose]
    - 애플리케이션 수명주기 동안 단일 설정 객체를 제공(Singleton)하기 위한 헬퍼 함수입니다.
    - LRU 캐시를 사용하여 파일 I/O 반복을 방지합니다.

    [Returns]
    - CaptureSettings: 로드된 설정 객체
    """
    return load_capture_settings()
