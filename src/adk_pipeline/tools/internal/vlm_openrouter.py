"""VLM 도구(OpenRouter/Qwen) - manifest.json과 captures를 읽어 vlm_raw.json/vlm.json 생성."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm


def _filter_manifest_by_time_range(
    manifest_payload: List[Dict],
    start_ms: int,
    end_ms: int,
) -> List[Dict]:
    """manifest에서 특정 시간 범위의 항목만 필터링."""
    filtered = []
    for item in manifest_payload:
        # timestamp_ms 또는 start_ms 필드 사용
        timestamp_ms = item.get("timestamp_ms", item.get("start_ms", 0))
        if timestamp_ms is None:
            continue
        timestamp_ms = int(timestamp_ms)
        if start_ms <= timestamp_ms < end_ms:
            filtered.append(item)
    return filtered


def run_vlm_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
    concurrency: int = 1,
    show_progress: bool = False,
) -> Dict[str, str]:
    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_base)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_paths: List[str] = []
    for item in sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", 0)), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))

    if not image_paths:
        raise ValueError("VLM 입력 이미지가 없습니다(manifest.json을 확인하세요).")

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    vlm_json = raw_path.with_name("vlm.json")
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=vlm_json,
    )
    raw_path.unlink(missing_ok=True)

    return {"vlm_raw_json": str(raw_path), "vlm_json": str(vlm_json)}


def run_vlm_for_batch(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_dir: Path,
    start_ms: int,
    end_ms: int,
    batch_size: Optional[int],
    concurrency: int = 1,
    show_progress: bool = False,
) -> Dict[str, str]:
    """특정 시간 범위의 캡처만 VLM 처리하여 배치별 vlm.json 생성.

    Args:
        captures_dir: 캡처 이미지 디렉토리
        manifest_json: manifest.json 경로
        video_name: 비디오 이름
        output_dir: 출력 디렉토리 (배치별 디렉토리)
        start_ms: 시작 시간 (밀리초)
        end_ms: 종료 시간 (밀리초)
        batch_size: VLM 요청 배치 크기
        concurrency: 병렬 요청 수
        show_progress: 진행 로그 출력 여부

    Returns:
        vlm_raw_json: vlm_raw.json 경로
        vlm_json: vlm.json 경로
        image_count: 처리된 이미지 수
    """
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_dir.parent)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    # 시간 범위로 필터링
    filtered_manifest = _filter_manifest_by_time_range(manifest_payload, start_ms, end_ms)

    image_paths: List[str] = []
    filtered_manifest_items: List[Dict] = []
    for item in sorted(
        (x for x in filtered_manifest if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", x.get("start_ms", 0))), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))
        filtered_manifest_items.append(item)

    if not image_paths:
        # 이 시간 범위에 이미지가 없으면 빈 vlm.json 생성
        empty_vlm = {"schema_version": 1, "items": [], "duration_ms": end_ms}
        vlm_json_path = output_dir / "vlm.json"
        vlm_json_path.write_text(json.dumps(empty_vlm, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "vlm_raw_json": "",
            "vlm_json": str(vlm_json_path),
            "image_count": 0,
        }

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )

    # 배치별 vlm_raw.json 저장
    raw_path = output_dir / "vlm_raw.json"
    write_vlm_raw_json(results, raw_path)

    # 배치별 manifest.json 임시 생성 (convert 함수용)
    temp_manifest_path = output_dir / "manifest_temp.json"
    temp_manifest_path.write_text(
        json.dumps(filtered_manifest_items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # vlm_raw.json → vlm.json 변환
    vlm_json_path = output_dir / "vlm.json"
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=temp_manifest_path,
        vlm_raw_json=raw_path,
        output_vlm_json=vlm_json_path,
    )

    # 임시 파일 정리 (vlm_raw.json은 유지)
    temp_manifest_path.unlink(missing_ok=True)

    return {
        "vlm_raw_json": str(raw_path),
        "vlm_json": str(vlm_json_path),
        "image_count": len(image_paths),
    }
