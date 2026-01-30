"""VLM raw 결과(vlm_raw.json)를 fusion 입력(vlm.json)으로 변환한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_fusion_vlm_payload(
    *,
    manifest_payload: Any,
    vlm_raw_payload: Any,
) -> Dict[str, Any]:
    """manifest와 vlm_raw를 join해 fusion 입력 구조를 만든다.

    동작 방식:
    1. manifest.json에서 각 이미지(file_name)별로 time_ranges를 그룹화.
    2. vlm_raw.json에서 file_name을 키로 하여 추출 텍스트를 매핑.
    3. 이미지별로 1개의 항목을 생성하고 time_ranges 배열을 포함.

    Args:
        manifest_payload: manifest.json 로드 결과 (List[Dict])
        vlm_raw_payload: vlm_raw.json 로드 결과 (List[Dict])

    Returns:
        Dict[str, Any]: Fusion 단계에서 사용할 vlm.json 구조
        Example:
            {
                "items": [
                    {
                        "id": "vlm_001",
                        "cap_id": "cap_001",
                        "extracted_text": "Slide 1 Content...",
                        "time_ranges": [
                            {"start_ms": 0, "end_ms": 84960},
                            {"start_ms": 90931, "end_ms": 94145}
                        ]
                    },
                    ...
                ]
            }

    Raises:
        ValueError: 입력 형식이 잘못되었거나, manifest의 파일이 vlm_raw에 없을 때.
    """
    # 1. Manifest 검증 및 이미지별 time_ranges 그룹화
    if not isinstance(manifest_payload, list):
        raise ValueError("Invalid manifest.json format (must be a list).")

    # 이미지별로 time_ranges를 그룹화 (순서 유지)
    image_time_ranges: Dict[str, Dict[str, Any]] = {}
    
    for item in manifest_payload:
        if not isinstance(item, dict):
            continue
        
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue

        cap_id = item.get("id")
        
        if file_name not in image_time_ranges:
            image_time_ranges[file_name] = {
                "cap_id": cap_id,
                "file_name": file_name,
                "time_ranges": [],
                "first_start_ms": float('inf'),  # 정렬용
            }
        
        # time_ranges 배열 추출
        time_ranges = item.get("time_ranges")
        if time_ranges and isinstance(time_ranges, list):
            for rng in time_ranges:
                if not isinstance(rng, dict):
                    continue
                start_ms = rng.get("start_ms")
                end_ms = rng.get("end_ms")
                if start_ms is not None and end_ms is not None:
                    image_time_ranges[file_name]["time_ranges"].append({
                        "start_ms": int(start_ms),
                        "end_ms": int(end_ms)
                    })
                    # 정렬 기준: 첫 번째 시작 시간
                    if int(start_ms) < image_time_ranges[file_name]["first_start_ms"]:
                        image_time_ranges[file_name]["first_start_ms"] = int(start_ms)
        # Fallback for old schema (start_ms, end_ms)
        elif "start_ms" in item:
            try:
                start_ms = int(item["start_ms"])
                end_ms = int(item.get("end_ms", start_ms))
                image_time_ranges[file_name]["time_ranges"].append({
                    "start_ms": start_ms,
                    "end_ms": end_ms
                })
                if start_ms < image_time_ranges[file_name]["first_start_ms"]:
                    image_time_ranges[file_name]["first_start_ms"] = start_ms
            except (ValueError, TypeError):
                continue

    if not image_time_ranges:
        raise ValueError("No valid entries found in manifest.json (checked 'time_ranges' and 'start_ms').")

    # 2. VLM Raw 결과 매핑 (Key: 파일명 -> Value: 추출 텍스트)
    if not isinstance(vlm_raw_payload, list):
        raise ValueError("Invalid vlm_raw.json format (must be a list).")

    image_text: Dict[str, str] = {}
    for item in vlm_raw_payload:
        if not isinstance(item, dict):
            continue
            
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
            
        # VLM 결과에서 텍스트 추출 (여러 박스가 있을 경우 개행으로 연결)
        raw_results = item.get("raw_results")
        parts: List[str] = []
        if isinstance(raw_results, list):
            for box in raw_results:
                if not isinstance(box, dict):
                    continue
                text = box.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        
        # 파일명만 추출하여 키로 사용 (경로 제외)
        image_text[Path(image_path).name] = "\n\n".join(parts).strip()

    if not image_text:
        raise ValueError("No valid image_path found in vlm_raw.json.")

    # 3. Join (Manifest + VLM Raw) - 이미지별로 1개 항목 생성
    missing: List[str] = []
    items: List[Dict[str, Any]] = []
    
    # 첫 번째 시작 시간 기준 정렬
    sorted_images = sorted(
        image_time_ranges.values(),
        key=lambda x: x["first_start_ms"]
    )
    
    for idx, img_data in enumerate(sorted_images, start=1):
        file_name = img_data["file_name"]
        
        # Manifest에는 있는데 VLM 결과가 없으면 에러 (누락 방지)
        if file_name not in image_text:
            missing.append(file_name)
            continue
        
        # time_ranges 정렬 (start_ms 기준)
        sorted_ranges = sorted(img_data["time_ranges"], key=lambda x: x["start_ms"])
            
        items.append({
            "id": f"vlm_{idx:03d}",
            "cap_id": img_data["cap_id"],
            "extracted_text": image_text[file_name],
            "time_ranges": sorted_ranges
        })

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Failed to join manifest.json and vlm_raw.json: {len(missing)} images are missing. Example: {preview}"
        )
        
    return {"items": items}


def convert_vlm_raw_to_fusion_vlm(
    *,
    manifest_json: Path,
    vlm_raw_json: Path,
    output_vlm_json: Path,
) -> None:
    """파일 경로를 받아 vlm_raw.json을 vlm.json으로 변환한다.

    Args:
        manifest_json: 캡처 메니페스트 파일 경로 (입력)
        vlm_raw_json: VLM 추출 결과 파일 경로 (입력)
        output_vlm_json: 변환된 vlm.json 저장 경로 (출력)
    """
    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    vlm_raw_payload = json.loads(vlm_raw_json.read_text(encoding="utf-8"))
    
    fusion_payload = build_fusion_vlm_payload(
        manifest_payload=manifest_payload,
        vlm_raw_payload=vlm_raw_payload,
    )
    
    output_vlm_json.parent.mkdir(parents=True, exist_ok=True)
    output_vlm_json.write_text(
        json.dumps(fusion_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

