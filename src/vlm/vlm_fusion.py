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
    1. capture.json에서 (timestamp_ms, file_name) 쌍을 추출해 시간순 정렬.
    2. vlm_raw.json에서 file_name(이미지 파일명)을 키로 하여 추출 텍스트를 매핑.
    3. manifest의 file_name으로 vlm_raw의 텍스트를 찾아 타임스탬프와 결합.
    4. 매칭되지 않는 이미지가 있으면 에러 발생.

    Args:
        manifest_payload: capture.json 로드 결과 (List[Dict])
        vlm_raw_payload: vlm_raw.json 로드 결과 (List[Dict])

    Returns:
        Dict[str, Any]: Fusion 단계에서 사용할 vlm.json 구조
        Example:
            {
                "items": [
                    {"timestamp_ms": 1000, "extracted_text": "Slide 1 Content..."},
                    {"timestamp_ms": 2000, "extracted_text": "Slide 2 Content..."},
                    ...
                ]
            }

    Raises:
        ValueError: 입력 형식이 잘못되었거나, manifest의 파일이 vlm_raw에 없을 때.
    """
    # 1. Manifest 검증 및 엔트리 추출
    if not isinstance(manifest_payload, list):
        raise ValueError("Invalid capture.json format (must be a list).")

    manifest_entries: List[Dict[str, Any]] = []
    for item in manifest_payload:
        if not isinstance(item, dict):
            continue
        # 필수 필드 확인
        if "start_ms" not in item or "file_name" not in item:
            continue
        try:
            start_ms = int(item["start_ms"])
        except (TypeError, ValueError):
            continue
            
        file_name = str(item["file_name"]).strip()
        if not file_name:
            continue
            
        manifest_entries.append({"timestamp_ms": start_ms, "file_name": file_name})

    if not manifest_entries:
        raise ValueError("No valid entries found in capture.json.")
    
    # 타임스탬프 기준 오름차순 정렬 (Fusion은 시간 순서 처리가 필수)
    manifest_entries.sort(key=lambda x: (int(x["timestamp_ms"]), str(x["file_name"])))

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
        # 예: data/captures/slide_001.jpg -> slide_001.jpg
        image_text[Path(image_path).name] = "\n\n".join(parts).strip()

    if not image_text:
        raise ValueError("No valid image_path found in vlm_raw.json.")

    # 3. Join (Manifest + VLM Raw)
    missing: List[str] = []
    items: List[Dict[str, Any]] = []
    vlm_index = 0
    
    for entry in manifest_entries:
        ts = int(entry["timestamp_ms"])
        file_name = str(entry["file_name"])
        
        # Manifest에는 있는데 VLM 결과가 없으면 에러 (누락 방지)
        if file_name not in image_text:
            missing.append(file_name)
            continue
        
        vlm_index += 1
        items.append({
            "timestamp_ms": ts,
            "extracted_text": image_text[file_name],
            "id": f"vlm_{vlm_index:03d}",
        })

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Failed to join capture.json and vlm_raw.json: {len(missing)} images are missing. Example: {preview}"
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
        json.dumps(fusion_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

