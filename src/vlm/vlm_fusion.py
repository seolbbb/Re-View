"""VLM 원시 결과(vlm_raw.json)를 fusion 입력(vlm.json)으로 변환한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def build_fusion_vlm_payload(
    *,
    manifest_payload: Any,
    vlm_raw_payload: Any,
) -> Dict[str, Any]:
    """manifest와 vlm_raw를 조인해 fusion 입력 구조를 만든다."""
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    manifest_entries: List[Dict[str, Any]] = []
    for item in manifest_payload:
        if not isinstance(item, dict):
            continue
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
        raise ValueError("manifest.json에서 유효한 항목을 찾을 수 없습니다.")
    manifest_entries.sort(key=lambda x: (int(x["timestamp_ms"]), str(x["file_name"])))

    if not isinstance(vlm_raw_payload, list):
        raise ValueError("vlm_raw.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_text: Dict[str, str] = {}
    for item in vlm_raw_payload:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        raw_results = item.get("raw_results")
        parts: List[str] = []
        if isinstance(raw_results, list):
            for box in raw_results:
                if not isinstance(box, dict):
                    continue
                text = box.get("text")
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                parts.append(text)
        image_text[Path(image_path).name] = "\n\n".join(parts).strip()

    if not image_text:
        raise ValueError("vlm_raw.json에서 유효한 image_path를 찾을 수 없습니다.")

    missing: List[str] = []
    items: List[Dict[str, Any]] = []
    for entry in manifest_entries:
        ts = int(entry["timestamp_ms"])
        file_name = str(entry["file_name"])
        if file_name not in image_text:
            missing.append(file_name)
            continue
        items.append({"timestamp_ms": ts, "extracted_text": image_text[file_name]})

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"manifest.json과 vlm_raw.json 조인 실패: {len(missing)}개 이미지가 누락되었습니다. 예: {preview}"
        )
    return {"items": items}


def convert_vlm_raw_to_fusion_vlm(
    *,
    manifest_json: Path,
    vlm_raw_json: Path,
    output_vlm_json: Path,
) -> None:
    """파일 경로를 받아 vlm_raw.json을 vlm.json으로 변환한다."""
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

