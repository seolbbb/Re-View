"""VLM 원시 결과(vlm_raw.json) + capture manifest를 fusion 입력 vlm.json으로 변환."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_manifest_entries(manifest: Any) -> List[Dict[str, Any]]:
    if not isinstance(manifest, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    normalized: List[Dict[str, Any]] = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        if "timestamp_ms" not in item or "file_name" not in item:
            continue
        try:
            timestamp_ms = int(item["timestamp_ms"])
        except (TypeError, ValueError):
            continue
        file_name = str(item["file_name"]).strip()
        if not file_name:
            continue
        normalized.append({"timestamp_ms": timestamp_ms, "file_name": file_name})

    if not normalized:
        raise ValueError("manifest.json에서 유효한 항목을 찾을 수 없습니다.")
    normalized.sort(key=lambda x: (int(x["timestamp_ms"]), str(x["file_name"])))
    return normalized


def _extract_text_from_raw_results(raw_results: Any) -> str:
    if not isinstance(raw_results, list):
        return ""
    parts: List[str] = []
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
    return "\n\n".join(parts).strip()


def _build_image_text_map(vlm_raw: Any) -> Dict[str, str]:
    if not isinstance(vlm_raw, list):
        raise ValueError("vlm_raw.json 형식이 올바르지 않습니다(배열이어야 함).")

    mapping: Dict[str, str] = {}
    for item in vlm_raw:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        file_name = Path(image_path).name
        extracted_text = _extract_text_from_raw_results(item.get("raw_results"))
        mapping[file_name] = extracted_text
    if not mapping:
        raise ValueError("vlm_raw.json에서 유효한 image_path를 찾을 수 없습니다.")
    return mapping


def build_fusion_vlm_payload(
    *,
    manifest_payload: Any,
    vlm_raw_payload: Any,
) -> Tuple[Dict[str, Any], List[str]]:
    manifest_entries = _normalize_manifest_entries(manifest_payload)
    image_text = _build_image_text_map(vlm_raw_payload)

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
    return {"schema_version": 1, "items": items}, [str(e["file_name"]) for e in manifest_entries]


def convert_vlm_raw_to_fusion_vlm(
    *,
    manifest_json: Path,
    vlm_raw_json: Path,
    output_vlm_json: Path,
) -> None:
    manifest_payload = _read_json(manifest_json)
    vlm_raw_payload = _read_json(vlm_raw_json)
    fusion_payload, _ = build_fusion_vlm_payload(
        manifest_payload=manifest_payload,
        vlm_raw_payload=vlm_raw_payload,
    )
    _write_json(output_vlm_json, fusion_payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vlm_raw.json + manifest.json → fusion 입력 vlm.json 변환")
    parser.add_argument("--manifest", required=True, help="capture manifest.json 경로")
    parser.add_argument("--vlm-raw", required=True, help="VLM 원시 결과(vlm_raw.json) 경로")
    parser.add_argument("--out", required=True, help="출력 vlm.json 경로")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=Path(args.manifest),
        vlm_raw_json=Path(args.vlm_raw),
        output_vlm_json=Path(args.out),
    )
    print(f"[OK] saved to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

