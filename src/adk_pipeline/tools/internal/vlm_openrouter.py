"""VLM 도구(OpenRouter/Qwen) - manifest.json과 captures를 읽어 vlm_raw.json/vlm.json 생성."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm


def run_vlm_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
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

    results = extractor.extract_features(image_paths, batch_size=batch_size)
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    vlm_json = raw_path.with_name("vlm.json")
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=vlm_json,
    )

    return {"vlm_raw_json": str(raw_path), "vlm_json": str(vlm_json)}
