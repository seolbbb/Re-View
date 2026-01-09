"""Fusion config.yaml 생성 도구.

기존 `src/run_video_pipeline.py:_generate_fusion_config` 동작을 그대로 재사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def generate_fusion_config(
    *,
    template_config: Path,
    output_config: Path,
    repo_root: Path,
    stt_json: Path,
    vlm_json: Path,
    manifest_json: Optional[Path],
    output_root: Path,
) -> None:
    payload: Dict[str, Any]
    with template_config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root)).replace("\\\\", "/")
        except ValueError:
            return str(p)

    paths_payload: Dict[str, Any] = {
        "stt_json": _rel(stt_json),
        "vlm_json": _rel(vlm_json),
        "output_root": _rel(output_root),
    }
    if manifest_json is not None:
        paths_payload["captures_manifest_json"] = _rel(manifest_json)

    payload["paths"] = paths_payload

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
