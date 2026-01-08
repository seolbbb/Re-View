"""Preprocessing Agent 도구들.

Preprocessing Agent가 사용하는 도구:
- load_data: Pre-ADK 산출물 검증
- run_vlm: VLM 실행 (캡처 → vlm.json)
- run_sync: Sync 실행 (STT + VLM → segments_units.jsonl)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from google.adk.tools import ToolContext

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..paths import DEFAULT_OUTPUT_BASE
from ..store import VideoStore

_OUTPUT_BASE = _PROJECT_ROOT / DEFAULT_OUTPUT_BASE


def load_data(tool_context: ToolContext) -> Dict[str, Any]:
    """Pre-ADK 산출물(stt.json, manifest.json, captures)을 검증합니다.

    Returns:
        success: 검증 성공 여부
        artifacts: 산출물 경로들
        error: 실패 시 에러 메시지
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    video_root = store.video_root()

    # 필수 산출물 검증
    missing = []
    if not store.stt_json().exists():
        missing.append("stt.json")
    if not store.manifest_json().exists():
        missing.append("manifest.json")
    if not store.captures_dir().exists():
        missing.append("captures/")

    if missing:
        return {
            "success": False,
            "error": f"Pre-ADK 산출물 누락: {', '.join(missing)}",
            "hint": "run_adk_pipeline.py --video 명령으로 Pre-ADK 단계를 먼저 실행하세요.",
        }

    return {
        "success": True,
        "video_name": video_name,
        "artifacts": {
            "stt_json": str(store.stt_json()),
            "manifest_json": str(store.manifest_json()),
            "captures_dir": str(store.captures_dir()),
        },
    }


def run_vlm(tool_context: ToolContext) -> Dict[str, Any]:
    """VLM을 실행하여 캡처 이미지에서 텍스트/UI 요소를 추출합니다.

    Returns:
        success: 실행 성공 여부
        vlm_json: 생성된 vlm.json 경로
        skipped: 이미 존재하여 스킵한 경우 True
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 이미 존재하면 스킵
    if store.vlm_json().exists():
        return {
            "success": True,
            "skipped": True,
            "vlm_json": str(store.vlm_json()),
            "message": "vlm.json이 이미 존재합니다. 스킵합니다.",
        }

    try:
        from .internal.vlm_openrouter import run_vlm_openrouter
        run_vlm_openrouter(
            captures_dir=store.captures_dir(),
            manifest_json=store.manifest_json(),
            video_name=video_name,
            output_base=_OUTPUT_BASE,
            batch_size=None,
        )

        return {
            "success": True,
            "skipped": False,
            "vlm_json": str(store.vlm_json()),
            "vlm_raw_json": str(store.vlm_raw_json()),
        }
    except Exception as e:
        return {"success": False, "error": f"VLM 실행 실패: {e}"}


def run_sync(tool_context: ToolContext) -> Dict[str, Any]:
    """Sync를 실행하여 STT와 VLM 결과를 동기화합니다.

    Returns:
        success: 실행 성공 여부
        segments_units_jsonl: 생성된 파일 경로
        skipped: 이미 존재하여 스킵한 경우 True
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    video_root = store.video_root()

    # VLM이 먼저 완료되어야 함
    if not store.vlm_json().exists():
        return {"success": False, "error": "vlm.json이 없습니다. run_vlm을 먼저 실행하세요."}

    # 이미 존재하면 스킵
    if store.segments_units_jsonl().exists():
        return {
            "success": True,
            "skipped": True,
            "segments_units_jsonl": str(store.segments_units_jsonl()),
            "message": "segments_units.jsonl이 이미 존재합니다. 스킵합니다.",
        }

    try:
        # Fusion config 생성
        from .internal.fusion_config import generate_fusion_config
        fusion_template = _PROJECT_ROOT / "src" / "fusion" / "config.yaml"
        generate_fusion_config(
            template_config=fusion_template,
            output_config=store.fusion_config_yaml(),
            repo_root=_PROJECT_ROOT,
            stt_json=store.stt_json(),
            vlm_json=store.vlm_json(),
            manifest_json=store.manifest_json(),
            output_root=video_root,
        )

        # Sync 실행
        from .internal.sync_data import run_sync as _run_sync
        _run_sync(fusion_config_path=store.fusion_config_yaml(), limit=None)

        return {
            "success": True,
            "skipped": False,
            "segments_jsonl": str(store.segments_jsonl()),
            "segments_units_jsonl": str(store.segments_units_jsonl()),
        }
    except Exception as e:
        return {"success": False, "error": f"Sync 실행 실패: {e}"}
