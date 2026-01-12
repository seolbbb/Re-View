"""Preprocessing Agent 도구들.

Preprocessing Agent가 사용하는 도구:
- load_data: Pre-ADK 산출물 검증
- run_vlm: VLM 실행 (캡처 → vlm.json)
- run_sync: Sync 실행 (STT + VLM → segments_units.jsonl)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

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

    배치 모드일 때는 현재 배치의 캡처 범위만 처리하고 batch 폴더에 저장합니다.

    Returns:
        success: 실행 성공 여부
        vlm_json: 생성된 vlm.json 경로
        skipped: 이미 존재하여 스킵한 경우 True
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    force_rerun = tool_context.state.get("force_preprocessing", False)
    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    batch_mode = tool_context.state.get("batch_mode", False)

    # 배치 모드: 현재 배치의 캡처 범위만 처리
    if batch_mode:
        batch_ranges = tool_context.state.get("batch_ranges", [])
        current_batch_index = tool_context.state.get("current_batch_index", 0)
        sorted_manifest = tool_context.state.get("sorted_manifest", [])

        if current_batch_index >= len(batch_ranges):
            return {"success": False, "error": f"배치 인덱스 {current_batch_index}가 범위를 벗어났습니다."}

        batch_info = batch_ranges[current_batch_index]
        start_idx = batch_info["start_idx"]
        end_idx = batch_info["end_idx"]
        batch_manifest = sorted_manifest[start_idx:end_idx]

        # 배치 폴더에 저장
        output_dir = store.batch_dir(current_batch_index)
        vlm_output_path = store.batch_vlm_json(current_batch_index)

        if vlm_output_path.exists() and not force_rerun:
            return {
                "success": True,
                "skipped": True,
                "vlm_json": str(vlm_output_path),
                "message": f"배치 {current_batch_index} vlm.json이 이미 존재합니다.",
            }

        raw_batch_size = tool_context.state.get("vlm_batch_size", 2)
        raw_concurrency = tool_context.state.get("vlm_concurrency", 3)
        show_progress = bool(tool_context.state.get("vlm_show_progress", True))

        try:
            vlm_batch_size = int(raw_batch_size) if raw_batch_size else None
            concurrency = int(raw_concurrency)
        except (TypeError, ValueError):
            vlm_batch_size = 2
            concurrency = 3

        try:
            from .internal.vlm_openrouter import run_vlm_for_batch
            result = run_vlm_for_batch(
                captures_dir=store.captures_dir(),
                manifest_json=store.manifest_json(),
                video_name=video_name,
                output_dir=output_dir,
                batch_manifest=batch_manifest,
                batch_size=vlm_batch_size,
                concurrency=concurrency,
                show_progress=show_progress,
            )

            return {
                "success": True,
                "skipped": False,
                "batch_index": current_batch_index,
                "vlm_json": result.get("vlm_json", ""),
                "image_count": result.get("image_count", 0),
                "message": f"배치 {current_batch_index} VLM 완료 ({result.get('image_count', 0)}장 처리)",
            }
        except Exception as e:
            return {"success": False, "error": f"배치 VLM 실행 실패: {e}"}

    # 일반 모드: 전체 캡처 처리
    # 강제 재실행 시 기존 파일 삭제
    if force_rerun and store.vlm_json().exists():
        store.vlm_json().unlink()
        if store.vlm_raw_json().exists():
            store.vlm_raw_json().unlink()

    # 이미 존재하면 스킵 (force_rerun이 아닌 경우)
    if store.vlm_json().exists():
        return {
            "success": True,
            "skipped": True,
            "vlm_json": str(store.vlm_json()),
            "message": "vlm.json이 이미 존재합니다. 스킵합니다.",
        }

    raw_batch_size = tool_context.state.get("vlm_batch_size", 2)
    raw_concurrency = tool_context.state.get("vlm_concurrency", 3)
    show_progress = bool(tool_context.state.get("vlm_show_progress", True))

    try:
        batch_size: Optional[int]
        if raw_batch_size is None:
            batch_size = None
        else:
            batch_size = int(raw_batch_size)
            if batch_size < 1:
                return {"success": False, "error": "vlm_batch_size는 1 이상이어야 합니다."}

        concurrency = int(raw_concurrency)
        if concurrency < 1:
            return {"success": False, "error": "vlm_concurrency는 1 이상의 정수여야 합니다."}
    except (TypeError, ValueError):
        return {"success": False, "error": "VLM 설정 값이 올바르지 않습니다."}

    try:
        from .internal.vlm_openrouter import run_vlm_openrouter
        run_vlm_openrouter(
            captures_dir=store.captures_dir(),
            manifest_json=store.manifest_json(),
            video_name=video_name,
            output_base=_OUTPUT_BASE,
            batch_size=batch_size,
            concurrency=concurrency,
            show_progress=show_progress,
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

    배치 모드일 때는 현재 배치의 vlm.json을 사용하고 batch 폴더에 저장합니다.

    Returns:
        success: 실행 성공 여부
        segments_units_jsonl: 생성된 파일 경로
        skipped: 이미 존재하여 스킵한 경우 True
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    force_rerun = tool_context.state.get("force_preprocessing", False)
    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)
    batch_mode = tool_context.state.get("batch_mode", False)

    # 배치 모드: 현재 배치의 vlm.json으로 sync
    if batch_mode:
        batch_ranges = tool_context.state.get("batch_ranges", [])
        current_batch_index = tool_context.state.get("current_batch_index", 0)

        if current_batch_index >= len(batch_ranges):
            return {"success": False, "error": f"배치 인덱스 {current_batch_index}가 범위를 벗어났습니다."}

        batch_info = batch_ranges[current_batch_index]
        start_ms = batch_info.get("start_ms", 0)
        end_ms = batch_info.get("end_ms", 0)

        # 배치 폴더 경로
        batch_dir = store.batch_dir(current_batch_index)
        vlm_path = store.batch_vlm_json(current_batch_index)
        segments_output = store.batch_segments_units_jsonl(current_batch_index)

        if not vlm_path.exists():
            return {"success": False, "error": f"배치 {current_batch_index} vlm.json이 없습니다."}

        if segments_output.exists() and not force_rerun:
            return {
                "success": True,
                "skipped": True,
                "segments_units_jsonl": str(segments_output),
                "message": f"배치 {current_batch_index} segments_units.jsonl이 이미 존재합니다.",
            }

        try:
            # 첫 배치에서 fusion config 생성
            if not store.fusion_config_yaml().exists():
                from .internal.fusion_config import generate_fusion_config
                fusion_template = _PROJECT_ROOT / "src" / "fusion" / "config.yaml"
                generate_fusion_config(
                    template_config=fusion_template,
                    output_config=store.fusion_config_yaml(),
                    repo_root=_PROJECT_ROOT,
                    stt_json=store.stt_json(),
                    vlm_json=vlm_path,
                    manifest_json=store.manifest_json(),
                    output_root=store.video_root(),
                )

            # cumulative segment count로 segment_id offset 관리
            cumulative_segment_count = tool_context.state.get("cumulative_segment_count", 0)

            # Sync 실행 (배치 범위 지정)
            from .internal.sync_data import run_batch_sync_engine
            result = run_batch_sync_engine(
                fusion_config_path=store.fusion_config_yaml(),
                vlm_path=vlm_path,
                output_path=segments_output,
                time_range=(start_ms, end_ms),
                segment_id_offset=cumulative_segment_count,
            )

            # segment count 누적
            new_segments_count = result.get("segments_count", 0)
            tool_context.state["cumulative_segment_count"] = cumulative_segment_count + new_segments_count

            return {
                "success": True,
                "skipped": False,
                "batch_index": current_batch_index,
                "segments_units_jsonl": str(segments_output),
                "segments_count": new_segments_count,
                "message": f"배치 {current_batch_index} Sync 완료 ({new_segments_count}개 세그먼트)",
            }
        except Exception as e:
            return {"success": False, "error": f"배치 Sync 실행 실패: {e}"}

    # 일반 모드: 전체 Sync
    video_root = store.video_root()

    # VLM이 먼저 완료되어야 함
    if not store.vlm_json().exists():
        return {"success": False, "error": "vlm.json이 없습니다. run_vlm을 먼저 실행하세요."}

    # 강제 재실행 시 기존 파일 삭제
    if force_rerun and store.segments_units_jsonl().exists():
        store.segments_units_jsonl().unlink()
        if store.segments_jsonl().exists():
            store.segments_jsonl().unlink()

    # 이미 존재하면 스킵 (force_rerun이 아닌 경우)
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

