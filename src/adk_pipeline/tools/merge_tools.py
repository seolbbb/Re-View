"""병합 도구 (Merge Tools).

모든 배치 결과를 병합하고 최종 요약을 생성합니다.
- merge_all_batches: 배치별 파일 병합
- generate_final_summary: LLM으로 전체 통합 요약 생성
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from google.adk.tools import ToolContext

# 프로젝트 루트를 sys.path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..paths import DEFAULT_OUTPUT_BASE
from ..store import VideoStore

_OUTPUT_BASE = _PROJECT_ROOT / DEFAULT_OUTPUT_BASE


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일을 읽어 리스트로 반환합니다."""
    if not path.exists():
        return []
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _write_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    """리스트를 JSONL 파일로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_all_batches(tool_context: ToolContext) -> Dict[str, Any]:
    """모든 배치 결과를 병합합니다.

    각 배치 디렉토리에서 vlm.json, segments_units.jsonl, segment_summaries.jsonl을
    읽어 전체 파일로 병합합니다.

    Returns:
        success: 실행 성공 여부
        merged_files: 병합된 파일 경로들
        batch_count: 병합된 배치 수
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    batch_mode = tool_context.state.get("batch_mode", False)
    if not batch_mode:
        return {"success": False, "error": "배치 모드가 비활성화 상태입니다."}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    total_batches = tool_context.state.get("total_batches", 0)
    if total_batches == 0:
        return {"success": False, "error": "배치 정보가 없습니다. init_batch_mode를 먼저 실행하세요."}

    # 병합할 데이터 수집
    all_vlm_data: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    merged_batch_count = 0

    for batch_idx in range(total_batches):
        batch_dir = store.batch_dir(batch_idx)
        batch_has_data = False

        # VLM 데이터 병합
        vlm_path = store.batch_vlm_json(batch_idx)
        if vlm_path.exists():
            with open(vlm_path, "r", encoding="utf-8") as f:
                vlm_data = json.load(f)
                # schema_version/items 형식
                if isinstance(vlm_data, dict):
                    items = vlm_data.get("items", [])
                    all_vlm_data.extend(items)
                    batch_has_data = True
                elif isinstance(vlm_data, list):
                    # 레거시 형식
                    all_vlm_data.extend(vlm_data)
                    batch_has_data = True

        # Segments 병합
        segments_path = store.batch_segments_units_jsonl(batch_idx)
        if segments_path.exists():
            batch_segments = _read_jsonl(segments_path)
            all_segments.extend(batch_segments)
            batch_has_data = True

        # Summaries 병합
        summaries_path = store.batch_segment_summaries_jsonl(batch_idx)
        if summaries_path.exists():
            batch_summaries = _read_jsonl(summaries_path)
            all_summaries.extend(batch_summaries)
            batch_has_data = True

        # 실제 데이터가 있는 배치만 카운트
        if batch_has_data:
            merged_batch_count += 1

    if merged_batch_count == 0:
        return {"success": False, "error": "병합할 배치가 없습니다."}

    merged_files = []

    # VLM 병합 결과 저장 (schema_version/items 형식)
    if all_vlm_data:
        vlm_output = store.vlm_json()
        vlm_output.parent.mkdir(parents=True, exist_ok=True)
        with open(vlm_output, "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "items": all_vlm_data}, f, ensure_ascii=False, indent=2)
        merged_files.append(str(vlm_output))

    # Segments 병합 결과 저장
    if all_segments:
        # segment_id 재정렬 (연속성 보장)
        all_segments.sort(key=lambda x: x.get("segment_id", 0))
        segments_output = store.segments_units_jsonl()
        _write_jsonl(segments_output, all_segments)
        merged_files.append(str(segments_output))

    # Summaries 병합 결과 저장
    if all_summaries:
        # segment_id 재정렬 (연속성 보장)
        all_summaries.sort(key=lambda x: x.get("segment_id", 0))
        summaries_output = store.segment_summaries_jsonl()
        _write_jsonl(summaries_output, all_summaries)
        merged_files.append(str(summaries_output))

    return {
        "success": True,
        "merged_files": merged_files,
        "batch_count": merged_batch_count,
        "vlm_entries": len(all_vlm_data),
        "segments_count": len(all_segments),
        "summaries_count": len(all_summaries),
    }


def generate_final_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """LLM으로 전체 통합 요약을 생성합니다.

    병합된 segment_summaries.jsonl을 기반으로 최종 요약 마크다운을 생성합니다.

    Returns:
        success: 실행 성공 여부
        final_summary_dir: 생성된 파일들이 있는 디렉터리
        files: 생성된 파일 목록
    """
    video_name = tool_context.state.get("video_name")
    if not video_name:
        return {"success": False, "error": "video_name 미설정"}

    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 병합된 summaries가 있어야 함
    if not store.segment_summaries_jsonl().exists():
        return {"success": False, "error": "segment_summaries.jsonl이 없습니다. merge_all_batches를 먼저 실행하세요."}

    fusion_dir = store.fusion_dir()

    try:
        from src.fusion.config import load_config
        config = load_config(str(store.fusion_config_yaml()))

        from .internal.final_summary import write_final_summaries as _write_final_summaries
        written = _write_final_summaries(
            summaries_jsonl=store.segment_summaries_jsonl(),
            outputs_dir=store.final_outputs_dir(),
            generate_formats=[str(x) for x in config.raw.final_summary.generate_formats],
            max_chars_per_format=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=None,
        )

        return {
            "success": True,
            "final_summary_dir": str(store.final_outputs_dir()),
            "files": written,
        }
    except Exception as e:
        return {"success": False, "error": f"Final Summary 생성 실패: {e}"}


def merge_and_finalize(tool_context: ToolContext) -> Dict[str, Any]:
    """배치 병합과 최종 요약을 한번에 수행합니다.

    merge_all_batches + generate_final_summary를 연속 실행합니다.

    Returns:
        success: 실행 성공 여부
        merge_result: 병합 결과
        final_result: 최종 요약 결과
    """
    # 1. 배치 병합
    merge_result = merge_all_batches(tool_context)
    if not merge_result.get("success"):
        return {
            "success": False,
            "error": f"배치 병합 실패: {merge_result.get('error')}",
            "merge_result": merge_result,
        }

    # 2. 최종 요약 생성
    final_result = generate_final_summary(tool_context)
    if not final_result.get("success"):
        return {
            "success": False,
            "error": f"최종 요약 생성 실패: {final_result.get('error')}",
            "merge_result": merge_result,
            "final_result": final_result,
        }

    return {
        "success": True,
        "merge_result": merge_result,
        "final_result": final_result,
    }
