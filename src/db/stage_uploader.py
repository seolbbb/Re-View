"""
Stage Uploader 모듈 - 로컬 우선 + DB Fallback + 단계별 업로드 함수.

이 모듈은 파이프라인 각 단계 완료 시 결과를 즉시 DB에 업로드하고,
로컬 파일이 없을 경우 DB에서 fallback으로 데이터를 가져오는 기능을 제공합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .supabase_adapter import SupabaseAdapter


def _get_timestamp() -> str:
    """[YYYY-MM-DD | HH:MM:SS.mmm] 형식의 타임스탬프를 반환한다."""
    from datetime import datetime
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


# =============================================================================
# Fallback 함수들 - 로컬 파일 우선, 없으면 DB에서 조회
# =============================================================================

def get_vlm_results_with_fallback(
    video_root: Path,
    video_id: str,
    processing_job_id: str,
    adapter: Optional["SupabaseAdapter"] = None,
) -> List[Dict[str, Any]]:
    """VLM 결과를 로컬에서 먼저 찾고, 없으면 DB에서 조회한다.

    Args:
        video_root: 비디오 출력 루트 디렉토리
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        adapter: Supabase 어댑터 (없으면 내부에서 생성)

    Returns:
        VLM 결과 리스트
    """
    vlm_items = []

    # 1. 로컬 파일 확인 (단일 모드)
    vlm_path = video_root / "vlm.json"
    if vlm_path.exists():
        vlm_payload = json.loads(vlm_path.read_text(encoding="utf-8"))
        vlm_items.extend(vlm_payload.get("items", []))
        return vlm_items

    # 2. 로컬 파일 확인 (배치 모드)
    batches_dir = video_root / "batches"
    if batches_dir.exists():
        for batch_dir in sorted(batches_dir.iterdir()):
            batch_vlm = batch_dir / "vlm.json"
            if batch_vlm.exists():
                vlm_payload = json.loads(batch_vlm.read_text(encoding="utf-8"))
                vlm_items.extend(vlm_payload.get("items", []))
        if vlm_items:
            return vlm_items

    # 3. DB Fallback
    if adapter is None:
        from .supabase_adapter import get_supabase_adapter
        adapter = get_supabase_adapter()

    if adapter:
        query = adapter.client.table("vlm_results").select("*").eq("video_id", video_id)
        if processing_job_id:
            query = query.eq("processing_job_id", processing_job_id)
        result = query.execute()
        if result.data:
            # 새 스키마에서는 payload 대신 개별 컬럼 사용
            vlm_items = [
                {
                    "id": row.get("cap_id"),
                    "timestamp_ms": row.get("timestamp_ms"),
                    "extracted_text": row.get("extracted_text", ""),
                }
                for row in result.data
            ]

    return vlm_items


def get_segments_with_fallback(
    video_root: Path,
    video_id: str,
    processing_job_id: str,
    adapter: Optional["SupabaseAdapter"] = None,
) -> List[Dict[str, Any]]:
    """Segments를 로컬에서 먼저 찾고, 없으면 DB에서 조회한다.

    Args:
        video_root: 비디오 출력 루트 디렉토리
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        adapter: Supabase 어댑터

    Returns:
        Segment 리스트
    """
    segments = []

    # 1. 로컬 파일 확인 (fusion 디렉토리)
    segments_path = video_root / "fusion" / "segments_units.jsonl"
    if segments_path.exists():
        with segments_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append(json.loads(line))
        return segments

    # 2. DB Fallback
    if adapter is None:
        from .supabase_adapter import get_supabase_adapter
        adapter = get_supabase_adapter()

    if adapter:
        query = (
            adapter.client.table("segments")
            .select("*")
            .eq("video_id", video_id)
        )
        if processing_job_id:
            query = query.eq("processing_job_id", processing_job_id)
        query = query.order("segment_index", desc=False)
        result = query.execute()
        if result.data:
            segments = result.data

    return segments


def get_summaries_with_fallback(
    video_root: Path,
    video_id: str,
    processing_job_id: str,
    adapter: Optional["SupabaseAdapter"] = None,
) -> List[Dict[str, Any]]:
    """Summaries를 로컬에서 먼저 찾고, 없으면 DB에서 조회한다.

    Args:
        video_root: 비디오 출력 루트 디렉토리
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        adapter: Supabase 어댑터

    Returns:
        Summary 리스트
    """
    summaries = []

    # 1. 로컬 파일 확인
    summaries_path = video_root / "fusion" / "segment_summaries.jsonl"
    if summaries_path.exists():
        with summaries_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))
        return summaries

    # 2. DB Fallback
    if adapter is None:
        from .supabase_adapter import get_supabase_adapter
        adapter = get_supabase_adapter()

    if adapter:
        query = (
            adapter.client.table("summaries")
            .select("*, segments(*)")
            .eq("video_id", video_id)
        )
        if processing_job_id:
            query = query.eq("processing_job_id", processing_job_id)
        result = query.execute()
        if result.data:
            summaries = result.data

    return summaries


# =============================================================================
# 단계별 업로드 함수들 - 각 단계 완료 후 즉시 DB 업로드
# =============================================================================

def upload_vlm_results_for_batch(
    adapter: "SupabaseAdapter",
    video_id: str,
    processing_job_id: str,
    vlm_json_path: Path,
) -> int:
    """배치의 VLM 결과를 DB에 업로드한다.

    Args:
        adapter: Supabase 어댑터
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        vlm_json_path: vlm.json 파일 경로

    Returns:
        업로드된 레코드 수
    """
    if not vlm_json_path.exists():
        return 0

    vlm_payload = json.loads(vlm_json_path.read_text(encoding="utf-8"))
    vlm_items = vlm_payload.get("items", [])

    if not vlm_items:
        return 0

    try:
        vlm_rows = adapter.insert_vlm_results(
            video_id,
            vlm_items,
            processing_job_id=processing_job_id,
        )
    except Exception as e:
        import traceback
        with open("error_db.log", "w", encoding="utf-8") as f:
            f.write(f"[DB] ERROR in upload_vlm_results_for_batch: {e}\n")
            traceback.print_exc(file=f)
        raise e

    return len(vlm_rows)


def upload_segments_for_batch(
    adapter: "SupabaseAdapter",
    video_id: str,
    processing_job_id: str,
    segments_path: Path,
    offset: int = 0,
) -> Dict[int, str]:
    """배치의 Segments를 DB에 업로드하고 segment_index -> DB ID 매핑을 반환한다.

    Args:
        adapter: Supabase 어댑터
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        segments_path: segments_units.jsonl 파일 경로
        offset: segment_index 오프셋 (이전 배치들의 세그먼트 수)

    Returns:
        segment_index(int) -> segment_id(uuid) 매핑 딕셔너리
    """
    segment_map: Dict[int, str] = {}

    if not segments_path.exists():
        return segment_map

    segments = []
    with segments_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))

    if not segments:
        return segment_map

    # Apply offset to segment_index to avoid collisions across batches
    if offset:
        for seg in segments:
            idx = seg.get("segment_id")
            if idx is not None:
                try:
                    seg["segment_id"] = int(idx) + offset
                except (TypeError, ValueError):
                    pass

    try:
        saved_segments = adapter.save_segments(
            video_id,
            segments,
            processing_job_id=processing_job_id,
        )
    except Exception as e:
        import traceback
        with open("error_db.log", "a", encoding="utf-8") as f:
            f.write(f"[DB] ERROR in upload_segments_for_batch: {e}\n")
            traceback.print_exc(file=f)
        raise e

    for seg in saved_segments:
        idx = seg.get("segment_index")
        uuid = seg.get("id")
        if idx is not None and uuid:
            segment_map[idx] = uuid

    return segment_map


def upload_summaries_for_batch(
    adapter: "SupabaseAdapter",
    video_id: str,
    processing_job_id: str,
    summaries_path: Path,
    segment_map: Dict[int, str],
    batch_index: Optional[int] = None,
) -> int:
    """배치의 Summaries를 DB에 업로드한다.

    Args:
        adapter: Supabase 어댑터
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        summaries_path: segment_summaries.jsonl 파일 경로
        segment_map: segment_index -> segment_id 매핑
        batch_index: 배치 인덱스 (0-indexed)

    Returns:
        업로드된 레코드 수
    """
    if not summaries_path.exists():
        return 0

    summaries = []
    with summaries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                summaries.append(json.loads(line))

    if not summaries:
        return 0

    try:
        saved_summaries = adapter.save_summaries(
            video_id,
            summaries,
            segment_map=segment_map,
            processing_job_id=processing_job_id,
            batch_index=batch_index,
        )
    except Exception as e:
        import traceback
        with open("error_db.log", "a", encoding="utf-8") as f:
            f.write(f"[DB] ERROR in upload_summaries_for_batch: {e}\n")
            traceback.print_exc(file=f)
        raise e

    return len(saved_summaries)


def upload_judge_result(
    adapter: "SupabaseAdapter",
    video_id: str,
    processing_job_id: str,
    judge_result: Union[Dict[str, Any], Path, str],
    batch_idx: int,
) -> Optional[Dict[str, Any]]:
    """Judge 결과를 DB에 업로드한다.

    Args:
        adapter: Supabase 어댑터
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        judge_result: judge 실행 결과 (Dict) 또는 파일 경로 (Path/str)
        batch_idx: 배치 인덱스 (1-indexed)

    Returns:
        저장된 judge 레코드 (또는 None)
    """
    # Path나 str이면 파일에서 로드
    if isinstance(judge_result, (Path, str)):
        path_obj = Path(judge_result)
        if not path_obj.exists():
            return None
        try:
            judge_result = json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[DB] Error loading judge JSON: {e}")
            return None

    if not isinstance(judge_result, dict):
        return None

    report = judge_result.get("report", {})
    final_score = float(report.get("scores_avg", {}).get("final", 0.0))

    try:
        saved_judge = adapter.insert_judge(
            video_id,
            score=final_score,
            report=report,
            processing_job_id=processing_job_id,
            status="DONE",
            batch_index=batch_idx,  # 별도 컬럼으로 저장
        )
        return saved_judge
    except Exception as e:
        print(f"[DB] Warning: Failed to upload judge result: {e}")
        return None


def upsert_final_summary_results(
    adapter: "SupabaseAdapter",
    video_id: str,
    processing_job_id: str,
    summaries_path: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """최종 summary_results를 UPSERT하고 videos 테이블을 업데이트한다.

    Args:
        adapter: Supabase 어댑터
        video_id: 비디오 ID
        processing_job_id: 처리 작업 ID
        summaries_path: segment_summaries.jsonl 파일 경로
        results_dir: 최종 결과물 디렉토리 (final_summary_*.md 파일 위치)

    Returns:
        UPSERT 결과 정보
    """
    result_info: Dict[str, Any] = {"saved": {}, "errors": []}

    # 지원하는 포맷들과 파일명 매핑
    format_file_map = {
        "timeline": "final_summary_timeline.md",
        "tldr": "final_summary_tldr.md",
        "tldr_timeline": "final_summary_tldr_timeline.md",
    }

    # 디버그: results_dir 확인
    if not results_dir.exists():
        result_info["errors"].append(f"results_dir not found: {results_dir}")
        print(f"{_get_timestamp()}   [DB] Warning: results_dir not found: {results_dir}")

    # 각 포맷별로 파일 확인 및 업로드
    found_any = False
    for fmt, filename in format_file_map.items():
        file_path = results_dir / filename
        if file_path.exists():
            found_any = True
            try:
                content = file_path.read_text(encoding="utf-8")
                upsert_result = adapter.upsert_summary_results(
                    video_id,
                    format=fmt,
                    payload={"content": content},
                    processing_job_id=processing_job_id,
                    status="DONE",
                )
                result_info["saved"][fmt] = upsert_result.get("id")
                print(f"{_get_timestamp()}   [DB] summary_results ({fmt}) uploaded: {upsert_result.get('id')}")
            except Exception as e:
                result_info["errors"].append(f"{fmt}: {str(e)}")

    if not found_any:
        result_info["errors"].append(f"No summary files found in {results_dir}")
        print(f"{_get_timestamp()}   [DB] Warning: No summary files found in {results_dir}")

    # 3. processing_job 상태를 DONE으로 업데이트
    try:
        adapter.update_processing_job_status(processing_job_id, "DONE")
    except Exception as e:
        result_info["errors"].append(f"processing_job_status: {str(e)}")

    # 4. videos.status를 DONE으로 업데이트
    try:
        adapter.client.table("videos").update({
            "status": "DONE"
        }).eq("id", video_id).execute()
    except Exception as e:
        result_info["errors"].append(f"videos_status: {str(e)}")

    return result_info


def accumulate_segments_to_fusion(
    batch_segments_path: Path,
    accumulated_segments_path: Path,
) -> None:
    """배치의 segments_units.jsonl을 fusion 디렉토리에 누적한다.

    Args:
        batch_segments_path: 배치의 segments_units.jsonl 경로
        accumulated_segments_path: fusion/segments_units.jsonl 경로
    """
    if not batch_segments_path.exists():
        return

    batch_content = batch_segments_path.read_text(encoding="utf-8")
    if not batch_content.strip():
        return

    # append 모드로 누적
    with accumulated_segments_path.open("a", encoding="utf-8") as f:
        # 이전 내용이 있고 개행으로 끝나지 않으면 개행 추가
        if accumulated_segments_path.exists() and accumulated_segments_path.stat().st_size > 0:
            # 마지막 문자 확인
            with accumulated_segments_path.open("r", encoding="utf-8") as check:
                check.seek(0, 2)  # 파일 끝으로 이동
                if check.tell() > 0:
                    check.seek(check.tell() - 1)
                    last_char = check.read(1)
                    if last_char != "\n":
                        f.write("\n")
        f.write(batch_content)
        # 마지막에 개행이 없으면 추가
        if not batch_content.endswith("\n"):
            f.write("\n")
