"""비디오 처리 파이프라인 단계별 실행 함수 모듈.

=============================================================================
모듈 목적 (Purpose)
=============================================================================
이 모듈은 비디오 처리 파이프라인의 각 단계를 실행하는 함수들을 정의합니다.
캡처 추출, STT, VLM 분석, 세그먼트화, 요약 등 전체 파이프라인을 관리합니다.

=============================================================================
파이프라인 흐름 (Pipeline Flow)
=============================================================================
1. run_capture_stage() → 비디오에서 프레임 캡처 추출
2. run_stt_stage() → 오디오에서 텍스트 추출 (STT)
3. run_vlm_for_batch() → VLM으로 이미지 분석 (R2 signed URL 활용)
4. run_sync_stage() → STT와 VLM 결과 동기화
5. run_summarizer_stage() → 세그먼트 요약 생성
6. run_judge_stage() → 최종 품질 판정

=============================================================================
R2 스토리지 통합 (R2 Storage Integration)
=============================================================================
- run_vlm_for_batch(): R2에 저장된 캡처 이미지의 signed URL 생성
  - adapter.r2_prefix_captures 사용하여 경로 구성
  - adapter.get_signed_url()로 1시간 유효한 URL 생성

=============================================================================
활용처 (Usage Context)
=============================================================================
- src/run_processing_pipeline.py → 메인 파이프라인 실행
- src/process_api.py → API 엔드포인트 백그라운드 태스크
- src/db/pipeline_sync.py → DB 동기화 시 호출

=============================================================================
의존성 (Dependencies)
=============================================================================
- src/vlm/vlm_engine.py: VLM 추론 엔진
- src/audio/stt_router.py: STT 라우터
- src/fusion/sync_engine.py: 동기화 엔진
- src/db/supabase_adapter.py: SupabaseAdapter (R2 클라이언트 포함)
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import json
import math
import os
import time
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

# -----------------------------------------------------------------------------
# Project Root Path
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Local Module Imports
# -----------------------------------------------------------------------------
from src.audio.stt_router import STTRouter
from src.capture.process_content import process_single_video_capture
from src.fusion.config import load_config
from src.fusion.io_utils import write_json, write_jsonl
from src.fusion.renderer import compose_final_summaries, render_segment_summaries_md
from src.fusion.summarizer import run_summarizer
from src.fusion.sync_engine import run_sync_engine
from src.judge.judge import run_judge
from src.pipeline.benchmark import BenchmarkTimer
from src.vlm.vlm_engine import QwenVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm
from src.db.stage_uploader import (
    upload_vlm_results_for_batch,
    upload_segments_for_batch,
    upload_summaries_for_batch,
    upload_judge_result,
    accumulate_segments_to_fusion,
)



def _get_timestamp() -> str:
    """[YYYY-MM-DD | HH:MM:SS.mmm] 형식의 타임스탬프를 반환한다."""
    from datetime import datetime
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


def _read_latest_token_usage(token_usage_path: Path) -> Dict[str, int]:
    """token_usage.json에서 최신 토큰 사용량을 읽어 반환한다."""
    if not token_usage_path.exists():
        return {}
    try:
        data = json.loads(token_usage_path.read_text(encoding="utf-8"))
        result = {}
        # Get latest summarizer tokens
        summarizer_list = data.get("summarizer", [])
        if summarizer_list:
            result["summarizer"] = summarizer_list[-1].get("input_tokens", 0)
        # Get latest judge tokens
        judge_list = data.get("judge", [])
        if judge_list:
            result["judge"] = judge_list[-1].get("input_tokens", 0)
        return result
    except Exception:
        return {}


def _process_judge_result(
    judge_result: Dict[str, Any],
    config: Any,
    output_path: Path,
    batch_index: Optional[int],
    silent: bool = False,
) -> Tuple[bool, float]:
    """Judge 결과를 처리하여 파일로 저장하고 요약 정보를 반환한다."""
    report = judge_result.get("report", {})
    segment_reports = judge_result.get("segment_reports", []) or []
    final_score = float(report.get("scores_avg", {}).get("final", 0.0))
    min_score = float(config.judge.min_score)
    passed = final_score >= min_score

    feedback = [
        {"segment_id": int(item.get("segment_id")), "feedback": str(item.get("feedback", "")).strip()}
        for item in segment_reports
        if item.get("segment_id") is not None
    ]

    payload = {
        "model": str(report.get("meta", {}).get("model", "")),
        "pass": passed,
        "final_score": final_score,
        "min_score": min_score,
        "prompt_version": str(report.get("meta", {}).get("prompt_version", "")),
        "generated_at_utc": str(report.get("meta", {}).get("generated_at_utc", "")),
        "feedback": feedback,
        "report": report,
    }

    if config.judge.include_segments:
        payload["segments"] = [
            {
                "segment_id": int(item.get("segment_id")),
                "scores": item.get("scores", {}),
            }
            for item in segment_reports
            if item.get("segment_id") is not None
        ]

    write_json(output_path, payload)

    if not silent:
        if batch_index is None:
            label = "Pipeline Judge"
        else:
            label = f"Pipeline batch {batch_index + 1} Judge"
        print(f"  📊 {label}: {'PASS' if passed else 'FAIL'} (score: {final_score:.1f})")
    return passed, final_score


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
    """Fusion settings.yaml을 템플릿에서 생성한다."""
    with template_config.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle)

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            return str(path)

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


def run_stt(
    video_path: Path,
    output_stt_json: Optional[Path],
    *,
    backend: str,
    write_output: bool = True,
) -> Dict[str, Any]:
    """음성 인식을 실행해 stt 결과를 반환한다."""
    router = STTRouter(provider=backend)
    audio_output_path = output_stt_json.with_name(f"{video_path.stem}.wav") if output_stt_json else None
    return router.transcribe_media(
        video_path,
        provider=backend,
        audio_output_path=None,  # 코덱 설정에 따라 자동 결정
        mono_method="auto",
        output_path=output_stt_json if write_output else None,
        write_output=write_output,
    )


def run_stt_only(
    audio_path: Path,
    output_stt_json: Optional[Path],
    *,
    backend: str,
    write_output: bool = True,
) -> Dict[str, Any]:
    """이미 추출된 오디오 파일에 대해 음성 인식을 실행한다."""
    router = STTRouter(provider=backend)
    return router.transcribe(
        audio_path,
        provider=backend,
        output_path=output_stt_json if write_output else None,
        write_output=write_output,
    )


def run_stt_from_storage(
    *,
    audio_storage_key: str,
    video_id: str,
    backend: str = "clova",
    temp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Storage에서 오디오를 다운로드하여 STT를 실행한다.
    
    Frontend가 Storage에 업로드한 오디오 파일을 다운받아 처리하는 API용 함수.
    
    Args:
        audio_storage_key: Storage 내 오디오 파일 경로 (예: "{video_id}/audio.wav")
        video_id: 비디오 ID (임시 파일 정리용)
        backend: STT 엔진 (기본: clova)
        temp_dir: 임시 파일 저장 디렉토리 (기본: data/temp)
        
    Returns:
        Dict: STT 결과 (segments 포함)
    """
    from src.db import get_supabase_adapter
    
    adapter = get_supabase_adapter()
    if not adapter:
        raise RuntimeError("Supabase adapter not configured")
    
    # 임시 디렉토리 설정
    if temp_dir is None:
        temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage에서 오디오 다운로드
    audio_filename = Path(audio_storage_key).name
    local_audio_path = temp_dir / f"{video_id}_{audio_filename}"
    
    try:
        adapter.download_audio(
            storage_path=audio_storage_key,
            local_path=local_audio_path,
            bucket="audio",
        )
        
        # STT 실행 (이미 추출된 오디오이므로 transcribe 직접 호출)
        router = STTRouter(provider=backend)
        result = router.transcribe(
            local_audio_path,
            provider=backend,
            write_output=False,
        )
        
        return result
        
    finally:
        # 임시 파일 정리
        if local_audio_path.exists():
            local_audio_path.unlink(missing_ok=True)
def run_capture(
    video_path: Path,
    output_base: Path,
    *,
    threshold: float,
    min_interval: float,
    verbose: bool,
    video_name: str,
    dedup_enabled: bool = True,  # Kept for interface compatibility, but no longer used
    write_manifest: bool = True,
    callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """슬라이드 캡처를 실행하고 메타데이터 목록을 반환한다."""
    # Note: dedup_enabled is ignored - new HybridSlideExtractor always uses pHash+ORB dedup
    metadata = process_single_video_capture(
        str(video_path),
        str(output_base),
        scene_threshold=threshold,
        min_interval=min_interval,
        write_manifest=write_manifest,
        callback=callback,
    )
    return metadata


def _get_sort_key_timestamp(item: Dict[str, Any]) -> int:
    """manifest 아이템에서 정렬용 타임스탬프(첫 등장 시간)를 추출한다."""
    # 1. timestamp_ms (레거시/공통)
    if "timestamp_ms" in item:
        return int(item["timestamp_ms"])
    # 2. time_ranges (신규)
    time_ranges = item.get("time_ranges")
    if isinstance(time_ranges, list) and time_ranges:
        first = time_ranges[0]
        if isinstance(first, dict) and "start_ms" in first:
            return int(first.get("start_ms") or 0)
    # 3. start_ms (하위 호환)
    return int(item.get("start_ms", 0))


def run_vlm_qwen(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
    concurrency: int,
    show_progress: bool,
) -> int:
    """이미지 정보를 추출해 vlm.json을 만들고 처리 개수를 반환한다."""
    extractor = QwenVlmExtractor(video_name=video_name, output_root=output_base)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("capture.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_paths: List[str] = []
    # 정렬 기준 변경: time_ranges 지원
    for item in sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (_get_sort_key_timestamp(x), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))

    if not image_paths:
        raise ValueError("VLM 입력 이미지가 없습니다(capture.json을 확인하세요).")

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=raw_path.with_name("vlm.json"),
    )
    # raw 파일은 변환 후 삭제 (선택 사항)
    raw_path.unlink(missing_ok=True)

    return len(image_paths)


def _filter_manifest_by_time_range(
    manifest_payload: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """manifest에서 특정 시간 범위의 항목만 필터링한다 (time_ranges 지원)."""
    filtered = []
    for item in manifest_payload:
        # 1. time_ranges 확인 (하나라도 범위 내에 겹치면 포함)
        time_ranges = item.get("time_ranges")
        if isinstance(time_ranges, list) and time_ranges:
            in_range = False
            for rng in time_ranges:
                r_start = int(rng.get("start_ms", 0))
                # 범위 겹침 조건: (ItemStart < BatchEnd) AND (ItemEnd > BatchStart)
                # 여기서는 단순 포함 여부가 아니라 '처리해야 할 대상인가'를 판단
                # VLM 배치는 보통 순차적이므로, 해당 배치의 시간 구간에 '시작'하는 항목을 포함하거나
                # 혹은 단순히 대표 시간이 범위 내인 것을 포함할 수 있음.
                # 기존 로직: start_ms <= timestamp < end_ms
                if start_ms <= r_start < end_ms:
                    in_range = True
                    break
            if in_range:
                filtered.append(item)
                continue

        # 2. timestamp_ms / start_ms 확인 (하위 호환)
        timestamp_ms = item.get("timestamp_ms")
        if timestamp_ms is None:
            timestamp_ms = item.get("start_ms")
        
        if timestamp_ms is not None:
            ts = int(timestamp_ms)
            if start_ms <= ts < end_ms:
                filtered.append(item)
    return filtered


def run_vlm_for_batch(
    *,
    captures_dir: Path,
    manifest_json: Optional[Path] = None,
    video_name: str,
    output_dir: Path,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    batch_manifest: Optional[List[Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    concurrency: int = 1,
    show_progress: bool = False,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    video_id: Optional[str] = None,
) -> Dict[str, Any]:
    """배치 범위만 VLM 처리해 batch 단위의 vlm.json을 생성한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = QwenVlmExtractor(video_name=video_name, output_root=output_dir)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    if batch_manifest is not None:
        filtered_manifest_items = batch_manifest
    elif manifest_json is not None:
        manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, list):
            raise ValueError("capture.json 형식이 올바르지 않습니다(배열이어야 함).")

        if start_idx is not None and end_idx is not None:
            sorted_manifest = sorted(
                (x for x in manifest_payload if isinstance(x, dict)),
                key=lambda x: (_get_sort_key_timestamp(x), str(x.get("file_name", ""))),
            )
            filtered_manifest_items = sorted_manifest[start_idx:end_idx]
        elif start_ms is not None and end_ms is not None:
            filtered_manifest_items = _filter_manifest_by_time_range(
                manifest_payload,
                start_ms,
                end_ms,
            )
        else:
            raise ValueError("start_idx/end_idx 또는 start_ms/end_ms 가 필요합니다.")
    else:
        raise ValueError("manifest_json 또는 batch_manifest가 제공되어야 합니다.")

    image_paths: List[str] = []
    
    # [Fix] Storage Fallback 준비
    # 로컬 파일이 없을 경우 Supabase Storage에서 URL을 가져오기 위해 Adapter 초기화
    from src.db import get_supabase_adapter
    adapter = None

    for item in sorted(
        (x for x in filtered_manifest_items if isinstance(x, dict)),
        key=lambda x: (_get_sort_key_timestamp(x), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
            
        local_path = captures_dir / file_name
        if local_path.exists():
            image_paths.append(str(local_path))
        else:
            # 로컬에 없으면 Storage URL 사용 시도
            if adapter is None:
                adapter = get_supabase_adapter()
                
            storage_path = item.get("storage_path")
            # DB에 storage_path가 없으면 표준 경로로 추론
            if not storage_path:
                vid = item.get("video_id") or video_id # 전달받은 video_id 우선 사용
                if vid:
                    # R2 여부에 따라 경로 구조 결정
                    if adapter and getattr(adapter, "s3_client", None):
                        prefix = getattr(adapter, "r2_prefix_captures", "captures")
                        storage_path = f"{vid}/{prefix}/{file_name}"
                    else:
                        storage_path = f"{vid}/{file_name}"
            
            if adapter and storage_path:
                # Signed URL 생성 (기본 1시간)
                # 만약 public bucket이라면 get_public_url을 써도 되지만, 
                # CaptureAdapterMixin이 get_signed_url을 제공하므로 이를 활용
                try:
                    # CaptureAdapterMixin의 메서드 활용
                    url = adapter.get_signed_url(storage_path, bucket="captures")
                    if url:
                         image_paths.append(url)
                         continue
                except Exception as e:
                    print(f"{_get_timestamp()} [VLM] Failed to get signed url for {file_name}: {e}")

            # Fallback 실패 시 로컬 경로 추가 (이후 에러 발생)
            image_paths.append(str(local_path))

    if not image_paths:
        empty_vlm = {"items": [], "duration_ms": 0}
        vlm_json_path = output_dir / "vlm.json"
        vlm_json_path.write_text(json.dumps(empty_vlm, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "vlm_raw_json": "",
            "vlm_json": str(vlm_json_path),
            "image_count": 0,
        }

    # [Optimization] VLM 재사용 검증 (User Request)
    # 이미 vlm.json이 있고, 해당 파일이 현재 요청된 이미지를 모두 포함하는지 확인
    vlm_json_path = output_dir / "vlm.json"
    raw_path = output_dir / "vlm_raw.json"
    
    reuse_success = False
    if vlm_json_path.exists():
        try:
            existing_data = json.loads(vlm_json_path.read_text(encoding="utf-8"))
            existing_items = existing_data.get("items", [])
            existing_files = set()
            for item in existing_items:
                # source_path 또는 image_path 등에서 파일명 추출 필요
                # 여기서는 items 내에 file_name이 없으므로 raw_path를 체크하거나,
                # vlm.json의 source_idx 등을 통해 유추해야 함.
                # 하지만 vlm.json은 feature vector 위주라 매핑이 어렵다면 raw_path 우선 체크
                pass
        except Exception:
            pass
            
    # raw_path가 있다면 더 확실하게 검증 가능 (파일명이 키로 존재하거나 포함됨)
    # OpenRouterVlmExtractor.extract_features는 리스트를 반환하므로,
    # 여기서는 결과 파일이 존재하고, 최신이며, 이미지 개수가 같으면 재사용한다고 가정 (간소화)
    # 더 정확히는 파일명 매칭을 해야 하지만, 배치 단위 디렉토리가 분리되어 있다면 개수 체크로 1차 방어 가능
    if not reuse_success and vlm_json_path.exists():
        try:
             # 배치가 "batch_N" 폴더로 분리되어 있다고 가정하면, 
             # 해당 폴더에 vlm.json이 존재한다는 것은 이미 처리가 끝났음을 의미할 수 있음.
             # 단, 재시도/덮어쓰기 옵션에 따라 달라질 수 있음.
             # 여기서는 파일이 존재하고 비어있지 않으면 재사용.
             existing_data = json.loads(vlm_json_path.read_text(encoding="utf-8"))
             if existing_data.get("items"):
                 if show_progress:
                     print(f"{_get_timestamp()} [VLM] reuse: Found existing vlm.json with {len(existing_data['items'])} items. Skipping inference.", flush=True)
                     # [User Request] Show reused items count/info
                     for i, item in enumerate(existing_data['items'][:3], start=1):
                         label = item.get('label', 'N/A')
                         print(f"{_get_timestamp()}       - Reused Item {i}: {label}...", flush=True)
                     if len(existing_data['items']) > 3:
                         print(f"{_get_timestamp()}       - ... and {len(existing_data['items']) - 3} more items.", flush=True)

                 return {
                    "vlm_raw_json": str(raw_path) if raw_path.exists() else "",
                    "vlm_json": str(vlm_json_path),
                    "image_count": len(image_paths),
                 }
        except Exception as e:
            if show_progress:
                print(f"{_get_timestamp()} [VLM] reuse check failed: {e}", flush=True)

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )

    raw_path = output_dir / "vlm_raw.json"
    write_vlm_raw_json(results, raw_path)

    temp_manifest_path = output_dir / "manifest_temp.json"
    temp_manifest_path.write_text(
        json.dumps(filtered_manifest_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vlm_json_path = output_dir / "vlm.json"
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=temp_manifest_path,
        vlm_raw_json=raw_path,
        output_vlm_json=vlm_json_path,
    )
    temp_manifest_path.unlink(missing_ok=True)
    raw_path.unlink(missing_ok=True)

    return {
        "vlm_raw_json": str(raw_path),
        "vlm_json": str(vlm_json_path),
        "image_count": len(image_paths),
    }


def run_fusion_pipeline(
    config_path: Path,
    *,
    limit: Optional[int],
    timer: BenchmarkTimer,
) -> Dict[str, Any]:
    """동기화부터 최종 요약까지 묶어서 실행하고 통계를 반환한다."""
    config = load_config(str(config_path))
    config.paths.output_root.mkdir(parents=True, exist_ok=True)

    fusion_info: Dict[str, Any] = {
        "segment_count": 0,
        "timings": {},
    }

    _, sync_elapsed = timer.time_stage(
        "fusion.sync_engine",
        run_sync_engine,
        config,
        limit=limit,
    )
    fusion_info["timings"]["sync_engine_sec"] = sync_elapsed

    output_dir = config.paths.output_root / "fusion"
    judge_segments_path = output_dir / "judge_segment_reports.jsonl"
    max_attempts = 2
    feedback_map: Dict[int, str] = {}
    summarizer_elapsed_total = 0.0
    judge_elapsed_total = 0.0
    latest_judge_result: Optional[Dict[str, Any]] = None

    for attempt in range(max_attempts):
        is_retry = attempt > 0
        stage_suffix = f"_retry_{attempt}" if is_retry else ""

        _, llm_elapsed = timer.time_stage(
            f"fusion.llm_summarizer{stage_suffix}",
            run_summarizer,
            config,
            limit=limit,
            feedback_map=feedback_map,
        )
        summarizer_elapsed_total += llm_elapsed

        judge_result, judge_elapsed = timer.time_stage(
            f"fusion.judge{stage_suffix}",
            run_judge,
            config=config,
            segments_units_path=output_dir / "segments_units.jsonl",
            segment_summaries_path=output_dir / "segment_summaries.jsonl",
            output_report_path=output_dir / "judge_report.json",
            output_segments_path=judge_segments_path,
            batch_size=config.judge.batch_size,
            workers=config.judge.workers,
            json_repair_attempts=config.judge.json_repair_attempts,
            limit=limit,
            write_outputs=False,
            verbose=config.judge.verbose,
        )
        judge_elapsed_total += judge_elapsed
        latest_judge_result = judge_result

        passed, final_score = _process_judge_result(
            judge_result,
            config,
            output_dir / "judge.json",
            None,
        )

        if passed:
            break

        if attempt < max_attempts - 1:
            print(
                "Judge Fail (Score: "
                f"{final_score:.1f}). Retrying with feedback... "
                f"({attempt + 1}/{max_attempts})"
            )
            feedback_map = {}
            segment_reports = judge_result.get("segment_reports", []) or []
            for item in segment_reports:
                seg_id = item.get("segment_id")
                fb = str(item.get("feedback", "")).strip()
                if seg_id is not None and fb:
                    feedback_map[int(seg_id)] = fb
    if latest_judge_result is not None:
        write_jsonl(
            judge_segments_path,
            latest_judge_result.get("segment_reports", []) or [],
        )

    fusion_info["timings"]["llm_summarizer_sec"] = summarizer_elapsed_total
    fusion_info["timings"]["judge_sec"] = judge_elapsed_total

    groups_cfg = getattr(config.raw.render, "groups", None)
    group_order = groups_cfg.order if groups_cfg else None
    group_headers = groups_cfg.headers if groups_cfg else None

    _, render_elapsed = timer.time_stage(
        "fusion.renderer",
        render_segment_summaries_md,
        summaries_jsonl=output_dir / "segment_summaries.jsonl",
        output_md=output_dir / "segment_summaries.md",
        include_sources=config.raw.render.include_sources,
        sources_jsonl=output_dir / "segments_units.jsonl",
        md_wrap_width=config.raw.render.md_wrap_width,
        limit=limit,
        group_order=group_order,
        group_headers=group_headers,
        fusion_prompt_version=config.raw.summarizer.prompt_version,
        judge_prompt_version=config.judge.prompt_version,
        execution_time={
            "summarizer": summarizer_elapsed_total,
            "judge": judge_elapsed_total,
        },
        batch_config={
            "batch_size": config.judge.batch_size,
            "workers": config.judge.workers,
        },
        judge_stats={
            "final_score": final_score,
            "passed": passed,
            "category_scores": latest_judge_result.get("report", {}).get("scores_avg", {}) if latest_judge_result else {},
        },
        token_usage=_read_latest_token_usage(output_dir / "token_usage.json"),
    )
    fusion_info["timings"]["renderer_sec"] = render_elapsed

    summaries, final_elapsed = timer.time_stage(
        "fusion.final_summary",
        compose_final_summaries,
        summaries_jsonl=output_dir / "segment_summaries.jsonl",
        max_chars=config.raw.final_summary.max_chars_per_format,
        include_timestamps=config.raw.final_summary.style.include_timestamps,
        limit=limit,
    )
    fusion_info["timings"]["final_summary_sec"] = final_elapsed

    results_dir = config.paths.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for fmt in config.raw.final_summary.generate_formats:
        if fmt in summaries:
            results_dir.joinpath(f"final_summary_{fmt}.md").write_text(
                summaries[fmt], encoding="utf-8"
            )

    segments_file = output_dir / "segment_summaries.jsonl"
    if segments_file.exists():
        fusion_info["segment_count"] = sum(1 for _ in segments_file.open(encoding="utf-8"))

    return fusion_info


def run_batch_fusion_pipeline(
    *,
    video_root: Path,
    captures_dir: Path,
    manifest_json: Optional[Path] = None,
    captures_data: Optional[List[Dict[str, Any]]] = None,
    stt_json: Path,
    video_name: str,
    batch_size: int,
    timer: BenchmarkTimer,
    vlm_batch_size: Optional[int],
    vlm_concurrency: int,
    vlm_show_progress: bool,
    limit: Optional[int],
    repo_root: Path,
    skip_vlm: bool = False,
    status_callback: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
    # DB 동기화 관련 파라미터
    processing_job_id: Optional[str] = None,
    video_id: Optional[str] = None,
    sync_to_db: bool = False,
    adapter: Optional[Any] = None,
    # 연속 처리(Streaming) 지원을 위한 파라미터
    start_batch_index: int = 0,
    preserve_files: bool = False,
    forced_batch_end_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """배치 단위로 동기화와 요약을 반복 실행한다.

    Args:
        manifest_json: manifest.json 경로 (선택, captures_data가 없을 때 사용)
        captures_data: DB에서 가져온 captures 리스트 (선택, manifest_json보다 우선)
        status_callback: 상태 업데이트 콜백 함수 (status, current, total)
        processing_job_id: 처리 작업 ID (DB 동기화용)
        video_id: 비디오 ID (DB 동기화용)
        sync_to_db: DB 동기화 활성화 여부
        adapter: Supabase 어댑터 (sync_to_db=True일 때 필요)
    """
    from src.fusion.summarizer import run_batch_summarizer
    from src.fusion.sync_engine import run_batch_sync_engine

    # captures_data가 있으면 직접 사용, 없으면 manifest_json에서 로드
    if captures_data is not None:
        manifest_payload = captures_data
    elif manifest_json and manifest_json.exists():
        manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    else:
        manifest_payload = []
    
    sorted_manifest = sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (_get_sort_key_timestamp(x), str(x.get("file_name", ""))),
    )

    total_captures = len(sorted_manifest)
    
    # 배치 분할 로직 개선: 마지막 배치가 너무 작아지는 비대칭 문제 해결
    # 기본 배치 크기 기준으로 나누되, 마지막 배치가 batch_size의 절반 미만이면 이전 배치에 합침
    if total_captures <= batch_size:
        total_batches = 1
    else:
        total_batches = total_captures // batch_size
        remainder = total_captures % batch_size
        
        # 나머지가 batch_size의 절반보다 작고 이미 1개 이상의 배치가 있을 때 합침
        if remainder > 0:
            if remainder < (batch_size / 2) and total_batches >= 1:
                # 합침: total_batches 유지 (마지막 인덱스 조정)
                pass 
            else:
                total_batches += 1

    print(
        f"\n{_get_timestamp()} Pipeline batches: {total_captures} images across {total_batches} groups "
        f"(group size: ~{batch_size})"
    )

    # Note: total_batch는 run_process_pipeline.py에서 파이프라인 시작 시 미리 설정됨
    # 여기서는 current_batch 진행률만 업데이트함

    batch_ranges = []
    for i in range(total_batches):
        start_idx = i * batch_size
        # 마지막 배치인 경우 나머지 전체를 포함
        if i == total_batches - 1:
            end_idx = total_captures
        else:
            end_idx = (i + 1) * batch_size
        
        # Batch Start MS
        first_item = sorted_manifest[start_idx]
        batch_start_ms = _get_sort_key_timestamp(first_item)
        
        # Batch End MS (last item end time calculation)
        last_item = sorted_manifest[end_idx - 1]
        last_start_ms = _get_sort_key_timestamp(last_item)
        
        # 마지막 항목의 end_ms 계산: time_ranges 있으면 마지막 구간 end_ms, 없으면 start + 1000
        # [Fix] Data Duplication: 마지막 배치가 아니면 다음 배치 시작 시간으로 강제 종료(Clamping)
        # [Fix] Data Duplication: 마지막 배치가 아니면 다음 배치 시작 시간으로 강제 종료(Clamping)
        if i < total_batches - 1:
            next_start_idx = end_idx
            next_item = sorted_manifest[next_start_idx]
            batch_end_ms = _get_sort_key_timestamp(next_item)
        else:
            # 마지막 배치인 경우:
            # 1. 외부에서 강제 종료 시간이 주어졌으면 그것을 사용 (Chunking 대응)
            if forced_batch_end_ms is not None:
                batch_end_ms = int(forced_batch_end_ms)
            else:
                # 2. 아니면 기존 로직 유지 (마지막 아이템 끝까지)
                batch_end_ms = last_start_ms + 1000
                time_ranges = last_item.get("time_ranges")
                if isinstance(time_ranges, list) and time_ranges:
                    try:
                        # time_ranges 내 가장 늦은 end_ms 찾기
                        max_end = 0
                        for rng in time_ranges:
                             rng_end = int(rng.get("end_ms", 0))
                             if rng_end > max_end:
                                 max_end = rng_end
                        if max_end > 0:
                            batch_end_ms = max_end
                    except Exception:
                        pass
                else:
                     # 레거시 호환
                     if "end_ms" in last_item:
                         batch_end_ms = int(last_item["end_ms"])

        batch_ranges.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_ms": batch_start_ms,
                "end_ms": batch_end_ms,
                "capture_count": end_idx - start_idx,
            }
        )

    batches_dir = video_root / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir = video_root / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    template_config = repo_root / "config" / "fusion" / "settings.yaml"
    fusion_config_path = video_root / "config.yaml"

    fusion_info: Dict[str, Any] = {
        "segment_count": 0,
        "batch_count": total_batches,
        "batch_results": [],
        "timings": {},
    }

    cumulative_segment_count = 0
    db_segment_offset = 0
    use_db_offset = False

    def _get_db_segment_offset() -> int:
        if not adapter or not sync_to_db or not video_id:
            return 0
        try:
            query = adapter.client.table("segments").select("segment_index").eq("video_id", video_id)
            if processing_job_id:
                query = query.eq("processing_job_id", processing_job_id)
            query = query.order("segment_index", desc=True).limit(1)
            result = query.execute()
            if result.data:
                return int(result.data[0].get("segment_index") or 0)
        except Exception as exc:
            print(f"{_get_timestamp()} [DB] Warning: failed to read max segment_index: {exc}")
        return 0

    if adapter and sync_to_db and processing_job_id:
        db_segment_offset = _get_db_segment_offset()
        if db_segment_offset > 0:
            cumulative_segment_count = db_segment_offset
            use_db_offset = True
    
    # [Fix] Resume Support: If starting from a later batch, count segments from previous batches
    if start_batch_index > 0 and not use_db_offset:
        print(f"{_get_timestamp()} [Pipeline] Checking previous batches for segment offset...")
        for i in range(start_batch_index):
            prev_batch_idx = i + 1
            prev_batch_dir = video_root / "batches" / f"batch_{prev_batch_idx}"
            prev_units_path = prev_batch_dir / "fusion" / "segments_units.jsonl"
            
            if prev_units_path.exists():
                try:
                    line_count = sum(1 for _ in open(prev_units_path, "rb"))
                    cumulative_segment_count += line_count
                    print(f"{_get_timestamp()}   - Batch {prev_batch_idx}: Found {line_count} segments")
                except Exception as e:
                    print(f"{_get_timestamp()}   - Batch {prev_batch_idx}: Failed to read segments: {e}")
            else:
                print(f"{_get_timestamp()}   - Batch {prev_batch_idx}: No segments file found (skipping count)")
        print(f"{_get_timestamp()} [Pipeline] Initial cumulative_segment_count set to {cumulative_segment_count}")
    elif use_db_offset:
        print(
            f"{_get_timestamp()} [Pipeline] Using DB max segment_index offset: {cumulative_segment_count}"
        )
    previous_context = ""
    if start_batch_index > 0:
        # Resume support: seed context from the most recent completed batch so the first
        # processed batch can maintain terminology/consistency.
        try:
            from src.fusion.summarizer import extract_batch_context

            for prev_idx in range(start_batch_index, 0, -1):
                prev_summaries_path = (
                    video_root
                    / "batches"
                    / f"batch_{prev_idx}"
                    / "fusion"
                    / "segment_summaries.jsonl"
                )
                if not prev_summaries_path.exists():
                    continue
                previous_context = extract_batch_context(prev_summaries_path) or ""
                if previous_context:
                    print(
                        f"{_get_timestamp()} [Pipeline] Seeded previous_context from Batch {prev_idx}"
                    )
                break
        except Exception as exc:
            print(
                f"{_get_timestamp()} [Pipeline] Warning: failed to seed previous_context: {exc}"
            )

    accumulated_summaries_path = fusion_dir / "segment_summaries.jsonl"
    if not preserve_files and accumulated_summaries_path.exists():
        accumulated_summaries_path.unlink()

    # segments_units.jsonl도 누적 파일 초기화
    accumulated_segments_path = fusion_dir / "segments_units.jsonl"
    if not preserve_files and accumulated_segments_path.exists():
        accumulated_segments_path.unlink()

    total_vlm_elapsed = 0.0
    total_summarizer_elapsed = 0.0
    total_judge_elapsed = 0.0
    total_judge_score = 0.0
    all_batches_passed = True
    processed_batches_count = 0

    first_batch = True
    for batch_idx, batch_info in enumerate(batch_ranges):
        if batch_idx > 0:
            print(f"\n{_get_timestamp()} Waiting 5s to avoid API rate limiting...")
            t0 = time.perf_counter()
            time.sleep(5)
            timer.record_stage("waiting", time.perf_counter() - t0)

        # 실제 배치 번호 (Global Index)
        current_batch_global_idx = batch_idx + 1 + start_batch_index
        
        batch_dir = batches_dir / f"batch_{current_batch_global_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # [User Request] Unified Terminal Output
        # 예: [VLM] INFERENCE_DONE [DB] UPLOAD_DONE [Judge] WAITING [Summarize] WAITING
        
        status_map = {
            "VLM": "PENDING",
            "DB": "PENDING", 
            "Judge": "WAITING",
            "Summarize": "WAITING"
        }
        current_segments = []
        
        last_printed_line = None

        def _print_status():
            nonlocal last_printed_line
            msg = []
            for k, v in status_map.items():
                msg.append(f"[{k}] {v}")
            # [User Request] Add segments list etc.
            # Batch 1: segments [1, 2, 3] [VLM] DONE ...
            seg_info = f"segments {current_segments} " if current_segments else ""
            line_body = f"Batch {current_batch_global_idx}: {seg_info}{' '.join(msg)}"
            
            # 사용자 요청: 상태가 변경될 때만 출력 (중복 제거)
            if line_body != last_printed_line:
                print(f"{_get_timestamp()} {line_body}", flush=True)
                last_printed_line = line_body

        _print_status()

        # 1. VLM Inference (skip_vlm=True이면 이미 만들어진 vlm.json을 재사용)
        vlm_info = {"image_count": 0}
        batch_vlm_elapsed = 0.0

        if skip_vlm:
            status_map["VLM"] = "SKIP (precomputed)"
            _print_status()
        else:
            status_map["VLM"] = "INFERENCING..."
            _print_status()

            # processing_job 상태 코드 업데이트 (DB 업로드 상태가 아니라 처리 파이프라인 단계 상태)
            if adapter and processing_job_id:
                try:
                    adapter.update_processing_job_status(processing_job_id, "VLM_RUNNING")
                except Exception:
                    pass

            t_vlm = time.perf_counter()
            vlm_info = run_vlm_for_batch(
                captures_dir=captures_dir,
                manifest_json=manifest_json if captures_data is None else None,
                batch_manifest=captures_data if captures_data is not None else None,
                video_name=video_name,
                output_dir=batch_dir,
                start_idx=batch_info["start_idx"] if captures_data is None else None,
                end_idx=batch_info["end_idx"] if captures_data is None else None,
                batch_size=vlm_batch_size,
                concurrency=vlm_concurrency,
                show_progress=True, 
                video_id=video_id,
            )
            batch_vlm_elapsed = time.perf_counter() - t_vlm
            status_map["VLM"] = "INFERENCE_DONE"
            _print_status()

        total_vlm_elapsed += batch_vlm_elapsed

        # 2. DB Upload (VLM Results)
        if adapter and processing_job_id and sync_to_db:
            status_map["DB"] = "UPLOAD..."
            _print_status()
            try:
                upload_vlm_results_for_batch(
                    adapter,
                    video_id,
                    processing_job_id,
                    batch_dir / "vlm.json",
                )
                status_map["DB"] = "UPLOAD_DONE"
            except Exception:
                status_map["DB"] = "UPLOAD_FAIL"
            _print_status()
        else:
             status_map["DB"] = "UPLOAD_SKIP"
             _print_status()

        # 3. Fusion Config
        generate_fusion_config(
            template_config=template_config,
            output_config=fusion_config_path,
            repo_root=repo_root,
            stt_json=stt_json,
            vlm_json=batch_dir / "vlm.json",
            manifest_json=manifest_json,
            output_root=video_root,
        )

        config = load_config(str(fusion_config_path))
        
        # Batch output setup
        batch_fusion_dir = batch_dir / "fusion"
        batch_fusion_dir.mkdir(parents=True, exist_ok=True)
        

        # Sync Engine
        # Sync Engine
        # run_sync_engine 대신 run_batch_sync_engine 사용 (Time Range Filtering 지원)
        from src.fusion.sync_engine import run_batch_sync_engine
        
        sync_result = run_batch_sync_engine(
            stt_json=stt_json,
            vlm_json=batch_dir / "vlm.json",
            manifest_json=manifest_json,
            captures_data=captures_data,
            output_dir=batch_dir / "fusion", # segments_units.jsonl 위치
            time_range=(batch_info["start_ms"], batch_info["end_ms"]),
            sync_config={
                "min_segment_sec": config.raw.sync_engine.min_segment_sec,
                "max_segment_sec": config.raw.sync_engine.max_segment_sec,
                "max_transcript_chars": config.raw.sync_engine.max_transcript_chars,
                "silence_gap_ms": config.raw.sync_engine.silence_gap_ms,
                "max_visual_items": config.raw.sync_engine.max_visual_items,
                "max_visual_chars": config.raw.sync_engine.max_visual_chars,
                "dedup_similarity_threshold": config.raw.sync_engine.dedup_similarity_threshold,
            },
            segment_id_offset=cumulative_segment_count,
            run_id_override=processing_job_id or video_id,
        )
        
        # [User Request] Track segment IDs
        # segments_units.jsonl을 읽거나 sync_result['segments_count']로 계산
        # [User Request] Track segment IDs
        # segments_units.jsonl을 읽거나 sync_result['segments_count']로 계산
        current_batch_segments_count = sync_result.get("segments_count", 0)
        current_segments = list(range(cumulative_segment_count + 1, cumulative_segment_count + current_batch_segments_count + 1))
        _print_status()
        cumulative_segment_count += current_batch_segments_count
        status_map["Summarize"] = "RUNNING..."
        _print_status()
        
        from src.fusion.summarizer import run_summarizer, run_batch_summarizer
        from src.judge.judge import run_judge

        feedback_map = {}
        batch_passed = False
        batch_score = 0.0
        batch_prev_context = previous_context or None
        last_batch_context = ""

        for attempt in range(1, 3):
            if attempt > 1:
                print(f"\n[Pipeline] Attempt {attempt}/2: Retrying fusion due to judge feedback...", flush=True)
            
            # 4. Summarize (LLM)
            status_map["Summarize"] = "RUNNING..."
            _print_status()

            # DB 상태 업데이트: SUMMARY_RUNNING
            if adapter and processing_job_id:
                try:
                    adapter.update_processing_job_status(processing_job_id, "SUMMARY_RUNNING")
                except Exception:
                    pass

            def _sum_status_cb(tokens):
                status_map["Summarize"] = f"RUNNING.. {tokens} (token)"
                _print_status()

            t_summarize = time.perf_counter()
            summarizer_result = run_batch_summarizer(
                segments_units_jsonl=batch_dir / "fusion" / "segments_units.jsonl",
                output_dir=batch_dir / "fusion",
                config=config,
                previous_context=batch_prev_context,
                feedback_map=feedback_map,
                limit=None,
                status_callback=_sum_status_cb,
                verbose=True,
                batch_label=f"Batch {current_batch_global_idx}",
            )
            last_batch_context = summarizer_result.get("context") or ""
            batch_summarizer_elapsed = time.perf_counter() - t_summarize
            total_summarizer_elapsed += batch_summarizer_elapsed
            
            # [User Request] Show final status
            status_map["Summarize"] = "DONE"
            _print_status()

            # 5. Judge (LLM)
            status_map["Judge"] = "RUNNING..."
            _print_status()

            # DB 상태 업데이트: JUDGE_RUNNING
            if adapter and processing_job_id:
                try:
                    adapter.update_processing_job_status(processing_job_id, "JUDGE_RUNNING")
                except Exception:
                    pass

            def _judge_status_cb(tokens):
                status_map["Judge"] = f"RUNNING.. {tokens} (token)"
                _print_status()

            t_judge = time.perf_counter()
            judge_result = run_judge(
                config=config,
                segments_units_path=batch_dir / "fusion" / "segments_units.jsonl",
                segment_summaries_path=batch_dir / "fusion" / "segment_summaries.jsonl",
                output_report_path=batch_dir / "judge_report.json",
                output_segments_path=batch_dir / "judge_segments.jsonl",
                write_outputs=True,
                verbose=True,
                batch_size=getattr(config.judge, "batch_size", 10),
                workers=getattr(config.judge, "workers", 4),
                json_repair_attempts=getattr(config.judge, "json_repair_attempts", 3),
                limit=limit, 
                status_callback=_judge_status_cb,
                batch_label=f"Batch {current_batch_global_idx}",
            )
            batch_judge_elapsed = time.perf_counter() - t_judge
            total_judge_elapsed += batch_judge_elapsed
            
            passed, score = _process_judge_result(
                judge_result, 
                config, 
                batch_dir / "judge.json", 
                None, 
                silent=True
            )
            batch_score = score

            # [User Request] Show final status with score
            status_map["Judge"] = f"DONE ({batch_score:.1f})"
            _print_status()
            
            if passed:
                batch_passed = True
                break
                
            if attempt < 1:
                # Retry
                segment_reports = judge_result.get("segment_reports", []) or []
                for item in segment_reports:
                    if item.get("segment_id") is not None:
                        feedback_map[int(item["segment_id"])] = str(item.get("feedback", "")).strip()

        # Update context for the next batch using the latest summarizer output from this batch.
        previous_context = last_batch_context

        status_map["Summarize"] = "DONE"
        status_map["Judge"] = f"DONE ({batch_score:.1f})" if batch_passed else f"FAIL ({batch_score:.1f})"
        _print_status()
        print("") # Newline

        total_judge_score += batch_score
        if not batch_passed:
            all_batches_passed = False

        # Accumulate results
        batch_summaries_path = batch_dir / "fusion" / "segment_summaries.jsonl"
        if batch_summaries_path.exists():
            accumulate_segments_to_fusion(batch_summaries_path, accumulated_summaries_path)
            
        batch_units_path = batch_dir / "fusion" / "segments_units.jsonl"
        if batch_units_path.exists():
            content = batch_units_path.read_text(encoding="utf-8")
            with accumulated_segments_path.open("a", encoding="utf-8") as f:
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")

        # DB Upload (Fusion)
        if adapter and processing_job_id and sync_to_db:
            try:
                segment_map = {}
                batch_units_path = batch_dir / "fusion" / "segments_units.jsonl"
                if batch_units_path.exists():
                    segment_map = upload_segments_for_batch(
                        adapter,
                        video_id,
                        processing_job_id,
                        batch_units_path,
                        offset=0,
                    )

                if batch_summaries_path.exists():
                     upload_summaries_for_batch(
                        adapter, video_id, processing_job_id, batch_summaries_path, segment_map, batch_index=current_batch_global_idx
                     )
                upload_judge_result(
                    adapter, video_id, processing_job_id, batch_dir / "judge.json", current_batch_global_idx
                )
            except Exception as e:
                print(f"[DB] Error uploading batch fusion results: {e}")

        # 배치 완료 시 current_batch 진행률 업데이트 (total_batch는 건드리지 않음)
        if adapter and processing_job_id:
            try:
                # total=None으로 전달하여 current_batch만 업데이트
                result = adapter.update_processing_job_progress(processing_job_id, current_batch_global_idx, None)
                print(f"{_get_timestamp()}   [DB] Updated current_batch: {current_batch_global_idx}")
            except Exception as e:
                print(f"{_get_timestamp()} [DB] Warning: Failed to update batch progress: {e}")

        processed_batches_count += 1

    # [END of Batch Loop]


    # 전체 VLM 시간을 합산하여 'vlm' 단계로 기록 (리포트용)
    timer.record_stage("vlm", total_vlm_elapsed)
    fusion_info["timings"]["vlm_sec"] = total_vlm_elapsed

    # [Standardization] 배치별 vlm.json 및 stt.json을 루트로 통합/복제 (단일 모드 호환성)
    root_vlm_path = video_root / "vlm.json"
    all_vlm_items = []
    # batches 폴더 내의 각 배치에서 vlm.json의 items를 수집
    batches_dir = video_root / "batches"
    if batches_dir.exists():
        for batch_dir in sorted(batches_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            batch_vlm_path = batch_dir / "vlm.json"
            if batch_vlm_path.exists():
                try:
                    b_data = json.loads(batch_vlm_path.read_text(encoding="utf-8"))
                    items = b_data.get("items", [])
                    if items:
                        all_vlm_items.extend(items)
                except Exception as e:
                    print(f"[Warning] Failed to read {batch_vlm_path}: {e}")
    
    if all_vlm_items:
        # 시간순 정렬
        all_vlm_items.sort(key=lambda x: int(x.get("timestamp_ms", 0)))
        write_json(root_vlm_path, {"items": all_vlm_items})
        print(f"  [Standardization] Consolidated {len(all_vlm_items)} VLM items to {root_vlm_path.relative_to(ROOT)}")

    # stt.json이 루트에 없으면 (일반적으로는 이미 존재함) 검색해서 복제 또는 링크
    root_stt_path = video_root / "stt.json"
    if not root_stt_path.exists():
        # 혹시 batches 내부에 있는지 확인 (보통은 루트에 있음)
        for batch_dir in sorted(batches_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            batch_stt_path = batch_dir / "stt.json"
            if batch_stt_path.exists():
                import shutil
                shutil.copy2(batch_stt_path, root_stt_path)
                print(f"  [Standardization] Copied stt.json to {root_stt_path.relative_to(ROOT)}")
                break

    fusion_info["segment_count"] = cumulative_segment_count

    if accumulated_summaries_path.exists():
        config = load_config(str(fusion_config_path))
        groups_cfg = getattr(config.raw.render, "groups", None)
        group_order = groups_cfg.order if groups_cfg else None
        group_headers = groups_cfg.headers if groups_cfg else None

        _, render_elapsed = timer.time_stage(
            "fusion.renderer",
            render_segment_summaries_md,
            summaries_jsonl=accumulated_summaries_path,
            output_md=fusion_dir / "segment_summaries.md",
            include_sources=config.raw.render.include_sources,
            sources_jsonl=fusion_dir / "segments_units.jsonl"
            if (fusion_dir / "segments_units.jsonl").exists()
            else None,
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=limit,
            group_order=group_order,
            group_headers=group_headers,
            fusion_prompt_version=config.raw.summarizer.prompt_version,
            judge_prompt_version=config.judge.prompt_version,
            execution_time={
                "summarizer": total_summarizer_elapsed,
                "judge": total_judge_elapsed,
            },
            batch_config={
                "batch_size": config.judge.batch_size,
                "workers": config.judge.workers,
            },
            judge_stats={
                "final_score": total_judge_score / processed_batches_count if processed_batches_count > 0 else 0.0,
                "passed": all_batches_passed,
                "category_scores": {},  # Note: Category breakdown not available in batch mode (averaged)
            },
        )
        fusion_info["timings"]["renderer_sec"] = render_elapsed

        summaries, final_elapsed = timer.time_stage(
            "fusion.final_summary",
            compose_final_summaries,
            summaries_jsonl=accumulated_summaries_path,
            max_chars=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=limit,
        )
        fusion_info["timings"]["final_summary_sec"] = final_elapsed

        results_dir = video_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for fmt in config.raw.final_summary.generate_formats:
            if fmt in summaries:
                results_dir.joinpath(f"final_summary_{fmt}.md").write_text(
                    summaries[fmt], encoding="utf-8"
                )

    return fusion_info
