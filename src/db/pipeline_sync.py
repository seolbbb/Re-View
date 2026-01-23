"""
Supabase DB 동기화 모듈 (Pipeline Integration).

이 모듈은 비디오 분석 파이프라인(`run_video_pipeline.py`)의 실행 결과를
Supabase 데이터베이스로 전송하는 연결 고리 역할을 합니다.

주요 기능:
1. SupabaseAdapter 초기화 및 연결 확인
2. Video 레코드 생성 (중복 방지 로직 포함)
3. 전처리/처리 작업 관리 (preprocessing_jobs/processing_jobs)
4. 모든 분석 결과물(Captures, STT, Segments, Summaries)의 일괄 업로드
5. 최종 실행 결과 요약 리포트 출력
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from .supabase_adapter import get_supabase_adapter

def sync_pipeline_results_to_db(
    video_path: Path,
    video_root: Path,
    run_meta: Dict[str, Any],
    duration_sec: Optional[int] = None,
    provider: str = "clova",
    user_id: Optional[str] = None,
    include_preprocess: bool = True,
) -> bool:
    """
    파이프라인 실행 완료 후 생성된 결과물을 Supabase DB와 동기화합니다.
    
    이 함수는 `run_video_pipeline.py`의 마지막 단계에서 호출되어야 하며,
    로컬에 저장된 JSON/JSONL 파일들을 읽어 DB에 저장하고, 필요한 경우 Storage에 파일을 업로드합니다.
    
    Args:
        video_path: 처리된 원본 비디오 파일의 경로 (파일명 추출 및 중복 확인용)
        video_root: 분석 결과물(JSON 등)이 저장된 루트 디렉토리 경로 (예: output/video_name)
        run_meta: 파이프라인 실행 통계(시간, 버전 등)를 담은 딕셔너리
        duration_sec: 비디오 재생 시간 (초)
        provider: 사용된 STT 모델 제공자 (예: 'clova', 'openai')
        user_id: 비디오 소유자 ID (Optional, 멀티 유저 환경용)
        include_preprocess: 캡처/음성 결과까지 업로드할지 여부
        
    Returns:
        bool: 동기화 성공 여부 (True: 성공, False: 실패)
    """
    # 0. 어댑터 초기화 및 환경변수 확인
    adapter = get_supabase_adapter()
    if not adapter:
        print("[DB] Supabase adapter not configured (check .env). Skipping upload.")
        # DB 설정이 없어도 파이프라인 자체는 성공으로 간주할 수 있으므로, 
        # 로컬 파일 생성까지만 하고 종료하는 시나리오를 위해 False 반환하지만 에러는 아님.
        return False
        
    # 데이터 타입 정규화 (DB의 duration_sec는 INTEGER임)
    if duration_sec is not None:
        try:
            duration_sec = int(float(str(duration_sec)))
        except (ValueError, TypeError):
            duration_sec = None
            
    try:
        print(f"[DB] Starting Supabase synchronization for {video_path.name}...")
        
        # 1. Video 레코드 관리 (생성 또는 조회)
        video_id = None
        
        # 사용자 ID가 있는 경우, 동일한 파일명의 비디오가 이미 존재하는지 확인 (중복 업로드 방지)
        if user_id:
            existing_video = adapter.get_video_by_filename(user_id, video_path.name)
            if existing_video:
                video_id = existing_video['id']
                print(f"[DB] Found existing video: {video_id} (Skipping creation)")
        
        # 기존 비디오가 없으면 새로 생성
        if not video_id:
            video_data = adapter.create_video(
                name=video_path.stem,           # 확장자를 제외한 파일명을 비디오 이름으로 사용
                original_filename=video_path.name,
                duration_sec=duration_sec,
                user_id=user_id
            )
            video_id = video_data['id']
            print(f"[DB] Created video record: {video_id} (Duration: {duration_sec}s)")
        
        # 2. 전처리 작업 생성 (preprocessing_jobs)
        preprocess_job_id = None
        if include_preprocess:
            config_hash = run_meta.get("args", {}).get("config_hash") if isinstance(run_meta, dict) else None
            job = adapter.create_preprocessing_job(
                video_id,
                source="SERVER",
                stt_backend=provider,
                config_hash=config_hash,
            )
            preprocess_job_id = job.get("id")
            print(f"[DB] Created preprocessing_job record: {preprocess_job_id}")

        # 3. 분석 결과물 일괄 업로드 (Core Logic)
        # SupabaseAdapter.save_all_pipeline_results가 모든 하위 콘텐츠(STT, Segments 등)의 저장을 조율함
        results = adapter.save_all_pipeline_results(
            video_id=video_id,
            video_root=video_root,
            provider=provider,
            preprocess_job_id=preprocess_job_id,
            include_preprocess=include_preprocess,
        )
            
        # 4. 결과 요약 및 로그 출력
        saved_counts = results.get("saved", {})
        errors = results.get("errors", [])
        
        print(f"[DB] Upload Summary:")
        print(f"  - Video ID: {video_id}")
        for k, v in saved_counts.items():
            print(f"  - {k}: {v} records")
            
        if errors:
            print(f"[DB] ⚠️ Completed with {len(errors)} errors:")
            for e in errors:
                print(f"  - {e}")
            return False # 에러 발생 시 실패로 간주 (정책에 따라 부분 성공 허용 가능)
        else:
            print("[DB] ✅ All artifacts uploaded successfully.")
            return True
            
    except Exception as e:
        print(f"[DB] ❌ Sync failed: {e}")
        # 디버깅을 위해 traceback이 필요한 경우 주석 해제
        # import traceback
        # traceback.print_exc()
        return False


def prepare_preprocess_db_sync(
    *,
    video_path: Path,
    video_root: Path,
    run_meta: Dict[str, Any],
    duration_sec: Optional[int] = None,
    user_id: Optional[str] = None,
    stt_backend: str = "clova",
    table_name: str = "captures",
) -> Optional[Tuple[Any, str, Optional[str]]]:
    """전처리 단계의 부분 업로드를 위한 DB 컨텍스트를 준비한다.
    
    새 ERD 기준으로 preprocessing_jobs 테이블을 사용합니다.
    
    Returns:
        Tuple[adapter, video_id, preprocessing_job_id] or None
    """
    adapter = get_supabase_adapter()
    if not adapter:
        print("[DB] Supabase adapter not configured (check .env). Skipping upload.")
        return None
    try:
        # preprocess 단계에서 재사용할 video_id/preprocessing_job_id를 미리 만든다.
        if duration_sec is not None:
            try:
                duration_sec = int(float(str(duration_sec)))
            except (ValueError, TypeError):
                duration_sec = None

        video_id = None
        if user_id:
            # 동일 파일명의 기존 video 레코드를 재사용한다.
            existing_video = adapter.get_video_by_filename(user_id, video_path.name)
            if existing_video:
                video_id = existing_video["id"]

        if not video_id:
            # 없으면 새로 만들고 이후 단계에서 재사용한다.
            video_data = adapter.create_video(
                name=video_path.stem,
                original_filename=video_path.name,
                duration_sec=duration_sec,
                user_id=user_id,
            )
            video_id = video_data["id"]

        # 새 ERD: preprocessing_jobs 생성 (QUEUED → RUNNING)
        preprocessing_job_id = None
        config_hash = run_meta.get("args", {}).get("config_hash") if isinstance(run_meta, dict) else None
        job = adapter.create_preprocessing_job(
            video_id,
            source="SERVER",
            stt_backend=stt_backend,
            config_hash=config_hash,
        )
        preprocessing_job_id = job.get("id")
        
        # 즉시 RUNNING 상태로 전환
        if preprocessing_job_id:
            adapter.update_preprocessing_job_status(preprocessing_job_id, "RUNNING")

        return adapter, video_id, preprocessing_job_id
    except Exception as exc:
        print(f"[DB] ❌ Sync init failed: {exc}")
        return None


def sync_preprocess_artifacts_to_db(
    *,
    adapter: Any,
    video_id: str,
    video_root: Path,
    provider: str,
    preprocess_job_id: Optional[str],
    include_stt: bool,
    include_captures: bool,
    include_audio: bool = True,
    stt_payload: Optional[Any] = None,
    captures_payload: Optional[List[Dict[str, Any]]] = None,
    audio_path: Optional[Path] = None,
    table_name: str = "captures",
) -> Dict[str, Any]:
    """STT/캡처/오디오 아티팩트를 부분 업로드한다.
    
    새 ERD 기준으로 preprocess_job_id를 사용합니다.
    """
    results: Dict[str, Any] = {"saved": {}, "errors": []}

    # 1. Audio 업로드 (가장 먼저 - 파일 크기가 클 수 있음)
    if include_audio:
        try:
            # 오디오 파일 경로 찾기
            if audio_path is None:
                # video_root 내 .wav 파일 검색
                wav_files = list(video_root.glob("*.wav"))
                if wav_files:
                    audio_path = wav_files[0]
            
            if audio_path and audio_path.exists():
                upload_result = adapter.upload_audio(video_id, audio_path)
                audio_storage_key = upload_result.get("storage_path")
                results["saved"]["audio"] = 1
                results["audio_storage_key"] = audio_storage_key  # STT에서 사용할 키
                
                # preprocessing_jobs에 audio_storage_key 기록
                if preprocess_job_id and audio_storage_key:
                    adapter.client.table("preprocessing_jobs").update({
                        "audio_storage_key": audio_storage_key
                    }).eq("id", preprocess_job_id).execute()
        except Exception as e:
            results["errors"].append(f"audio: {str(e)}")

    # 2. Captures 업로드
    if include_captures:
        try:
            captures_dir = video_root / "captures"
            if captures_payload is not None:
                captures_result = adapter.save_captures_with_upload_payload(
                    video_id,
                    captures_payload,
                    captures_dir,
                    preprocess_job_id=preprocess_job_id,
                    table_name=table_name,
                )
            else:
                manifest_path = video_root / "manifest.json"
                if not manifest_path.exists():
                    captures_result = None
                else:
                    captures_result = adapter.save_captures_with_upload(
                        video_id,
                        manifest_path,
                        captures_dir,
                        preprocess_job_id=preprocess_job_id,
                        table_name=table_name,
                    )
            if captures_result:
                results["saved"]["captures"] = captures_result.get("db_saved", 0)
                results["errors"].extend(captures_result.get("errors", []))
        except Exception as e:
            results["errors"].append(f"captures: {str(e)}")

    # 3. STT 결과 저장
    if include_stt:
        try:
            if stt_payload is not None:
                segments = stt_payload.get("segments", stt_payload) if isinstance(stt_payload, dict) else stt_payload
                if isinstance(segments, dict):
                    segments = segments.get("segments", [])
                if not isinstance(segments, list):
                    segments = []
                stt_rows = adapter.save_stt_result(
                    video_id,
                    segments,
                    preprocess_job_id=preprocess_job_id,
                    provider=provider,
                )
            else:
                stt_path = video_root / "stt.json"
                if not stt_path.exists():
                    stt_rows = None
                else:
                    stt_rows = adapter.save_stt_from_file(
                        video_id,
                        stt_path,
                        provider=provider,
                        preprocess_job_id=preprocess_job_id,
                    )
            results["saved"]["stt_results"] = len(stt_rows) if stt_rows else 0
        except Exception as e:
            results["errors"].append(f"stt_results: {str(e)}")

    return results



def finalize_preprocess_db_sync(
    *,
    adapter: Any,
    video_id: str,
    preprocess_job_id: Optional[str],
    run_meta: Dict[str, Any],
    errors: List[str],
) -> None:
    """전처리 업로드 이후 상태/메타데이터를 갱신한다.
    
    새 ERD 기준으로 preprocessing_jobs 상태를 업데이트합니다.
    - 성공: RUNNING → DONE
    - 에러: RUNNING → FAILED (에러 메시지 기록)
    """
    if not preprocess_job_id:
        return
        
    run_status = run_meta.get("status") if isinstance(run_meta, dict) else None
    error_message = "; ".join(errors) if errors else None
    
    if run_status == "error" or errors:
        # 파이프라인 실패 또는 업로드 중 에러 발생
        if not error_message and isinstance(run_meta, dict):
            error_message = run_meta.get("error")
        adapter.update_preprocessing_job_status(
            preprocess_job_id, 
            "FAILED", 
            error_message=error_message
        )
        # videos 테이블에도 에러 기록
        adapter.update_video_status(video_id, "FAILED", error=error_message)
    else:
        # 정상 완료
        adapter.update_preprocessing_job_status(preprocess_job_id, "DONE")
        adapter.update_video_status(video_id, "PREPROCESS_DONE")


def sync_processing_results_to_db(
    *,
    video_root: Path,
    video_id: str,
    processing_job_id: str,
) -> Dict[str, Any]:
    """처리 파이프라인 결과물(VLM, Segments, Summaries)을 DB에 업로드한다.
    
    새 ERD 기준으로 processing_jobs 테이블과 연동됩니다.
    """
    adapter = get_supabase_adapter()
    if not adapter:
        print("[DB] Supabase adapter not configured. Skipping processing upload.")
        return {"saved": {}, "errors": ["No adapter"]}

    results: Dict[str, Any] = {"saved": {}, "errors": []}

    try:
        # 1. VLM Results (vlm.json 또는 batches/batch_N/vlm.json)
        vlm_items_all = []
        
        # 단일 모드: vlm.json 체크
        vlm_path = video_root / "vlm.json"
        if vlm_path.exists():
            import json
            vlm_payload = json.loads(vlm_path.read_text(encoding="utf-8"))
            vlm_items_all.extend(vlm_payload.get("items", []))
        
        # 배치 모드: batches/batch_N/vlm.json 체크
        batches_dir = video_root / "batches"
        if batches_dir.exists():
            import json
            for batch_dir in sorted(batches_dir.iterdir()):
                batch_vlm = batch_dir / "vlm.json"
                if batch_vlm.exists():
                    vlm_payload = json.loads(batch_vlm.read_text(encoding="utf-8"))
                    vlm_items_all.extend(vlm_payload.get("items", []))
        
        if vlm_items_all:
            vlm_rows = adapter.insert_vlm_results(
                video_id,
                vlm_items_all,
                processing_job_id=processing_job_id,
            )
            results["saved"]["vlm_results"] = len(vlm_rows)
    except Exception as e:
        results["errors"].append(f"vlm_results: {str(e)}")

    # 2. Segments (fusion/segments_units.jsonl)
    segment_map = {}
    try:
        segments_path = video_root / "fusion" / "segments_units.jsonl"
        if segments_path.exists():
            segments = adapter.save_segments_from_file(
                video_id, 
                segments_path, 
                processing_job_id=processing_job_id
            )
            results["saved"]["segments"] = len(segments)
            for seg in segments:
                idx = seg.get("segment_index")
                uuid = seg.get("id")
                if idx is not None and uuid:
                    segment_map[idx] = uuid
    except Exception as e:
        results["errors"].append(f"segments: {str(e)}")

    # 3. Summaries (fusion/segment_summaries.jsonl)
    try:
        summaries_path = video_root / "fusion" / "segment_summaries.jsonl"
        if summaries_path.exists():
            summaries = adapter.save_summaries_from_file(
                video_id,
                summaries_path,
                segment_map=segment_map,
                processing_job_id=processing_job_id,
            )
            results["saved"]["summaries"] = len(summaries)
    except Exception as e:
        results["errors"].append(f"summaries: {str(e)}")

    return results

