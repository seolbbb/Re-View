"""
Supabase DB 동기화 모듈 (Pipeline Integration).

이 모듈은 비디오 분석 파이프라인(`run_video_pipeline.py`)의 실행 결과를
Supabase 데이터베이스로 전송하는 연결 고리 역할을 합니다.

주요 기능:
1. SupabaseAdapter 초기화 및 연결 확인
2. Video 레코드 생성 (중복 방지 로직 포함)
3. Pipeline Run 메타데이터 저장 (실행 이력 관리)
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
                storage_path=str(video_root),   # 로컬 결과물 경로 저장
                duration_sec=duration_sec,
                user_id=user_id
            )
            video_id = video_data['id']
            print(f"[DB] Created video record: {video_id} (Duration: {duration_sec}s)")
        
        # 2. 실행 메타데이터 저장 (Pipeline Runs)
        # 이번 파이프라인 실행에 대한 이력을 별도 테이블에 저장하여, 
        # 언제, 어떤 설정으로 분석이 수행되었는지 추적 가능하게 함.
        pipeline_run_id = None
        if run_meta:
            run_data = adapter.save_pipeline_run(video_id, run_meta)
            pipeline_run_id = run_data.get('id')
            print(f"[DB] Created pipeline_run record: {pipeline_run_id}")

        # 3. 분석 결과물 일괄 업로드 (Core Logic)
        # SupabaseAdapter.save_all_pipeline_results가 모든 하위 콘텐츠(STT, Segments 등)의 저장을 조율함
        results = adapter.save_all_pipeline_results(
            video_id=video_id,
            video_root=video_root,
            provider=provider,
            pipeline_run_id=pipeline_run_id,
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
) -> Optional[Tuple[Any, str, Optional[str]]]:
    """전처리 단계의 부분 업로드를 위한 DB 컨텍스트를 준비한다."""
    adapter = get_supabase_adapter()
    if not adapter:
        print("[DB] Supabase adapter not configured (check .env). Skipping upload.")
        return None
    try:
        # preprocess 단계에서 재사용할 video_id/pipeline_run_id를 미리 만든다.
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
                storage_path=str(video_root),
                duration_sec=duration_sec,
                user_id=user_id,
            )
            video_id = video_data["id"]

        pipeline_run_id = None
        if run_meta:
            # 전처리 시작 시점의 run_meta를 먼저 저장해 run_id를 확보한다.
            run_data = adapter.save_pipeline_run(video_id, run_meta)
            pipeline_run_id = run_data.get("id")

        return adapter, video_id, pipeline_run_id
    except Exception as exc:
        print(f"[DB] ❌ Sync init failed: {exc}")
        return None


def sync_preprocess_artifacts_to_db(
    *,
    adapter: Any,
    video_id: str,
    video_root: Path,
    provider: str,
    pipeline_run_id: Optional[str],
    include_stt: bool,
    include_captures: bool,
) -> Dict[str, Any]:
    """STT/캡처 아티팩트를 부분 업로드한다."""
    results: Dict[str, Any] = {"saved": {}, "errors": []}

    if include_captures:
        manifest_path = video_root / "manifest.json"
        if manifest_path.exists():
            try:
                # manifest.json + captures/ 이미지를 읽어 업로드한다.
                captures_dir = video_root / "captures"
                captures_result = adapter.save_captures_with_upload(
                    video_id,
                    manifest_path,
                    captures_dir,
                    pipeline_run_id=pipeline_run_id,
                )
                results["saved"]["captures"] = captures_result.get("db_saved", 0)
                results["errors"].extend(captures_result.get("errors", []))
            except Exception as e:
                results["errors"].append(f"captures: {str(e)}")

    if include_stt:
        stt_path = video_root / "stt.json"
        if stt_path.exists():
            try:
                # stt.json을 읽어 DB에 직접 저장한다.
                stt_rows = adapter.save_stt_from_file(
                    video_id,
                    stt_path,
                    provider=provider,
                    pipeline_run_id=pipeline_run_id,
                )
                results["saved"]["stt_results"] = len(stt_rows) if stt_rows else 0
            except Exception as e:
                results["errors"].append(f"stt_results: {str(e)}")

    return results


def finalize_preprocess_db_sync(
    *,
    adapter: Any,
    video_id: str,
    pipeline_run_id: Optional[str],
    run_meta: Dict[str, Any],
    errors: List[str],
) -> None:
    """전처리 업로드 이후 상태/메타데이터를 갱신한다."""
    if pipeline_run_id:
        # 전처리 완료 시점의 run_meta로 최신 상태를 덮어쓴다.
        adapter.update_pipeline_run(pipeline_run_id, run_meta)

    run_status = run_meta.get("status") if isinstance(run_meta, dict) else None
    error_message = "; ".join(errors) if errors else None
    if run_status == "error":
        # 파이프라인 자체 실패는 failed로 표시한다.
        if not error_message and isinstance(run_meta, dict):
            error_message = run_meta.get("error")
        adapter.update_video_status(video_id, "failed", error=error_message)
    elif errors:
        # 부분 업로드 실패는 completed_with_errors로 표시한다.
        adapter.update_video_status(video_id, "completed_with_errors", error=error_message)
    else:
        # 문제 없으면 completed로 마무리한다.
        adapter.update_video_status(video_id, "completed")
