"""처리 파이프라인을 API로 실행하는 FastAPI 래퍼.

사용 방법:

1. 백엔드 서버 실행
uvicorn src.process_api:app --port 8080

-> 서버가 실행이 되었으면 아래 예시와 같이 뜸
INFO:     Started server process [92286]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

2. 프론트엔드 실행
-> 다른 터미널에서 cd frontend
-> npm install (첫 실행시)
-> npm run dev

API 직접 테스트 (백엔드만 사용할 경우):
다른 터미널을 추가로 열고 아래 요청 예시를 입력, 전처리가 되어 있어야 함
POST 실행
요청 예시:
  curl -X POST http://localhost:8080/process \\
    -H "Content-Type: application/json" \\
    -d '{"video_name":"diffusion","force_db":true}'

diffusion.mp4이면 video_name은 diffusion
force_db를 사용하면 supabase에서 정보를 가져옴
db 설정하기 어렵다 -> force_db true 빼고 실행하면 로컬에 저장된 전처리 정보를 가져와서 실행함

GET 실행
서버 체크:
    curl http://localhost:8080/health
마지막 실행 로그:
    curl http://localhost:8080/runs/diffusion
비디오 상태:
    curl http://localhost:8080/videos/{video_id}/status
처리 진행률:
    curl http://localhost:8080/videos/{video_id}/progress
최신 요약:
    curl http://localhost:8080/videos/{video_id}/summary
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from src.db import get_supabase_adapter
from src.run_process_pipeline import run_processing_pipeline
from src.services.chat_session_store import ChatSessionStore


app = FastAPI(title="Screentime Processing API")

_cors_origins_str = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:5174"
)
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


_chat_sessions = ChatSessionStore(ttl=3600)


def _chat_session_cleanup_loop() -> None:
    while True:
        time.sleep(300)
        _chat_sessions.cleanup_expired()


threading.Thread(
    target=_chat_session_cleanup_loop,
    name="chat-session-cleanup",
    daemon=True,
).start()


class ProcessRequest(BaseModel):
    video_name: Optional[str] = None
    video_id: Optional[str] = None
    output_base: str = "data/outputs"
    batch_mode: Optional[bool] = None
    limit: Optional[int] = None
    force_db: Optional[bool] = None
    sync_to_db: Optional[bool] = None


class ProcessResponse(BaseModel):
    status: str
    message: str


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/runs/{video_name}")
def get_last_run(video_name: str, output_base: str = "data/outputs") -> dict:
    output_root = Path(output_base) / video_name
    run_meta_path = output_root / "pipeline_run.json"
    if not run_meta_path.exists():
        raise HTTPException(status_code=404, detail="pipeline_run.json not found.")
    return json.loads(run_meta_path.read_text(encoding="utf-8"))


@app.post("/process", response_model=ProcessResponse)
def process_pipeline(request: ProcessRequest, background_tasks: BackgroundTasks) -> ProcessResponse:
    if not request.video_name and not request.video_id:
        raise HTTPException(status_code=400, detail="video_name or video_id is required.")

    background_tasks.add_task(
        run_processing_pipeline,
        video_name=request.video_name,
        video_id=request.video_id,
        output_base=request.output_base,
        batch_mode=request.batch_mode,
        limit=request.limit,
        force_db=request.force_db,
        sync_to_db=request.sync_to_db,
    )

    return ProcessResponse(status="started", message="processing started")


# =============================================================================
# 새 ERD 기반 엔드포인트
# =============================================================================


def _build_status_payload(adapter, video_id: str) -> Dict[str, Any]:
    """비디오 상태 페이로드를 생성합니다 (SSE/REST 공용)."""
    video = adapter.get_video(video_id)
    if not video:
        return None

    result = {
        "video_id": video_id,
        "video_status": video.get("status"),
        "error_message": video.get("error_message"),
    }

    # 전처리 작업 정보
    preprocess_job_id = video.get("current_preprocess_job_id")
    if preprocess_job_id:
        preprocess_job = adapter.get_preprocessing_job(preprocess_job_id)
        if preprocess_job:
            result["preprocess_job"] = {
                "id": preprocess_job_id,
                "status": preprocess_job.get("status"),
                "started_at": preprocess_job.get("started_at"),
                "ended_at": preprocess_job.get("ended_at"),
            }

    # 처리 작업 정보
    processing_job_id = video.get("current_processing_job_id")
    if processing_job_id:
        processing_job = adapter.get_processing_job(processing_job_id)
        if processing_job:
            result["processing_job"] = {
                "id": processing_job_id,
                "status": processing_job.get("status"),
                "progress_current": processing_job.get("current_batch"),
                "progress_total": processing_job.get("total_batch"),
                "started_at": processing_job.get("started_at"),
                "ended_at": processing_job.get("ended_at"),
            }

    return result


def _build_summaries_payload(adapter, video_id: str) -> Dict[str, Any]:
    """요약 페이로드를 생성합니다 (SSE/REST 공용)."""
    rows = adapter.get_summaries(video_id)

    items = []
    for r in rows:
        seg_info = r.get("segments") or {}
        items.append({
            "summary_id": r.get("id"),
            "segment_id": r.get("segment_id"),
            "segment_index": seg_info.get("segment_index"),
            "start_ms": seg_info.get("start_ms"),
            "end_ms": seg_info.get("end_ms"),
            "summary": r.get("summary"),
            "created_at": r.get("created_at"),
        })

    items.sort(key=lambda x: x.get("segment_index") or 0)

    return {
        "video_id": video_id,
        "count": len(items),
        "items": items,
    }


@app.get("/videos/{video_id}/status")
def get_video_status(video_id: str) -> Dict[str, Any]:
    """비디오 상태 및 현재 작업 정보 조회."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    result = _build_status_payload(adapter, video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return result


@app.get("/videos/{video_id}/status/stream")
def stream_video_status(video_id: str):
    """비디오 상태 및 요약을 SSE로 스트리밍합니다.

    이벤트 타입:
    - status: 상태 변경 시 (video_status, processing_job, error_message)
    - summaries: 새 요약 추가 시 (count, items)
    - done: 처리 완료/실패 시
    """
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    # 초기 비디오 존재 확인
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    def event_generator():
        last_status_json = None
        last_summary_count = 0

        while True:
            # 상태 조회
            status_data = _build_status_payload(adapter, video_id)
            if status_data is None:
                yield _format_sse_event("error", {"error": "Video not found"})
                break

            # 상태 변경 시에만 이벤트 발생
            status_json = json.dumps(status_data, sort_keys=True, default=str)
            if status_json != last_status_json:
                yield _format_sse_event("status", status_data)
                last_status_json = status_json

            # 현재 상태 확인
            current_status = (status_data.get("video_status") or "").upper()

            # 처리 중일 때만 요약 체크
            if current_status not in ("DONE", "FAILED"):
                summaries = _build_summaries_payload(adapter, video_id)
                if summaries["count"] > last_summary_count:
                    yield _format_sse_event("summaries", summaries)
                    last_summary_count = summaries["count"]

            # 완료/실패 시 done 이벤트 후 종료
            if current_status in ("DONE", "FAILED"):
                # 최종 요약 전송
                final_summaries = _build_summaries_payload(adapter, video_id)
                if final_summaries["count"] > last_summary_count:
                    yield _format_sse_event("summaries", final_summaries)
                yield _format_sse_event("done", {"video_status": current_status})
                break

            time.sleep(1)  # 1초 간격 DB 체크

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/videos/{video_id}/progress")
def get_video_progress(video_id: str) -> Dict[str, Any]:
    """처리 진행률 조회 (폴링용)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 정보 조회
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    processing_job_id = video.get("current_processing_job_id")
    if not processing_job_id:
        return {
            "video_id": video_id,
            "has_processing_job": False,
            "video_status": video.get("status"),
        }
    
    processing_job = adapter.get_processing_job(processing_job_id)
    if not processing_job:
        return {
            "video_id": video_id,
            "has_processing_job": False,
            "video_status": video.get("status"),
        }
    
    progress_current = processing_job.get("current_batch") or 0
    progress_total = processing_job.get("total_batch") or 1
    progress_percent = round((progress_current / progress_total) * 100, 1) if progress_total > 0 else 0
    
    return {
        "video_id": video_id,
        "has_processing_job": True,
        "processing_job_id": processing_job_id,
        "status": processing_job.get("status"),
        "progress_current": progress_current,
        "progress_total": progress_total,
        "progress_percent": progress_percent,
        "is_complete": processing_job.get("status") in ("DONE", "FAILED"),
    }


@app.get("/videos/{video_id}/summary")
def get_video_summary(video_id: str, format: Optional[str] = None) -> Dict[str, Any]:
    """최신 요약 결과 조회 (IN_PROGRESS 포함)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 존재 확인
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # 최신 summary_results 조회
    summary_result = adapter.get_latest_summary_results(video_id, format)
    if not summary_result:
        return {
            "video_id": video_id,
            "has_summary": False,
            "video_status": video.get("status"),
        }
    
    return {
        "video_id": video_id,
        "has_summary": True,
        "summary_id": summary_result.get("id"),
        "format": summary_result.get("format"),
        "status": summary_result.get("status"),
        "is_complete": summary_result.get("status") == "DONE",
        "payload": summary_result.get("payload"),
        "created_at": summary_result.get("created_at"),
    }


def _parse_id_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items = []
    seen = set()
    for raw in value.split(","):
        cleaned = raw.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        items.append(cleaned)
    return items


@app.get("/videos/{video_id}/summaries")
def get_video_summaries(video_id: str) -> Dict[str, Any]:
    """세부 요약 세그먼트 리스트 조회 (Chatbot용)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return _build_summaries_payload(adapter, video_id)


@app.get("/videos/{video_id}/evidence")
def get_video_evidence(
    video_id: str,
    stt_ids: Optional[str] = None,
    cap_ids: Optional[str] = None,
) -> Dict[str, Any]:
    """Evidence lookup by STT/VLM IDs (Chatbot thinking mode)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    stt_list = _parse_id_list(stt_ids)
    cap_list = _parse_id_list(cap_ids)

    stt_rows = adapter.get_stt_results_by_ids(video_id, stt_list) if stt_list else []
    vlm_rows = adapter.get_vlm_results_by_ids(video_id, cap_list) if cap_list else []

    return {
        "video_id": video_id,
        "stt": stt_rows,
        "vlm": vlm_rows,
    }


# =============================================================================
# 프론트엔드 연동 엔드포인트
# =============================================================================

INPUT_DIR = Path("data/inputs")
OUTPUT_DIR = Path("data/outputs")


def _sanitize_filename(name: str) -> str:
    """파일명에서 안전하지 않은 문자를 제거합니다 (ASCII, 숫자, -, . 만 허용)."""
    # 한글 등 비-ASCII 문자는 _로 치환하여 Storage/FS 호환성 확보
    return re.sub(r'[^a-zA-Z0-9.\-_]', '_', name)


def _ensure_preprocess_job_finalized(adapter, video_id: str) -> None:
    """preprocessing_job이 RUNNING 상태로 남아있으면 DONE으로 전환합니다."""
    try:
        video = adapter.get_video(video_id)
        if not video:
            return
        job_id = video.get("current_preprocess_job_id")
        if not job_id:
            return
        job = adapter.get_preprocessing_job(job_id)
        if job and job.get("status") == "RUNNING":
            adapter.update_preprocessing_job_status(job_id, "DONE")
    except Exception:
        pass


def _update_video_duration(adapter, video_id: str, video_path: str) -> None:
    """비디오 파일에서 duration_sec를 추출하여 DB에 업데이트합니다."""
    try:
        from src.pipeline.benchmark import get_video_info
        video_info = get_video_info(Path(video_path))
        duration = video_info.get("duration_sec")
        if duration is not None:
            duration = int(float(duration))
            adapter.client.table("videos").update(
                {"duration_sec": duration}
            ).eq("id", video_id).execute()
    except Exception:
        pass


def _run_full_pipeline(video_path: str, video_id: str) -> None:
    """전처리 → 처리 파이프라인을 순차 실행합니다 (BackgroundTasks용).

    업로드 엔드포인트에서 이미 videos 레코드를 생성했으므로,
    existing_video_id를 전달하여 파이프라인 내부에서 중복 생성을 방지한다.
    """
    from src.run_preprocess_pipeline import run_preprocess_pipeline

    adapter = get_supabase_adapter()
    try:
        print("\n" + "=" * 60)
        print(f"  Re:View API Pipeline: {video_id}")
        print("=" * 60)
        print(f"1. Starting 'Preprocessing'...")
        
        # 1. 전처리 (existing_video_id로 기존 레코드 재사용, DB 중복 생성 방지)
        run_preprocess_pipeline(
            video=video_path,
            sync_to_db=True,
            write_local_json=True,
            existing_video_id=video_id,
        )
        if adapter:
            _update_video_duration(adapter, video_id, video_path)
            _ensure_preprocess_job_finalized(adapter, video_id)
            adapter.update_video_status(video_id, "PREPROCESS_DONE")

        print(f"\n2. Starting 'Processing'...")
        # 2. 처리 파이프라인
        video_name = Path(video_path).stem
        run_processing_pipeline(
            video_name=video_name,
            video_id=video_id,
            sync_to_db=True,
            force_db=True,
        )
    except Exception as exc:
        if adapter:
            adapter.update_video_status(video_id, "FAILED", error=str(exc))


class UploadResponse(BaseModel):
    video_id: str
    video_name: str
    status: str


@app.post("/api/videos/upload", response_model=UploadResponse)
async def upload_video(
    http_request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> UploadResponse:
    """비디오 업로드 + 자동 파이프라인 실행."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_name = _sanitize_filename(file.filename)
    save_path = INPUT_DIR / f"{timestamp}_{safe_name}"

    content = await file.read()
    save_path.write_bytes(content)

    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video_name = Path(file.filename).stem
    user_id = _get_user_id_from_request(adapter, http_request)
    video = adapter.create_video(
        name=video_name,
        original_filename=file.filename,
        user_id=user_id,
    )
    video_id = video["id"]
    
    # [User Request] Video 원본 업로드 (video_storage_key 채우기)
    try:
        adapter.upload_video(video_id, save_path)
    except Exception as e:
        print(f"[API] Warning: Video upload failed but continuing preprocess: {e}")

    adapter.update_video_status(video_id, "PREPROCESSING")

    background_tasks.add_task(_run_full_pipeline, str(save_path), video_id)

    return UploadResponse(
        video_id=video_id,
        video_name=video_name,
        status="PREPROCESSING",
    )


# =============================================================================
# Signed URL 기반 업로드 엔드포인트
# =============================================================================


class UploadInitRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


class UploadInitResponse(BaseModel):
    video_id: str
    video_name: str
    upload_url: str
    storage_key: str


@app.post("/api/videos/upload/init", response_model=UploadInitResponse)
def init_upload(payload: UploadInitRequest, http_request: Request):
    """
    비디오 업로드를 위한 Presigned URL 발급 및 DB 레코드 생성.
    
    Purpose:
        프론트엔드에서 비디오 파일을 직접 스토리지에 업로드할 수 있도록
        Presigned URL을 발급합니다. R2 환경에서는 boto3 generate_presigned_url을,
        Supabase 환경에서는 create_signed_upload_url을 사용합니다.
    
    Storage Path Structure:
        - R2: {video_id}/{r2_prefix_videos}/{filename}
          예: abc123/videos/sample.mp4
        - Supabase: {video_id}/{filename}
    
    API Flow:
        1. Frontend → POST /api/videos/upload/init (파일명 전달)
        2. Backend → videos 테이블에 레코드 생성
        3. Backend → Presigned URL 반환
        4. Frontend → PUT {upload_url} (파일 직접 업로드)
        5. Frontend → POST /api/videos/upload/complete (업로드 완료 알림)
    
    Args:
        request (UploadInitRequest): 업로드할 파일명 포함
            - filename: 원본 파일명 (확장자 포함)
    
    Returns:
        UploadInitResponse:
            - video_id: 생성된 비디오 ID (UUID)
            - video_name: 파일명 (확장자 제외)
            - upload_url: Presigned URL (1시간 유효)
            - storage_key: 스토리지 경로
    
    Raises:
        HTTPException(503): DB 연결 실패
    
    Related:
        - complete_upload(): 업로드 완료 후 파이프라인 실행
        - _run_full_pipeline_from_storage(): 전처리 및 처리 실행
    """
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video_name = Path(payload.filename).stem
    safe_name = _sanitize_filename(payload.filename)
    user_id = _get_user_id_from_request(adapter, http_request)
    # 임시 ID 없이 레코드 먼저 생성하여 video_id 확보
    video = adapter.create_video(
        name=video_name,
        original_filename=payload.filename,
        user_id=user_id,
    )
    video_id = video["id"]

    # MIME 타입 추론 (확장자 기반)
    import mimetypes
    content_type, _ = mimetypes.guess_type(safe_name)
    if not content_type:
        content_type = "video/mp4"  # fallback

    # R2 또는 Supabase Storage에 따라 경로 및 URL 생성
    if adapter.s3_client:
        # R2 presigned URL 생성
        storage_key = f"{video_id}/{adapter.r2_prefix_videos}/{safe_name}"
        adapter.update_video_storage_key(video_id, storage_key)
        
        upload_url = adapter.s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': adapter.r2_bucket,
                'Key': storage_key,
                'ContentType': content_type
            },
            ExpiresIn=3600  # 1시간 유효
        )
        print(f"[R2] Generated presigned upload URL for {storage_key} (ContentType: {content_type})")
    else:
        if getattr(adapter, "r2_only", False):
            raise HTTPException(status_code=503, detail="R2 storage is required (check R2_* env vars)")
        # Supabase Storage fallback
        storage_key = f"{video_id}/{safe_name}"
        adapter.update_video_storage_key(video_id, storage_key)
        
        signed = adapter.client.storage.from_("videos").create_signed_upload_url(
            storage_key
        )
        upload_url = signed.get("signed_url") or signed.get("signedURL", "")

    return UploadInitResponse(
        video_id=video_id,
        video_name=video_name,
        upload_url=upload_url,
        storage_key=storage_key,
    )


class UploadCompleteRequest(BaseModel):
    video_id: str
    storage_key: str


class UploadCompleteResponse(BaseModel):
    video_id: str
    status: str


@app.post("/api/videos/upload/complete", response_model=UploadCompleteResponse)
def complete_upload(
    request: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
):
    """Storage 업로드 완료 알림 → 파이프라인 실행."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(request.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    adapter.update_video_status(request.video_id, "PREPROCESSING")

    background_tasks.add_task(
        _run_full_pipeline_from_storage,
        request.video_id,
        request.storage_key,
    )

    return UploadCompleteResponse(
        video_id=request.video_id,
        status="PREPROCESSING",
    )


def _run_full_pipeline_from_storage(video_id: str, storage_key: str) -> None:
    """
    스토리지에서 비디오를 다운로드하여 전처리 → 처리 파이프라인을 순차 실행.
    
    Purpose:
        프론트엔드에서 업로드 완료 후 백그라운드에서 실행되는 메인 파이프라인.
        R2 또는 Supabase Storage에서 비디오를 다운로드하고, 전처리(캡처 추출,
        STT) 및 처리(VLM, Fusion) 파이프라인을 순차적으로 실행합니다.
    
    Storage Integration (R2 Priority):
        1. adapter.s3_client가 초기화되어 있으면 R2 사용
        2. R2 미설정 시 Supabase Storage fallback
        3. 다운로드 경로: {bucket}/{storage_key}
    
    Pipeline Flow:
        1. 스토리지에서 임시 디렉토리로 비디오 다운로드
        2. run_preprocess_pipeline() 실행 (캡처/오디오 추출, STT)
        3. run_processing_pipeline() 실행 (VLM, Judge, Fusion)
        4. 임시 디렉토리 정리
    
    Args:
        video_id (str): 비디오 UUID (DB 레코드 ID)
        storage_key (str): 스토리지 경로
            - R2: {video_id}/videos/{filename}
            - Supabase: {video_id}/{filename}
    
    Returns:
        None (백그라운드 태스크로 실행)
    
    Side Effects:
        - videos.status 업데이트 (PREPROCESSING → PREPROCESS_DONE → PROCESSING → DONE)
        - captures, stt_results, segments, summaries 테이블에 결과 저장
        - R2/Supabase Storage에 캡처 이미지, 오디오 업로드
    
    Error Handling:
        - 예외 발생 시 videos.status를 FAILED로 업데이트
        - 임시 디렉토리는 finally 블록에서 항상 정리
    
    Called By:
        - complete_upload() 엔드포인트 (BackgroundTasks)
    """
    from src.run_preprocess_pipeline import run_preprocess_pipeline

    adapter = get_supabase_adapter()
    tmp_dir = None
    try:
        print("\n" + "=" * 60)
        print(f"  Re:View API Pipeline (Storage): {video_id}")
        print("=" * 60)
        
        # 1. Storage에서 임시 디렉토리로 다운로드 (R2 우선, Supabase fallback)
        tmp_dir = tempfile.mkdtemp()
        original_name = Path(storage_key).name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        tmp_filename = f"{timestamp}_{original_name}"
        tmp_path = Path(tmp_dir) / tmp_filename

        if adapter.s3_client:
            # R2에서 다운로드
            adapter.s3_client.download_file(adapter.r2_bucket, storage_key, str(tmp_path))
            print(f"[R2] Downloaded video from {adapter.r2_bucket}/{storage_key}")
        else:
            if getattr(adapter, "r2_only", False):
                raise RuntimeError("R2 storage is required (check R2_* env vars)")
            # Supabase Storage fallback
            file_data = adapter.client.storage.from_("videos").download(storage_key)
            tmp_path.write_bytes(file_data)

        print(f"\n1. Starting 'Preprocessing'...")
        # 2. 전처리
        run_preprocess_pipeline(
            video=str(tmp_path),
            sync_to_db=True,
            write_local_json=True,
            existing_video_id=video_id,
        )
        if adapter:
            _update_video_duration(adapter, video_id, str(tmp_path))
            _ensure_preprocess_job_finalized(adapter, video_id) # Added this line
            adapter.update_video_status(video_id, "PREPROCESS_DONE")

        print(f"\n2. Starting 'Processing'...") # Added print header
        # 3. 처리 파이프라인
        video_name = tmp_path.stem
        run_processing_pipeline(
            video_name=video_name,
            video_id=video_id,
            sync_to_db=True,
            force_db=True,
        )
    except Exception as exc:
        if adapter:
            adapter.update_video_status(video_id, "FAILED", error=str(exc))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/videos")
def list_videos() -> Dict[str, Any]:
    """비디오 목록 조회 (최신순)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    videos = adapter.list_videos()
    return {"videos": videos}


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    """비디오 파일 서빙 (Storage signed URL 또는 로컬 fallback)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Storage에 업로드된 비디오가 있으면 signed URL로 리다이렉트
    storage_key = video.get("video_storage_key")
    if storage_key:
        signed_url = adapter.get_signed_url(storage_key, bucket="videos")
        if signed_url:
            return RedirectResponse(url=signed_url)

    # Fallback: 로컬 파일시스템에서 제공
    original_filename = video.get("original_filename", "")
    if INPUT_DIR.exists():
        for f in sorted(INPUT_DIR.iterdir(), reverse=True):
            if f.name.endswith(original_filename) or original_filename in f.name:
                # MIME 타입 추론 (확장자 기반)
                import mimetypes
                media_type, _ = mimetypes.guess_type(str(f))
                if not media_type:
                    media_type = "video/mp4"  # fallback
                return FileResponse(
                    path=str(f),
                    media_type=media_type,
                    filename=original_filename,
                )

    raise HTTPException(status_code=404, detail="Video file not found")


@app.get("/api/videos/{video_id}/thumbnail")
def get_video_thumbnail(video_id: str):
    """비디오 썸네일 이미지 서빙 (captures 버킷 signed URL 또는 로컬 fallback)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # DB에서 첫 번째 캡처의 storage_path 조회
    try:
        caps = (
            adapter.client.table("captures")
            .select("storage_path")
            .eq("video_id", video_id)
            .limit(1)
            .execute()
        )
        if caps.data and caps.data[0].get("storage_path"):
            signed_url = adapter.get_signed_url(
                caps.data[0]["storage_path"], bucket="captures"
            )
            if signed_url:
                return RedirectResponse(url=signed_url)
    except Exception:
        pass

    # Fallback: 로컬 파일시스템
    video_name = video.get("name", "")
    captures_dir = OUTPUT_DIR / video_name / "captures"

    if captures_dir.exists():
        images = sorted(captures_dir.glob("*.jpg")) + sorted(captures_dir.glob("*.png"))
        if images:
            return FileResponse(
                path=str(images[0]),
                media_type="image/jpeg",
            )

    raise HTTPException(status_code=404, detail="Thumbnail not found")


class ChatRequest(BaseModel):
    video_id: str
    message: str
    session_id: Optional[str] = None
    reasoning_mode: Optional[str] = None  # "flash" or "thinking"


class ChatResponse(BaseModel):
    response: str
    session_id: str


def _resolve_video_name(video: Dict[str, Any], fallback: str) -> str:
    for key in ("name", "video_name", "original_filename"):
        value = video.get(key)
        if value:
            return str(value)
    return fallback


def _get_or_create_chat_session(request: ChatRequest):
    session = None
    if request.session_id:
        session = _chat_sessions.get(request.session_id)
    if session:
        # Update reasoning_mode if provided in request
        if request.reasoning_mode and request.reasoning_mode in ("flash", "thinking"):
            session._state["reasoning_mode"] = request.reasoning_mode
        return session

    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(request.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_name = _resolve_video_name(video, request.video_id)
    initial_state = {
        "video_id": request.video_id,
        "video_name": video_name,
        "chat_mode": "full",
    }
    # Set reasoning_mode if provided
    if request.reasoning_mode and request.reasoning_mode in ("flash", "thinking"):
        initial_state["reasoning_mode"] = request.reasoning_mode

    return _chat_sessions.create(
        request.video_id,
        video_name,
        process_api_url="http://localhost:8080",
        initial_state=initial_state,
    )


def _format_sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].lower(), parts[1].strip()
    if scheme != "bearer" or not token:
        return None
    return token


def _get_user_id_from_request(adapter, request: Request) -> Optional[str]:
    """Authorization 헤더의 Supabase JWT로 user_id를 추출한다."""
    token = _get_bearer_token(request)
    if not token:
        return None
    try:
        user_resp = adapter.client.auth.get_user(token)
        # supabase-py v2: UserResponse.user.id
        user_obj = getattr(user_resp, "user", None)
        user_id = getattr(user_obj, "id", None) if user_obj else None
        if user_id:
            return user_id
        # fallback: dict-style response
        if isinstance(user_resp, dict):
            user_dict = user_resp.get("user") or user_resp.get("data", {}).get("user")
            if isinstance(user_dict, dict):
                return user_dict.get("id")
    except Exception:
        return None
    return None


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """LangGraph 기반 챗봇 API (비스트리밍)."""
    session = _get_or_create_chat_session(request)
    messages = session.send_message(request.message)
    response_text = "".join([getattr(msg, "text", "") for msg in messages if msg])
    return ChatResponse(
        response=response_text,
        session_id=session.session_id,
    )


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """LangGraph 기반 챗봇 API (SSE 스트리밍)."""
    session = _get_or_create_chat_session(request)

    def event_generator():
        yield _format_sse_event("session", {"session_id": session.session_id})
        try:
            last_message_id: Optional[str] = None
            for msg in session.stream_message(request.message):
                payload = {
                    "text": getattr(msg, "text", ""),
                    "is_final": bool(getattr(msg, "is_final", False)),
                }
                message_id = getattr(msg, "message_id", None)
                if message_id:
                    payload["message_id"] = message_id
                    last_message_id = message_id
                yield _format_sse_event("message", payload)

            suggestion_payload = None
            if hasattr(session, "consume_latest_suggestions"):
                suggestion_payload = session.consume_latest_suggestions()
            if suggestion_payload:
                questions = suggestion_payload.get("questions") or []
                if isinstance(questions, list) and questions:
                    yield _format_sse_event(
                        "suggestions",
                        {
                            "session_id": session.session_id,
                            "message_id": suggestion_payload.get("message_id") or last_message_id,
                            "questions": questions,
                            "source": suggestion_payload.get("source") or "graph_node",
                        },
                    )
            yield _format_sse_event("done", {"session_id": session.session_id})
        except Exception as exc:
            yield _format_sse_event("error", {"error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# STT 처리 엔드포인트 (Frontend → Backend Storage 기반)
# =============================================================================


class STTProcessRequest(BaseModel):
    """STT 처리 요청 모델.
    
    Frontend가 Storage에 오디오를 업로드한 후 호출하는 API용.
    """
    video_id: str
    audio_storage_key: str  # Storage 내 경로 (예: "{video_id}/audio.wav")
    preprocess_job_id: Optional[str] = None
    provider: str = "clova"


class STTProcessResponse(BaseModel):
    status: str
    message: str
    stt_results_count: Optional[int] = None


@app.post("/stt/process", response_model=STTProcessResponse)
def process_stt_from_storage(request: STTProcessRequest) -> STTProcessResponse:
    """Storage에서 오디오를 다운받아 STT를 처리하고 결과를 DB에 저장한다.
    
    사용 예시:
        curl -X POST http://localhost:8001/stt/process \\
            -H "Content-Type: application/json" \\
            -d '{"video_id":"...", "audio_storage_key":"video_id/audio.wav"}'
    """
    from src.pipeline.stages import run_stt_from_storage
    
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 존재 확인
    video = adapter.get_video(request.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # 1. Storage에서 오디오 다운로드 → STT 실행
        stt_result = run_stt_from_storage(
            audio_storage_key=request.audio_storage_key,
            video_id=request.video_id,
            backend=request.provider,
        )
        
        # 2. STT 결과를 DB에 저장
        segments = stt_result.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        
        saved_rows = adapter.save_stt_result(
            video_id=request.video_id,
            segments=segments,
            preprocess_job_id=request.preprocess_job_id,
            provider=request.provider,
        )
        
        # 3. preprocessing_job 상태 업데이트 (있는 경우)
        if request.preprocess_job_id:
            adapter.update_preprocessing_job_status(request.preprocess_job_id, "DONE")
            adapter.update_video_status(request.video_id, "PREPROCESS_DONE")
        
        return STTProcessResponse(
            status="ok",
            message=f"STT processing completed. {len(saved_rows)} results saved.",
            stt_results_count=len(saved_rows),
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Audio file not found in storage: {e}")
    except Exception as e:
        # 에러 시 preprocessing_job 상태 업데이트
        if request.preprocess_job_id:
            adapter.update_preprocessing_job_status(
                request.preprocess_job_id, 
                "FAILED", 
                error_message=str(e)
            )
            adapter.update_video_status(request.video_id, "FAILED", error=str(e))
        raise HTTPException(status_code=500, detail=f"STT processing failed: {e}")
