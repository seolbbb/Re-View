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
import base64
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Request, Response
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
def process_pipeline(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
) -> ProcessResponse:
    if not request.video_name and not request.video_id:
        raise HTTPException(status_code=400, detail="video_name or video_id is required.")

    # video_id 기반 요청만 인증/권한 체크 (video_name-only는 기존 dev/local 사용을 위해 허용)
    if request.video_id:
        adapter = get_supabase_adapter()
        if not adapter:
            raise HTTPException(status_code=503, detail="Database not configured")

        if _is_internal_request(http_request):
            video = _require_video_exists(adapter, request.video_id)
        else:
            user_id = _require_user_id(adapter, http_request)
            video = _require_video_owner(adapter, user_id=user_id, video_id=request.video_id)

        # 이미 PROCESSING 중이면 중복 실행 방지
        if (video.get("status") or "").upper() == "PROCESSING":
            raise HTTPException(
                status_code=409,
                detail="Pipeline is already running for this video",
            )

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
        "video_name": video.get("name"),
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

    # pipeline_mode 판별: preprocess_job과 processing_job이 동시에 존재하며
    # preprocess_job이 아직 완료되지 않았으면 async
    pp_job = result.get("preprocess_job")
    proc_job = result.get("processing_job")
    if pp_job and proc_job and (pp_job.get("status") or "").upper() not in ("DONE", ""):
        result["pipeline_mode"] = "async"
    else:
        result["pipeline_mode"] = "sequential"

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
def get_video_status(video_id: str, http_request: Request) -> Dict[str, Any]:
    """비디오 상태 및 현재 작업 정보 조회."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    if _is_internal_request(http_request):
        _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        _require_video_owner(adapter, user_id=user_id, video_id=video_id)

    result = _build_status_payload(adapter, video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return result


@app.get("/videos/{video_id}/status/stream")
def stream_video_status(video_id: str, http_request: Request):
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
    if _is_internal_request(http_request):
        _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        _require_video_owner(adapter, user_id=user_id, video_id=video_id)

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
def get_video_progress(video_id: str, http_request: Request) -> Dict[str, Any]:
    """처리 진행률 조회 (폴링용)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 정보 조회
    if _is_internal_request(http_request):
        video = _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        video = _require_video_owner(adapter, user_id=user_id, video_id=video_id)
    
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
def get_video_summary(video_id: str, http_request: Request, format: Optional[str] = None) -> Dict[str, Any]:
    """최신 요약 결과 조회 (IN_PROGRESS 포함)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 존재 확인
    if _is_internal_request(http_request):
        video = _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        video = _require_video_owner(adapter, user_id=user_id, video_id=video_id)
    
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
def get_video_summaries(video_id: str, http_request: Request) -> Dict[str, Any]:
    """세부 요약 세그먼트 리스트 조회 (Chatbot용)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    if _is_internal_request(http_request):
        _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        _require_video_owner(adapter, user_id=user_id, video_id=video_id)

    return _build_summaries_payload(adapter, video_id)


@app.get("/videos/{video_id}/evidence")
def get_video_evidence(
    video_id: str,
    http_request: Request,
    stt_ids: Optional[str] = None,
    cap_ids: Optional[str] = None,
) -> Dict[str, Any]:
    """Evidence lookup by STT/VLM IDs (Chatbot thinking mode)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    if _is_internal_request(http_request):
        _require_video_exists(adapter, video_id)
    else:
        user_id = _require_user_id(adapter, http_request)
        _require_video_owner(adapter, user_id=user_id, video_id=video_id)

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
    user_id = _require_user_id(adapter, http_request)
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
    user_id = _require_user_id(adapter, http_request)
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
    pipeline_mode: str = "async"


class UploadCompleteResponse(BaseModel):
    video_id: str
    status: str


@app.post("/api/videos/upload/complete", response_model=UploadCompleteResponse)
def complete_upload(
    request: UploadCompleteRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """Storage 업로드 완료 알림 → 파이프라인 실행."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    video = _require_video_owner(adapter, user_id=user_id, video_id=request.video_id)

    # Safety: don't allow arbitrary keys outside the video's prefix.
    if not request.storage_key.startswith(f"{request.video_id}/"):
        raise HTTPException(status_code=400, detail="Invalid storage_key")

    expected_storage_key = video.get("video_storage_key")
    if expected_storage_key and expected_storage_key != request.storage_key:
        raise HTTPException(status_code=400, detail="storage_key does not match the video")

    adapter.update_video_status(request.video_id, "PREPROCESSING")

    if request.pipeline_mode == "async":
        background_tasks.add_task(
            _run_async_pipeline_from_storage,
            request.video_id,
            request.storage_key,
        )
    else:
        background_tasks.add_task(
            _run_full_pipeline_from_storage,
            request.video_id,
            request.storage_key,
        )

    return UploadCompleteResponse(
        video_id=request.video_id,
        status="PREPROCESSING",
    )


def _download_from_storage(adapter, video_id: str, storage_key: str) -> tuple:
    """스토리지(R2/Supabase)에서 비디오를 임시 디렉토리로 다운로드.

    Returns:
        (tmp_dir, tmp_path): 임시 디렉토리 경로와 다운로드된 파일 경로.
        호출자가 finally에서 tmp_dir를 정리해야 합니다.
    """
    tmp_dir = tempfile.mkdtemp()
    original_name = Path(storage_key).name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    tmp_filename = f"{timestamp}_{original_name}"
    tmp_path = Path(tmp_dir) / tmp_filename

    if adapter.s3_client:
        adapter.s3_client.download_file(adapter.r2_bucket, storage_key, str(tmp_path))
        print(f"[R2] Downloaded video from {adapter.r2_bucket}/{storage_key}")
    else:
        if getattr(adapter, "r2_only", False):
            raise RuntimeError("R2 storage is required (check R2_* env vars)")
        file_data = adapter.client.storage.from_("videos").download(storage_key)
        tmp_path.write_bytes(file_data)

    return tmp_dir, tmp_path


def _run_full_pipeline_from_storage(video_id: str, storage_key: str) -> None:
    """스토리지에서 비디오를 다운로드하여 전처리 → 처리 파이프라인을 순차 실행."""
    from src.run_preprocess_pipeline import run_preprocess_pipeline

    adapter = get_supabase_adapter()
    tmp_dir = None
    try:
        print("\n" + "=" * 60)
        print(f"  Re:View API Pipeline (Storage): {video_id}")
        print("=" * 60)

        tmp_dir, tmp_path = _download_from_storage(adapter, video_id, storage_key)

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


def _run_async_pipeline_from_storage(video_id: str, storage_key: str) -> None:
    """비동기 파이프라인으로 전처리+분석을 병렬 실행."""
    import asyncio
    from src.run_pipeline_demo_async import run_async_demo, _apply_qwen_vlm_overrides, _load_pipeline_defaults, _load_yaml_runtime_defaults

    adapter = get_supabase_adapter()
    tmp_dir = None
    try:
        print("\n" + "=" * 60)
        print(f"  Re:View API Async Pipeline (Storage): {video_id}")
        print("=" * 60)

        tmp_dir, tmp_path = _download_from_storage(adapter, video_id, storage_key)

        # 2. 비디오 duration 업데이트
        _update_video_duration(adapter, video_id, str(tmp_path))

        # 3. Qwen VLM override 적용
        ROOT = Path(__file__).resolve().parents[1]
        _apply_qwen_vlm_overrides(ROOT)

        # 4. 비동기 파이프라인 실행
        defaults = _load_yaml_runtime_defaults()
        output_base = Path(defaults.get("output_base", "data/outputs"))
        if not output_base.is_absolute():
            output_base = (ROOT / output_base).resolve()
        else:
            output_base = output_base.resolve()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        asyncio.run(
            run_async_demo(
                video_path=tmp_path,
                output_base=output_base,
                run_id=run_id,
                stt_backend=str(defaults.get("stt_backend", "clova")),
                capture_batch_size=int(defaults.get("capture_batch_size", 6)),
                vlm_parallelism=int(defaults.get("vlm_parallelism", 3)),
                vlm_inner_concurrency=int(defaults.get("vlm_inner_concurrency", 1)),
                vlm_batch_size=int(defaults.get("vlm_batch_size", 6)),
                vlm_show_progress=bool(defaults.get("vlm_show_progress", True)),
                max_inflight_chunks=8,
                queue_maxsize=0,
                strict_batch_order=True,
                sync_to_db=True,
                upload_video_to_r2=False,
                upload_audio_to_r2=True,
                existing_video_id=video_id,
            )
        )
    except Exception as exc:
        if adapter:
            adapter.update_video_status(video_id, "FAILED", error=str(exc))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/videos")
def list_videos(http_request: Request) -> Dict[str, Any]:
    """비디오 목록 조회 (최신순)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    videos = adapter.list_videos_for_user(user_id)
    return {"videos": videos}


class MediaTicketResponse(BaseModel):
    ticket: str
    expires_in: int


@app.post("/api/media/ticket", response_model=MediaTicketResponse)
def issue_media_ticket(http_request: Request) -> MediaTicketResponse:
    """Issue a short-lived, signed token for media endpoints (stream/thumbnail)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    ttl_sec = _MEDIA_TICKET_TTL_SEC
    ticket = _create_media_ticket(user_id, ttl_sec=ttl_sec)
    return MediaTicketResponse(ticket=ticket, expires_in=ttl_sec)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(video_id: str, http_request: Request) -> Response:
    """Delete a user's video (DB rows + R2 objects under {video_id}/)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    video = _require_video_owner(adapter, user_id=user_id, video_id=video_id)

    if _is_video_actively_processing(adapter, video):
        raise HTTPException(status_code=409, detail="Video is still processing")

    # Storage cleanup first so failures are retryable (DB row remains).
    try:
        storage_result = _delete_storage_objects_for_video(adapter, video_id, video)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Storage delete failed: {exc}")

    errors = storage_result.get("errors") or []
    if errors:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete storage objects ({len(errors)} errors)",
        )

    # DB delete (FK cascades will clean up children).
    if not adapter.delete_video(video_id, user_id):
        raise HTTPException(status_code=500, detail="Failed to delete video")

    return Response(status_code=204)


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str, http_request: Request, ticket: Optional[str] = None):
    """비디오 파일 서빙 (Storage signed URL 또는 로컬 fallback)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id_from_request_or_ticket(adapter, http_request, ticket)
    video = _require_video_owner(adapter, user_id=user_id, video_id=video_id)

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
def get_video_thumbnail(video_id: str, http_request: Request, ticket: Optional[str] = None):
    """비디오 썸네일 이미지 서빙 (captures 버킷 signed URL 또는 로컬 fallback)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id_from_request_or_ticket(adapter, http_request, ticket)
    video = _require_video_owner(adapter, user_id=user_id, video_id=video_id)

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


def _get_or_create_chat_session(
    request: ChatRequest,
    *,
    video: Dict[str, Any],
    user_id: str,
):
    session = None
    if request.session_id:
        session = _chat_sessions.get(request.session_id)
    if session:
        # Safety: do not allow session reuse across different videos.
        state = getattr(session, "_state", None)
        if isinstance(state, dict):
            session_video_id = state.get("video_id")
            if session_video_id and str(session_video_id) != str(request.video_id):
                session = None

    if session:
        # Update reasoning_mode if provided in request
        if request.reasoning_mode and request.reasoning_mode in ("flash", "thinking"):
            session._state["reasoning_mode"] = request.reasoning_mode
        return session

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
        user_id=user_id,
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


# Default to 1h so video playback (seek/range requests) remains stable for longer sessions.
_MEDIA_TICKET_TTL_SEC = int(os.getenv("MEDIA_TICKET_TTL_SEC", "3600"))
_DELETE_STALE_PREPROCESS_SEC = int(os.getenv("DELETE_STALE_PREPROCESS_SEC", "1800"))  # 30m
_DELETE_STALE_PROCESSING_SEC = int(os.getenv("DELETE_STALE_PROCESSING_SEC", "7200"))  # 2h

_warned_media_ticket_secret_fallback = False
_warned_internal_api_token_fallback = False


def _media_ticket_secret() -> str:
    # Prefer a dedicated secret, but fall back to SUPABASE_KEY (server-side secret) so
    # local/dev setups still work without extra config.
    global _warned_media_ticket_secret_fallback
    secret = os.getenv("MEDIA_TICKET_SECRET")
    if secret:
        return secret
    fallback = os.getenv("SUPABASE_KEY") or ""
    if fallback and not _warned_media_ticket_secret_fallback:
        role = _supabase_jwt_role(fallback)
        msg = "Warning: MEDIA_TICKET_SECRET is not set. Falling back to SUPABASE_KEY"
        if role:
            msg += f" (role={role})"
        print(msg)
        _warned_media_ticket_secret_fallback = True
    return fallback


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _supabase_jwt_role(token: str) -> Optional[str]:
    """Best-effort decode of a Supabase JWT to extract the 'role' claim.

    This is used only for warning logs when we fall back to SUPABASE_KEY.
    """
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_raw = _b64url_decode(parts[1]).decode("utf-8")
        payload = json.loads(payload_raw)
        role = payload.get("role")
        return str(role) if role else None
    except Exception:
        return None


def _create_media_ticket(user_id: str, *, ttl_sec: int = _MEDIA_TICKET_TTL_SEC) -> str:
    if not user_id:
        raise ValueError("user_id is required")
    secret = _media_ticket_secret().encode("utf-8")
    if not secret:
        raise RuntimeError("MEDIA_TICKET_SECRET (or SUPABASE_KEY) is required")

    exp = int(time.time()) + int(ttl_sec)
    payload = {"uid": user_id, "exp": exp}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{payload_b64}.{sig_b64}"


def _verify_media_ticket(ticket: str) -> Optional[str]:
    if not ticket:
        return None
    secret = _media_ticket_secret().encode("utf-8")
    if not secret:
        return None
    try:
        payload_b64, sig_b64 = ticket.split(".", 1)
        expected_sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).digest()
        expected_sig_b64 = _b64url_encode(expected_sig)
        if not hmac.compare_digest(expected_sig_b64, sig_b64):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = int(payload.get("exp") or 0)
        if exp <= int(time.time()):
            return None
        uid = payload.get("uid")
        return str(uid) if uid else None
    except Exception:
        return None


def _require_user_id(adapter, request: Request) -> str:
    user_id = _get_user_id_from_request(adapter, request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id


def _require_user_id_from_request_or_ticket(adapter, request: Request, ticket: Optional[str]) -> str:
    user_id = _get_user_id_from_request(adapter, request) or _verify_media_ticket(ticket or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_id


def _require_video_owner(adapter, *, user_id: str, video_id: str) -> Dict[str, Any]:
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(video.get("user_id") or "") != str(user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    return video


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _record_age_sec(record: Dict[str, Any]) -> Optional[int]:
    now = datetime.now(timezone.utc)
    # Prefer updated_at as a heartbeat signal (processing jobs update progress frequently).
    for key in ("updated_at", "started_at", "created_at"):
        dt = _parse_dt(record.get(key))
        if not dt:
            continue
        return int((now - dt).total_seconds())
    return None


def _is_video_actively_processing(adapter, video: Dict[str, Any]) -> bool:
    """Return True if a video appears to have an active (non-stale) running job.

    This is used to prevent deletes while a pipeline job is *actually* running, while
    still allowing deletes for stuck videos that were force-canceled and never
    transitioned to DONE/FAILED.
    """
    pre_id = video.get("current_preprocess_job_id")
    if pre_id:
        try:
            pre = adapter.get_preprocessing_job(pre_id)
        except Exception:
            pre = None
        if pre and (str(pre.get("status") or "").upper() == "RUNNING"):
            age = _record_age_sec(pre)
            # If we can't determine age, be conservative and block deletion.
            if age is None or age < _DELETE_STALE_PREPROCESS_SEC:
                return True

    proc_id = video.get("current_processing_job_id")
    if proc_id:
        try:
            proc = adapter.get_processing_job(proc_id)
        except Exception:
            proc = None
        if proc:
            status = str(proc.get("status") or "").upper()
            if status in ("VLM_RUNNING", "SUMMARY_RUNNING", "JUDGE_RUNNING"):
                age = _record_age_sec(proc)
                if age is None or age < _DELETE_STALE_PROCESSING_SEC:
                    return True

    return False


def _dedupe_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in paths:
        if not raw:
            continue
        value = str(raw)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _supabase_storage_remove_paths(
    adapter,
    *,
    bucket: str,
    paths: List[str],
    chunk_size: int = 100,
) -> Dict[str, Any]:
    """Best-effort delete a list of object paths from Supabase Storage."""
    paths = _dedupe_paths(paths)
    total = len(paths)
    if total == 0:
        return {"bucket": bucket, "total": 0, "deleted": 0, "errors": []}

    deleted = 0
    errors: List[Dict[str, Any]] = []
    for i in range(0, total, chunk_size):
        chunk = paths[i : i + chunk_size]
        try:
            # supabase-py storage API expects a list of file paths.
            adapter.client.storage.from_(bucket).remove(chunk)
            deleted += len(chunk)
        except Exception as exc:
            errors.append({"bucket": bucket, "code": "RemoveFailed", "message": str(exc)})

    return {"bucket": bucket, "total": total, "deleted": deleted, "errors": errors}


def _delete_storage_objects_for_video(adapter, video_id: str, video: Dict[str, Any]) -> Dict[str, Any]:
    """Delete storage objects for a given video.

    - R2 configured: delete everything under `{video_id}/` prefix (covers video/captures/audios).
    - R2 not configured:
        - if r2_only: raise
        - else: best-effort delete known paths from Supabase Storage buckets.
    """
    if getattr(adapter, "s3_client", None):
        return adapter.r2_delete_prefix(f"{video_id}/")

    if getattr(adapter, "r2_only", False):
        raise RuntimeError("R2 storage is required (check R2_* env vars)")

    prefix = f"{video_id}/"

    # videos bucket
    video_key = video.get("video_storage_key")
    video_paths = [video_key] if video_key and str(video_key).startswith(prefix) else []

    # captures bucket (best-effort; ignore missing rows)
    capture_paths: List[str] = []
    try:
        caps = (
            adapter.client.table("captures")
            .select("storage_path")
            .eq("video_id", video_id)
            .execute()
        )
        for row in caps.data or []:
            p = row.get("storage_path") if isinstance(row, dict) else None
            if p and str(p).startswith(prefix):
                capture_paths.append(str(p))
    except Exception:
        pass

    # audio bucket (preprocessing_jobs.audio_storage_key)
    audio_paths: List[str] = []
    try:
        jobs = (
            adapter.client.table("preprocessing_jobs")
            .select("audio_storage_key")
            .eq("video_id", video_id)
            .execute()
        )
        for row in jobs.data or []:
            p = row.get("audio_storage_key") if isinstance(row, dict) else None
            if p and str(p).startswith(prefix):
                audio_paths.append(str(p))
    except Exception:
        pass

    results = [
        _supabase_storage_remove_paths(adapter, bucket="videos", paths=video_paths),
        _supabase_storage_remove_paths(adapter, bucket="captures", paths=capture_paths),
        _supabase_storage_remove_paths(adapter, bucket="audio", paths=audio_paths),
    ]

    total = sum(r.get("total", 0) for r in results)
    deleted = sum(r.get("deleted", 0) for r in results)
    errors: List[Dict[str, Any]] = []
    for r in results:
        errors.extend(r.get("errors") or [])

    return {"total": total, "deleted": deleted, "errors": errors}


def _internal_api_token() -> str:
    # Used for server-to-server calls (e.g. chatbot code calling back into this API).
    global _warned_internal_api_token_fallback
    configured = os.getenv("PROCESS_API_INTERNAL_TOKEN") or os.getenv("INTERNAL_API_TOKEN")
    if configured:
        return configured
    fallback = os.getenv("SUPABASE_KEY") or ""
    if fallback and not _warned_internal_api_token_fallback:
        role = _supabase_jwt_role(fallback)
        msg = "Warning: PROCESS_API_INTERNAL_TOKEN is not set. Falling back to SUPABASE_KEY"
        if role:
            msg += f" (role={role})"
        print(msg)
        _warned_internal_api_token_fallback = True
    return fallback


def _is_internal_request(request: Request) -> bool:
    expected = _internal_api_token()
    if not expected:
        return False
    token = request.headers.get("x-internal-token") or request.headers.get("X-Internal-Token") or ""
    if not token:
        return False
    return hmac.compare_digest(token, expected)


def _require_video_exists(adapter, video_id: str) -> Dict[str, Any]:
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """LangGraph 기반 챗봇 API (비스트리밍)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    video = _require_video_owner(adapter, user_id=user_id, video_id=request.video_id)
    session = _get_or_create_chat_session(request, video=video, user_id=user_id)
    messages = session.send_message(request.message)
    response_text = "".join([getattr(msg, "text", "") for msg in messages if msg])
    return ChatResponse(
        response=response_text,
        session_id=session.session_id,
    )


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    """LangGraph 기반 챗봇 API (SSE 스트리밍)."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    user_id = _require_user_id(adapter, http_request)
    video = _require_video_owner(adapter, user_id=user_id, video_id=request.video_id)
    session = _get_or_create_chat_session(request, video=video, user_id=user_id)

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
def process_stt_from_storage(request: STTProcessRequest, http_request: Request) -> STTProcessResponse:
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
    user_id = _require_user_id(adapter, http_request)
    video = _require_video_owner(adapter, user_id=user_id, video_id=request.video_id)
    
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
