"""처리 파이프라인을 API로 실행하는 FastAPI 래퍼.

사용 방법:
서버 실행: uvicorn src.process_api:app --host 0.0.0.0 --port 8000
-> 서버가 실행이 되었으면 아래 예시와 같이 뜸
INFO:     Started server process [92286]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

다른 터미널을 추가로 열고 아래 요청 예시를 입력, 전처리가 되어 있어야 함
POST 실행
요청 예시:
  curl -X POST http://localhost:8000/process \\
    -H "Content-Type: application/json" \\
    -d '{"video_name":"diffusion","force_db":true}'

diffusion.mp4이면 video_name은 diffusion
force_db를 사용하면 supabase에서 정보를 가져옴
db 설정하기 어렵다 -> force_db true 빼고 실행하면 로컬에 저장된 전처리 정보를 가져와서 실행함

GET 실행
서버 체크:
    curl http://localhost:8000/health
마지막 실행 로그:
    curl http://localhost:8000/runs/diffusion
비디오 상태:
    curl http://localhost:8000/videos/{video_id}/status
처리 진행률:
    curl http://localhost:8000/videos/{video_id}/progress
최신 요약:
    curl http://localhost:8000/videos/{video_id}/summary
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.db import get_supabase_adapter
from src.run_process_pipeline import run_processing_pipeline


app = FastAPI(title="Screentime Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/videos/{video_id}/status")
def get_video_status(video_id: str) -> Dict[str, Any]:
    """비디오 상태 및 현재 작업 정보 조회."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")
    
    # 비디오 정보 조회
    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
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
        
    # summaries 테이블 조회 (segments 조인됨)
    rows = adapter.get_summaries(video_id)
    
    # 응답 포맷 구성
    items = []
    for r in rows:
        seg_info = r.get("segments") or {}
        items.append({
            "summary_id": r.get("id"),
            "segment_id": r.get("segment_id"),
            "segment_index": seg_info.get("segment_index"),
            "start_ms": seg_info.get("start_ms"),
            "end_ms": seg_info.get("end_ms"),
            "summary": r.get("summary"),  # JSONB
            "created_at": r.get("created_at"),
        })
        
    # segment_index 기준 정렬
    items.sort(key=lambda x: x.get("segment_index") or 0)
    
    return {
        "video_id": video_id,
        "count": len(items),
        "items": items,
    }


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
    """파일명에서 안전하지 않은 문자를 제거합니다."""
    return re.sub(r'[^\w\-.]', '_', name)


def _run_full_pipeline(video_path: str, video_id: str) -> None:
    """전처리 → 처리 파이프라인을 순차 실행합니다 (BackgroundTasks용)."""
    from src.run_preprocess_pipeline import run_preprocess_pipeline

    adapter = get_supabase_adapter()
    try:
        # 1. 전처리
        run_preprocess_pipeline(
            video=video_path,
            sync_to_db=True,
            write_local_json=True,
        )
        if adapter:
            adapter.update_video_status(video_id, "PREPROCESS_DONE")

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
    video = adapter.create_video(
        name=video_name,
        original_filename=file.filename,
    )
    video_id = video["id"]
    adapter.update_video_status(video_id, "PREPROCESSING")

    background_tasks.add_task(_run_full_pipeline, str(save_path), video_id)

    return UploadResponse(
        video_id=video_id,
        video_name=video_name,
        status="PREPROCESSING",
    )


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
    """비디오 파일 서빙."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    original_filename = video.get("original_filename", "")

    # data/inputs/ 디렉토리에서 파일 검색
    if INPUT_DIR.exists():
        for f in sorted(INPUT_DIR.iterdir(), reverse=True):
            if f.name.endswith(original_filename) or original_filename in f.name:
                return FileResponse(
                    path=str(f),
                    media_type="video/mp4",
                    filename=original_filename,
                )

    raise HTTPException(status_code=404, detail="Video file not found on disk")


@app.get("/api/videos/{video_id}/thumbnail")
def get_video_thumbnail(video_id: str):
    """비디오 썸네일 이미지 서빙."""
    adapter = get_supabase_adapter()
    if not adapter:
        raise HTTPException(status_code=503, detail="Database not configured")

    video = adapter.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

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


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """챗봇 API (placeholder — LangGraph 구현 전)."""
    session_id = request.session_id or str(uuid.uuid4())
    return ChatResponse(
        response="챗봇 준비 중입니다. LangGraph 연결 후 실제 답변이 제공됩니다.",
        session_id=session_id,
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
