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
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.run_process_pipeline import run_processing_pipeline
import json


app = FastAPI(title="Screentime Processing API")


class ProcessRequest(BaseModel):
    video_name: Optional[str] = None
    video_id: Optional[str] = None
    output_base: str = "data/outputs"
    batch_mode: Optional[bool] = None
    limit: Optional[int] = None
    force_db: bool = False
    sync_to_db: bool = False


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
