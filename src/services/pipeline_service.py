from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.adk_chatbot.agent import root_agent as chatbot_root_agent
from src.adk_chatbot.paths import DEFAULT_OUTPUT_BASE, sanitize_video_name
from src.adk_chatbot.store import VideoStore
from src.run_preprocess_pipeline import run_preprocess_pipeline as run_preprocess_pipeline_job

from .adk_session import AdkMessage, AdkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (PROJECT_ROOT / DEFAULT_OUTPUT_BASE).resolve()


def get_default_output_base() -> Path:
    return DEFAULT_OUTPUT_ROOT


def _clear_preprocess_artifacts(
    *,
    output_base: Path,
    video_name: str,
    raw_stem: str,
) -> None:
    store = VideoStore(output_base=output_base, video_name=video_name)
    targets = [store.stt_json(), store.manifest_json()]
    for target in targets:
        if target.exists():
            target.unlink()

    captures_dir = store.captures_dir()
    if captures_dir.exists():
        shutil.rmtree(captures_dir)

    audio_path = store.video_root() / f"{raw_stem}.wav"
    if audio_path.exists():
        audio_path.unlink()


def run_preprocess_pipeline(
    *,
    video_path: Path,
    output_base: Path,
    stt_backend: str,
    parallel: bool,
    capture_threshold: float,
    capture_dedupe_threshold: float,
    capture_min_interval: float,
    force: bool = False,
) -> Dict[str, Any]:
    video_name = sanitize_video_name(video_path.stem)
    output_base = output_base.resolve()

    if force:
        _clear_preprocess_artifacts(
            output_base=output_base,
            video_name=video_name,
            raw_stem=video_path.stem,
        )

    run_preprocess_pipeline_job(
        video=str(video_path),
        output_base=str(output_base),
        stt_backend=stt_backend,
        parallel=parallel,
        capture_threshold=capture_threshold,
        capture_dedupe_threshold=capture_dedupe_threshold,
        capture_min_interval=capture_min_interval,
    )
    store = VideoStore(output_base=output_base, video_name=video_name)
    meta_path = store.pipeline_run_json()
    meta_payload: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta_payload = {}
    return {
        "video_name": video_name,
        "video_root": str(store.video_root()),
        "run_meta": meta_payload,
    }


def build_adk_state(
    *,
    video_name: str,
    force_preprocessing: bool,
    max_reruns: int,
    vlm_batch_size: Optional[int],
    vlm_concurrency: int,
    vlm_show_progress: bool,
    judge_min_score: float,
) -> Dict[str, Any]:
    return {
        "video_name": video_name,
        "force_preprocessing": force_preprocessing,
        "max_reruns": max_reruns,
        "vlm_batch_size": vlm_batch_size,
        "vlm_concurrency": vlm_concurrency,
        "vlm_show_progress": vlm_show_progress,
        "judge_min_score": judge_min_score,
    }


def start_adk_session(
    *,
    state: Dict[str, Any],
    app_name: str = "screentime_pipeline",
    user_id: str = "streamlit",
) -> AdkSession:
    return AdkSession(
        root_agent=chatbot_root_agent,
        app_name=app_name,
        user_id=user_id,
        initial_state=state,
    )


def send_adk_message(session: AdkSession, message: str) -> List[AdkMessage]:
    return session.send_message(message)
