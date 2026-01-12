"""Pre-DB 단계: mp4 -> STT + Capture.

- 입력(mp4): `data/inputs` (로컬 업로드 대체)
- 출력(DB 대체): `data/outputs/{video_name}`

이 단계는 ADK 밖(기존 로직 재사용)으로 두고, Root Agent 이후부터 ADK로 구성한다.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.audio.stt_router import STTRouter
from src.capture.process_content import process_single_video_capture


def _sync_capture_outputs(*, capture_root: Path, target_root: Path) -> None:
    """capture 모듈이 만든 산출물을 target_root로 복사한다.

    capture 모듈은 기본적으로 `output_base/{video_path.stem}`에 쓰기 때문에,
    sanitize된 video_name과 stem이 다른 경우에도 target_root를 일관되게 사용하기 위함.
    """

    src_manifest = capture_root / "manifest.json"
    src_captures = capture_root / "captures"

    if not src_manifest.exists() or not src_captures.exists():
        raise FileNotFoundError(
            f"capture 산출물을 찾을 수 없습니다: {src_manifest} / {src_captures}"
        )

    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_manifest, target_root / "manifest.json")

    target_captures = target_root / "captures"
    target_captures.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_captures, target_captures, dirs_exist_ok=True)


def run_stt(*, video_path: Path, output_stt_json: Path, backend: str) -> None:
    router = STTRouter(provider=backend)
    audio_output_path = output_stt_json.with_name(f"{video_path.stem}.wav")
    router.transcribe_media(
        video_path,
        provider=backend,
        audio_output_path=audio_output_path,
        mono_method="auto",
        output_path=output_stt_json,
    )


def run_capture(
    *,
    video_path: Path,
    output_base: Path,
    scene_threshold: float,
    dedupe_threshold: float,
    min_interval: float,
) -> List[Dict[str, Any]]:
    metadata = process_single_video_capture(
        str(video_path),
        str(output_base),
        scene_threshold=scene_threshold,
        dedupe_threshold=dedupe_threshold,
        min_interval=min_interval,
    )
    return metadata


def ensure_pre_db_artifacts(
    *,
    video_path: Path,
    video_root: Path,
    output_base: Path,
    stt_backend: str,
    parallel: bool,
    capture_threshold: float,
    capture_dedupe_threshold: float,
    capture_min_interval: float,
) -> Dict[str, Path]:
    """stt.json + manifest.json + captures/가 없으면 생성한다."""

    stt_json = video_root / "stt.json"
    manifest_json = video_root / "manifest.json"
    captures_dir = video_root / "captures"

    has_pre_db = stt_json.exists() and manifest_json.exists() and captures_dir.exists()
    if has_pre_db:
        return {
            "stt_json": stt_json,
            "manifest_json": manifest_json,
            "captures_dir": captures_dir,
        }


    video_root.mkdir(parents=True, exist_ok=True)

    stt_elapsed: Optional[float] = None
    capture_elapsed: Optional[float] = None

    def _timed(func, *args, **kwargs) -> Tuple[Any, float]:
        import time

        started = time.perf_counter()
        result = func(*args, **kwargs)
        return result, time.perf_counter() - started

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(
                _timed,
                run_stt,
                video_path=video_path,
                output_stt_json=stt_json,
                backend=stt_backend,
            )
            capture_future = executor.submit(
                _timed,
                run_capture,
                video_path=video_path,
                output_base=output_base,
                scene_threshold=capture_threshold,
                dedupe_threshold=capture_dedupe_threshold,
                min_interval=capture_min_interval,
            )
            _, stt_elapsed = stt_future.result()
            metadata, capture_elapsed = capture_future.result()
    else:
        _, stt_elapsed = _timed(
            run_stt,
            video_path=video_path,
            output_stt_json=stt_json,
            backend=stt_backend,
        )
        metadata, capture_elapsed = _timed(
            run_capture,
            video_path=video_path,
            output_base=output_base,
            scene_threshold=capture_threshold,
            dedupe_threshold=capture_dedupe_threshold,
            min_interval=capture_min_interval,
        )

    raw_capture_root = output_base / video_path.stem
    if raw_capture_root.resolve() != video_root.resolve():
        _sync_capture_outputs(capture_root=raw_capture_root, target_root=video_root)
    else:
        manifest_json.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    captures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "stt_json": stt_json,
        "manifest_json": manifest_json,
        "captures_dir": captures_dir,
    }
