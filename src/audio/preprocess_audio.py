#!/usr/bin/env python3
"""
ffmpeg 기반 안전한 mono WAV 전처리.

입력 MP4/WAV에서 다음 4가지 후보를 만들고 volumedetect로 음량을 비교해
가장 유효한(덜 음수인) mean_volume 후보를 output.wav로 선택합니다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MEAN_VOLUME_SILENCE_THRESHOLD_DB = -50.0
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


@dataclass
class CandidateResult:
    key: str
    label: str
    output_path: Path
    mean_volume_db: Optional[float] = None
    max_volume_db: Optional[float] = None
    error: Optional[str] = None


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} 실행 파일을 찾을 수 없습니다. 설치 후 PATH에 추가하세요.")
    return path


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _get_audio_stream_info(ffprobe: str, input_path: Path) -> Tuple[int, int]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index,channels",
        "-of",
        "json",
        str(input_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 실패:\n{proc.stderr}")
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        return 0, 0
    first = streams[0] if isinstance(streams[0], dict) else {}
    channels = int(first.get("channels", 0) or 0)
    return len(streams), channels


def _extract_candidate(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    audio_filter: Optional[str],
) -> None:
    cmd = [ffmpeg, "-y", "-i", str(input_path), "-vn"]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += [
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg 변환 실패")


def _parse_volume(stderr_text: str) -> Tuple[Optional[float], Optional[float]]:
    mean_match = re.search(r"mean_volume:\s*(-?inf|-?\d+(\.\d+)?)\s*dB", stderr_text)
    max_match = re.search(r"max_volume:\s*(-?inf|-?\d+(\.\d+)?)\s*dB", stderr_text)

    def _to_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        if value == "-inf":
            return -9999.0
        try:
            return float(value)
        except ValueError:
            return None

    mean_db = _to_float(mean_match.group(1) if mean_match else None)
    max_db = _to_float(max_match.group(1) if max_match else None)
    return mean_db, max_db


def _measure_volume(ffmpeg: str, input_path: Path) -> Tuple[Optional[float], Optional[float]]:
    cmd = [ffmpeg, "-hide_banner", "-i", str(input_path), "-af", "volumedetect", "-f", "null", "-"]
    proc = _run(cmd)
    if proc.returncode != 0:
        return None, None
    return _parse_volume(proc.stderr)


def _build_candidates(
    channels: int,
    temp_dir: Path,
) -> Tuple[List[CandidateResult], Dict[str, str]]:
    right_ref = "c1" if channels >= 2 else "c0"
    phase_ref = "0.5*c0-0.5*c1" if channels >= 2 else "c0"

    return (
        [
        CandidateResult(
            key="A",
            label="standard_downmix",
            output_path=temp_dir / "candidate_A.wav",
        ),
        CandidateResult(
            key="B",
            label="left_only",
            output_path=temp_dir / "candidate_B.wav",
        ),
        CandidateResult(
            key="C",
            label="right_only",
            output_path=temp_dir / "candidate_C.wav",
        ),
        CandidateResult(
            key="D",
            label="phase_fixed",
            output_path=temp_dir / "candidate_D.wav",
        ),
        ],
        {
            "B": "pan=mono|c0=c0",
            "C": f"pan=mono|c0={right_ref}",
            "D": f"pan=mono|c0={phase_ref}",
        },
    )


def _select_best(candidates: List[CandidateResult]) -> CandidateResult:
    valid = [c for c in candidates if c.mean_volume_db is not None and c.error is None]
    if not valid:
        raise RuntimeError("유효한 후보를 만들지 못했습니다. 입력 오디오를 확인하세요.")

    non_silent = [c for c in valid if c.mean_volume_db >= MEAN_VOLUME_SILENCE_THRESHOLD_DB]
    pool = non_silent or valid
    best = max(pool, key=lambda c: c.mean_volume_db if c.mean_volume_db is not None else -9999.0)
    return best


def preprocess_audio(input_path: Path, output_path: Path) -> None:
    ffmpeg = _require_binary("ffmpeg")
    ffprobe = _require_binary("ffprobe")

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    stream_count, channels = _get_audio_stream_info(ffprobe, input_path)
    if stream_count < 1:
        raise RuntimeError("입력 파일에 오디오 트랙이 없습니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="audio_preprocess_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        candidates, filter_map = _build_candidates(channels, temp_dir)

        for cand in candidates:
            try:
                audio_filter = None
                if cand.key in filter_map:
                    audio_filter = filter_map[cand.key]
                _extract_candidate(ffmpeg, input_path, cand.output_path, audio_filter=audio_filter)
                mean_db, max_db = _measure_volume(ffmpeg, cand.output_path)
                cand.mean_volume_db = mean_db
                cand.max_volume_db = max_db
            except Exception as exc:
                cand.error = str(exc)

        print("[전처리] 후보별 mean_volume(dBFS):")
        for cand in candidates:
            if cand.error:
                print(f"  - {cand.key} {cand.label}: 실패 ({cand.error})")
            else:
                print(
                    f"  - {cand.key} {cand.label}: mean={cand.mean_volume_db}, max={cand.max_volume_db}"
                )

        best = _select_best(candidates)
        if best.mean_volume_db is not None and best.mean_volume_db < MEAN_VOLUME_SILENCE_THRESHOLD_DB:
            print(
                f"[경고] 모든 후보가 무음 기준({MEAN_VOLUME_SILENCE_THRESHOLD_DB} dBFS)보다 낮습니다."
            )

        output_path.write_bytes(best.output_path.read_bytes())
        print(
            f"[OK] 선택된 후보: {best.key} ({best.label}), output={output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ffmpeg 기반 안전한 mono 오디오 전처리")
    parser.add_argument("input", help="입력 미디어 파일 경로 (mp4/wav)")
    parser.add_argument("output", help="출력 WAV 경로 (mono/16kHz/pcm_s16le)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_audio(Path(args.input).expanduser(), Path(args.output).expanduser())


if __name__ == "__main__":
    main()
