"""ffmpeg로 미디어에서 오디오 트랙을 추출한다."""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(
    media_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    mono_method: str = "auto",
) -> Path:
    """미디어 파일에서 오디오를 추출해 파일로 저장한다."""
    media_path = Path(media_path).expanduser()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    output_path = Path(output_path) if output_path else Path("src/data/input") / f"{media_path.stem}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mono_method in ("left", "right", "phase-fix", "auto") and channels != 1:
        raise ValueError("mono-method requires channels=1.")

    if mono_method == "auto":
        mono_method = _select_best_mono_method(media_path, sample_rate=sample_rate)

    command = ["ffmpeg", "-y", "-i", str(media_path), "-vn"]
    mono_filter = _mono_filter_for_method(mono_method)
    if mono_filter:
        command += ["-af", mono_filter]

    command += [
        "-acodec",
        codec,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg executable not found. Please install FFmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {message}") from exc

    return output_path


def _mono_filter_for_method(mono_method: str) -> str | None:
    """모노 변환 방식에 따른 ffmpeg 필터 문자열을 반환한다."""
    if mono_method == "downmix":
        return "pan=mono|c0=0.5*c0+0.5*c1"
    if mono_method == "left":
        return "pan=mono|c0=c0"
    if mono_method == "right":
        return "pan=mono|c0=c1"
    if mono_method == "phase-fix":
        return "pan=mono|c0=0.5*c0-0.5*c1"
    return None


def _select_best_mono_method(media_path: Path, *, sample_rate: int, probe_seconds: int = 60) -> str:
    """여러 모노 방식 중 평균 볼륨이 가장 큰 방식을 선택한다."""
    candidates = ["downmix", "left", "right", "phase-fix"]
    volumes: dict[str, float] = {}
    for candidate in candidates:
        volume = _measure_mean_volume(
            media_path, mono_method=candidate, sample_rate=sample_rate, probe_seconds=probe_seconds
        )
        if volume is None:
            print(f"[INFO] mono-method candidate={candidate} mean_volume=NA")
            continue
        volumes[candidate] = volume
        print(f"[INFO] mono-method candidate={candidate} mean_volume={volume} dB")

    if not volumes:
        return "downmix"

    best_method = max(volumes, key=volumes.get)
    print(f"[INFO] mono-method auto selected: {best_method} (mean_volume={volumes[best_method]} dB)")
    return best_method


def _measure_mean_volume(
    media_path: Path,
    *,
    mono_method: str,
    sample_rate: int,
    probe_seconds: int,
) -> float | None:
    """지정된 모노 방식으로 일정 구간의 평균 볼륨을 측정한다."""
    filter_chain = []
    mono_filter = _mono_filter_for_method(mono_method)
    if mono_filter:
        filter_chain.append(mono_filter)
    filter_chain.append("volumedetect")

    command = [
        "ffmpeg",
        "-i",
        str(media_path),
        "-vn",
        "-t",
        str(probe_seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        ",".join(filter_chain),
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not result.stderr:
        return None
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            value = line.split("mean_volume:", 1)[1].strip().split(" ")[0]
            if value == "-inf":
                return -9999.0
            try:
                return float(value)
            except ValueError:
                return None
    return None

