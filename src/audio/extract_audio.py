# 실행: python src/audio/extract_audio.py --media-path src/data/input/screentime-mvp-video.mp4
# 옵션: --output-path src/data/input/sample.wav (기본: src/data/input/<입력파일명>.wav)
# 옵션: --sample-rate 16000 (샘플레이트 Hz, 기본 16000)
# 옵션: --channels 1 (오디오 채널 수, 기본 1=모노)
# 옵션: --codec pcm_s16le (오디오 코덱, 기본 WAV용 pcm_s16le)
# 옵션: --mono-method downmix|left|right|phase-fix|auto (모노 생성 방식, 기본: auto)
# 참고: ffmpeg 설치 필요
"""Extract audio track from a media file using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

from src.audio.settings import load_audio_settings


def default_output_path(media_path: Path) -> Path:
    return Path("src/data/input") / f"{media_path.stem}.wav"


def extract_audio(
    media_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    codec: Optional[str] = None,
    mono_method: Optional[str] = None,
) -> Path:
    settings = load_audio_settings()
    extract_settings = settings.get("extract", {})
    if not isinstance(extract_settings, dict):
        raise ValueError("extract 설정 형식이 올바르지 않습니다(맵이어야 함).")

    if sample_rate is None:
        sample_rate = int(extract_settings.get("sample_rate", 16000))
    if channels is None:
        channels = int(extract_settings.get("channels", 1))
    if codec is None:
        codec = str(extract_settings.get("codec", "pcm_s16le"))
    if mono_method is None:
        mono_method = str(extract_settings.get("mono_method", "auto"))

    media_path = Path(media_path).expanduser()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    output_path = Path(output_path) if output_path else default_output_path(media_path)
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


def parse_args() -> argparse.Namespace:
    settings = load_audio_settings()
    extract_settings = settings.get("extract", {})
    if not isinstance(extract_settings, dict):
        raise ValueError("extract 설정 형식이 올바르지 않습니다(맵이어야 함).")

    parser = argparse.ArgumentParser(description="Extract audio from a media file.")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument("--output-path", help="Output audio file path.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(extract_settings.get("sample_rate", 16000)),
        help="Sample rate (Hz).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=int(extract_settings.get("channels", 1)),
        help="Audio channels.",
    )
    parser.add_argument(
        "--codec",
        default=str(extract_settings.get("codec", "pcm_s16le")),
        help="Audio codec (ffmpeg).",
    )
    parser.add_argument(
        "--mono-method",
        default=str(extract_settings.get("mono_method", "auto")),
        choices=("downmix", "left", "right", "phase-fix", "auto"),
        help="Mono creation method (downmix or channel select/phase-fix/auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = extract_audio(
        args.media_path,
        args.output_path,
        sample_rate=args.sample_rate,
        channels=args.channels,
        codec=args.codec,
        mono_method=args.mono_method,
    )
    print(f"[OK] Audio saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
