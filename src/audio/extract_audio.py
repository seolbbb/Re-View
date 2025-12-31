# 실행: python src/audio/extract_audio.py --media-path src/data/input/screentime-mvp-video.mp4
# 옵션: --output-path src/data/output/sample/audio.wav (기본: src/data/output/<입력파일명>/audio.wav)
# 옵션: --sample-rate 16000 --channels 1 --codec pcm_s16le
# 참고: ffmpeg 설치 필요
"""Extract audio track from a media file using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def default_output_path(media_path: Path) -> Path:
    return Path("src/data/input") / media_path.stem / "audio.wav"


def extract_audio(
    media_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    media_path = Path(media_path).expanduser()
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    output_path = Path(output_path) if output_path else default_output_path(media_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract audio from a media file.")
    parser.add_argument("--media-path", required=True, help="Path to local media file (video/audio).")
    parser.add_argument("--output-path", help="Output audio file path.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (Hz).")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels.")
    parser.add_argument("--codec", default="pcm_s16le", help="Audio codec (ffmpeg).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = extract_audio(
        args.media_path,
        args.output_path,
        sample_rate=args.sample_rate,
        channels=args.channels,
        codec=args.codec,
    )
    print(f"[OK] Audio saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
