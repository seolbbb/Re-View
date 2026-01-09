"""Pre-ADK 파이프라인 CLI.

ADK 파이프라인 실행 전에 필요한 전처리를 수행합니다:
- STT: 음성을 텍스트로 변환 → stt.json
- Capture: 화면 캡처 추출 → manifest.json, captures/

Usage:
    python src/pre_adk_pipeline.py --video "video_name.mp4"
    python src/pre_adk_pipeline.py --video "C:/path/to/video.mp4"

산출물:
    data/outputs/{video_name}/
    ├── stt.json          # STT 결과
    ├── manifest.json     # 캡처 메타데이터
    └── captures/         # 캡처 이미지들

이후 ADK 파이프라인 실행:
    adk web src/adk_pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adk_pipeline.paths import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_BASE,
    resolve_video_path,
    sanitize_video_name,
)
from src.adk_pipeline.store import VideoStore
from src.adk_pipeline.tools.internal.pre_db import ensure_pre_db_artifacts


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pre_adk(
    video_path: Path,
    video_name: str,
    output_base: Path,
    stt_backend: str = "clova",
    parallel: bool = True,
    capture_threshold: float = 3.0,
    capture_dedupe_threshold: float = 3.0,
    capture_min_interval: float = 0.5,
) -> dict:
    """Pre-ADK 단계(STT + Capture)를 실행합니다.

    Args:
        video_path: 비디오 파일 경로
        video_name: 비디오 이름 (출력 폴더명)
        output_base: 출력 베이스 디렉토리
        stt_backend: STT 백엔드 ("clova" 또는 "whisper")
        parallel: STT와 Capture 병렬 실행 여부
        capture_threshold: 장면 전환 감지 임계값
        capture_dedupe_threshold: 중복 제거 임계값
        capture_min_interval: 캡처 최소 간격(초)

    Returns:
        생성된 산출물 경로 딕셔너리
    """
    store = VideoStore(output_base=output_base, video_name=video_name)
    video_root = store.video_root()
    video_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Pre-ADK pipeline for: {video_name}")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Output: {video_root}")

    started = time.perf_counter()

    artifacts = ensure_pre_db_artifacts(
        video_path=video_path,
        video_root=video_root,
        output_base=output_base,
        stt_backend=stt_backend,
        parallel=parallel,
        capture_threshold=capture_threshold,
        capture_dedupe_threshold=capture_dedupe_threshold,
        capture_min_interval=capture_min_interval,
    )

    elapsed = round(time.perf_counter() - started, 1)
    logger.info(f"Pre-ADK completed in {elapsed}s")
    logger.info(f"  stt.json: {artifacts['stt_json']}")
    logger.info(f"  manifest.json: {artifacts['manifest_json']}")
    logger.info(f"  captures/: {artifacts['captures_dir']}")

    return {
        "video_name": video_name,
        "video_root": str(video_root),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Pre-ADK 파이프라인: STT + Capture 전처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pre_adk_pipeline.py --video "my_video.mp4"
  python src/pre_adk_pipeline.py --video "C:/path/to/video.mp4"
  python src/pre_adk_pipeline.py --video "my_video.mp4" --stt-backend whisper

완료 후 ADK 파이프라인 실행:
  adk web src/adk_pipeline
""",
    )

    # 필수 인자
    parser.add_argument(
        "--video",
        required=True,
        help="data/inputs 기준 mp4 파일명 또는 절대 경로",
    )

    # 경로 옵션
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"입력 디렉토리 (기본: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help=f"출력 베이스 (기본: {DEFAULT_OUTPUT_BASE})",
    )

    # STT 옵션
    parser.add_argument(
        "--stt-backend",
        choices=["clova", "whisper"],
        default="clova",
        help="STT 백엔드 (기본: clova)",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="STT+Capture 병렬 실행 (기본: True)",
    )

    # Capture 옵션
    parser.add_argument(
        "--capture-threshold",
        type=float,
        default=3.0,
        help="장면 전환 감지 임계값 (기본: 3.0)",
    )
    parser.add_argument(
        "--capture-dedupe-threshold",
        type=float,
        default=3.0,
        help="중복 제거 임계값 (기본: 3.0)",
    )
    parser.add_argument(
        "--capture-min-interval",
        type=float,
        default=0.5,
        help="캡처 최소 간격(초) (기본: 0.5)",
    )

    return parser.parse_args()


def main() -> None:
    """CLI 엔트리 포인트."""
    args = parse_args()

    # 경로 설정
    input_dir = (ROOT / Path(args.input_dir)).resolve()
    output_base = (ROOT / Path(args.output_base)).resolve()

    # 비디오 경로 해석
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        video_path = resolve_video_path(input_dir, args.video)
    else:
        video_path = video_path.resolve()

    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"mp4만 지원합니다: {video_path}")

    video_name = sanitize_video_name(video_path.stem)

    # 실행
    run_pre_adk(
        video_path=video_path,
        video_name=video_name,
        output_base=output_base,
        stt_backend=args.stt_backend,
        parallel=args.parallel,
        capture_threshold=args.capture_threshold,
        capture_dedupe_threshold=args.capture_dedupe_threshold,
        capture_min_interval=args.capture_min_interval,
    )


if __name__ == "__main__":
    main()
