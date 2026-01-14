"""VLM 단독 실행을 위한 CLI 엔트리포인트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="OpenRouter VLM 실행 (이미지 → Markdown 텍스트)")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to a local image (repeatable).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the OpenRouter model name.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel batch requests (default: 1).",
    )
    parser.add_argument(
        "--video-name",
        required=True,
        help="data/outputs/{video_name}/vlm_raw.json 경로를 만들 때 사용할 이름",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="prompts.yaml의 프롬프트 버전 ID (예: vlm_v1.0)",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="prompts.yaml 경로 (기본: config/vlm/prompts.yaml)",
    )
    parser.add_argument(
        "--settings-path",
        default=None,
        help="settings.yaml 경로 (기본: config/vlm/settings.yaml)",
    )
    parser.add_argument(
        "--output-root",
        default="data/outputs",
        help="원시 결과 출력 베이스 디렉토리 (기본: data/outputs)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI에서 전달된 인자로 VLM 추출을 실행한다."""
    args = parse_args()

    prompt_path = Path(args.prompt_path) if args.prompt_path else None
    settings_path = Path(args.settings_path) if args.settings_path else None
    extractor = OpenRouterVlmExtractor(
        video_name=args.video_name,
        output_root=Path(args.output_root),
        prompt_version=args.prompt_version,
        prompt_path=prompt_path,
        settings_path=settings_path,
    )
    if args.model:
        extractor.model_name = args.model

    results = extractor.extract_features(
        args.image,
        show_progress=True,
        concurrency=args.concurrency,
    )
    output_path = extractor.get_output_path()
    write_vlm_raw_json(results, output_path)
    print(f"[OK] saved to {output_path}")


if __name__ == "__main__":
    main()
