"""sync_engine 실행 CLI."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config
from src.fusion.sync_engine import run_sync_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="STT/VLM sync_engine 실행")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="config YAML 경로")
    parser.add_argument("--limit", type=int, default=None, help="처리할 segment 수 제한")
    parser.add_argument("--dry_run", action="store_true", help="출력 미생성 모드")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_sync_engine(config, limit=args.limit, dry_run=args.dry_run)
    except Exception as exc:
        print(f"[ERROR] sync_engine 실패: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
