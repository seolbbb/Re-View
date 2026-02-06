"""
Re-upload capture images that have missing storage_path in DB.

Usage:
    python src/run_reupload_captures.py --video-id <uuid> [options]
    python src/run_reupload_captures.py --video-name <name> [options]

Options:
    --output-base   Output base directory (default: data/outputs)
    --bucket        Storage bucket name (default: captures)
    --limit         Max number of rows to process
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure local imports work when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from src.adk_chatbot.paths import sanitize_video_name
from src.db import get_supabase_adapter


def _resolve_video(adapter: Any, video_id: Optional[str], video_name: Optional[str]) -> Dict[str, Any]:
    if video_id:
        result = (
            adapter.client.table("videos")
            .select("id,name")
            .eq("id", video_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if rows:
            return rows[0]
        raise ValueError(f"Video not found: {video_id}")

    if not video_name:
        raise ValueError("video_id or video_name is required.")

    result = (
        adapter.client.table("videos")
        .select("id,name")
        .eq("name", video_name)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if rows:
        return rows[0]
    raise ValueError(f"Video not found by name: {video_name}")


def _fetch_missing_captures(adapter: Any, video_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    query = (
        adapter.client.table("captures")
        .select("id,file_name,storage_path")
        .eq("video_id", video_id)
        .filter("storage_path", "is", "null")
    )
    if limit is not None:
        query = query.limit(limit)
    result = query.execute()
    return result.data or []


def _upload_capture(adapter: Any, video_id: str, image_path: Path, bucket: str) -> str:
    storage_path = f"{video_id}/{image_path.name}"
    with image_path.open("rb") as handle:
        file_data = handle.read()
    if getattr(adapter, "r2_only", False):
        raise RuntimeError("R2 storage is required (check R2_* env vars)")
    adapter.client.storage.from_(bucket).upload(
        path=storage_path,
        file=file_data,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )
    return storage_path


def _update_capture(adapter: Any, capture_id: str, storage_path: str) -> None:
    adapter.client.table("captures").update(
        {"storage_path": storage_path}
    ).eq("id", capture_id).execute()


def run_reupload(
    *,
    video_id: Optional[str],
    video_name: Optional[str],
    output_base: str,
    bucket: str,
    limit: Optional[int],
) -> int:
    adapter = get_supabase_adapter()
    if not adapter:
        raise RuntimeError("Supabase adapter not configured. Check SUPABASE_URL/SUPABASE_KEY.")

    video = _resolve_video(adapter, video_id, video_name)
    video_id = video.get("id")
    video_name = video.get("name")
    if not video_id or not video_name:
        raise RuntimeError("Failed to resolve video_id or video_name.")

    output_root = (ROOT / Path(output_base)).resolve()
    video_root = output_root / sanitize_video_name(video_name)
    captures_dir = video_root / "captures"
    if not captures_dir.exists():
        raise FileNotFoundError(f"captures dir not found: {captures_dir}")

    missing = _fetch_missing_captures(adapter, video_id, limit)
    if not missing:
        print("No missing storage_path rows found.")
        return 0

    print(f"Found {len(missing)} captures with missing storage_path.")
    repaired = 0
    skipped = 0
    failed = 0
    for row in missing:
        file_name = row.get("file_name")
        capture_id = row.get("id")
        if not file_name or not capture_id:
            skipped += 1
            continue

        image_path = captures_dir / file_name
        if not image_path.exists():
            print(f"[skip] missing file: {image_path}")
            skipped += 1
            continue

        try:
            storage_path = _upload_capture(adapter, video_id, image_path, bucket)
            _update_capture(adapter, capture_id, storage_path)
            repaired += 1
        except Exception as exc:
            print(f"[fail] {file_name}: {exc}")
            failed += 1

    print(f"Done. repaired={repaired} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-upload captures with NULL storage_path.")
    parser.add_argument("--video-id", default=None, help="Video id (UUID)")
    parser.add_argument("--video-name", default=None, help="Video name (DB videos.name)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--bucket", default="captures", help="Storage bucket name")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")

    args = parser.parse_args()

    exit_code = run_reupload(
        video_id=args.video_id,
        video_name=args.video_name,
        output_base=args.output_base,
        bucket=args.bucket,
        limit=args.limit,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
