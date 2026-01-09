"""Insert stt.json segments and manifest items into Supabase Postgres."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
STT_TABLE = "stt_segments"
MANIFEST_TABLE = "manifest_items"
TABLES = (STT_TABLE, MANIFEST_TABLE)
BATCH_SIZE = 200


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _raise_on_error(response: Any, context: str) -> None:
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"{context}: {error}")


def _ensure_tables(client: Client) -> None:
    for table in TABLES:
        response = client.table(table).select("id").limit(1).execute()
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"Table check failed for {table}: {error}")


def _chunked(records: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [records[i : i + size] for i in range(0, len(records), size)]


def _normalize_stt_segments(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("stt.json must be an object.")
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("stt.json segments must be a list.")
    normalized: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
            text = str(seg["text"]).strip()
        except KeyError as exc:
            raise ValueError(f"stt.json missing required keys: {seg}") from exc
        confidence = seg.get("confidence")
        confidence = float(confidence) if confidence is not None else None
        normalized.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
                "confidence": confidence,
            }
        )
    if not normalized:
        raise ValueError("No valid stt segments found.")
    return normalized


def _normalize_manifest_items(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("manifest.json must be a list.")
    normalized: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "timestamp_ms" not in item:
            continue
        timestamp_ms = int(item["timestamp_ms"])
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        normalized_item: Dict[str, Any] = {
            "file_name": file_name,
            "timestamp_ms": timestamp_ms,
        }
        timestamp_human = str(item.get("timestamp_human", "")).strip()
        if timestamp_human:
            normalized_item["timestamp_human"] = timestamp_human
        normalized.append(normalized_item)
    if not normalized:
        raise ValueError("No valid manifest items found.")
    return normalized


def _insert_rows(
    client: Client,
    *,
    table: str,
    source_path: str,
    rows: List[Dict[str, Any]],
) -> None:
    records: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["source_path"] = source_path
        records.append(record)
    for batch in _chunked(records, BATCH_SIZE):
        response = client.table(table).insert(batch).execute()
        _raise_on_error(response, f"Insert failed for {table}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert stt.json segments and manifest items into Supabase."
    )
    parser.add_argument("--stt", required=True, help="Path to stt.json")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--url", default=None, help="Supabase URL (default: SUPABASE_URL)")
    parser.add_argument(
        "--key",
        default=None,
        help="Supabase API key (default: SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY)",
    )
    parser.add_argument("--no-create", action="store_true", help="Skip table existence check")
    return parser.parse_args()


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()
        
    args = _parse_args()

    url = args.url or os.getenv("SUPABASE_URL")
    if not url:
        raise RuntimeError("supabase_url is required")
    key = args.key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("supabase_key is required")

    stt_path = Path(args.stt).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    if not stt_path.exists():
        raise FileNotFoundError(f"stt.json not found: {stt_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    stt_payload = _read_json(stt_path)
    manifest_payload = _read_json(manifest_path)
    stt_segments = _normalize_stt_segments(stt_payload)
    manifest_items = _normalize_manifest_items(manifest_payload)

    client = create_client(url, key)
    if not args.no_create:
        _ensure_tables(client)
    _insert_rows(
        client,
        table=STT_TABLE,
        source_path=str(stt_path),
        rows=stt_segments,
    )
    _insert_rows(
        client,
        table=MANIFEST_TABLE,
        source_path=str(manifest_path),
        rows=manifest_items,
    )

    print(
        f"[OK] inserted stt segments={len(stt_segments)} + manifest items={len(manifest_items)}"
    )


if __name__ == "__main__":
    main()
