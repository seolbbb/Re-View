"""Insert stt.json + manifest.json into Supabase Postgres."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import Client, create_client


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
TABLES = ("stt_runs", "manifest_runs")


def _load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_param(cli_value: Optional[str], env_key: str, fallback: Optional[str]) -> Optional[str]:
    return cli_value or os.getenv(env_key) or fallback


def _resolve_supabase_key(cli_value: Optional[str]) -> Optional[str]:
    return (
        cli_value
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )


def _create_supabase_client(url: Optional[str], key: Optional[str]) -> Client:
    if not url:
        raise ValueError("SUPABASE_URL is not set.")
    if not key:
        raise ValueError(
            "SUPABASE_KEY is not set. Set SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, or SUPABASE_ANON_KEY."
        )
    return create_client(url, key)


def _raise_on_error(response: Any, context: str) -> None:
    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"{context}: {error}")


def _ensure_tables(client: Client) -> None:
    for table in TABLES:
        response = client.table(table).select("id").limit(1).execute()
        _raise_on_error(response, f"Table check failed for {table}")


def _insert_payload(
    client: Client,
    *,
    table: str,
    source_path: str,
    payload: Any,
) -> None:
    record: Dict[str, Any] = {
        "source_path": source_path,
        "payload": payload,
    }
    response = client.table(table).insert(record).execute()
    _raise_on_error(response, f"Insert failed for {table}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert stt.json and manifest.json into Supabase.")
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
    _load_env()
    args = _parse_args()

    url = _resolve_param(args.url, "SUPABASE_URL", None)
    key = _resolve_supabase_key(args.key)

    stt_path = Path(args.stt).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    if not stt_path.exists():
        raise FileNotFoundError(f"stt.json not found: {stt_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    stt_payload = _read_json(stt_path)
    manifest_payload = _read_json(manifest_path)

    client = _create_supabase_client(url, key)
    if not args.no_create:
        _ensure_tables(client)
    _insert_payload(
        client,
        table="stt_runs",
        source_path=str(stt_path),
        payload=stt_payload,
    )
    _insert_payload(
        client,
        table="manifest_runs",
        source_path=str(manifest_path),
        payload=manifest_payload,
    )

    print("[OK] inserted stt.json + manifest.json")


if __name__ == "__main__":
    main()
