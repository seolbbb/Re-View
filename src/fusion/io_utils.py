"""JSON/JSONL 입출력과 공통 유틸."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional


def read_json(path: Path, label: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON 파싱 실패: {path}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)


def read_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def ensure_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    test_path = output_root / ".write_test.tmp"
    with test_path.open("w", encoding="utf-8") as handle:
        handle.write("ok")
    test_path.unlink(missing_ok=True)


def print_jsonl_head(path: Path, max_lines: int = 2) -> None:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(max_lines):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    if lines:
        print("\n".join(lines))


def format_ms(ms: int) -> str:
    minutes = max(ms, 0) // 60000
    seconds = (max(ms, 0) // 1000) % 60
    return f"{minutes:02d}:{seconds:02d}"


def compute_run_id(config_path: Path, stt_path: Path, vlm_path: Path, manifest_path: Optional[Path]) -> str:
    hasher = hashlib.sha256()
    hasher.update(config_path.read_bytes())
    hasher.update(stt_path.read_bytes())
    hasher.update(vlm_path.read_bytes())
    if manifest_path and manifest_path.exists():
        hasher.update(manifest_path.read_bytes())
    return f"run_{hasher.hexdigest()[:12]}"


def update_token_usage(
    output_dir: Path,
    component: str,
    input_tokens: int,
    model: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Update token_usage.json with token counts for a component.

    Records are appended to a history array for each component.

    Args:
        output_dir: Directory to save token_usage.json (e.g., fusion/)
        component: Component name (e.g., "summarizer", "judge")
        input_tokens: Number of input tokens counted
        model: Model name used for counting
        extra: Optional extra fields to include
    """
    from datetime import datetime, timezone

    token_usage_path = output_dir / "token_usage.json"

    # Load existing data or create new
    if token_usage_path.exists():
        try:
            existing = read_json(token_usage_path, "token_usage.json")
        except (FileNotFoundError, ValueError):
            existing = {}
    else:
        existing = {}

    # Build new entry
    entry = {
        "input_tokens": input_tokens,
        "model": model,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        entry.update(extra)

    # Append to history array
    if component not in existing:
        existing[component] = []
    elif not isinstance(existing[component], list):
        # Migrate old format (single object) to array
        existing[component] = [existing[component]]

    existing[component].append(entry)

    # Save
    write_json(token_usage_path, existing)
