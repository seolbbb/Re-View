# 실행: python src/utils/token_counter.py src/data/output/screentime-mvp-video/stt.json
# 옵션: --model gpt-4o-mini (tiktoken 모델명 지정)
# 옵션: --per-segment (세그먼트별 토큰 수 출력)
# 예시: python /data/ephemeral/home/Screentime-MVP/src/utils/token_counter.py src/data/output/sample/stt.json --model gpt-4o-mini --per-segment
"""Estimate GPT-style token usage for STT JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token counter for STT JSON.")
    parser.add_argument(
        "json_path",
        help="Path to STT JSON (schema_version=1 or raw provider response).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Tiktoken model name (fallback to cl100k_base if unknown).",
    )
    parser.add_argument(
        "--per-segment",
        action="store_true",
        help="Print token counts per segment.",
    )
    return parser.parse_args()


def _extract_texts(data: object) -> List[str]:
    texts: List[str] = []

    if isinstance(data, dict):
        segments = data.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                text = segment.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

        if not texts:
            for key in ("text", "fullText", "result"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())

        if not texts:
            results = data.get("results")
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())

    return texts


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_encoder(model: str) -> Tuple[object | None, str | None]:
    try:
        import tiktoken
    except ModuleNotFoundError:
        return None, None

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return enc, getattr(enc, "name", None)


def _count_tokens(enc: object, text: str) -> int:
    return len(enc.encode(text))  # type: ignore[attr-defined]


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def main() -> None:
    args = parse_args()
    path = Path(args.json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    data = _load_json(path)
    texts = _extract_texts(data)
    if not texts:
        raise ValueError("No text found in the JSON. Expected segments/text fields.")

    full_text = " ".join(texts)
    enc, enc_name = _load_encoder(args.model)

    if enc is None:
        token_count = _approx_tokens(full_text)
        print("tiktoken not installed; using rough estimate (chars/4).")
    else:
        token_count = _count_tokens(enc, full_text)

    print(f"model: {args.model}")
    if enc_name:
        print(f"encoding: {enc_name}")
    print(f"segments: {len(texts)}")
    print(f"chars: {len(full_text)}")
    print(f"tokens: {token_count}")

    if args.per_segment:
        print("per_segment:")
        if enc is None:
            for idx, text in enumerate(texts, start=1):
                print(f"{idx}\t{_approx_tokens(text)}")
        else:
            for idx, text in enumerate(texts, start=1):
                print(f"{idx}\t{_count_tokens(enc, text)}")


if __name__ == "__main__":
    main()
