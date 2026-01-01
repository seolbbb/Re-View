"""자연어 요약 버전 summarizer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigBundle
from .io_utils import ensure_output_root, format_ms, read_jsonl
from .summarizer import _extract_text_from_response, _init_gemini_client, _strip_code_fences


PROMPT_VERSION = "sum_nl_v1.0"
RESPONSE_MIME_TYPE = "text/plain"


def _build_nl_prompt(segments: List[Dict[str, Any]]) -> str:
    compact_rows: List[Dict[str, Any]] = []
    for segment in segments:
        segment_id = int(segment.get("segment_id"))
        start_ms = int(segment.get("start_ms"))
        end_ms = int(segment.get("end_ms"))
        compact_rows.append(
            {
                "segment_id": segment_id,
                "time_range": f"{format_ms(start_ms)}–{format_ms(end_ms)}",
                "transcript_text": str(segment.get("transcript_text", "")).strip(),
                "visual_text": str(segment.get("visual_text", "")).strip(),
            }
        )

    jsonl_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in compact_rows)
    prompt = f"""
당신은 20년 경력의 학원 강사로서, 어려운 개념을 짧고 명확하게 풀어 설명하는 전문가입니다.
단, 당신이 사용할 수 있는 근거는 아래 JSONL 입력뿐이며, 입력에 없는 새로운 사실(정의/정리/응용/예시)을 단정적으로 추가하면 안 됩니다.
모든 출력은 한국어로 작성하되, 아래 용어 표기 규칙을 반드시 지키세요.

출력은 반드시 Markdown 문서로 작성하세요. (JSON/코드블록/표/목록 금지)
형식은 아래를 정확히 따릅니다:
# Segment Summaries (Natural Language)
## Segment {{segment_id}} ({{time_range}})
여기에 자연어 문단으로 2~4문장 요약.

작성 규칙:
- 세그먼트 순서를 유지하세요.
- 각 세그먼트는 해당 transcript_text/visual_text만 근거로 사용하세요.
- 전환/맥락 표현이 있으면 자연스럽게 포함하되, 다른 세그먼트의 내용을 섞지 마세요.
- 수식이 등장하면 문장으로 풀어 설명하되, 입력에 없는 일반 성질을 단정하지 마세요.

용어 표기 규칙(중요, 다른 규칙보다 우선):
- 출력은 기본적으로 한국어로 작성한다.
- 단, 아래 범주의 전문 용어는 한국어 번역 대신 영어 원어 표기를 우선 사용한다.
  1) 알고리즘/모델/방법론 명칭 (예: Expectation-Maximization (EM), Variational Inference, Mean-field Variational Inference, ELBO)
  2) 확률/통계/최적화 핵심 개념 (예: log marginal likelihood, posterior, prior, likelihood, KL divergence, stationary point, stationary function, functional derivative)
  3) 수식에 직접 등장하는 심볼/표현 (예: log p(x), q(z), p(z|x), \\mathcal{{L}}(q))

- 괄호 규칙:
  * 처음 등장할 때만 "영어 (약어)" 또는 "영어 (한국어 번역)" 중 하나로 병기한다.
    - 기본: 영어 (약어) 형태 권장. 예: Expectation-Maximization (EM)
    - 한국어 병기는 필요할 때만. 예: log marginal likelihood (로그 마지널 라이클리후드)
  * 이후에는 영어/약어만 사용한다.

입력(JSONL, 1줄=1세그먼트):
{jsonl_text}
"""
    return prompt


def _generate_text(
    client_bundle: Any,
    prompt: str,
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
) -> str:
    client = client_bundle.client
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
    }
    try:
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
            timeout=timeout_sec,
        )
    except TypeError:
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
        )
    return _extract_text_from_response(response)


def _run_with_retries_text(
    client_bundle: Any,
    prompt: str,
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: List[int],
) -> str:
    attempt = 0
    while True:
        try:
            return _generate_text(
                client_bundle,
                prompt,
                temperature,
                response_mime_type,
                timeout_sec,
            )
        except Exception:
            if attempt >= max_retries:
                raise
            sleep_for = backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
            time.sleep(max(sleep_for, 0))
            attempt += 1


def _print_head(path: Path, max_lines: int = 2) -> None:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(max_lines):
            line = handle.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    if lines:
        print("\n".join(lines))


def run_summarizer_nl(
    config: ConfigBundle, limit: Optional[int] = None, dry_run: bool = False
) -> None:
    paths = config.paths
    ensure_output_root(paths.output_root)
    output_dir = paths.output_root / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = output_dir / "segments_units.jsonl"
    if not input_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {input_jsonl}")

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(input_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        raise ValueError("요약할 세그먼트가 없습니다.")

    prompt = _build_nl_prompt(segments)

    if dry_run:
        print(f"[DRY RUN] segments={len(segments)} (LLM 미호출, 출력 미생성)")
        return

    client_bundle = _init_gemini_client(config)
    llm_text = _run_with_retries_text(
        client_bundle,
        prompt,
        config.raw.summarizer.temperature,
        RESPONSE_MIME_TYPE,
        config.raw.llm_gemini.timeout_sec,
        config.raw.llm_gemini.max_retries,
        config.raw.llm_gemini.backoff_sec,
    )
    cleaned = _strip_code_fences(llm_text).strip()
    if not cleaned:
        raise ValueError("LLM 응답이 비어 있습니다.")

    output_md = output_dir / "segment_summaries_nl.md"
    with output_md.open("w", encoding="utf-8") as handle:
        handle.write(cleaned)
        if not cleaned.endswith("\n"):
            handle.write("\n")

    _print_head(output_md, max_lines=2)
