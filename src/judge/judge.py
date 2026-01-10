# Usage:
# python src/judge/judge.py --config src/fusion/config.yaml
# python src/judge/judge.py --config src/fusion/config.yaml --limit 2 --return-reasons
#
# Inputs:
# - segments_units.jsonl (default: {output_root}/fusion/segments_units.jsonl)
# - segment_summaries.jsonl (default: {output_root}/fusion/segment_summaries.jsonl)
#
# Outputs:
# - judge_report.json (default: {output_root}/fusion/judge/judge_report.json)
# - judge_segment_reports.jsonl (default: {output_root}/fusion/judge/judge_segment_reports.jsonl)
#
# Options:
# --segments-units PATH
# --segment-summaries PATH
# --output-report PATH
# --output-segments PATH
# --batch-size N
# --workers N
# --verbose
# --limit N
# --json-repair-attempts N
# --return-reasons (or env JUDGE_RETURN_REASONS=1)
#
"""LLM judge for segment summaries.

This module forwards raw segment JSON and summary JSON to an LLM and collects
scores. Rule-based logic is intentionally minimal; the LLM decides using
evidence from the provided inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import ConfigBundle
from src.fusion.io_utils import read_jsonl, write_json, write_jsonl, update_token_usage


PROMPT_VERSION = "judge_v3"
JUDGE_TEMPERATURE = 0.2
MAX_SCORE = 10
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class GeminiClientBundle:
    """Holds the Gemini client and selected model metadata."""

    client: Any
    backend: str
    model: str


def _utc_now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _load_genai() -> Any:
    """Import google genai client with a clear error message."""
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError("google-genai package is required. Check requirements.txt.") from exc
    return genai


def _init_gemini_client(config: ConfigBundle) -> GeminiClientBundle:
    """Initialize a Gemini client from the fusion config."""
    genai = _load_genai()
    llm_cfg = config.raw.llm_gemini
    if llm_cfg.backend == "developer_api":
        api_key = None
        for env_name in llm_cfg.developer_api.api_key_env_candidates:
            api_key = os.getenv(env_name)
            if api_key:
                break
        if not api_key:
            raise ValueError("Developer API key is missing. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        client = genai.Client(api_key=api_key)
        return GeminiClientBundle(client=client, backend="developer_api", model=llm_cfg.model)

    if llm_cfg.backend == "vertex_ai":
        project = llm_cfg.vertex_ai.project
        location = llm_cfg.vertex_ai.location
        if not project or not location:
            raise ValueError("Vertex AI requires project/location in config.")
        if llm_cfg.vertex_ai.auth_mode == "adc":
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("Vertex AI express_api_key mode requires an API key.")
            client = genai.Client(vertexai=True, project=project, location=location, api_key=api_key)
        return GeminiClientBundle(client=client, backend="vertex_ai", model=llm_cfg.model)

    raise ValueError(f"Unsupported backend: {llm_cfg.backend}")


def _get_thread_client_bundle(config: ConfigBundle) -> GeminiClientBundle:
    bundle = getattr(_THREAD_LOCAL, "client_bundle", None)
    if bundle is None:
        bundle = _init_gemini_client(config)
        _THREAD_LOCAL.client_bundle = bundle
    return bundle


def _extract_text_from_response(response: Any) -> str:
    """Extract the text payload from a Gemini response."""
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text
    raise ValueError("Failed to extract text from Gemini response.")


def _generate_content(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
) -> str:
    """Call Gemini with a response schema and return the text response."""
    client = client_bundle.client
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
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


def _run_with_retries(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: List[int],
) -> str:
    """Retry Gemini calls with backoff if an error occurs."""
    attempt = 0
    while True:
        try:
            return _generate_content(
                client_bundle,
                prompt,
                response_schema,
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


def _strip_code_fences(text: str) -> str:
    """Strip ``` fences if present in LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_json_response(text: str) -> Any:
    """Parse the LLM output as JSON."""
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _repair_prompt(bad_json: str) -> str:
    """Return a prompt asking the LLM to fix invalid JSON."""
    return (
        "You returned invalid JSON. Return a valid JSON array only, "
        "matching the required schema. No markdown or extra text.\n\n"
        f"Bad output:\n{bad_json}"
    )


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """Split a list into fixed-size chunks."""
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _evaluate_batch(
    *,
    config: ConfigBundle,
    response_schema: Dict[str, Any],
    batch: List[Dict[str, Any]],
    batch_index: int,
    batch_total: int,
    return_reasons: bool,
    json_repair_attempts: int,
    verbose: bool,
) -> Dict[int, Dict[str, Any]]:
    seg_ids = [int(item.get("segment_id", -1)) for item in batch]
    if verbose:
        print(f"[JUDGE] batch {batch_index}/{batch_total} start (segments={seg_ids})")
    started = time.perf_counter()
    client_bundle = _get_thread_client_bundle(config)
    prompt = _build_prompt(batch, return_reasons)
    llm_text = _run_with_retries(
        client_bundle,
        prompt,
        response_schema,
        temperature=JUDGE_TEMPERATURE,
        response_mime_type=config.raw.llm_gemini.response_mime_type,
        timeout_sec=config.raw.llm_gemini.timeout_sec,
        max_retries=config.raw.llm_gemini.max_retries,
        backoff_sec=config.raw.llm_gemini.backoff_sec,
    )
    last_error: Optional[Exception] = None
    results: Dict[int, Dict[str, Any]] = {}
    for _ in range(json_repair_attempts + 1):
        try:
            payload = _parse_json_response(llm_text)
            if not isinstance(payload, list):
                raise ValueError("LLM response is not a JSON array.")
            for item in payload:
                if not isinstance(item, dict):
                    raise ValueError("LLM response item is not an object.")
                seg_id = int(item.get("segment_id"))
                results[seg_id] = item
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            llm_text = _run_with_retries(
                client_bundle,
                _repair_prompt(llm_text),
                response_schema,
                temperature=JUDGE_TEMPERATURE,
                response_mime_type=config.raw.llm_gemini.response_mime_type,
                timeout_sec=config.raw.llm_gemini.timeout_sec,
                max_retries=config.raw.llm_gemini.max_retries,
                backoff_sec=config.raw.llm_gemini.backoff_sec,
            )
    if last_error:
        raise RuntimeError(f"LLM JSON parse failed: {last_error}")
    if verbose:
        elapsed = time.perf_counter() - started
        print(f"[JUDGE] batch {batch_index}/{batch_total} done in {elapsed:.2f}s")
    return results


def _load_segments_units(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load segments_units.jsonl into a dict keyed by segment_id."""
    segments: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        segments[int(segment_id)] = row
    return segments


def _load_segment_summaries(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load segment_summaries.jsonl into a dict keyed by segment_id."""
    summaries: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        summaries[int(segment_id)] = row
    return summaries


def _merge_segments(
    segments_units: Dict[int, Dict[str, Any]],
    segment_summaries: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], List[int], List[int]]:
    """Return matched segment ids and missing ids on either side."""
    unit_ids = set(segments_units.keys())
    summary_ids = set(segment_summaries.keys())
    missing_units = sorted(summary_ids - unit_ids)
    missing_summaries = sorted(unit_ids - summary_ids)
    matched = sorted(unit_ids & summary_ids)
    return matched, missing_units, missing_summaries


def _build_prompt(segments_payload: List[Dict[str, Any]], include_reasons: bool) -> str:
    """Build the LLM judge prompt."""
    lines = [
        "You are a strict evaluator for segment summaries.",
        "Do not use external knowledge to validate correctness; evaluate grounding/compliance/quality only.",
        "Background is allowed only when labeled as background with notes.",
        "",
        "Score each segment on three axes (0-10 integers) aligned with the summarizer prompt rules,",
        "plus one reference-only field for visual usage (multimodal_use).",
        "",
        "Scoring philosophy (applies to all criteria):",
        "10: exceptional and rare; beyond requirements with near-zero ambiguity.",
        "7-9: good to very good; production-ready with minor flaws only.",
        "6: acceptable baseline (pass line).",
        "3-5: needs improvement; notable issues or weak grounding/compliance/quality.",
        "0-2: poor/failing; unreliable or unusable.",
        "",
        "1) Groundedness (source alignment):",
        "10: exceptional and rare; every direct/inferred claim is fully supported by evidence_refs;",
        "    background is explicitly labeled in notes; no unsupported assertions.",
        "7-9: good to very good; mostly grounded with only minor bold inferences or weak refs.",
        "6: baseline; core is grounded but several items have weak refs or overstated source_type.",
        "3-5: needs improvement; unsupported assertions appear or background/inferred boundary is blurry.",
        "0-2: poor/failing; evidence conflicts or grounding collapses (unsupported dominates).",
        "",
        "2) Spec Compliance (format/rules adherence):",
        "Hard fail (score 0) if output is not a JSON array or required structure is missing.",
        "Check at least:",
        "- Required structure: summary has bullets/definitions/explanations/open_questions arrays.",
        "- Required fields: each item has source_type/evidence_refs/confidence/notes as required.",
        "- source_type rules: direct/inferred must have evidence_refs; background may have empty refs",
        "  but notes must state the background knowledge.",
        "- explanations >= 3, cover roles A/B/C: (A) formula/notation explanation if any formula",
        "  appears, (B) comparison/contrast, (C) motivation/usage. At least one explicit why/how.",
        "- If a formula appears, show its full form before explanation using $...$ or $$...$$.",
        "- Definitions cover all key terms/symbols in transcript/visual/summary output.",
        "- Banned deictic words must not appear: 슬라이드, 화면, 그림, 보시면, 위/아래, 여기, 방금,",
        "  앞에서, 다음으로, 이 수식, 마지막 항.",
        "- JSON string safety: no unescaped double quotes inside string values.",
        "10: exceptional and rare; fully compliant with all rules.",
        "7-9: good to very good; minor slips only, no banned words.",
        "6: baseline; repeated minor issues but usable output (no critical violations).",
        "3-5: needs improvement; frequent violations or any banned word/required rule missed.",
        "0-2: poor/failing; broken JSON, missing required structure, or widespread violations.",
        "",
        "3) Note Quality (independent tutor notes):",
        "10: exceptional and rare; fully understandable without video; bullets provide backbone",
        "    (first bullet includes why it matters); definitions stand alone; explanations follow a",
        "    clear flow and differ in role; strong concept linkage; minimal fluff; very consistent",
        "    terminology.",
        "7-9: good to very good; useful notes with small weaknesses (slightly verbose or a weak connection).",
        "6: baseline; understandable but contains paraphrase-like parts or weaker independence.",
        "3-5: needs improvement; informative but weak as teaching notes; weak why/how and linkage.",
        "0-2: poor/failing; incoherent or internally contradictory.",
        "",
        "Final score: final = round(0.45*groundedness + 0.35*note_quality + 0.20*compliance, 2).",
        "",
        "4) Multimodal Use (visual usage, reference only; NOT used in final):",
        "10: exceptional and rare; explicitly restates key visual evidence and uses it to",
        "    explain/define/compare/motivate, along with audio evidence.",
        "7-9: good to very good; visual info supports 1-2 key points with clear meaning.",
        "6: baseline; visual info is mentioned but shallow or mostly decorative.",
        "3-5: needs improvement; visual evidence exists but is barely used or ignored.",
        "0-2: poor/failing; hallucinated visual-specific content (claims about visuals with no evidence).",
        "If segment_summary has no evidence_refs starting with v, cap multimodal_use to 0-3.",
        "",
        "Evidence guidance:",
        "- Evidence is inside each item's segments_units (transcript_units, visual_units)",
        "  and should be compared against segment_summary.",
        "- Do not infer facts that are not grounded in evidence.",
        "",
        "Output rules:",
        "- Return a JSON array only. No markdown or extra text.",
        "- Use integer scores only (0-10).",
        "- Do not output final; it will be computed downstream.",
    ]
    if include_reasons:
        lines.append("- Include reasons_ko as 4 short Korean items in this order:")
        lines.append("  1) 근거: groundedness 판단 이유")
        lines.append("  2) 규칙: compliance 판단 이유")
        lines.append("  3) 노트: note_quality 판단 이유")
        lines.append("  4) 시각: multimodal_use 판단 이유")
        lines.append("- Each item should be 1 sentence and start with the label (근거/규칙/노트/시각).")
        lines.append(
            "- Make reasons_ko concrete: mention 1-2 specific signals (e.g., banned word, missing"
            " definitions, missing role A/B/C, unsupported claim)."
        )
        lines.append(
            "- For 규칙: if banned words are the only issue, say so; if no issues, say '그 외 위반 없음'."
        )
    else:
        lines.append("- Do NOT include reasons_ko.")

    lines.append("Output format for each segment:")
    lines.append("{")
    lines.append('  "segment_id": int,')
    lines.append(
        '  "scores": {"groundedness": int, "compliance": int, '
        '"note_quality": int, "multimodal_use": int}'
    )
    if include_reasons:
        lines.append('  ,"reasons_ko": ["..."]')
    lines.append("}")
    lines.extend(
        [
            "",
            "Input JSON:",
            json.dumps(segments_payload, ensure_ascii=False),
        ]
    )
    return "\n".join(lines)


def _build_response_schema(include_reasons: bool) -> Dict[str, Any]:
    """Return a response schema for Gemini."""
    score_schema = {
        "type": "object",
        "required": ["groundedness", "compliance", "note_quality", "multimodal_use"],
        "properties": {
            "groundedness": {"type": "integer"},
            "compliance": {"type": "integer"},
            "note_quality": {"type": "integer"},
            "multimodal_use": {"type": "integer"},
        },
    }
    item_schema: Dict[str, Any] = {
        "type": "object",
        "required": ["segment_id", "scores"],
        "properties": {
            "segment_id": {"type": "integer"},
            "scores": score_schema,
        },
    }
    if include_reasons:
        item_schema["required"].append("reasons_ko")
        item_schema["properties"]["reasons_ko"] = {
            "type": "array",
            "items": {"type": "string"},
        }
    return {"type": "array", "items": item_schema}


def _clamp_score(value: Any, max_score: int = MAX_SCORE) -> int:
    """Normalize a score to an int within [0, max_score]."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(max_score, score))


def _normalize_str_list(value: Any) -> List[str]:
    """Normalize a value into a list of non-empty strings."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    normalized: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _compute_final_score(groundedness: int, compliance: int, note_quality: int) -> float:
    """Compute the weighted final score."""
    return round(
        0.45 * groundedness + 0.35 * note_quality + 0.20 * compliance,
        2,
    )


def run_judge(
    *,
    config: ConfigBundle,
    segments_units_path: Path,
    segment_summaries_path: Path,
    output_report_path: Path,
    output_segments_path: Path,
    batch_size: int,
    workers: int,
    json_repair_attempts: int,
    limit: Optional[int],
    return_reasons: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run the judge and write JSON outputs."""
    segments_units = _load_segments_units(segments_units_path)
    segment_summaries = _load_segment_summaries(segment_summaries_path)
    matched_ids, missing_units, missing_summaries = _merge_segments(
        segments_units, segment_summaries
    )

    if limit is not None:
        matched_ids = matched_ids[:limit]

    if not matched_ids:
        raise ValueError("No matched segments between segments_units and segment_summaries.")
    if workers < 1:
        raise ValueError("workers must be >= 1.")

    payloads: List[Dict[str, Any]] = []
    for seg_id in matched_ids:
        payloads.append(
            {
                "segment_id": seg_id,
                "segments_units": segments_units[seg_id],
                "segment_summary": segment_summaries[seg_id],
            }
        )

    response_schema = _build_response_schema(return_reasons)

    # Count input tokens for all payloads and save to token_usage.json
    try:
        full_prompt = _build_prompt(payloads, return_reasons)
        client_bundle = _init_gemini_client(config)
        token_result = client_bundle.client.models.count_tokens(
            model=client_bundle.model,
            contents=full_prompt
        )
        input_tokens = token_result.total_tokens
        output_dir = segments_units_path.parent
        update_token_usage(
            output_dir=output_dir,
            component="judge",
            input_tokens=input_tokens,
            model=client_bundle.model,
            extra={"segments_count": len(payloads)}
        )
        print(f"[TOKEN] judge input_tokens={input_tokens}")
    except Exception as exc:
        print(f"[TOKEN] count_tokens failed: {exc}")

    llm_results: Dict[int, Dict[str, Any]] = {}
    batches = _chunked(payloads, batch_size)
    if verbose:
        print(
            f"[JUDGE] batches={len(batches)} batch_size={batch_size} workers={workers}"
        )
    if workers == 1 or len(batches) == 1:
        for idx, batch in enumerate(batches, start=1):
            llm_results.update(
                _evaluate_batch(
                    config=config,
                    response_schema=response_schema,
                    batch=batch,
                    batch_index=idx,
                    batch_total=len(batches),
                    return_reasons=return_reasons,
                    json_repair_attempts=json_repair_attempts,
                    verbose=verbose,
                )
            )
    else:
        max_workers = min(workers, len(batches))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_batch,
                    config=config,
                    response_schema=response_schema,
                    batch=batch,
                    batch_index=idx,
                    batch_total=len(batches),
                    return_reasons=return_reasons,
                    json_repair_attempts=json_repair_attempts,
                    verbose=verbose,
                )
                for idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                llm_results.update(future.result())

    segment_reports: List[Dict[str, Any]] = []
    groundedness_scores: List[int] = []
    compliance_scores: List[int] = []
    note_quality_scores: List[int] = []
    multimodal_use_scores: List[int] = []
    final_scores: List[float] = []

    for seg_id in matched_ids:
        llm_item = llm_results.get(seg_id)
        if not llm_item:
            raise ValueError(f"Missing LLM result for segment_id={seg_id}")
        llm_scores = llm_item.get("scores", {}) if isinstance(llm_item, dict) else {}
        groundedness = _clamp_score(llm_scores.get("groundedness"))
        compliance = _clamp_score(llm_scores.get("compliance"))
        note_quality = _clamp_score(llm_scores.get("note_quality"))
        multimodal_use = _clamp_score(llm_scores.get("multimodal_use"))
        final_score = _compute_final_score(groundedness, compliance, note_quality)
        reasons_ko = (
            _normalize_str_list(llm_item.get("reasons_ko")) if return_reasons else []
        )

        segment_reports.append(
            {
                "schema_version": 1,
                "segment_id": seg_id,
                "scores": {
                    "groundedness": groundedness,
                    "compliance": compliance,
                    "note_quality": note_quality,
                    "multimodal_use": multimodal_use,
                    "final": final_score,
                },
                "reasons_ko": reasons_ko,
                "meta": {
                    "model": config.raw.llm_gemini.model,
                    "prompt_version": PROMPT_VERSION,
                },
            }
        )

        groundedness_scores.append(groundedness)
        compliance_scores.append(compliance)
        note_quality_scores.append(note_quality)
        multimodal_use_scores.append(multimodal_use)
        final_scores.append(final_score)

    avg_groundedness = round(sum(groundedness_scores) / len(groundedness_scores), 2)
    avg_compliance = round(sum(compliance_scores) / len(compliance_scores), 2)
    avg_note_quality = round(sum(note_quality_scores) / len(note_quality_scores), 2)
    avg_multimodal_use = round(sum(multimodal_use_scores) / len(multimodal_use_scores), 2)
    avg_final = round(sum(final_scores) / len(final_scores), 2)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "score_scale": {"min": 0, "max": MAX_SCORE},
        "scores": {
            "groundedness": avg_groundedness,
            "compliance": avg_compliance,
            "note_quality": avg_note_quality,
            "multimodal_use": avg_multimodal_use,
            "final": avg_final,
        },
        "segments": {
            "matched": len(matched_ids),
            "missing_units": missing_units,
            "missing_summaries": missing_summaries,
        },
        "meta": {
            "model": config.raw.llm_gemini.model,
            "prompt_version": PROMPT_VERSION,
            "return_reasons": return_reasons,
            "generated_at_utc": _utc_now_iso(),
            "segments_units_path": str(segments_units_path),
            "segment_summaries_path": str(segment_summaries_path),
        },
    }

    write_json(output_report_path, report)
    write_jsonl(output_segments_path, segment_reports)
    return report


def _resolve_path(explicit: Optional[str], fallback: Path) -> Path:
    """Resolve a path from CLI argument or fallback."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return fallback.resolve()


def _load_config(config_path: str) -> ConfigBundle:
    """Load the fusion config from YAML."""
    from src.fusion.config import load_config

    return load_config(config_path)


def main() -> None:
    """CLI entrypoint for the LLM judge."""
    parser = argparse.ArgumentParser(description="LLM judge for segment summaries")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="fusion config YAML path")
    parser.add_argument("--segments-units", default=None, help="segments_units.jsonl path")
    parser.add_argument("--segment-summaries", default=None, help="segment_summaries.jsonl path")
    parser.add_argument("--output-report", default=None, help="output report path")
    parser.add_argument("--output-segments", default=None, help="output segment reports JSONL path")
    parser.add_argument("--batch-size", type=int, default=3, help="LLM batch size")
    parser.add_argument("--workers", type=int, default=1, help="parallel LLM requests")
    parser.add_argument("--limit", type=int, default=None, help="limit segment count")
    parser.add_argument("--json-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print batch-level progress logs",
    )
    parser.add_argument(
        "--return-reasons",
        action="store_true",
        help="include reasons_ko in LLM output",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    output_root = config.paths.output_root
    default_segments_units = output_root / "fusion" / "segments_units.jsonl"
    default_segment_summaries = output_root / "fusion" / "segment_summaries.jsonl"
    default_report = output_root / "fusion" / "judge" / "judge_report.json"
    default_segments_report = output_root / "fusion" / "judge" / "judge_segment_reports.jsonl"

    segments_units_path = _resolve_path(args.segments_units, default_segments_units)
    segment_summaries_path = _resolve_path(args.segment_summaries, default_segment_summaries)
    output_report_path = _resolve_path(args.output_report, default_report)
    output_segments_path = _resolve_path(args.output_segments, default_segments_report)

    return_reasons = args.return_reasons or _env_flag("JUDGE_RETURN_REASONS", False)

    start_time = time.perf_counter()
    report = run_judge(
        config=config,
        segments_units_path=segments_units_path,
        segment_summaries_path=segment_summaries_path,
        output_report_path=output_report_path,
        output_segments_path=output_segments_path,
        batch_size=args.batch_size,
        workers=args.workers,
        json_repair_attempts=args.json_repair_attempts,
        limit=args.limit,
        return_reasons=return_reasons,
        verbose=args.verbose,
    )
    elapsed_sec = time.perf_counter() - start_time
    print(f"[OK] judge report: {output_report_path}")
    print(f"[OK] avg final score: {report['scores']['final']}")
    print(f"[OK] elapsed: {elapsed_sec:.2f}s")


if __name__ == "__main__":
    main()
