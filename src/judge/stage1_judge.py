"""Stage1 LLM judge for segment summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.io_utils import read_jsonl, write_json, write_jsonl


PROMPT_VERSION = "judge_stage1_v1"
QUALITY_MIN_SCORE = 3


@dataclass(frozen=True)
class GeminiClientBundle:
    client: Any
    backend: str
    model: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_genai() -> Any:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError("google-genai package is required. Check requirements.txt.") from exc
    return genai


def _init_gemini_client(config: Any) -> GeminiClientBundle:
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


def _extract_text_from_response(response: Any) -> str:
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
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _build_response_schema() -> Dict[str, Any]:
    score_schema = {
        "type": "object",
        "required": ["faithfulness"],
        "properties": {
            "faithfulness": {"type": "integer"},
        },
    }
    core_missing_point_schema = {
        "type": "object",
        "required": ["priority", "text", "ref_id"],
        "properties": {
            "priority": {"type": "string", "enum": ["P0", "P1"]},
            "text": {"type": "string"},
            "ref_id": {"type": "string"},
        },
    }
    optional_missing_point_schema = {
        "type": "object",
        "required": ["priority", "text", "ref_id"],
        "properties": {
            "priority": {"type": "string", "enum": ["P2"]},
            "text": {"type": "string"},
            "ref_id": {"type": "string"},
        },
    }
    item_schema = {
        "type": "object",
        "required": [
            "segment_id",
            "decision",
            "scores",
            "unsupported_claims",
            "missing_core_points",
            "missing_optional_visual_details",
            "faithfulness_reasoning_ko",
            "reasons_ko",
            "fixes_ko",
            "retry_instructions_en",
        ],
        "properties": {
            "segment_id": {"type": "integer"},
            "decision": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
            "scores": score_schema,
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "missing_core_points": {"type": "array", "items": core_missing_point_schema},
            "missing_optional_visual_details": {"type": "array", "items": optional_missing_point_schema},
            "faithfulness_reasoning_ko": {"type": "string"},
            "reasons_ko": {"type": "array", "items": {"type": "string"}},
            "fixes_ko": {"type": "array", "items": {"type": "string"}},
            "retry_instructions_en": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {"type": "array", "items": item_schema}


def _repair_prompt(bad_json: str) -> str:
    return (
        "You returned invalid JSON. Return a valid JSON array only, "
        "matching the required schema. No markdown or extra text.\n\n"
        f"Bad output:\n{bad_json}"
    )


def _truncate(text: str, max_chars: Optional[int]) -> str:
    if not max_chars or max_chars <= 0:
        return text
    return text[:max_chars]


def _normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(4, score))


def _normalize_decision(value: Any) -> str:
    decision = str(value or "uncertain").lower()
    if decision not in {"pass", "fail", "uncertain"}:
        return "uncertain"
    return decision


def _score_to_100(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) * 25.0, 2)


def _rate_to_100(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_missing_points(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            priority = str(item.get("priority", "P1")).strip().upper()
            if priority not in {"P0", "P1", "P2"}:
                priority = "P1"
            entry: Dict[str, Any] = {
                "priority": priority,
                "text": text,
                "ref_id": None,
            }
            ref_id_raw = item.get("ref_id", None)
            if ref_id_raw is not None:
                ref_id = str(ref_id_raw or "").strip()
                if ref_id and ref_id.lower() != "null":
                    entry["ref_id"] = ref_id
            normalized.append(entry)
            continue
        text = str(item or "").strip()
        if text:
            normalized.append({"priority": "P1", "text": text, "ref_id": None})
    return normalized


def _assign_missing_ids(
    segment_id: int,
    core_points: List[Dict[str, str]],
    optional_points: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    counts: Dict[str, int] = {"P0": 0, "P1": 0, "P2": 0}

    def _assign(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        updated: List[Dict[str, str]] = []
        for item in items:
            priority = item.get("priority", "P1")
            if priority not in counts:
                priority = "P1"
            counts[priority] += 1
            item_id = f"seg{segment_id}-{priority}-{counts[priority]:02d}"
            updated_item = dict(item)
            updated_item["id"] = item_id
            updated.append(updated_item)
        return updated

    core_updated = _assign(core_points)
    optional_updated = _assign(optional_points)
    return core_updated, optional_updated


def _collect_summary_items(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def _append(item_type: str, text_key: str, obj: Dict[str, Any]) -> None:
        text = str(obj.get(text_key, "") or "").strip()
        evidence_refs = obj.get("evidence_refs") or []
        if not isinstance(evidence_refs, list):
            evidence_refs = []
        items.append({"type": item_type, "text": text, "evidence_refs": evidence_refs})

    for bullet in summary.get("bullets", []) or []:
        if isinstance(bullet, dict):
            _append("bullet", "claim", bullet)
    for definition in summary.get("definitions", []) or []:
        if isinstance(definition, dict):
            term = str(definition.get("term", "") or "").strip()
            definition_text = str(definition.get("definition", "") or "").strip()
            combined = f"{term}: {definition_text}".strip(": ").strip()
            evidence_refs = definition.get("evidence_refs") or []
            if not isinstance(evidence_refs, list):
                evidence_refs = []
            items.append(
                {"type": "definition", "text": combined, "evidence_refs": evidence_refs}
            )
    for explanation in summary.get("explanations", []) or []:
        if isinstance(explanation, dict):
            _append("explanation", "point", explanation)
    for question in summary.get("open_questions", []) or []:
        if isinstance(question, dict):
            _append("open_question", "question", question)

    return items


def _extract_unit_ids(segment: Dict[str, Any]) -> Tuple[set, set, set]:
    transcript_units = segment.get("transcript_units", []) or []
    visual_units = segment.get("visual_units", []) or []
    transcript_ids = {str(item.get("unit_id")) for item in transcript_units if item.get("unit_id")}
    visual_ids = {str(item.get("unit_id")) for item in visual_units if item.get("unit_id")}
    return transcript_ids, visual_ids, transcript_ids | visual_ids


def _compute_adherence(
    summary: Dict[str, Any],
    unit_ids: set,
    max_list_items: int,
) -> Tuple[int, List[Dict[str, Any]], int, int, int]:
    items = _collect_summary_items(summary)
    total_items = len(items)
    if total_items == 0:
        return 0, [{"rule_id": "summary.empty", "message": "no summary items"}], 0, 0, 0

    missing_evidence = 0
    invalid_evidence = 0
    violations: List[Dict[str, Any]] = []

    for item in items:
        evidence_refs = [str(ref) for ref in item.get("evidence_refs", [])]
        if not evidence_refs:
            missing_evidence += 1
            violations.append(
                {"rule_id": "evidence_refs.missing", "message": f"missing evidence_refs for {item['type']}"}
            )
            continue
        invalid_refs = [ref for ref in evidence_refs if ref not in unit_ids]
        if invalid_refs:
            invalid_evidence += 1
            violations.append(
                {
                    "rule_id": "evidence_refs.invalid",
                    "message": f"invalid evidence_refs for {item['type']}: {invalid_refs[:3]}",
                }
            )

    violation_ratio = (missing_evidence + invalid_evidence) / total_items
    if violation_ratio == 0:
        score = 4
    elif violation_ratio <= 0.1:
        score = 3
    elif violation_ratio <= 0.25:
        score = 2
    elif violation_ratio <= 0.5:
        score = 1
    else:
        score = 0

    if len(violations) > max_list_items:
        violations = violations[:max_list_items]
    return score, violations, missing_evidence, invalid_evidence, total_items


def _compute_visual_metrics(
    summary: Dict[str, Any],
    visual_unit_ids: set,
) -> Dict[str, Any]:
    bullets = summary.get("bullets", []) or []
    total_bullets = len(bullets)
    bullets_with_visual = 0
    referenced_visual_units: set = set()
    for bullet in bullets:
        if not isinstance(bullet, dict):
            continue
        refs = bullet.get("evidence_refs") or []
        if not isinstance(refs, list):
            refs = []
        refs = [str(ref) for ref in refs]
        valid_visual_refs = [ref for ref in refs if ref in visual_unit_ids]
        if valid_visual_refs:
            bullets_with_visual += 1
        for ref in valid_visual_refs:
            referenced_visual_units.add(ref)
    visual_ref_usage_rate = (
        bullets_with_visual / total_bullets if total_bullets else None
    )
    total_visual_units = len(visual_unit_ids)
    visual_unit_coverage_rate = (
        len(referenced_visual_units) / total_visual_units if total_visual_units else None
    )
    return {
        "visual_ref_usage_rate": visual_ref_usage_rate,
        "visual_unit_coverage_rate": visual_unit_coverage_rate,
        "total_bullets": total_bullets,
        "bullets_with_visual_refs": bullets_with_visual,
        "total_visual_units": total_visual_units,
        "referenced_visual_units": len(referenced_visual_units),
        "referenced_visual_unit_ids": referenced_visual_units,
    }


def _build_segment_payload(
    segment: Dict[str, Any],
    summary: Dict[str, Any],
    max_transcript_chars: Optional[int],
    max_visual_chars: Optional[int],
) -> Dict[str, Any]:
    transcript_text = _truncate(str(segment.get("transcript_text", "") or ""), max_transcript_chars)
    visual_text = _truncate(str(segment.get("visual_text", "") or ""), max_visual_chars)
    payload = {
        "segment_id": segment.get("segment_id"),
        "evidence": {
            "transcript_text": transcript_text,
            "visual_text": visual_text,
        },
        "summary": {
            "bullets": summary.get("bullets", []),
            "definitions": summary.get("definitions", []),
            "explanations": summary.get("explanations", []),
        },
    }
    return payload


def _build_prompt(segments_payload: List[Dict[str, Any]]) -> str:
    return (
        "You are a strict evaluator for segment summaries.\n"
        "Each segment includes evidence (transcript_text + visual_text) and summary items.\n"
        "Score each segment with the rubric below and return JSON only.\n\n"
        "Rubric (0-4):\n"
        "- Faithfulness: claims must be supported by evidence. Unsupported claims reduce score.\n\n"
        "Hard constraints:\n"
        "- If any claim is not supported by evidence, you MUST include it in unsupported_claims.\n"
        "- If coverage is not perfect, you MUST include at least one item in missing_core_points or missing_optional_visual_details.\n"
        "- missing_core_points must be objects with {priority, text, ref_id}:\n"
        "  * priority: P0 (must-have), P1 (important)\n"
        "  * text: short description of the missing core point\n"
        "  * ref_id: evidence id if known, otherwise empty string\n"
        "  * note: P2 items should be reported under missing_optional_visual_details, not missing_core_points\n"
        "- missing_optional_visual_details must be objects with {priority, text, ref_id}:\n"
        "  * priority: P2 only\n"
        "  * text: short English description only (do not use Korean)\n"
        "  * ref_id: evidence id if known, otherwise empty string\n"
        "- Visual decorations, watermarks, logos, slide numbers, and references are usually optional_visual/P2.\n\n"
        "- Be conservative: do not default to 4 unless evidence is clearly complete.\n\n"
        "- If there are no fixes, return an empty array for fixes_ko.\n"
        "Language rules:\n"
        "- faithfulness_reasoning_ko, reasons_ko, fixes_ko must be written in Korean.\n"
        "- retry_instructions_en must be written in English (imperative style).\n\n"
        "Output JSON array only. For each segment:\n"
        "{\n"
        '  "segment_id": int,\n'
        '  "decision": "pass|fail|uncertain",\n'
        '  "scores": {"faithfulness": int},\n'
        '  "unsupported_claims": ["..."],\n'
        '  "missing_core_points": [{"priority": "P0|P1", "text": "...", "ref_id": "string"}],\n'
        '  "missing_optional_visual_details": [{"priority": "P2", "text": "...", "ref_id": "string"}],\n'
        '  "faithfulness_reasoning_ko": "1-2 sentences in Korean",\n'
        '  "reasons_ko": ["..."],\n'
        '  "fixes_ko": ["..."],\n'
        '  "retry_instructions_en": ["..."]\n'
        "}\n\n"
        "Input JSON:\n"
        f"{json.dumps(segments_payload, ensure_ascii=False)}"
    )


def _chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_segments_units(path: Path) -> Dict[int, Dict[str, Any]]:
    segments: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        segments[int(segment_id)] = row
    return segments


def _load_segment_summaries(path: Path) -> Dict[int, Dict[str, Any]]:
    summaries: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        summaries[int(segment_id)] = row.get("summary", {}) or {}
    return summaries


def _merge_segments(
    segments_units: Dict[int, Dict[str, Any]],
    segment_summaries: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], List[int], List[int]]:
    unit_ids = set(segments_units.keys())
    summary_ids = set(segment_summaries.keys())
    missing_units = sorted(summary_ids - unit_ids)
    missing_summaries = sorted(unit_ids - summary_ids)
    matched = sorted(unit_ids & summary_ids)
    return matched, missing_units, missing_summaries


def _collect_report_lists(values: List[Any], max_items: int) -> List[Any]:
    if max_items <= 0 or len(values) <= max_items:
        return values
    return values[:max_items]


def run_stage1(
    *,
    config: ConfigBundle,
    segments_units_path: Path,
    segment_summaries_path: Path,
    output_report_path: Path,
    output_segments_path: Path,
    batch_size: int,
    max_transcript_chars: Optional[int],
    max_visual_chars: Optional[int],
    json_repair_attempts: int,
    max_list_items: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    segments_units = _load_segments_units(segments_units_path)
    segment_summaries = _load_segment_summaries(segment_summaries_path)
    matched_ids, missing_units, missing_summaries = _merge_segments(segments_units, segment_summaries)

    if limit is not None:
        matched_ids = matched_ids[:limit]

    if not matched_ids:
        raise ValueError("No matched segments between segments_units and segment_summaries.")

    payloads: List[Dict[str, Any]] = []
    rule_records: Dict[int, Dict[str, Any]] = {}
    total_visual_units = 0
    visual_refs_all: set = set()
    total_bullets = 0
    bullets_with_visual = 0

    for seg_id in matched_ids:
        segment = segments_units[seg_id]
        summary = segment_summaries[seg_id]
        transcript_ids, visual_ids, unit_ids = _extract_unit_ids(segment)

        adherence_score, adherence_violations, missing_evidence, invalid_evidence, total_items = _compute_adherence(
            summary, unit_ids, max_list_items
        )
        visual_metrics = _compute_visual_metrics(summary, visual_ids)

        total_bullets += visual_metrics["total_bullets"]
        bullets_with_visual += visual_metrics["bullets_with_visual_refs"]
        total_visual_units += visual_metrics["total_visual_units"]
        visual_refs_all.update(visual_metrics["referenced_visual_unit_ids"])

        rule_records[seg_id] = {
            "adherence_score": adherence_score,
            "adherence_violations": adherence_violations,
            "missing_evidence_count": missing_evidence,
            "invalid_evidence_count": invalid_evidence,
            "total_summary_items": total_items,
            "visual_ref_usage_rate": visual_metrics["visual_ref_usage_rate"],
            "visual_unit_coverage_rate": visual_metrics["visual_unit_coverage_rate"],
            "visual_ref_usage_denominator": visual_metrics["total_bullets"],
            "visual_ref_usage_numerator": visual_metrics["bullets_with_visual_refs"],
            "visual_unit_coverage_denominator": visual_metrics["total_visual_units"],
            "visual_unit_coverage_numerator": visual_metrics["referenced_visual_units"],
        }

        payloads.append(
            _build_segment_payload(
                segment,
                summary,
                max_transcript_chars=max_transcript_chars,
                max_visual_chars=max_visual_chars,
            )
        )

    client_bundle = _init_gemini_client(config)
    response_schema = _build_response_schema()

    llm_results: Dict[int, Dict[str, Any]] = {}
    for batch in _chunked(payloads, batch_size):
        prompt = _build_prompt(batch)
        llm_text = _run_with_retries(
            client_bundle,
            prompt,
            response_schema,
            temperature=0.2,
            response_mime_type=config.raw.llm_gemini.response_mime_type,
            timeout_sec=config.raw.llm_gemini.timeout_sec,
            max_retries=config.raw.llm_gemini.max_retries,
            backoff_sec=config.raw.llm_gemini.backoff_sec,
        )
        last_error: Optional[Exception] = None
        for _ in range(json_repair_attempts + 1):
            try:
                payload = _parse_json_response(llm_text)
                if not isinstance(payload, list):
                    raise ValueError("LLM response is not a JSON array.")
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("LLM response item is not an object.")
                    seg_id = int(item.get("segment_id"))
                    llm_results[seg_id] = item
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                llm_text = _run_with_retries(
                    client_bundle,
                    _repair_prompt(llm_text),
                    response_schema,
                    temperature=0.2,
                    response_mime_type=config.raw.llm_gemini.response_mime_type,
                    timeout_sec=config.raw.llm_gemini.timeout_sec,
                    max_retries=config.raw.llm_gemini.max_retries,
                    backoff_sec=config.raw.llm_gemini.backoff_sec,
                )
        if last_error:
            raise RuntimeError(f"LLM JSON parse failed: {last_error}")

    segment_reports: List[Dict[str, Any]] = []
    unsupported_claims_all: List[str] = []
    missing_core_points_all: List[Dict[str, Any]] = []
    missing_optional_points_all: List[Dict[str, Any]] = []
    quality_warnings_all: List[Dict[str, Any]] = []
    faithfulness_scores: List[int] = []
    coverage_scores: List[int] = []
    adherence_scores: List[int] = []
    decisions: List[str] = []

    no_fix_phrases_ko = {
        "수정 사항이 없습니다.",
        "수정사항이 없습니다.",
        "수정 없음",
        "없음",
    }

    for seg_id in matched_ids:
        llm_item = llm_results.get(seg_id, {})
        llm_scores = llm_item.get("scores", {}) if isinstance(llm_item, dict) else {}
        faithfulness = _normalize_score(llm_scores.get("faithfulness"))
        llm_decision = _normalize_decision(llm_item.get("decision"))
        faith_reason_ko = str(llm_item.get("faithfulness_reasoning_ko", "") or "").strip()
        reasons_ko = _normalize_str_list(llm_item.get("reasons_ko"))
        fixes_ko = [
            item for item in _normalize_str_list(llm_item.get("fixes_ko"))
            if item not in no_fix_phrases_ko
        ]
        retry_instructions_en = _normalize_str_list(llm_item.get("retry_instructions_en"))
        unsupported_claims = _normalize_str_list(llm_item.get("unsupported_claims"))
        missing_core_raw = llm_item.get("missing_core_points") or []
        missing_optional_raw = llm_item.get("missing_optional_visual_details") or []
        missing_core = _normalize_missing_points(missing_core_raw)
        missing_optional = _normalize_missing_points(missing_optional_raw)
        if missing_core:
            core_filtered: List[Dict[str, Any]] = []
            for item in missing_core:
                if item.get("priority") == "P2":
                    missing_optional.append(item)
                else:
                    core_filtered.append(item)
            missing_core = core_filtered
        for item in missing_optional:
            item["priority"] = "P2"
        missing_core, missing_optional = _assign_missing_ids(seg_id, missing_core, missing_optional)

        rule_info = rule_records[seg_id]
        adherence_score = int(rule_info["adherence_score"])
        adherence_violations = rule_info["adherence_violations"]
        missing_evidence = rule_info["missing_evidence_count"]
        invalid_evidence = rule_info["invalid_evidence_count"]
        adherence_fail = (missing_evidence + invalid_evidence) > 0

        validity_decision = "fail" if adherence_fail else "pass"
        validity_reasons: List[str] = []
        if missing_evidence:
            validity_reasons.append("missing_evidence_refs")
        if invalid_evidence:
            validity_reasons.append("invalid_evidence_refs")

        core_p0 = sum(1 for p in missing_core if p["priority"] == "P0")
        core_p1 = sum(1 for p in missing_core if p["priority"] == "P1")
        unsupported_count = len(unsupported_claims)

        if unsupported_count == 0:
            faithfulness = 4
        elif unsupported_count == 1:
            faithfulness = 3
        else:
            faithfulness = 2

        if core_p0 >= 1:
            coverage = 1
        elif core_p1 >= 2:
            coverage = 2
        elif core_p1 == 1:
            coverage = 3
        else:
            coverage = 4

        quality_decision = "pass"
        quality_reasons: List[str] = []
        quality_warnings: List[Dict[str, Any]] = []
        if core_p1 > 0:
            quality_reasons.append("missing_core_p1_present")
            quality_warnings.append(
                {"segment_id": seg_id, "missing_core_p1_count": core_p1}
            )
        if faithfulness < QUALITY_MIN_SCORE:
            quality_decision = "fail"
            quality_reasons.append("faithfulness_below_threshold")
        if coverage < QUALITY_MIN_SCORE:
            quality_decision = "fail"
            quality_reasons.append("coverage_below_threshold")
        if core_p0 > 0:
            quality_decision = "fail"
            quality_reasons.append("missing_core_p0")
        if quality_decision != "fail" and unsupported_count > 0:
            quality_decision = "uncertain"
            quality_reasons.append("unsupported_claims_present")
        if llm_decision == "fail":
            quality_decision = "fail"
            quality_reasons.append("llm_decision_fail")
        elif llm_decision == "uncertain" and quality_decision == "pass":
            quality_decision = "uncertain"
            quality_reasons.append("llm_decision_uncertain")

        if validity_decision == "fail":
            decision = "fail"
        elif quality_decision == "fail":
            decision = "fail"
        elif quality_decision == "uncertain":
            decision = "uncertain"
        else:
            decision = "pass"

        segment_reports.append(
            {
                "schema_version": 1,
                "segment_id": seg_id,
                "decision": decision,
                "scores": {
                    "faithfulness": faithfulness,
                    "coverage": coverage,
                    "adherence": adherence_score,
                    "clarity": None,
                    "usefulness": None,
                },
                "scores_100": {
                    "faithfulness": _score_to_100(faithfulness),
                    "coverage": _score_to_100(coverage),
                    "adherence": _score_to_100(adherence_score),
                    "clarity": None,
                    "usefulness": None,
                },
                "validity_gate": {
                    "decision": validity_decision,
                    "reasons": validity_reasons,
                    "missing_evidence_count": missing_evidence,
                    "invalid_evidence_count": invalid_evidence,
                },
                "quality_gate": {
                    "decision": quality_decision,
                    "llm_decision": llm_decision,
                    "min_score": QUALITY_MIN_SCORE,
                    "reasons": quality_reasons,
                    "unsupported_claims_count": unsupported_count,
                    "missing_core_p0_count": core_p0,
                    "missing_core_p1_count": core_p1,
                    "warnings": quality_warnings,
                },
                "evaluation_details": {
                    "faithfulness_reasoning_ko": faith_reason_ko,
                    "reasons_ko": _collect_report_lists(reasons_ko, max_list_items),
                    "fixes_ko": _collect_report_lists(fixes_ko, max_list_items),
                    "retry_instructions_en": _collect_report_lists(
                        retry_instructions_en, max_list_items
                    ),
                    "unsupported_claims": _collect_report_lists(unsupported_claims, max_list_items),
                    "missing_core_points": _collect_report_lists(missing_core, max_list_items),
                    "missing_optional_visual_details": _collect_report_lists(missing_optional, max_list_items),
                    "adherence_violations": adherence_violations,
                },
                "rule_metrics": {
                    "visual_ref_usage_rate": rule_info["visual_ref_usage_rate"],
                    "visual_unit_coverage_rate": rule_info["visual_unit_coverage_rate"],
                    "visual_ref_usage_numerator": rule_info["visual_ref_usage_numerator"],
                    "visual_ref_usage_denominator": rule_info["visual_ref_usage_denominator"],
                    "visual_unit_coverage_numerator": rule_info["visual_unit_coverage_numerator"],
                    "visual_unit_coverage_denominator": rule_info["visual_unit_coverage_denominator"],
                    "missing_evidence_count": missing_evidence,
                    "invalid_evidence_count": invalid_evidence,
                },
                "rule_metrics_100": {
                    "visual_ref_usage_rate": _rate_to_100(rule_info["visual_ref_usage_rate"]),
                    "visual_unit_coverage_rate": _rate_to_100(rule_info["visual_unit_coverage_rate"]),
                },
                "meta": {
                    "stage": "stage1",
                    "model": config.raw.llm_gemini.model,
                    "prompt_version": PROMPT_VERSION,
                },
            }
        )

        unsupported_claims_all.extend([str(item) for item in unsupported_claims])
        missing_core_points_all.extend(
            [
                {"segment_id": seg_id, **item}
                for item in missing_core
            ]
        )
        missing_optional_points_all.extend(
            [
                {"segment_id": seg_id, **item}
                for item in missing_optional
            ]
        )
        if quality_warnings:
            quality_warnings_all.extend(quality_warnings)
        faithfulness_scores.append(faithfulness)
        coverage_scores.append(coverage)
        adherence_scores.append(adherence_score)
        decisions.append(decision)

    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
    avg_coverage = round(sum(coverage_scores) / len(coverage_scores), 3)
    avg_adherence = round(sum(adherence_scores) / len(adherence_scores), 3)

    fail_count = sum(1 for d in decisions if d == "fail")
    uncertain_count = sum(1 for d in decisions if d == "uncertain")
    if fail_count > 0:
        overall_decision = "fail"
    elif uncertain_count > 0:
        overall_decision = "uncertain"
    else:
        overall_decision = "pass"

    validity_fail_segments = [
        int(r["segment_id"])
        for r in segment_reports
        if r.get("validity_gate", {}).get("decision") == "fail"
    ]
    quality_fail_segments = [
        int(r["segment_id"])
        for r in segment_reports
        if r.get("quality_gate", {}).get("decision") == "fail"
    ]
    quality_uncertain_segments = [
        int(r["segment_id"])
        for r in segment_reports
        if r.get("quality_gate", {}).get("decision") == "uncertain"
    ]

    segment_report_map = {int(r["segment_id"]): r for r in segment_reports}
    final_reason_codes: List[str] = []
    final_reasons_ko: List[str] = []

    def _append_reason_ko(reason: str) -> None:
        if reason and reason not in final_reasons_ko:
            final_reasons_ko.append(reason)

    def _map_quality_reason(seg_id: int, code: str) -> str:
        mapping = {
            "faithfulness_below_threshold": "faithfulness 점수 부족",
            "coverage_below_threshold": "coverage 점수 부족",
            "missing_core_p0": "핵심(P0) 누락",
            "missing_core_p1_present": "핵심(P1) 누락",
            "unsupported_claims_present": "근거 없는 주장 존재",
            "llm_decision_fail": "LLM 판단 fail",
            "llm_decision_uncertain": "LLM 판단 uncertain",
        }
        reason_text = mapping.get(code, code)
        return f"segment {seg_id} {reason_text}"

    if validity_fail_segments:
        final_reason_codes.append("VALIDITY_GATE_FAIL")
        for seg_id in validity_fail_segments:
            reasons = segment_report_map.get(seg_id, {}).get("validity_gate", {}).get("reasons", [])
            if "missing_evidence_refs" in reasons:
                _append_reason_ko(f"segment {seg_id} evidence_refs 누락")
            if "invalid_evidence_refs" in reasons:
                _append_reason_ko(f"segment {seg_id} evidence_refs invalid")

    if quality_fail_segments:
        final_reason_codes.append("QUALITY_GATE_FAIL")
        for seg_id in quality_fail_segments:
            reasons = segment_report_map.get(seg_id, {}).get("quality_gate", {}).get("reasons", [])
            for reason_code in reasons:
                _append_reason_ko(_map_quality_reason(seg_id, reason_code))
    elif quality_uncertain_segments:
        final_reason_codes.append("QUALITY_GATE_UNCERTAIN")
        for seg_id in quality_uncertain_segments:
            reasons = segment_report_map.get(seg_id, {}).get("quality_gate", {}).get("reasons", [])
            for reason_code in reasons:
                _append_reason_ko(_map_quality_reason(seg_id, reason_code))

    visual_ref_usage_rate = (bullets_with_visual / total_bullets) if total_bullets else None
    visual_unit_coverage_rate = (
        (len(visual_refs_all) / total_visual_units) if total_visual_units else None
    )

    evaluation_details: Dict[str, Any] = {}
    if missing_core_points_all:
        evaluation_details["missing_core_points"] = _collect_report_lists(
            missing_core_points_all, max_list_items
        )
    if missing_optional_points_all:
        evaluation_details["missing_optional_visual_details"] = _collect_report_lists(
            missing_optional_points_all, max_list_items
        )

    actionable_feedback: Dict[str, Any] = {}
    if unsupported_claims_all:
        actionable_feedback["unsupported_claims"] = _collect_report_lists(
            unsupported_claims_all, max_list_items
        )
    if actionable_feedback:
        actionable_feedback.setdefault("suggested_fixes", [])

    report: Dict[str, Any] = {
        "schema_version": 1,
        "decision": overall_decision,
        "final_decision_reason_codes": final_reason_codes,
        "final_decision_reasons_ko": final_reasons_ko,
        "scores": {
            "faithfulness": avg_faithfulness,
            "coverage": avg_coverage,
            "adherence": avg_adherence,
            "clarity": None,
            "usefulness": None,
        },
        "scores_100": {
            "faithfulness": _score_to_100(avg_faithfulness),
            "coverage": _score_to_100(avg_coverage),
            "adherence": _score_to_100(avg_adherence),
            "clarity": None,
            "usefulness": None,
        },
        "visual_metrics": {
            "visual_ref_usage_rate": round(visual_ref_usage_rate, 4)
            if visual_ref_usage_rate is not None
            else None,
            "visual_unit_coverage_rate": round(visual_unit_coverage_rate, 4)
            if visual_unit_coverage_rate is not None
            else None,
            "visual_ref_usage_numerator": bullets_with_visual,
            "visual_ref_usage_denominator": total_bullets,
            "visual_unit_coverage_numerator": len(visual_refs_all),
            "visual_unit_coverage_denominator": total_visual_units,
            "llm_visual_reflection_score": None,
        },
        "visual_metrics_100": {
            "visual_ref_usage_rate": _rate_to_100(visual_ref_usage_rate),
            "visual_unit_coverage_rate": _rate_to_100(visual_unit_coverage_rate),
            "llm_visual_reflection_score": None,
        },
        "validity_gate": {
            "decision": "fail" if validity_fail_segments else "pass",
            "failed_segments": _collect_report_lists(validity_fail_segments, max_list_items),
        },
        "quality_gate": {
            "decision": "fail" if quality_fail_segments else "pass",
            "failed_segments": _collect_report_lists(quality_fail_segments, max_list_items),
            "uncertain_segments": _collect_report_lists(quality_uncertain_segments, max_list_items),
            "min_score": QUALITY_MIN_SCORE,
        },
        "meta": {
            "stage": "stage1",
            "model": config.raw.llm_gemini.model,
            "prompt_version": PROMPT_VERSION,
            "generated_at_utc": _utc_now_iso(),
            "segments_units_path": str(segments_units_path),
            "segment_summaries_path": str(segment_summaries_path),
        },
        "segment_coverage": {
            "summary_present_segments": len(segment_summaries),
            "units_present_segments": len(segments_units),
            "missing_units": missing_units,
            "missing_summaries": missing_summaries,
            "matched_segments": len(matched_ids),
        },
    }

    if quality_warnings_all:
        report["quality_gate"]["warnings"] = _collect_report_lists(
            quality_warnings_all, max_list_items
        )
    if evaluation_details:
        report["evaluation_details"] = evaluation_details
    if actionable_feedback:
        report["actionable_feedback"] = actionable_feedback

    write_json(output_report_path, report)
    write_jsonl(output_segments_path, segment_reports)
    return report


def _resolve_path(explicit: Optional[str], fallback: Path) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return fallback.resolve()


def _load_config(config_path: str):
    from src.fusion.config import load_config

    return load_config(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 LLM judge for segment summaries")
    parser.add_argument("--config", default="src/fusion/config.yaml", help="fusion config YAML path")
    parser.add_argument("--segments-units", default=None, help="segments_units.jsonl path")
    parser.add_argument("--segment-summaries", default=None, help="segment_summaries.jsonl path")
    parser.add_argument("--output-report", default=None, help="output report path")
    parser.add_argument("--output-segments", default=None, help="output segment reports JSONL path")
    parser.add_argument("--batch-size", type=int, default=3, help="LLM batch size")
    parser.add_argument("--limit", type=int, default=None, help="limit segment count")
    parser.add_argument("--max-transcript-chars", type=int, default=0)
    parser.add_argument("--max-visual-chars", type=int, default=0)
    parser.add_argument("--json-repair-attempts", type=int, default=1)
    parser.add_argument("--max-list-items", type=int, default=20)
    args = parser.parse_args()

    config = _load_config(args.config)
    output_root = config.paths.output_root
    default_segments_units = output_root / "fusion" / "segments_units.jsonl"
    default_segment_summaries = output_root / "fusion" / "segment_summaries.jsonl"
    default_report = output_root / "fusion" / "judge" / "stage1_report.json"
    default_segments_report = output_root / "fusion" / "judge" / "stage1_segment_reports.jsonl"

    segments_units_path = _resolve_path(args.segments_units, default_segments_units)
    segment_summaries_path = _resolve_path(args.segment_summaries, default_segment_summaries)
    output_report_path = _resolve_path(args.output_report, default_report)
    output_segments_path = _resolve_path(args.output_segments, default_segments_report)

    report = run_stage1(
        config=config,
        segments_units_path=segments_units_path,
        segment_summaries_path=segment_summaries_path,
        output_report_path=output_report_path,
        output_segments_path=output_segments_path,
        batch_size=args.batch_size,
        max_transcript_chars=args.max_transcript_chars or None,
        max_visual_chars=args.max_visual_chars or None,
        json_repair_attempts=args.json_repair_attempts,
        max_list_items=args.max_list_items,
        limit=args.limit,
    )
    print(f"[OK] stage1 report: {output_report_path}")
    print(f"[OK] decision: {report['decision']}")


if __name__ == "__main__":
    main()
