"""Gemini 기반 구간 요약 생성."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ConfigBundle
from .io_utils import ensure_output_root, print_jsonl_head, read_jsonl


PROMPT_VERSION = "sum_v1.0"


@dataclass(frozen=True)
class GeminiClientBundle:
    client: Any
    backend: str
    model: str


def _load_genai() -> Any:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError("google-genai 패키지가 필요합니다. requirements.txt를 확인하세요.") from exc
    return genai


def _init_gemini_client(config: ConfigBundle) -> GeminiClientBundle:
    genai = _load_genai()
    llm_cfg = config.raw.llm_gemini
    if llm_cfg.backend == "developer_api":
        api_key = None
        for env_name in llm_cfg.developer_api.api_key_env_candidates:
            api_key = os.getenv(env_name)
            if api_key:
                break
        if not api_key:
            raise ValueError("Developer API 키가 없습니다. GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정하세요.")
        client = genai.Client(api_key=api_key)
        return GeminiClientBundle(client=client, backend="developer_api", model=llm_cfg.model)

    if llm_cfg.backend == "vertex_ai":
        project = llm_cfg.vertex_ai.project
        location = llm_cfg.vertex_ai.location
        if not project or not location:
            raise ValueError("Vertex AI는 project/location이 필요합니다. config를 확인하세요.")
        if llm_cfg.vertex_ai.auth_mode == "adc":
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("Vertex AI express_api_key 모드용 API 키가 없습니다.")
            client = genai.Client(vertexai=True, project=project, location=location, api_key=api_key)
        return GeminiClientBundle(client=client, backend="vertex_ai", model=llm_cfg.model)

    raise ValueError(f"지원하지 않는 backend: {llm_cfg.backend}")


def _build_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["bullets", "definitions", "open_questions"],
        "properties": {
            "bullets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["bullet_id", "claim", "evidence_refs", "confidence", "notes"],
                    "properties": {
                        "bullet_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "evidence_refs": {
                            "type": "object",
                            "required": ["transcript_unit_ids", "visual_unit_ids"],
                            "properties": {
                                "transcript_unit_ids": {"type": "array", "items": {"type": "string"}},
                                "visual_unit_ids": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "notes": {"type": "string"},
                    },
                },
            },
            "definitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["term", "definition", "evidence_refs"],
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                        "evidence_refs": {
                            "type": "object",
                            "required": ["transcript_unit_ids", "visual_unit_ids"],
                            "properties": {
                                "transcript_unit_ids": {"type": "array", "items": {"type": "string"}},
                                "visual_unit_ids": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
    }


def _build_prompt(segment: Dict[str, Any], claim_max_chars: int, bullets_min: int, bullets_max: int) -> str:
    transcript_units = segment.get("transcript_units", [])
    visual_units = segment.get("visual_units", [])
    transcript_text = segment.get("transcript_text", "")
    visual_text = segment.get("visual_text", "")

    prompt = f"""당신은 강의/발표 구간 요약 전문가입니다.
모든 출력은 한국어로 작성하고, 아래 입력 구간만 근거로 사용하세요. 새 사실을 만들지 마세요.

출력은 반드시 순수 JSON만 반환하세요. (설명, 코드블록 금지)

요약 규칙:
- bullets는 {bullets_min}~{bullets_max}개
- claim은 {claim_max_chars}자 이하의 한 문장
- 확신이 없으면 notes에 "확인 불가", confidence는 low
- evidence_refs는 unit_id(t*, v*)만 사용
- bullet_id 형식: "{segment.get('segment_id')}-INDEX" (INDEX는 1부터 시작)

입력:
segment_id: {segment.get('segment_id')}
start_ms: {segment.get('start_ms')}
end_ms: {segment.get('end_ms')}
transcript_units: {json.dumps(transcript_units, ensure_ascii=False)}
visual_units: {json.dumps(visual_units, ensure_ascii=False)}
transcript_text: {json.dumps(transcript_text, ensure_ascii=False)}
visual_text: {json.dumps(visual_text, ensure_ascii=False)}
"""
    return prompt


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
    raise ValueError("Gemini 응답에서 텍스트를 추출할 수 없습니다.")


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


def _normalize_evidence_refs(evidence_refs: Any) -> Dict[str, List[str]]:
    if not isinstance(evidence_refs, dict):
        return {"transcript_unit_ids": [], "visual_unit_ids": []}
    transcript_ids = evidence_refs.get("transcript_unit_ids") or []
    visual_ids = evidence_refs.get("visual_unit_ids") or []
    transcript_ids = [str(item) for item in transcript_ids if str(item).startswith("t")]
    visual_ids = [str(item) for item in visual_ids if str(item).startswith("v")]
    return {"transcript_unit_ids": transcript_ids, "visual_unit_ids": visual_ids}


def _validate_summary_payload(
    payload: Dict[str, Any],
    segment_id: int,
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
) -> Dict[str, Any]:
    bullets = payload.get("bullets") or []
    definitions = payload.get("definitions") or []
    open_questions = payload.get("open_questions") or []

    if not isinstance(bullets, list):
        raise ValueError("bullets 형식이 올바르지 않습니다.")
    if len(bullets) < bullets_min:
        raise ValueError("bullets 개수가 부족합니다.")

    normalized_bullets: List[Dict[str, Any]] = []
    for idx, bullet in enumerate(bullets[:bullets_max], start=1):
        if not isinstance(bullet, dict):
            continue
        claim = str(bullet.get("claim", "")).strip()
        if len(claim) > claim_max_chars:
            raise ValueError("claim 길이가 초과되었습니다.")
        evidence_refs = _normalize_evidence_refs(bullet.get("evidence_refs"))
        confidence = str(bullet.get("confidence", "low")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        notes = str(bullet.get("notes", "")).strip()
        bullet_id = f"{segment_id}-{idx}"
        normalized_bullets.append(
            {
                "bullet_id": bullet_id,
                "claim": claim,
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "notes": notes,
            }
        )

    normalized_definitions: List[Dict[str, Any]] = []
    if isinstance(definitions, list):
        for item in definitions:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            definition = str(item.get("definition", "")).strip()
            evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
            if term and definition:
                normalized_definitions.append(
                    {"term": term, "definition": definition, "evidence_refs": evidence_refs}
                )

    normalized_questions: List[str] = []
    if isinstance(open_questions, list):
        normalized_questions = [str(item).strip() for item in open_questions if str(item).strip()]

    return {
        "bullets": normalized_bullets,
        "definitions": normalized_definitions,
        "open_questions": normalized_questions,
    }


def _parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _repair_prompt(bad_json: str, bullets_min: int, bullets_max: int, claim_max_chars: int) -> str:
    return f"""아래는 잘못된 JSON 출력입니다. 반드시 유효한 JSON만 반환하세요.
스키마는 bullets/definitions/open_questions 구조입니다. 설명 없이 JSON만 출력하세요.

규칙:
- bullets는 {bullets_min}~{bullets_max}개
- claim은 {claim_max_chars}자 이하
- evidence_refs는 transcript_unit_ids/visual_unit_ids를 포함
- confidence는 low|medium|high

잘못된 출력:
{bad_json}
"""


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
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_for = backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
            time.sleep(max(sleep_for, 0))
            attempt += 1


def run_summarizer(config: ConfigBundle, limit: Optional[int] = None, dry_run: bool = False) -> None:
    paths = config.paths
    ensure_output_root(paths.output_root)
    output_dir = paths.output_root / "fusion"
    input_jsonl = output_dir / "segments_units.jsonl"
    if not input_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {input_jsonl}")

    response_schema = _build_response_schema()
    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars

    client_bundle = None
    if not dry_run:
        client_bundle = _init_gemini_client(config)

    output_jsonl = output_dir / "segment_summaries.jsonl"
    output_handle = None
    processed = 0
    try:
        if not dry_run:
            output_handle = output_jsonl.open("w", encoding="utf-8")

        for segment in read_jsonl(input_jsonl):
            if limit is not None and processed >= limit:
                break
            processed += 1

            prompt = _build_prompt(segment, claim_max_chars, bullets_min, bullets_max)

            if dry_run:
                continue

            llm_text = _run_with_retries(
                client_bundle,
                prompt,
                response_schema,
                config.raw.summarizer.temperature,
                config.raw.llm_gemini.response_mime_type,
                config.raw.llm_gemini.timeout_sec,
                config.raw.llm_gemini.max_retries,
                config.raw.llm_gemini.backoff_sec,
            )

            last_error: Optional[Exception] = None
            attempts = config.raw.summarizer.json_repair_attempts
            for _ in range(attempts + 1):
                try:
                    payload = _parse_json_response(llm_text)
                    summary = _validate_summary_payload(
                        payload,
                        int(segment.get("segment_id")),
                        claim_max_chars,
                        bullets_min,
                        bullets_max,
                    )
                    record = {
                        "run_id": segment.get("run_id"),
                        "segment_id": segment.get("segment_id"),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "summary": summary,
                        "version": {
                            "schema_version": 1,
                            "prompt_version": PROMPT_VERSION,
                            "llm_model_id": config.raw.llm_gemini.model,
                            "temperature": config.raw.summarizer.temperature,
                            "backend": config.raw.llm_gemini.backend,
                        },
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    output_handle.write("\n")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    repair_prompt = _repair_prompt(
                        llm_text, bullets_min, bullets_max, claim_max_chars
                    )
                    llm_text = _run_with_retries(
                        client_bundle,
                        repair_prompt,
                        response_schema,
                        config.raw.summarizer.temperature,
                        config.raw.llm_gemini.response_mime_type,
                        config.raw.llm_gemini.timeout_sec,
                        config.raw.llm_gemini.max_retries,
                        config.raw.llm_gemini.backoff_sec,
                    )
            if last_error:
                raise RuntimeError(
                    f"segment_id={segment.get('segment_id')} JSON/검증 실패: {last_error}"
                )
    finally:
        if output_handle:
            output_handle.close()

    if dry_run:
        print(f"[DRY RUN] segments={processed} (LLM 미호출, 출력 미생성)")
        return

    print_jsonl_head(output_jsonl, max_lines=2)
