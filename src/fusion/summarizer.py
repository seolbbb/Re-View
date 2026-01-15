"""Gemini 기반 구간 요약 생성."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import ConfigBundle
from .io_utils import ensure_output_root, read_jsonl, update_token_usage


PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "fusion" / "prompts.yaml"
DEFAULT_PROMPT_VERSION = "sum_v1.6"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiClientBundle:
    client: Any
    backend: str
    model: str


def _load_genai() -> Any:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai 패키지가 필요합니다. requirements.txt를 확인하세요."
        ) from exc
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
            raise ValueError(
                "Developer API 키가 없습니다. GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정하세요."
            )
        client = genai.Client(api_key=api_key)
        return GeminiClientBundle(
            client=client, backend="developer_api", model=llm_cfg.model
        )

    if llm_cfg.backend == "vertex_ai":
        project = llm_cfg.vertex_ai.project
        location = llm_cfg.vertex_ai.location
        if not project or not location:
            raise ValueError(
                "Vertex AI는 project/location이 필요합니다. config를 확인하세요."
            )
        if llm_cfg.vertex_ai.auth_mode == "adc":
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("Vertex AI express_api_key 모드용 API 키가 없습니다.")
            client = genai.Client(
                vertexai=True, project=project, location=location, api_key=api_key
            )
        return GeminiClientBundle(
            client=client, backend="vertex_ai", model=llm_cfg.model
        )

    raise ValueError(f"지원하지 않는 backend: {llm_cfg.backend}")


def _build_response_schema() -> Dict[str, Any]:
    evidence_schema = {"type": "array", "items": {"type": "string"}}
    source_type_schema = {
        "type": "string",
        "enum": ["direct", "inferred", "background"],
    }
    bullet_schema = {
        "type": "object",
        "required": [
            "bullet_id",
            "claim",
            "source_type",
            "evidence_refs",
            "confidence",
            "notes",
        ],
        "properties": {
            "bullet_id": {"type": "string"},
            "claim": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    definition_schema = {
        "type": "object",
        "required": [
            "term",
            "definition",
            "source_type",
            "evidence_refs",
            "confidence",
            "notes",
        ],
        "properties": {
            "term": {"type": "string"},
            "definition": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    explanation_schema = {
        "type": "object",
        "required": ["point", "source_type", "evidence_refs", "confidence", "notes"],
        "properties": {
            "point": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    question_schema = {
        "type": "object",
        "required": ["question", "source_type", "evidence_refs", "confidence", "notes"],
        "properties": {
            "question": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    summary_schema = {
        "type": "object",
        "required": ["bullets", "definitions", "explanations", "open_questions"],
        "properties": {
            "bullets": {"type": "array", "items": bullet_schema},
            "definitions": {"type": "array", "items": definition_schema},
            "explanations": {"type": "array", "items": explanation_schema},
            "open_questions": {"type": "array", "items": question_schema},
        },
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["segment_id", "summary"],
            "properties": {
                "segment_id": {"type": "integer"},
                "summary": summary_schema,
            },
        },
    }


def load_prompt_template(
    *,
    prompt_version: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> str:
    """프롬프트 템플릿을 설정 파일에서 읽는다."""
    version_id = prompt_version or DEFAULT_PROMPT_VERSION
    path = prompt_path or PROMPT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"요약 프롬프트 설정을 찾을 수 없습니다: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("요약 프롬프트 설정 형식이 올바르지 않습니다(맵이어야 함).")
    prompt_block = payload.get(version_id)
    if not isinstance(prompt_block, dict):
        available = ", ".join(sorted(payload.keys()))
        raise ValueError(
            f"프롬프트 버전을 찾을 수 없습니다: {version_id} (가능: {available})"
        )
    template = prompt_block.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"프롬프트 템플릿이 비어 있습니다: {version_id}")
    return template.strip()


def _render_prompt_template(template: str, replacements: Dict[str, str]) -> str:
    """템플릿 토큰을 치환해 최종 프롬프트를 만든다."""
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered

def _build_batch_prompt(
    segments: List[Dict[str, Any]],
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
    previous_context: Optional[str] = None,
    prompt_version: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> str:
    """설정된 템플릿을 사용해 배치 프롬프트를 생성한다."""
    claim_rule = (
        f"- claim은 {claim_max_chars}자 이하의 한 문장"
        if claim_max_chars > 0
        else "- claim은 한 문장으로 작성 (길이 제한 없음)"
    )
    if bullets_min > 0 and bullets_max > 0:
        bullets_rule = f"- bullets는 {bullets_min}~{bullets_max}개 (권장)"
    elif bullets_min > 0:
        bullets_rule = f"- bullets는 최소 {bullets_min}개 (상한 없음)"
    else:
        bullets_rule = "- bullets는 필요한 만큼 작성"

    segments_text_parts: List[str] = []
    for seg in segments:
        segments_text_parts.append(f"--- Segment {seg.get('segment_id')} ---")
        segments_text_parts.append(json.dumps(seg, ensure_ascii=False))
    segments_text = "\n".join(segments_text_parts)
    jsonl_text = "\n".join(json.dumps(seg, ensure_ascii=False) for seg in segments)

    context_section = ""
    if previous_context:
        context_section = f"""
========================
이전 배치 요약 (맥락 유지용)
========================
{previous_context}

위 내용은 이전 배치에서 다룬 핵심 내용입니다.
- 현재 배치 요약 시 위 내용과 일관성을 유지하세요.
- 동일한 용어/개념은 같은 방식으로 설명하세요.
========================

"""

    template = load_prompt_template(
        prompt_version=prompt_version, prompt_path=prompt_path
    )
    replacements = {
        "CLAIM_MAX_CHARS": str(claim_max_chars),
        "BULLETS_MIN": str(bullets_min),
        "BULLETS_MAX": str(bullets_max),
        "BULLETS_RULE": bullets_rule,
        "CLAIM_RULE": claim_rule,
        "SEGMENTS_TEXT": segments_text,
        "JSONL_TEXT": jsonl_text,
        "bullets_rule": bullets_rule,
        "claim_rule": claim_rule,
        "jsonl_text": jsonl_text,
    }
    prompt = _render_prompt_template(template, replacements)
    if (
        "{{SEGMENTS_TEXT}}" not in template
        and "{SEGMENTS_TEXT}" not in template
        and "{{JSONL_TEXT}}" not in template
        and "{jsonl_text}" not in template
    ):
        prompt = f"{prompt}\n\n{segments_text}"

    return f"{context_section}{prompt}"


def _generate_content(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
) -> str:
    import concurrent.futures

    client = client_bundle.client
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
    }

    def _call_api():
        try:
            return client.models.generate_content(
                model=client_bundle.model,
                contents=prompt,
                config=config,
                timeout=timeout_sec,
            )
        except TypeError:
            return client.models.generate_content(
                model=client_bundle.model,
                contents=prompt,
                config=config,
            )

    # 강제 타임아웃 적용 (SDK timeout이 안 먹힐 경우 대비)
    print(f"[DEBUG] Gemini API 호출 시작 (timeout={timeout_sec}s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_api)
        try:
            response = future.result(timeout=timeout_sec + 10)  # 여유 10초 추가
            print(f"[DEBUG] Gemini API 호출 완료")
        except concurrent.futures.TimeoutError:
            print(f"[DEBUG] Gemini API 타임아웃!")
            raise TimeoutError(f"Gemini API 호출이 {timeout_sec + 10}초 후 타임아웃되었습니다.")

    return _extract_text_from_response(response)


def _normalize_evidence_refs(evidence_refs: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(evidence_refs, list):
        candidates = [str(item) for item in evidence_refs]
    elif isinstance(evidence_refs, dict):
        candidates.extend(
            [str(item) for item in evidence_refs.get("transcript_unit_ids", [])]
        )
        candidates.extend(
            [str(item) for item in evidence_refs.get("visual_unit_ids", [])]
        )
    elif isinstance(evidence_refs, str):
        candidates = [evidence_refs]
    else:
        return []

    normalized: List[str] = []
    seen = set()
    for item in candidates:
        if not (item.startswith("t") or item.startswith("v")):
            continue
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _normalize_confidence(value: Any) -> str:
    confidence = str(value or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        return "low"
    return confidence


def _normalize_source_type(value: Any) -> str:
    source_type = str(value or "direct").lower()
    if source_type not in {"direct", "inferred", "background"}:
        return "direct"
    return source_type


def _validate_summary_payload(
    payload: Dict[str, Any],
    segment_id: int,
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
) -> Dict[str, Any]:
    bullets = payload.get("bullets") or []
    definitions = payload.get("definitions") or []
    explanations = payload.get("explanations") or []
    open_questions = payload.get("open_questions") or []

    if not isinstance(bullets, list):
        raise ValueError("bullets 형식이 올바르지 않습니다.")
    if not isinstance(definitions, list):
        raise ValueError("definitions 형식이 올바르지 않습니다.")
    if not isinstance(explanations, list):
        raise ValueError("explanations 형식이 올바르지 않습니다.")
    if not isinstance(open_questions, list):
        raise ValueError("open_questions 형식이 올바르지 않습니다.")
    # bullets 개수, claim 길이 검증 제거 (Judge에서 품질 평가)

    normalized_bullets: List[Dict[str, Any]] = []
    max_items = bullets_max if bullets_max > 0 else len(bullets)
    for idx, bullet in enumerate(bullets[:max_items], start=1):
        if not isinstance(bullet, dict):
            continue
        claim = str(bullet.get("claim", "")).strip()
        source_type = _normalize_source_type(bullet.get("source_type"))
        evidence_refs = _normalize_evidence_refs(bullet.get("evidence_refs"))
        confidence = _normalize_confidence(bullet.get("confidence"))
        notes = str(bullet.get("notes", "")).strip()
        bullet_id = f"{segment_id}-{idx}"
        normalized_bullets.append(
            {
                "bullet_id": bullet_id,
                "claim": claim,
                "source_type": source_type,
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "notes": notes,
            }
        )

    normalized_definitions: List[Dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not term or not definition:
            continue
        source_type = _normalize_source_type(item.get("source_type"))
        evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
        confidence = _normalize_confidence(item.get("confidence"))
        notes = str(item.get("notes", "")).strip()
        normalized_definitions.append(
            {
                "term": term,
                "definition": definition,
                "source_type": source_type,
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "notes": notes,
            }
        )

    normalized_explanations: List[Dict[str, Any]] = []
    for item in explanations:
        if isinstance(item, dict):
            point = str(item.get("point", "")).strip()
            if not point:
                continue
            source_type = _normalize_source_type(item.get("source_type"))
            evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
            confidence = _normalize_confidence(item.get("confidence"))
            notes = str(item.get("notes", "")).strip()
            normalized_explanations.append(
                {
                    "point": point,
                    "source_type": source_type,
                    "evidence_refs": evidence_refs,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized_explanations.append(
            {
                "point": text,
                "source_type": "direct",
                "evidence_refs": [],
                "confidence": "low",
                "notes": "확인 불가",
            }
        )

    normalized_questions: List[Dict[str, Any]] = []
    for item in open_questions:
        if isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            source_type = _normalize_source_type(item.get("source_type"))
            evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
            confidence = _normalize_confidence(item.get("confidence"))
            notes = str(item.get("notes", "")).strip()
            normalized_questions.append(
                {
                    "question": question,
                    "source_type": source_type,
                    "evidence_refs": evidence_refs,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized_questions.append(
            {
                "question": text,
                "source_type": "direct",
                "evidence_refs": [],
                "confidence": "low",
                "notes": "확인 불가",
            }
        )

    return {
        "bullets": normalized_bullets,
        "definitions": normalized_definitions,
        "explanations": normalized_explanations,
        "open_questions": normalized_questions,
    }


def _parse_json_response(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _repair_prompt(
    bad_json: str, bullets_min: int, bullets_max: int, claim_max_chars: int
) -> str:
    claim_rule = (
        f"- claim은 {claim_max_chars}자 이하"
        if claim_max_chars > 0
        else "- claim은 한 문장 (길이 제한 없음)"
    )
    if bullets_min > 0 and bullets_max > 0:
        bullets_rule = f"- bullets는 {bullets_min}~{bullets_max}개"
    elif bullets_min > 0:
        bullets_rule = f"- bullets는 최소 {bullets_min}개 (상한 없음)"
    else:
        bullets_rule = "- bullets는 필요한 만큼"
    return f"""아래는 잘못된 JSON 출력입니다. 반드시 유효한 JSON만 반환하세요.
출력은 JSON 배열이며, 각 원소는 segment_id와 summary를 포함해야 합니다.
summary는 bullets/definitions/explanations/open_questions 구조입니다. 설명 없이 JSON만 출력하세요.

규칙:
{bullets_rule}
{claim_rule}
- evidence_refs는 unit_id(t*, v*) 배열만 사용
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
                logger.error(f"GenerateContent failed after {max_retries} retries: {exc}")
                raise
            sleep_for = (
                backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
            )
            logger.warning(
                f"GenerateContent failed (attempt {attempt+1}/{max_retries}). "
                f"Retrying in {sleep_for}s. Error: {exc}"
            )
            time.sleep(max(sleep_for, 0))
            attempt += 1


def run_summarizer(
    config: ConfigBundle, limit: Optional[int] = None, dry_run: bool = False
) -> None:
    paths = config.paths
    ensure_output_root(paths.output_root)
    output_dir = paths.output_root / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = output_dir / "segments_units.jsonl"
    if not input_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {input_jsonl}")

    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars
    prompt_version = config.raw.summarizer.prompt_version or DEFAULT_PROMPT_VERSION

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(input_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        raise ValueError("요약할 세그먼트가 없습니다.")

    prompt = _build_batch_prompt(
        segments,
        claim_max_chars,
        bullets_min,
        bullets_max,
        prompt_version=prompt_version,
    )

    if dry_run:
        print(f"[DRY RUN] segments={len(segments)} (LLM 미호출, 출력 미생성)")
        return

    response_schema = _build_response_schema()
    client_bundle = _init_gemini_client(config)

    # Count input tokens and save to token_usage.json
    try:
        token_result = client_bundle.client.models.count_tokens(
            model=client_bundle.model,
            contents=prompt
        )
        input_tokens = token_result.total_tokens
        update_token_usage(
            output_dir=output_dir,
            component="summarizer",
            input_tokens=input_tokens,
            model=client_bundle.model,
            extra={"segments_count": len(segments)}
        )
        print(f"[TOKEN] summarizer input_tokens={input_tokens}")
    except Exception as exc:
        print(f"[TOKEN] count_tokens failed: {exc}")

    output_jsonl = output_dir / "segment_summaries.jsonl"
    output_handle = None
    try:
        output_handle = output_jsonl.open("w", encoding="utf-8")

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
                if not isinstance(payload, list):
                    raise ValueError("응답이 JSON 배열 형식이 아닙니다.")

                summary_map: Dict[int, Dict[str, Any]] = {}
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("응답 배열의 항목 형식이 올바르지 않습니다.")
                    if "segment_id" not in item:
                        raise ValueError("응답에 segment_id가 없습니다.")
                    segment_id = int(item.get("segment_id"))
                    if segment_id in summary_map:
                        raise ValueError(f"중복 segment_id 발견: {segment_id}")
                    summary_map[segment_id] = _validate_summary_payload(
                        item.get("summary", {}),
                        segment_id,
                        claim_max_chars,
                        bullets_min,
                        bullets_max,
                    )

                expected_ids = [int(seg.get("segment_id")) for seg in segments]
                if set(summary_map.keys()) != set(expected_ids):
                    missing = sorted(set(expected_ids) - set(summary_map.keys()))
                    extra = sorted(set(summary_map.keys()) - set(expected_ids))
                    raise ValueError(
                        f"segment_id 불일치 (missing={missing}, extra={extra})"
                    )

                for segment in segments:
                    segment_id = int(segment.get("segment_id"))
                    summary = summary_map[segment_id]
                    record = {
                        "run_id": segment.get("run_id"),
                        "segment_id": segment.get("segment_id"),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "summary": summary,
                        "version": {
                            "prompt_version": prompt_version,
                            "llm_model_id": config.raw.llm_gemini.model,
                            "temperature": config.raw.summarizer.temperature,
                            "backend": config.raw.llm_gemini.backend,
                        },
                    }
                    output_handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                    )
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
            raise RuntimeError(f"LLM JSON/검증 실패: {last_error}")
    finally:
        if output_handle:
            output_handle.close()


def run_batch_summarizer(
    *,
    segments_units_jsonl: Path,
    output_dir: Path,
    config: ConfigBundle,
    previous_context: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """배치별 세그먼트 요약을 생성합니다.

    Args:
        segments_units_jsonl: 배치의 segments_units.jsonl 경로
        output_dir: 배치별 출력 디렉토리
        config: ConfigBundle 인스턴스
        previous_context: 이전 배치의 요약 context (선택)
        limit: 처리할 세그먼트 수 제한 (선택)

    Returns:
        success: 실행 성공 여부
        segment_summaries_jsonl: 생성된 파일 경로
        segments_count: 처리된 세그먼트 수
        context: 다음 배치에 전달할 context
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not segments_units_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {segments_units_jsonl}")

    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(segments_units_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        # 세그먼트가 없으면 빈 파일 생성
        output_jsonl = output_dir / "segment_summaries.jsonl"
        output_jsonl.write_text("", encoding="utf-8")
        return {
            "success": True,
            "segment_summaries_jsonl": str(output_jsonl),
            "segments_count": 0,
            "context": "",
        }

    prompt = _build_batch_prompt(
        segments,
        claim_max_chars,
        bullets_min,
        bullets_max,
        previous_context,
        prompt_version=prompt_version,
    )

    response_schema = _build_response_schema()
    client_bundle = _init_gemini_client(config)

    # Token 사용량 기록
    try:
        token_result = client_bundle.client.models.count_tokens(
            model=client_bundle.model,
            contents=prompt
        )
        input_tokens = token_result.total_tokens
        update_token_usage(
            output_dir=output_dir,
            component="batch_summarizer",
            input_tokens=input_tokens,
            model=client_bundle.model,
            extra={"segments_count": len(segments), "has_context": previous_context is not None}
        )
        print(f"[TOKEN] batch_summarizer input_tokens={input_tokens}")
    except Exception as exc:
        print(f"[TOKEN] count_tokens failed: {exc}")

    output_jsonl = output_dir / "segment_summaries.jsonl"
    output_handle = None
    try:
        output_handle = output_jsonl.open("w", encoding="utf-8")

        llm_text = _run_with_retries(
            client_bundle,
            prompt,
            response_schema,
            config.raw.summarizer.temperature,
            config.raw.llm_gemini.response_mime_type,
            300,  # config.raw.llm_gemini.timeout_sec
            config.raw.llm_gemini.max_retries,
            config.raw.llm_gemini.backoff_sec,
        )

        last_error: Optional[Exception] = None
        attempts = config.raw.summarizer.json_repair_attempts
        for _ in range(attempts + 1):
            try:
                payload = _parse_json_response(llm_text)
                if not isinstance(payload, list):
                    raise ValueError("응답이 JSON 배열 형식이 아닙니다.")

                summary_map: Dict[int, Dict[str, Any]] = {}
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("응답 배열의 항목 형식이 올바르지 않습니다.")
                    if "segment_id" not in item:
                        raise ValueError("응답에 segment_id가 없습니다.")
                    sid = int(item["segment_id"])
                    summary_map[sid] = _validate_summary_payload(
                        item.get("summary", {}), sid, claim_max_chars, bullets_min, bullets_max
                    )

                # 파일에 기록
                for segment in segments:
                    segment_id = int(segment.get("segment_id"))
                    summary = summary_map[segment_id]
                    record = {
                        "run_id": segment.get("run_id"),
                        "segment_id": segment.get("segment_id"),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "summary": summary,
                        "version": {
                            "prompt_version": prompt_version,
                            "llm_model_id": config.raw.llm_gemini.model,
                            "temperature": config.raw.summarizer.temperature,
                            "backend": config.raw.llm_gemini.backend,
                        },
                    }
                    output_handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                    )
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
            raise RuntimeError(f"LLM JSON/검증 실패: {last_error}")
    finally:
        if output_handle:
            output_handle.close()

    # 다음 배치를 위한 context 추출
    next_context = extract_batch_context(output_jsonl)

    return {
        "success": True,
        "segment_summaries_jsonl": str(output_jsonl),
        "segments_count": len(segments),
        "context": next_context,
    }


def extract_batch_context(summaries_jsonl: Path, max_chars: int = 500) -> str:
    """배치 요약에서 다음 배치를 위한 context를 추출합니다.

    각 segment의 첫 번째 bullet의 claim만 추출하여 간결한 context를 생성합니다.

    Args:
        summaries_jsonl: segment_summaries.jsonl 경로
        max_chars: 최대 문자 수 (기본: 500)

    Returns:
        context 문자열 (max_chars 이하)
    """
    if not summaries_jsonl.exists():
        return ""

    context_parts = []
    for summary_record in read_jsonl(summaries_jsonl):
        segment_id = summary_record.get("segment_id", "?")
        summary = summary_record.get("summary", {})
        bullets = summary.get("bullets", [])
        if bullets:
            # 첫 번째 bullet의 claim만 추출
            first_claim = bullets[0].get("claim", "")
            if first_claim:
                context_parts.append(f"[Seg {segment_id}] {first_claim}")

    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    return context
