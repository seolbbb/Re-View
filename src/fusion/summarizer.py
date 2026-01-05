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


def _build_batch_prompt(
    segments: List[Dict[str, Any]],
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
) -> str:
    jsonl_text = "\n".join(
        json.dumps(segment, ensure_ascii=False) for segment in segments
    )
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

    prompt = f"""
당신은 20년 경력의 학원 강사로서, 어려운 개념을 짧고 명확하게 풀어 설명하는 전문가입니다.
모든 출력은 한국어로 작성하되, 아래 용어 표기 규칙을 반드시 지키세요.

출력은 반드시 "순수 JSON 배열"만 반환하세요. (설명 문장, 코드블록, 마크다운 금지)
배열의 각 원소는 다음 형식입니다:
{{
  "segment_id": int,
  "summary": {{
    "bullets": [...],
    "definitions": [...],
    "explanations": [...],
    "open_questions": [...]
  }}
}}

요약 규칙:
{bullets_rule}
{claim_rule}

source_type 분류 규칙(핵심):
각 항목(bullets/definitions/explanations/open_questions)에는 source_type 필드를 반드시 포함하세요.
- "direct": 입력 텍스트를 직접 인용/패러프레이징. evidence_refs 필수.
- "inferred": 입력 정보들을 종합하여 논리적으로 추론. evidence_refs 필수 (근거가 된 unit_id들).
- "background": 수학적 성질, 널리 알려진 정의 등 일반 배경지식. evidence_refs는 빈 배열([])도 허용.

Judge 검증 기준:
- direct/inferred: evidence_refs로 검증 가능해야 함
- background: 검증 스킵 (수학적 사실, 공리, 널리 알려진 정의 등)

배경지식 사용 권장 예시:
- "KL divergence는 항상 0 이상이다" → source_type: "background"
- "확률의 총합은 1이다" → source_type: "background"
- 입력에 있는 수식의 수학적 의미 설명 → source_type: "background" + notes에 근거 명시

출력 포맷 규칙:
- bullet_id 형식: "SEGMENT_ID-INDEX" (INDEX는 1부터 시작)
- bullets 항목 스키마:
  {{
    "bullet_id": "1-1",
    "claim": "한 문장 요약(구체적으로)",
    "source_type": "direct|inferred|background",
    "evidence_refs": ["t1","v2"],
    "confidence": "high|medium|low",
    "notes": ""
  }}
- definitions 항목 스키마:
  {{
    "term": "용어(규칙에 따라 영어 우선)",
    "definition": "입력 근거로부터 정리한 정의(1~2문장)",
    "source_type": "direct|inferred|background",
    "evidence_refs": ["v2"],
    "confidence": "high|medium|low",
    "notes": ""
  }}
- explanations 항목 스키마(강사 해설 전용):
  {{
    "point": "학생이 헷갈릴만한 지점을 짚는 설명(2~4문장). 배경지식을 활용한 풍부한 설명 권장.",
    "source_type": "direct|inferred|background",
    "evidence_refs": ["t3","v2"],
    "confidence": "high|medium|low",
    "notes": "background 사용 시 어떤 일반 지식인지 명시"
  }}
- open_questions 항목 스키마:
  {{
    "question": "입력에 근거해 자연스럽게 생기는 질문(1문장)",
    "source_type": "direct|inferred|background",
    "evidence_refs": ["t2"],
    "confidence": "high|medium|low",
    "notes": ""
  }}

품질 가이드(설명 강화):
- 각 segment마다 explanations를 최소 2개 작성하세요.
- explanations는 다음 중 최소 1개를 포함해야 합니다:
  1) 수식이 등장하면: 수식의 각 항이 무엇을 나타내는지 '문장으로 해설' (background 활용 가능)
  2) 비교가 등장하면: 두 개념의 차이를 입력 문구 기반으로 대비 설명
  3) 용어가 등장하면: 왜 그 용어가 필요한지(입력 맥락) 한 문장으로 풀기
- 배경지식(source_type: "background")을 적극 활용하여 풍부한 설명을 제공하세요.
  예: "ELBO가 log p(x)의 하한이 되는 이유는 KL divergence가 항상 0 이상이기 때문입니다."
- 각 segment bullets는 transcript_units 개수에 따라 최소 개수를 강제합니다:
  * 1~3개: 최소 3개
  * 4~6개: 최소 5개
  * 7개 이상: 최소 7개

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
    if bullets_min > 0 and len(bullets) < bullets_min:
        raise ValueError("bullets 개수가 부족합니다.")

    normalized_bullets: List[Dict[str, Any]] = []
    max_items = bullets_max if bullets_max > 0 else len(bullets)
    for idx, bullet in enumerate(bullets[:max_items], start=1):
        if not isinstance(bullet, dict):
            continue
        claim = str(bullet.get("claim", "")).strip()
        if claim_max_chars > 0 and len(claim) > claim_max_chars:
            raise ValueError("claim 길이가 초과되었습니다.")
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
                raise
            sleep_for = (
                backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
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

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(input_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        raise ValueError("요약할 세그먼트가 없습니다.")

    prompt = _build_batch_prompt(segments, claim_max_chars, bullets_min, bullets_max)

    if dry_run:
        print(f"[DRY RUN] segments={len(segments)} (LLM 미호출, 출력 미생성)")
        return

    response_schema = _build_response_schema()
    client_bundle = _init_gemini_client(config)

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
                            "schema_version": 1,
                            "prompt_version": PROMPT_VERSION,
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

    print_jsonl_head(output_jsonl, max_lines=2)
