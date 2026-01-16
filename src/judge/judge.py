"""세그먼트 요약을 LLM으로 평가하는 Judge 모듈.

파이프라인에서 호출되며 입력/출력 경로는 호출자가 전달한다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]

from src.fusion.config import ConfigBundle
from src.fusion.io_utils import read_jsonl, write_json, write_jsonl, update_token_usage
from src.fusion.gemini import init_gemini_client, run_with_retries


PROMPTS_PATH = ROOT / "config" / "judge" / "prompts.yaml"
PROMPT_PAYLOAD_TOKEN = "{{SEGMENTS_JSON}}"
JUDGE_TEMPERATURE = 0.2
MAX_SCORE = 10
_THREAD_LOCAL = threading.local()




def _get_thread_client_bundle(config: ConfigBundle) -> GeminiClientBundle:
    """thread 로컬에 Gemini 클라이언트를 캐시한다."""
    bundle = getattr(_THREAD_LOCAL, "client_bundle", None)
    if bundle is None:
        bundle = init_gemini_client(config)
        _THREAD_LOCAL.client_bundle = bundle
    return bundle




def _strip_code_fences(text: str) -> str:
    """LLM 출력의 코드 블록 펜스를 제거한다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _repair_prompt(bad_json: str) -> str:
    """잘못된 JSON을 수정하도록 요청하는 프롬프트를 만든다."""
    return (
        "You returned invalid JSON. Return a valid JSON array only, "
        "matching the required schema. No markdown or extra text.\n\n"
        f"Bad output:\n{bad_json}"
    )


def _chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """리스트를 고정 크기의 배치로 분할한다."""
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_prompt_template(prompt_version: Optional[str]) -> Tuple[str, str]:
    """config/judge/prompts.yaml에서 프롬프트 템플릿을 로드한다."""
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Judge prompt config not found: {PROMPTS_PATH}")
    payload = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Judge prompt config must be a mapping.")
    selected_version = prompt_version or next(iter(payload.keys()))
    entry = payload.get(selected_version)
    if entry is None:
        available = ", ".join(payload.keys())
        raise ValueError(
            f"Judge prompt template is missing: {selected_version} (available: {available})"
        )
    if isinstance(entry, dict):
        template = entry.get("template")
    else:
        template = entry
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Judge prompt template is empty: {selected_version}")
    return selected_version, template.strip()


def _evaluate_batch(
    *,
    config: ConfigBundle,
    response_schema: Dict[str, Any],
    prompt_template: str,
    batch: List[Dict[str, Any]],
    batch_index: int,
    batch_total: int,
    json_repair_attempts: int,
    verbose: bool,
) -> Dict[int, Dict[str, Any]]:
    """배치 단위로 LLM 평가를 실행하고 결과를 반환한다."""
    # 1. 배치 처리 시작 시간 측정
    seg_ids = [int(item.get("segment_id", -1)) for item in batch]
    if verbose:
        print(f"[JUDGE] batch {batch_index}/{batch_total} start (segments={seg_ids})")
    started = time.perf_counter()
    
    # 2. 클라이언트/프롬프트 준비
    client_bundle = _get_thread_client_bundle(config)
    prompt = _build_prompt(prompt_template, batch)
    
    # 3. LLM 호출 (Retry 로직 포함)
    llm_text = run_with_retries(
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
    
    # 4. JSON 파싱 및 구조 검증 (실패 시 복구 프롬프트로 재시도)
    for _ in range(json_repair_attempts + 1):
        try:
            payload = json.loads(_strip_code_fences(llm_text))
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
            # 수정 프롬프트로 재시도
            llm_text = run_with_retries(
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
    
    # 5. 처리 결과 반환
    if verbose:
        elapsed = time.perf_counter() - started
        print(f"[JUDGE] batch {batch_index}/{batch_total} done in {elapsed:.2f}s")
    return results


def _load_segments_units(path: Path) -> Dict[int, Dict[str, Any]]:
    """segments_units.jsonl을 segment_id 기준 dict로 로드한다."""
    segments: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        segments[int(segment_id)] = row
    return segments


def _load_segment_summaries(path: Path) -> Dict[int, Dict[str, Any]]:
    """segment_summaries.jsonl을 segment_id 기준 dict로 로드한다."""
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
    """매칭/누락된 segment_id 목록을 반환한다."""
    unit_ids = set(segments_units.keys())
    summary_ids = set(segment_summaries.keys())
    missing_units = sorted(summary_ids - unit_ids)
    missing_summaries = sorted(unit_ids - summary_ids)
    matched = sorted(unit_ids & summary_ids)
    return matched, missing_units, missing_summaries


def _build_prompt(prompt_template: str, segments_payload: List[Dict[str, Any]]) -> str:
    """프롬프트 템플릿에 입력 payload를 주입한다."""
    if PROMPT_PAYLOAD_TOKEN not in prompt_template:
        raise ValueError(f"Judge prompt missing token: {PROMPT_PAYLOAD_TOKEN}")
    payload_json = json.dumps(segments_payload, ensure_ascii=False)
    return prompt_template.replace(PROMPT_PAYLOAD_TOKEN, payload_json)


def _build_response_schema() -> Dict[str, Any]:
    """Gemini 응답 스키마를 구성한다."""
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
        "required": ["segment_id", "scores", "feedback"],
        "properties": {
            "segment_id": {"type": "integer"},
            "scores": score_schema,
            "feedback": {"type": "string"},
        },
    }
    return {"type": "array", "items": item_schema}


def _clamp_score(value: Any, max_score: int = MAX_SCORE) -> int:
    """점수를 [0, max_score] 범위로 정규화한다."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(max_score, score))


def _normalize_feedback(value: Any) -> str:
    """피드백 문자열을 한 줄로 정리한다."""
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _compute_final_score(groundedness: int, compliance: int, note_quality: int) -> float:
    """가중치 기반 최종 점수를 계산한다."""
    return round(0.45 * groundedness + 0.35 * note_quality + 0.20 * compliance, 2)


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
    verbose: bool,
    write_outputs: bool,
) -> Dict[str, Any]:
    """Judge를 실행하고 결과를 반환한다."""
    # 1. 데이터 로드 및 정합성 체크
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

    # 2. 평가 페이로드 구성
    payloads: List[Dict[str, Any]] = []
    for seg_id in matched_ids:
        payloads.append(
            {
                "segment_id": seg_id,
                "segments_units": segments_units[seg_id],
                "segment_summary": segment_summaries[seg_id],
            }
        )

    response_schema = _build_response_schema()
    prompt_version, prompt_template = _load_prompt_template(config.judge.prompt_version)

    # 3. 토큰 사용량 측정
    try:
        full_prompt = _build_prompt(prompt_template, payloads)
        client_bundle = init_gemini_client(config)
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
    
    # 4. 배치 평가 실행 (단일/병렬)
    if workers == 1 or len(batches) == 1:
        for idx, batch in enumerate(batches, start=1):
            llm_results.update(
                _evaluate_batch(
                    config=config,
                    response_schema=response_schema,
                    prompt_template=prompt_template,
                    batch=batch,
                    batch_index=idx,
                    batch_total=len(batches),
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
                    prompt_template=prompt_template,
                    batch=batch,
                    batch_index=idx,
                    batch_total=len(batches),
                    json_repair_attempts=json_repair_attempts,
                    verbose=verbose,
                )
                for idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                llm_results.update(future.result())

    # 5. 결과 집계 및 점수 계산
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
        groundedness = _clamp_score(llm_scores.get("groundedness"))      # 근거 충실도 (출처/컨텍스트 기반 응답)
        compliance = _clamp_score(llm_scores.get("compliance"))          # 프롬프트/규칙 준수도
        note_quality = _clamp_score(llm_scores.get("note_quality"))      # 결과물 품질 (명확성·구조·요약력)
        multimodal_use = _clamp_score(llm_scores.get("multimodal_use"))  # 멀티모달 정보 활용 적절성
        final_score = _compute_final_score(groundedness, compliance, note_quality)
        feedback = _normalize_feedback(llm_item.get("feedback"))
        if not feedback:
            raise ValueError(f"segment_id={seg_id} feedback is empty.")

        segment_reports.append(
            {
                "segment_id": seg_id,
                "scores": {
                    "groundedness": groundedness,
                    "compliance": compliance,
                    "note_quality": note_quality,
                    "multimodal_use": multimodal_use,
                    "final": final_score,
                },
                "feedback": feedback,
                "meta": {
                    "model": config.raw.llm_gemini.model,
                    "prompt_version": prompt_version,
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
            "prompt_version": prompt_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    result = {"report": report, "segment_reports": segment_reports}
    if write_outputs:
        report["meta"]["segments_units_path"] = str(segments_units_path)
        report["meta"]["segment_summaries_path"] = str(segment_summaries_path)
        write_json(output_report_path, report)
        write_jsonl(output_segments_path, segment_reports)
    return result
