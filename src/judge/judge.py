"""
[Intent - 모듈 목적]
Fusion 파이프라인에서 생성된 세그먼트 요약(Segment Summary)을 LLM 기반으로 평가하는 Judge 모듈입니다.
Groundedness(근거 충실도), Note Quality(노트 품질), Multimodal Use(멀티모달 활용도)를 기준으로
0-10점 척도로 평가하며, 최종 점수는 Groundedness와 Note Quality의 가중 평균으로 계산됩니다.

[Usage - 활용처]
1. 파이프라인 내부 호출:
   - src/pipeline/stages.py의 process_fusion_pipeline() → run_judge()
   - src/pipeline/fusion_worker_async.py의 AsyncFusionSummaryJudgeWorker → run_judge()

2. 독립 실행:
   - python -m src.judge --segments-units <path> --segment-summaries <path>

3. API 호출:
   - 서버 환경에서 Supabase 연동 시 processing_job_id 기반 평가 결과 저장

[Usage Method - 사용 방식]
1. 입력 파일 준비:
   - segments_units.jsonl: Fusion 단계에서 생성된 세그먼트별 원본 데이터
   - segment_summaries.jsonl: Summarizer 단계에서 생성된 세그먼트별 요약

2. 설정 파일:
   - config/judge/settings.yaml: 배치 크기, 워커 수, JSON 복구 시도 횟수 등
   - config/fusion/settings.yaml: LLM 설정 (모델명, 타임아웃 등)

3. 실행 흐름:
   run_judge()
     → [1단계] 데이터 로드 및 정합성 체크
     → [2단계] 평가 페이로드 구성
     → [3단계] 프롬프트 템플릿 로드
     → [4단계] 배치 평가 실행 (병렬)
     → [5단계] 결과 집계 및 점수 계산
     → [6단계] 결과 저장 (옵션)
"""

from __future__ import annotations

# 표준 라이브러리
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# 외부 라이브러리
import yaml

# 프로젝트 경로 설정
ROOT = Path(__file__).resolve().parents[2]

# 내부 모듈
from src.fusion.config import ConfigBundle
from src.fusion.gemini import GeminiClientBundle, init_gemini_client, run_with_retries
from src.fusion.io_utils import read_jsonl, update_token_usage, write_json, write_jsonl

# 설정 경로 및 상수
PROMPT_PAYLOAD_TOKEN = "{{SEGMENTS_JSON}}"

# 스레드 로컬 스토리지

_THREAD_LOCAL = threading.local()


# ========================================
# [섹션 1: 유틸리티 함수 - 범용 헬퍼]
# ========================================

def _get_timestamp() -> str:
    """
    [사용 파일]
    - src/judge/judge.py의 _evaluate_batch() (로그 출력용)
    - src/judge/judge.py의 run_judge() (로그 출력용)

    [목적]
    현재 시각을 [YYYY-MM-DD | HH:MM:SS.mmm] 형식의 문자열로 반환하여
    Judge 실행 로그에서 배치별 시작/완료 시간을 추적합니다.

    [Returns]
    str: [YYYY-MM-DD | HH:MM:SS.mmm] 형식 타임스탬프
    """
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


def _clamp_score(value: Any, max_score: int) -> int:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (결과 집계 시)

    [목적]
    LLM이 반환한 점수를 [0, max_score] 범위로 정규화합니다.
    비정상 값(None, 문자열 등)을 안전하게 처리하여 항상 유효한 정수를 반환합니다.

    [Args 설명]
    value (Any): LLM에서 반환된 점수 값
    max_score (int): 최대 점수

    [Returns]
    int: 정규화된 점수 (0 이상 max_score 이하)
    """
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(max_score, score))


def _normalize_feedback(value: Any) -> str:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (결과 집계 시)

    [목적]
    LLM이 반환한 피드백 문자열을 한 줄로 정리합니다.
    줄바꿈, 탭, 연속된 공백을 단일 공백으로 변환하여 JSON 저장 시 가독성을 높입니다.

    [Args 설명]
    value (Any): LLM에서 반환된 피드백 값

    [Returns]
    str: 한 줄로 정리된 피드백 문자열
    """
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _compute_final_score(groundedness: int, note_quality: int) -> float:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (결과 집계 시)

    [목적]
    Groundedness와 Note Quality의 가중 평균으로 최종 점수를 계산합니다.
    Judge v3 기준 가중치는 각각 50%입니다.

    [Args 설명]
    groundedness (int): 근거 충실도 점수 (0-10)
    note_quality (int): 노트 품질 점수 (0-10)

    [Returns]
    float: 최종 점수 (소수점 둘째 자리까지 반올림)
    """
    return round(0.50 * groundedness + 0.50 * note_quality, 2)


def _chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (배치 분할 시)

    [목적]
    세그먼트 리스트를 고정 크기의 배치로 분할합니다.
    병렬 처리 및 LLM API 호출 최적화를 위해 사용됩니다.

    [Args 설명]
    items (List[Dict[str, Any]]): 분할할 세그먼트 페이로드 리스트
    size (int): 배치 크기

    [Returns]
    List[List[Dict[str, Any]]]: 배치 리스트
    """
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _strip_code_fences(text: str) -> str:
    """
    [사용 파일]
    - src/judge/judge.py의 _evaluate_batch() (LLM 응답 파싱 시)

    [목적]
    LLM이 반환한 JSON 응답에서 마크다운 코드 블록 펜스(```)를 제거합니다.
    일부 LLM이 JSON을 ```json ... ``` 형태로 감싸 반환하는 경우를 처리합니다.

    [Args 설명]
    text (str): LLM 응답 원문

    [Returns]
    str: 코드 펜스가 제거된 순수 JSON 문자열
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


# ========================================
# [섹션 2: 클라이언트 관리]
# ========================================

def _get_thread_client_bundle(config: ConfigBundle) -> GeminiClientBundle:
    """
    [사용 파일]
    - src/judge/judge.py의 _evaluate_batch() (배치 평가 시)

    [목적]
    스레드 로컬 스토리지에 Gemini 클라이언트를 캐시하여 재사용합니다.
    병렬 처리(ThreadPoolExecutor) 시 각 스레드마다 독립적인 클라이언트를 유지하며,
    중복 생성을 방지하여 성능을 최적화합니다.

    [연결 여부]
    - API 연결: Gemini API (init_gemini_client 호출 시)
    - 파일 I/O: 없음
    - DB 연결: 없음

    [Args 설명]
    config (ConfigBundle): Fusion 설정 번들

    [Returns]
    GeminiClientBundle: Gemini 클라이언트 번들
    """
    bundle = getattr(_THREAD_LOCAL, "client_bundle", None)
    if bundle is None:
        bundle = init_gemini_client(config)
        _THREAD_LOCAL.client_bundle = bundle
    return bundle


# ========================================
# [섹션 3: 데이터 로드 함수]
# ========================================

def _load_segments_units(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (1단계: 데이터 로드)

    [목적]
    Fusion 단계에서 생성된 segments_units.jsonl 파일을 읽어
    segment_id를 키로 하는 딕셔너리로 변환합니다.

    [연결 여부]
    - API 연결: 없음
    - 파일 I/O: segments_units.jsonl 읽기
    - DB 연결: 없음

    [Args 설명]
    path (Path): segments_units.jsonl 파일 경로

    [Returns]
    Dict[int, Dict[str, Any]]: segment_id → 세그먼트 데이터 매핑
    """
    segments: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        segment_id = row.get("segment_id")
        if segment_id is None:
            continue
        segments[int(segment_id)] = row
    return segments


def _load_segment_summaries(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (1단계: 데이터 로드)

    [목적]
    Summarizer 단계에서 생성된 segment_summaries.jsonl 파일을 읽어
    segment_id를 키로 하는 딕셔너리로 변환합니다.

    [연결 여부]
    - API 연결: 없음
    - 파일 I/O: segment_summaries.jsonl 읽기
    - DB 연결: 없음

    [Args 설명]
    path (Path): segment_summaries.jsonl 파일 경로

    [Returns]
    Dict[int, Dict[str, Any]]: segment_id → 요약 데이터 매핑
    """
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
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (1단계: 데이터 정합성 체크)

    [목적]
    segments_units와 segment_summaries의 segment_id를 비교하여
    매칭되는 ID와 누락된 ID를 파악합니다. 데이터 정합성 검증에 사용됩니다.

    [Args 설명]
    segments_units (Dict[int, Dict[str, Any]]): segment_id → units 데이터
    segment_summaries (Dict[int, Dict[str, Any]]): segment_id → summary 데이터

    [Returns]
    Tuple[List[int], List[int], List[int]]: (matched, missing_units, missing_summaries)
    """
    unit_ids = set(segments_units.keys())
    summary_ids = set(segment_summaries.keys())
    missing_units = sorted(summary_ids - unit_ids)
    missing_summaries = sorted(unit_ids - summary_ids)
    matched = sorted(unit_ids & summary_ids)
    return matched, missing_units, missing_summaries


# ========================================
# [섹션 4: 프롬프트 및 스키마 구성]
# ========================================

def _load_prompt_template(prompts_path: Path, prompt_version: Optional[str]) -> Tuple[str, str]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (2단계: 프롬프트 로드)

    [목적]
    지정된 경로의 prompts.yaml에서 Judge 프롬프트 템플릿을 로드합니다.
    Legacy 형식(template 키)과 Modular 형식(system/criteria/protocol 등 키)을 모두 지원합니다.

    [연결 여부]
    - API 연결: 없음
    - 파일 I/O: prompts.yaml 읽기
    - DB 연결: 없음

    [Args 설명]
    prompts_path (Path): prompts.yaml 파일 경로
    prompt_version (Optional[str]): 사용할 프롬프트 버전 (None이면 첫 번째 키 사용)

    [Returns]
    Tuple[str, str]: (selected_version, prompt_template)
    """
    if not prompts_path.exists():
        raise FileNotFoundError(f"Judge prompt config not found: {prompts_path}")
    payload = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Judge prompt config must be a mapping.")
    selected_version = prompt_version or next(iter(payload.keys()))
    entry = payload.get(selected_version)
    if entry is None:
        available = ", ".join(payload.keys())
        raise ValueError(
            f"Judge prompt template is missing: {selected_version} (available: {available})"
        )
    
    # Legacy 형식: template 키가 있는 경우
    if isinstance(entry, dict):
        template = entry.get("template")
        if isinstance(template, str) and template.strip():
            return selected_version, template.strip()
        
        # Modular 형식: system/criteria/protocol/input_format/output_format 키가 있는 경우
        parts = []
        for key in ["system", "criteria", "protocol", "input_format", "output_format"]:
            if key in entry:
                value = entry[key]
                if isinstance(value, str) and value.strip():
                    parts.append(f"## {key.upper().replace('_', ' ')}\n{value.strip()}")
        
        if parts:
            return selected_version, "\n\n".join(parts)
    elif isinstance(entry, str) and entry.strip():
        return selected_version, entry.strip()
    
    raise ValueError(f"Judge prompt template is empty: {selected_version}")


def _build_response_schema() -> Dict[str, Any]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (2단계: 스키마 구성)
    - src/judge/judge.py의 _evaluate_batch() (LLM 호출 시)

    [목적]
    Gemini API의 구조화된 출력(Structured Output)을 위한 JSON 스키마를 생성합니다.
    LLM이 반드시 지정된 형식의 JSON 배열을 반환하도록 강제합니다.

    [연결 여부]
    - API 연결: Gemini API (response_schema 파라미터로 전달)
    - 파일 I/O: 없음
    - DB 연결: 없음

    [Returns]
    Dict[str, Any]: JSON Schema 정의
    """
    score_schema = {
        "type": "object",
        "required": ["groundedness", "note_quality", "multimodal_use"],
        "properties": {
            "groundedness": {"type": "integer"},
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


def _build_prompt(prompt_template: str, segments_payload: List[Dict[str, Any]]) -> str:
    """
    [사용 파일]
    - src/judge/judge.py의 _evaluate_batch() (LLM 호출 시)

    [목적]
    프롬프트 템플릿에 평가 대상 세그먼트 데이터를 주입하여 최종 프롬프트를 생성합니다.
    템플릿의 {{SEGMENTS_JSON}} 토큰을 실제 JSON 데이터로 대체합니다.

    [Args 설명]
    prompt_template (str): Judge 프롬프트 템플릿 원문
    segments_payload (List[Dict[str, Any]]): 평가할 세그먼트 데이터 리스트

    [Returns]
    str: 세그먼트 데이터가 주입된 최종 프롬프트
    """
    if PROMPT_PAYLOAD_TOKEN not in prompt_template:
        raise ValueError(f"Judge prompt missing token: {PROMPT_PAYLOAD_TOKEN}")
    payload_json = json.dumps(segments_payload, ensure_ascii=False)
    return prompt_template.replace(PROMPT_PAYLOAD_TOKEN, payload_json)


def _repair_prompt(bad_json: str) -> str:
    """
    [사용 파일]
    - src/judge/judge.py의 _evaluate_batch() (JSON 파싱 실패 시)

    [목적]
    LLM이 반환한 JSON이 파싱 불가능할 때, 수정을 요청하는 프롬프트를 생성합니다.
    잘못된 JSON 원문을 포함하여 LLM이 오류를 인식하고 수정하도록 유도합니다.

    [Args 설명]
    bad_json (str): LLM이 반환한 잘못된 JSON 문자열

    [Returns]
    str: JSON 수정 요청 프롬프트
    """
    return (
        "You returned invalid JSON. Return a valid JSON array only, "
        "matching the required schema. No markdown or extra text.\n\n"
        f"Bad output:\n{bad_json}"
    )


# ========================================
# [섹션 5: 배치 평가 실행]
# ========================================

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
    batch_label: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """
    [사용 파일]
    - src/judge/judge.py의 run_judge() (4단계: 배치 평가)

    [목적]
    세그먼트 배치를 LLM에 전달하여 평가하고, JSON 응답을 파싱하여 결과를 반환합니다.
    JSON 파싱 실패 시 복구 프롬프트로 재시도하여 안정성을 높입니다.

    [연결 여부]
    - API 연결: Gemini API (run_with_retries 호출)
    - 파일 I/O: 없음
    - DB 연결: 없음

    [Args 설명]
    config (ConfigBundle): Fusion 설정 번들
    response_schema (Dict[str, Any]): Gemini 응답 스키마
    prompt_template (str): Judge 프롬프트 템플릿
    batch (List[Dict[str, Any]]): 평가할 세그먼트 배치
    batch_index (int): 현재 배치 인덱스
    batch_total (int): 전체 배치 수
    json_repair_attempts (int): JSON 복구 재시도 횟수
    verbose (bool): 상세 로그 출력 여부
    batch_label (Optional[str]): 배치 라벨

    [Returns]
    Tuple[Dict[int, Dict[str, Any]], int]: (llm_results, total_tokens)
    """
    # 1. 배치 처리 시작 시간 측정
    seg_ids = [int(item.get("segment_id", -1)) for item in batch]
    if verbose:
        print(f"{_get_timestamp()} [JUDGE] Batch {batch_index}/{batch_total} start (segments={seg_ids})")
    started = time.perf_counter()
    
    # 2. 클라이언트/프롬프트 준비
    client_bundle = _get_thread_client_bundle(config)
    prompt = _build_prompt(prompt_template, batch)
    
    # 3. LLM 호출 (Retry 로직 포함)
    llm_text, total_tokens, _ = run_with_retries(
        client_bundle,
        prompt,
        response_schema,
        temperature=config.judge.temperature,
        response_mime_type=config.raw.llm_gemini.response_mime_type,
        timeout_sec=config.raw.llm_gemini.timeout_sec,
        max_retries=config.raw.llm_gemini.max_retries,
        backoff_sec=config.raw.llm_gemini.backoff_sec,
        context=f"{batch_label}: Judge" if batch_label else "Judge",
        verbose=verbose,
        role="judge",
        key_fail_cooldown_sec=config.raw.llm_gemini.key_fail_cooldown_sec,
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
                if verbose:
                    prefix = f"{batch_label}: " if batch_label else "      "
                    print(f"{_get_timestamp()} {prefix}- [Judge] Evaluated segment {seg_id}", flush=True)
                results[seg_id] = item
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            # 수정 프롬프트로 재시도
            llm_text, tokens, _ = run_with_retries(
                client_bundle,
                _repair_prompt(llm_text),
                response_schema,
                temperature=config.judge.temperature,
                response_mime_type=config.raw.llm_gemini.response_mime_type,
                timeout_sec=config.raw.llm_gemini.timeout_sec,
                max_retries=config.raw.llm_gemini.max_retries,
                backoff_sec=config.raw.llm_gemini.backoff_sec,
                context=f"{batch_label}: [Judge]" if batch_label else "Judge",
                verbose=verbose,
                role="judge",
                key_fail_cooldown_sec=config.raw.llm_gemini.key_fail_cooldown_sec,
            )
            total_tokens += tokens
    if last_error:
        raise RuntimeError(f"LLM JSON parse failed: {last_error}")
    
    # 5. 처리 결과 반환
    if verbose:
        elapsed = time.perf_counter() - started
        print(f"{_get_timestamp()} [JUDGE] Batch {batch_index}/{batch_total} done in {elapsed:.2f}s")
    return results, total_tokens


# ========================================
# [섹션 6: 메인 함수]
# ========================================

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
    status_callback: Optional[Callable[[int], None]] = None,
    batch_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [사용 파일]
    - src/pipeline/stages.py의 process_fusion_pipeline()
    - src/pipeline/fusion_worker_async.py의 AsyncFusionSummaryJudgeWorker
    - src/judge/__main__.py

    [목적]
    Fusion 파이프라인에서 생성된 세그먼트 요약을 LLM으로 평가하는 Judge 모듈의 메인 함수입니다.
    데이터 로드 → 프롬프트 구성 → 배치 평가 → 결과 집계 → 파일 저장의 전체 흐름을 관리합니다.

    [연결 여부]
    - API 연결: Gemini API (LLM 평가 및 토큰 카운팅)
    - 파일 I/O:
      - 입력: segments_units.jsonl, segment_summaries.jsonl
      - 출력: judge_report.json, judge_segments.jsonl (write_outputs=True 시)
    - DB 연결: 없음

    [Args 설명]
    config (ConfigBundle): Fusion 설정 번들
    segments_units_path (Path): segments_units.jsonl 파일 경로
    segment_summaries_path (Path): segment_summaries.jsonl 파일 경로
    output_report_path (Path): 전체 평가 리포트 저장 경로
    output_segments_path (Path): 세그먼트별 평가 결과 저장 경로
    batch_size (int): 배치 크기
    workers (int): 병렬 평가 스레드 수
    json_repair_attempts (int): JSON 복구 재시도 횟수
    limit (Optional[int]): 평가할 세그먼트 수 제한 (테스트용)
    verbose (bool): 상세 로그 출력 여부
    write_outputs (bool): 결과 파일 저장 여부
    status_callback (Optional[Callable]): 토큰 사용량 콜백
    batch_label (Optional[str]): 배치 라벨

    [Returns]
    Dict[str, Any]: Judge 실행 결과 (report, segment_reports, token_usage 포함)
    """
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

    # 2. 평가 페이로드 구성 (Token 절약을 위해 불필요한 필드 제거)
    payloads: List[Dict[str, Any]] = []
    for seg_id in matched_ids:
        unit = segments_units[seg_id].copy()
        
        # transcript_units가 있으면 전체 텍스트인 transcript_text는 중복이므로 제거
        if "transcript_units" in unit and unit["transcript_units"]:
            unit.pop("transcript_text", None)
            
        payloads.append(
            {
                "segment_id": seg_id,
                "segments_units": unit,
                "segment_summary": segment_summaries[seg_id],
            }
        )


def _validate_segment(unit: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    [Purpose]
    세그먼트 요약의 정합성(ID 존재 여부, 금지어 등)을 시스템 레벨에서 검증합니다.

    [Args]
    unit (Dict): 원본 데이터 (transcript_units, visual_units 포함)
    summary (Dict): LLM이 생성한 요약 (evidence_refs 포함)

    [Returns]
    Dict: validation_report (valid refs, compliance issues, critical_fail 여부 등)
    """
    # 1. ID 검증 (Reference Check)
    evidence_refs = summary.get("evidence_refs", [])
    if not isinstance(evidence_refs, list):
        evidence_refs = []

    stt_ids = {u.get("id") for u in unit.get("transcript_units", []) if u.get("id")}
    vlm_ids = {u.get("id") for u in unit.get("visual_units", []) if u.get("id")}
    
    invalid_refs = []
    stt_ref_count = 0
    vlm_ref_count = 0

    for ref in evidence_refs:
        if not isinstance(ref, str):
            continue
        if ref in stt_ids:
            stt_ref_count += 1
        elif ref in vlm_ids:
            vlm_ref_count += 1
        else:
            invalid_refs.append(ref)

    refs_status = "VALID" if not invalid_refs else "INVALID"

    # 2. 규칙 준수 (Compliance Check)
    banned_words = ["슬라이드", "화면", "그림", "보시면", "위/아래", "여기", "방금", "앞에서", "다음으로", "이 수식", "마지막 항"]
    content_text = json.dumps(summary, ensure_ascii=False)
    
    compliance_issues = []
    # 3. 필수 키 검증 (Missing Keys)
    # - summary: Critical Failure (평가 불가)
    # - confidence, source_type: Compliance Issue (감점 요인)
    # [Update for v4.1] confidence/source_type are now nested in bullets/definitions/explanations
    critical_keys = ["summary"]
    
    # Check top-level critical keys
    for key in critical_keys:
        if key not in summary or not summary[key]:
            compliance_issues.append(f"Missing critical key: {key}")
            
    # Check nested recommended keys (confidence, source_type)
    # v4.1 구조: summary -> bullets -> [ {confidence, ...}, ... ]
    required_nested_keys = ["confidence", "source_type"]
    critical_nested_errors = []
    sub_lists = ["bullets", "definitions", "explanations"]
    
    if "summary" in summary and isinstance(summary["summary"], dict):
        summary_content = summary["summary"]
        for list_key in sub_lists:
            if list_key in summary_content and isinstance(summary_content[list_key], list):
                for idx, item in enumerate(summary_content[list_key]):
                    for req_key in required_nested_keys:
                        if req_key not in item:
                            critical_nested_errors.append(f"Missing key in {list_key}[{idx}]: {req_key}")
    
    for err in critical_nested_errors:
        compliance_issues.append(err)

    compliance_status = "PASS" if not compliance_issues else "ISSUES"

    # 4. 치명적 오류 (Critical Failure) 판별
    # - 요약 내용(summary)이 아예 없는 경우
    # - 필수 메타데이터(confidence, source_type)가 누락된 경우 (Strict Mode)
    # - [Optimized] 참조 오류(invalid_refs)가 있는 경우도 Critical로 처리하여 LLM 비용 절감
    critical_fail = False
    if "summary" not in summary or not summary["summary"]:
        critical_fail = True
    if critical_nested_errors:
        critical_fail = True
    if invalid_refs:
        critical_fail = True
    
    return {
        "refs_status": refs_status,
        "invalid_refs": invalid_refs,
        "stt_ref_count": stt_ref_count,
        "vlm_ref_count": vlm_ref_count,
        "compliance_status": compliance_status,
        "compliance_issues": compliance_issues,
        "critical_fail": critical_fail,
    }

def run_judge(
    segments_units_path: Path,
    segment_summaries_path: Path,
    output_report_json: Path,
    output_segments_jsonl: Path,
    config: ConfigBundle,
    limit: Optional[int] = None,
    verbose: bool = False,
    write_outputs: bool = True,
):
    """
    [Intent]
    Judge 모듈의 메인 엔트리포인트입니다.
    데이터 로드 -> 프롬프트 준비 -> 배치 평가 -> 결과 집계를 수행합니다.
    """
    t0 = _get_timestamp()
    if verbose:
        print(f"[{t0}] Starting Judge module...")

    # 1. 데이터 로드 및 병합
    units_map = _load_segments_units(segments_units_path)
    if verbose:
        print(f"Loaded {len(units_map)} segment units.")

    summaries = _load_segment_summaries(segment_summaries_path)
    if verbose:
        print(f"Loaded {len(summaries)} segment summaries.")

    matched_ids, missing_units_ids, missing_summaries_ids = _merge_segments(units_map, summaries)
    
    # 데이터 매핑 재구성 (Loop에서 사용)
    merged_data = {
        seg_id: (units_map[seg_id], summaries[seg_id]) 
        for seg_id in matched_ids
    }

    if limit and limit > 0:
        matched_ids = matched_ids[:limit]
        if verbose:
            print(f"Limiting evaluation to first {limit} segments.")

    missing_units = len(missing_units_ids)
    missing_summaries = len(missing_summaries_ids)

    # 2. 프롬프트 템플릿 로드
    prompt_version, prompt_template = _load_prompt_template(
        config.judge_prompts_path, config.judge.prompt_version
    )
    if verbose:
        print(f"Using prompt version: {prompt_version}")

    response_schema = _build_response_schema()

    # 3. 평가 실행 (배치 단위 + System Validation Short-circuit)
    llm_results: Dict[int, Any] = {}
    batch_size = config.judge.batch_size
    batches = list(_chunked(matched_ids, batch_size))
    total_judge_tokens = 0
    
    # 설정: System Filter 활성화 여부 (Config에 없으면 기본값 False 처리)
    enable_system_filter = getattr(config.judge, "enable_system_filter", False)

    if verbose:
        print(
            f"{_get_timestamp()} [JUDGE] batches={len(batches)} batch_size={batch_size} workers={config.judge.workers} sys_filter={enable_system_filter}"
        )

    with ThreadPoolExecutor(max_workers=config.judge.workers) as executor:
        future_to_batch = {}
        
        for idx, batch_ids in enumerate(batches, start=1):
            # 3-1. System Validation 및 Payload 구성
            batch_items = []
            valid_batch_ids = []
            skipped_results = {} # seg_id -> result dict

            for seg_id in batch_ids:
                unit, summary = merged_data[seg_id]
                val_report = _validate_segment(unit, summary)
                
                # 최적화: Critical Failure 시 LLM 스킵
                if enable_system_filter and val_report["critical_fail"]:
                    skipped_results[seg_id] = {
                        "segment_id": seg_id,
                        "scores": {
                            "groundedness": 0,
                            "note_quality": 0,
                            "multimodal_use": 0
                        },
                        "feedback": f"SYSTEM_FAIL: {'; '.join(val_report['compliance_issues'] or ['Critical Error'])}"
                    }
                    continue

                # 정상 항목은 LLM Payload 구성
                item_payload = {
                    "segment_id": seg_id,
                    "validation_report": {
                        "refs_status": val_report["refs_status"],
                        "invalid_refs": val_report["invalid_refs"],
                        "compliance_status": val_report["compliance_status"],
                        "compliance_issues": val_report["compliance_issues"],
                        "vlm_count": val_report["vlm_ref_count"]
                    },
                    "source_refs": {
                        "stt_ids": sorted(list({u.get("id") for u in unit.get("transcript_units", []) if u.get("id")})),
                        "vlm_ids": sorted(list({u.get("id") for u in unit.get("visual_units", []) if u.get("id")}))
                    },
                    "summary": summary
                }
                batch_items.append(item_payload)
                valid_batch_ids.append(seg_id)
            
            # 미리 처리된 결과(Short-circuit) 저장
            llm_results.update(skipped_results)

            # LLM 호출이 필요한 항목이 있다면 실행
            if batch_items:
                # _evaluate_batch는 내부적으로 _build_prompt를 호출하므로 여기서 직접 호출하지 않고 템플릿과 데이터를 전달합니다.
                future = executor.submit(
                    _evaluate_batch, 
                    config=config,
                    response_schema=response_schema,
                    prompt_template=prompt_template,
                    batch=batch_items, # Pass the PROCESSED items with validation report
                    batch_index=idx,
                    batch_total=len(batches),
                    json_repair_attempts=config.judge.json_repair_attempts,
                    verbose=verbose,
                    batch_label=f"Batch {idx}/{len(batches)}"
                )
                future_to_batch[future] = valid_batch_ids
        # 결과 수집
        for future in as_completed(future_to_batch):
            batch_ids = future_to_batch[future]
            try:
                batch_result, batch_tokens = future.result()
                llm_results.update(batch_result)
                total_judge_tokens += batch_tokens
            except Exception as exc:
                print(f"Batch {batch_ids} generated an exception: {exc}")
                # 에러 발생 시 0점 처리
                for seg_id in batch_ids:
                    if seg_id not in llm_results:
                         llm_results[seg_id] = {
                            "scores": {"groundedness": 0, "note_quality": 0, "multimodal_use": 0}, 
                            "feedback": "Evaluation Failed"
                        }

    # 5. 결과 집계 및 점수 계산
    segment_reports: List[Dict[str, Any]] = []
    groundedness_scores: List[int] = []
    note_quality_scores: List[int] = []
    multimodal_use_scores: List[int] = []
    final_scores: List[float] = []

    for seg_id in matched_ids:
        llm_item = llm_results.get(seg_id)
        if not llm_item:
            raise ValueError(f"Missing LLM result for segment_id={seg_id}")
        llm_scores = llm_item.get("scores", {}) if isinstance(llm_item, dict) else {}
        groundedness = _clamp_score(llm_scores.get("groundedness"), config.judge.max_score)
        note_quality = _clamp_score(llm_scores.get("note_quality"), config.judge.max_score)
        multimodal_use = _clamp_score(llm_scores.get("multimodal_use"), config.judge.max_score)
        final_score = _compute_final_score(groundedness, note_quality)
        feedback = _normalize_feedback(llm_item.get("feedback"))
        if not feedback:
            raise ValueError(f"segment_id={seg_id} feedback is empty.")

        segment_reports.append(
            {
                "segment_id": seg_id,
                "scores": {
                    "groundedness": groundedness,
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
        note_quality_scores.append(note_quality)
        multimodal_use_scores.append(multimodal_use)
        final_scores.append(final_score)

    avg_groundedness = round(sum(groundedness_scores) / len(groundedness_scores), 2)
    avg_note_quality = round(sum(note_quality_scores) / len(note_quality_scores), 2)
    avg_multimodal_use = round(sum(multimodal_use_scores) / len(multimodal_use_scores), 2)
    avg_final = round(sum(final_scores) / len(final_scores), 2)

    report: Dict[str, Any] = {
        "score_scale": {"min": 0, "max": config.judge.max_score},
        "scores_avg": {
            "groundedness": avg_groundedness,
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

    if verbose:
        print(f"{_get_timestamp()} [JUDGE] Total Tokens: {total_judge_tokens}")

    result = {
        "report": report, 
        "segment_reports": segment_reports,
        "token_usage": {"total_tokens": total_judge_tokens}
    }
    if write_outputs:
        report["meta"]["segments_units_path"] = str(segments_units_path)
        report["meta"]["segment_summaries_path"] = str(segment_summaries_path)
        write_json(output_report_json, report)
        write_jsonl(output_segments_jsonl, segment_reports)
    return result
