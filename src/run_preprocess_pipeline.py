"""
[Intent]
STT(음성 인식)와 비디오 캡처 과정을 통합하여 실행하는 전처리 파이프라인의 메인 진입점입니다.
오디오 추출, STT 분석, 슬라이드 캡처 작업을 병렬 또는 순차적으로 수행하며,
결과물을 로컬 파일 시스템에 저장하거나 Supabase DB와 연동하여 관리합니다.

[Usage]
- 터미널이나 외부 스크립트에서 `--video` 인자와 함께 호출하여 실행합니다.
- 예: python src/run_preprocess_pipeline.py --video data/input/lecture.mp4 --local-json

[Usage Method]
1. CLI 인자를 파싱하여 실행 모드(병렬 여부, DB 동기화 여부 등)를 결정합니다.
2. 입력 비디오의 메타데이터를 확인하고 출력 디렉토리를 생성합니다.
3. run_preprocess_pipeline() 함수를 통해 실제 작업을 오케스트레이션합니다.
4. 작업 완료 후 벤치마크 리포트를 생성하고 최종 상태를 기록합니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 내부 모듈 임포트
from src.audio.extract_audio import extract_audio
from src.capture.settings import load_capture_settings
from src.db.pipeline_sync import (
    finalize_preprocess_db_sync,
    prepare_preprocess_db_sync,
    sync_preprocess_artifacts_to_db,
)
from src.pipeline.benchmark import (
    BenchmarkTimer,
    get_video_info,
    print_benchmark_report,
)
from src.pipeline.logger import pipeline_logger
from src.pipeline.stages import run_capture, run_stt, run_stt_from_storage

# 환경 변수 로딩
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def _sanitize_video_name(stem: str) -> str:
    """[Purpose] 파일명을 안전한 디렉토리 명으로 변환합니다 (공백 및 특수문자 제거)."""
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"


def _validate_and_log_path(path_obj: Path, label: str) -> Path:
    """
    [Purpose] 경로가 허용된 디렉토리(data/inputs, data/outputs) 하위에 있는지 검증하고 로깅합니다.
    [Args]
    - path_obj (Path): 검증할 경로 객체
    - label (str): 로깅 시 표시할 라벨
    """
    abs_path = path_obj.resolve()
    # 프로젝트 루트 기준 data/inputs, data/outputs 절대 경로
    allowed_inputs = (ROOT / "data" / "inputs").resolve()
    allowed_outputs = (ROOT / "data" / "outputs").resolve()
    
    # data/inputs 또는 data/outputs 하위인지 확인 (자신 포함)
    is_valid = False
    if abs_path == allowed_inputs or abs_path == allowed_outputs:
        is_valid = True
    else:
        try:
            # relative_to는 하위 경로일 때만 성공함
            abs_path.relative_to(allowed_inputs)
            is_valid = True
        except ValueError:
            try:
                abs_path.relative_to(allowed_outputs)
                is_valid = True
            except ValueError:
                pass
            
    if not is_valid:
        # 절대 경로가 아닌 프로젝트 루트 기준 상대 경로로 표시하여 개인정보 보호
        try:
            rel_path = abs_path.relative_to(ROOT)
        except ValueError:
            rel_path = abs_path
            
        error_msg = f"[Path Error] {label}이 허용된 범위를 벗어났습니다: {rel_path}\n(허용 범위: data/inputs 또는 data/outputs)"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # 정상 경로도 상대 경로로 출력
    try:
        display_path = abs_path.relative_to(ROOT)
    except ValueError:
        display_path = abs_path
        
    print(f"[Path Check] {label} verified: {display_path}")
    return abs_path


def _append_benchmark_report(path: Path, report_md: str, pipeline_label: str) -> None:
    """[Purpose] 기존 벤치마크 리포트 파일이 있으면 뒤에 내용을 추가합니다."""
    timestamp = datetime.now(timezone.utc).isoformat()
    if path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n---\n")
            handle.write(f"Benchmark Append: {pipeline_label} | {timestamp}\n\n")
            handle.write(report_md)
    else:
        path.write_text(report_md, encoding="utf-8")


def run_preprocess_pipeline(
    *,
    video: Optional[str],
    output_base: str = "data/outputs",
    stt_backend: str = "clova",
    parallel: bool = True,
    capture_threshold: Optional[float] = None,
    capture_dedupe_threshold: Optional[float] = None,
    capture_min_interval: Optional[float] = None,
    capture_dedup_enabled: bool = True,
    capture_verbose: bool = False,
    limit: Optional[int] = None,
    write_local_json: Optional[bool] = None,
    sync_to_db: Optional[bool] = None,
    db_table_name: str = "captures",
) -> Optional[str]:
    """
    [Usage File] main()
    [Purpose] 오디오 추출, STT, 캡처를 포함한 전체 전처리 과정을 제어합니다.
    [Connection]
    - src.pipeline.stages: 개별 단계(run_stt, run_capture 등) 호출
    - src.db.pipeline_sync: DB 동기화 처리
    - src.pipeline.benchmark: 실행 통계 기록
    
    [Args]
    - video (str): 입력 비디오 경로
    - output_base (str): 결과물 저장 루트 경로
    - parallel (bool): 병렬 처리 엔진 사용 여부
    - sync_to_db (bool): 결과물을 DB로 자동 전송할지 여부
    
    [Returns]
    - Optional[str]: DB 동기화 성공 시 생성된 video_id, 아니면 None
    """
    # 1. 설정 및 경로 초기화
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    
    pipeline_config = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    
    video_path_str = video or pipeline_config.get("video", {}).get("preprocess_path")
    if not video_path_str:
        raise ValueError("입력 비디오 경로가 지정되지 않았습니다.")

    video_path = Path(video_path_str).expanduser().resolve()
    _validate_and_log_path(video_path, "Input Video")
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    # 캡처 설정 보정 (settings.yaml의 기본값 활용)
    capture_settings = load_capture_settings()
    capture_threshold = capture_threshold if capture_threshold is not None else capture_settings.persistence_drop_ratio
    capture_dedupe_threshold = capture_dedupe_threshold if capture_dedupe_threshold is not None else capture_settings.sensitivity_diff
    capture_min_interval = capture_min_interval if capture_min_interval is not None else capture_settings.min_interval
    
    # 출력 경로 생성 및 검증
    output_base_path = (ROOT / Path(output_base)).resolve()
    _validate_and_log_path(output_base_path, "Output Base")
    
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base_path / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    # 벤치마크/로깅 준비
    timer = BenchmarkTimer()
    video_info = get_video_info(video_path)
    timer.start_total()

    # 2. DB 동기화 컨텍스트 준비
    db_context: Optional[Tuple[Any, str, Optional[str]]] = None
    db_errors: List[str] = []
    run_meta = {
        "video_path": str(video_path),
        "video_name": video_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running"
    }

    if sync_to_db is not False: # Explicit False가 아니면 시도
        db_context = prepare_preprocess_db_sync(
            video_path=video_path,
            video_root=video_root,
            run_meta=run_meta,
            duration_sec=video_info.get("duration_sec"),
            stt_backend=stt_backend,
        )

    # 캡처 개수 추적용
    capture_count = 0
    stt_payload = None

    # 내부 헬퍼: 각 단계 완료 후 DB 연동 및 공통 로깅
    def _finalize_stage(stage: str, proc_elapsed: float, **kwargs) -> None:
        comp_map = {"audio": "Audio", "capture": "Capture", "stt": "STT"}
        comp_name = comp_map.get(stage, "System")
        
        db_elapsed = 0.0
        if db_context:
            db_start = time.perf_counter()
            adapter, video_id, job_id = db_context
            stage_results = sync_preprocess_artifacts_to_db(
                adapter=adapter,
                video_id=video_id,
                video_root=video_root,
                provider=stt_backend,
                preprocess_job_id=job_id,
                include_stt=(stage == "stt"),
                include_captures=(stage == "capture"),
                include_audio=(stage == "audio"),
                **kwargs
            )
            db_elapsed = time.perf_counter() - db_start
            if stage_results.get("errors"):
                db_errors.extend(stage_results["errors"])

        # 통합 로깅 포맷 적용
        res_info = ""
        if stage == "stt" and "stt_payload" in kwargs:
            res_info = f" | seg: {len(kwargs['stt_payload'].get('segments', []))}"
        elif stage == "capture" and "captures_payload" in kwargs:
            res_info = f" | cap: {len(kwargs['captures_payload'])}"

        msg = f"DONE (Task: {proc_elapsed:.1f}s | DB: {db_elapsed:.1f}s{res_info})"
        pipeline_logger.log(comp_name, msg)

    # 3. 파이프라인 실행 엔진
    try:
        pipeline_logger.log("System", f"Starting Preprocessing (Parallel={parallel})")
        
        # 병렬 또는 순차 실행을 위한 내부 함수들
        def handle_audio_stt_chain():
            # (1) Audio Extract
            # STT 제공자가 기대하는 기본 포맷인 .wav를 사용하도록 권장 (또는 .mp3)
            audio_path = video_root / f"{video_name}.mp3"
            pipeline_logger.log("Audio", "Extracting...")
            start = time.perf_counter()
            extract_audio(video_path, output_path=audio_path)
            elapsed_audio = time.perf_counter() - start
            timer.record_stage("audio", elapsed_audio)
            _finalize_stage("audio", elapsed_audio, audio_path=audio_path)
            
            # (2) STT Analysis
            pipeline_logger.log("STT", "Analyzing...")
            start = time.perf_counter()
            stt_json = video_root / "stt.json"
            
            # Note: run_stt 대신 run_stt_only를 사용하여 이미 추출된 오디오를 사용함 (중복 추출 방지)
            from src.pipeline.stages import run_stt_only
            payload = run_stt_only(audio_path, stt_json, backend=stt_backend, write_output=write_local_json)
            
            elapsed_stt = time.perf_counter() - start
            timer.record_stage("stt", elapsed_stt)
            _finalize_stage("stt", elapsed_stt, stt_payload=payload)
            return payload

        def handle_capture():
            pipeline_logger.log("Capture", "Extracting...")
            start = time.perf_counter()
            results = run_capture(
                video_path,
                output_base_path,
                threshold=capture_threshold,
                dedupe_threshold=capture_dedupe_threshold,
                min_interval=capture_min_interval,
                verbose=capture_verbose,
                video_name=video_name,
                write_manifest=write_local_json
            )
            elapsed = time.perf_counter() - start
            timer.record_stage("capture", elapsed)
            _finalize_stage("capture", elapsed, captures_payload=results)
            return results

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_audio_stt = executor.submit(handle_audio_stt_chain)
                f_capture = executor.submit(handle_capture)
                stt_payload = f_audio_stt.result()
                capture_results = f_capture.result()
        else:
            stt_payload = handle_audio_stt_chain()
            capture_results = handle_capture()

        capture_count = len(capture_results)
        timer.end_total()

        # 4. 결과 리포팅
        md_report = print_benchmark_report(
            video_info=video_info,
            timer=timer,
            capture_count=capture_count,
            segment_count=len(stt_payload.get("segments", [])) if stt_payload else 0,
            video_path=video_path,
            output_root=video_root,
            parallel=parallel
        )
        _append_benchmark_report(video_root / "benchmark_report.md", md_report, "Preprocess")

        # 5. DB 동기화 마무리
        if db_context:
            adapter, video_id, job_id = db_context
            finalize_preprocess_db_sync(
                adapter=adapter,
                video_id=video_id,
                preprocess_job_id=job_id,
                run_meta=run_meta,
                errors=db_errors
            )
            return video_id

        print(f"\n[Success] Pipeline finished for {video_name}")
        return None

    except Exception as exc:
        pipeline_logger.log("System", f"ERROR: {str(exc)}")
        timer.end_total()
        raise exc


def main() -> None:
    """[Purpose] CLI 실행을 위한 메인 엔트리포인트입니다."""
    parser = argparse.ArgumentParser(description="Feature Capture Pipeline - Preprocessor")
    parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output-base", default="data/outputs", help="출력 디렉토리 루트")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel", help="병렬 처리 모드 비활성화")
    parser.add_argument("--no-db-sync", action="store_false", dest="db_sync", help="Supabase 동기화 건너뛰기")
    parser.add_argument("--local-json", action="store_true", help="로컬 전용 JSON 아티팩트 생성")
    
    args = parser.parse_args()
    
    run_preprocess_pipeline(
        video=args.video,
        output_base=args.output_base,
        parallel=args.parallel,
        sync_to_db=args.db_sync,
        write_local_json=args.local_json
    )


if __name__ == "__main__":
    main()
