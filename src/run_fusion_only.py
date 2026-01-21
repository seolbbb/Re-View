"""
VLM 처리를 제외하고 Fusion(Summarizer + Judge) 단계만 실행하는 파이프라인 엔트리포인트.

이미 VLM 처리(`vlm.json`)가 완료된 상태에서, Summarizer와 Judge 프롬프트만 변경하여
빠르게 재실행하고 벤치마크를 수행할 때 사용합니다.

Usage:
    python src/run_fusion_only.py --video-name sample_video [options]

Arguments:
    --video-name       (Required) 실행할 비디오 폴더명 (data/outputs/{video_name})
    --output-base      (Optional) 출력 루트 디렉토리 (기본값: data/outputs)
    --limit            (Optional) 처리할 최대 세그먼트 수 (테스트용)
    --summarizer-version, -sv (Optional) 사용할 요약 프롬프트 버전 (예: v1.5, v1.8, v3.2)
    --judge-version, -jv      (Optional) 사용할 평가 프롬프트 버전 (예: v2, v3)

Examples:
    # 기본 실행 (설정 파일의 버전 사용)
    python src/run_fusion_only.py --video-name sample4

    # 특정 버전으로 실험 (v3.2 요약, v3 평가)
    python src/run_fusion_only.py --video-name sample4 -sv v3.2 -jv v3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import yaml

# 스크립트 실행 시 로컬 import가 동작하도록 레포 루트를 설정한다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# API 키와 로컬 설정을 위해 환경 변수를 로드한다.
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


from src.pipeline.benchmark import BenchmarkTimer, print_benchmark_report
from src.pipeline.stages import (
    generate_fusion_config,
    run_fusion_pipeline,
    run_batch_fusion_pipeline,
)


def _sanitize_video_name(stem: str) -> str:
    """파일명 stem을 안전한 출력 폴더명으로 정규화한다."""
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:80] if value else "video"

def _append_benchmark_report(path: Path, report_md: str, pipeline_label: str) -> None:
    """기존 리포트가 있으면 구분선+타임스탬프로 이어 붙인다."""
    timestamp = datetime.now(timezone.utc).isoformat()
    if path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n---\n")
            handle.write(f"Benchmark Append: {pipeline_label} | {timestamp}\n\n")
            handle.write(report_md)
    else:
        path.write_text(report_md, encoding="utf-8")


def run_fusion_only_pipeline(
    *,
    video_name: str,
    output_base: str = "data/outputs",
    limit: Optional[int] = None,
    summarizer_version: Optional[str] = None,
    judge_version: Optional[str] = None,
    batch_mode: bool = False,
) -> None:
    """VLM 결과가 이미 존재하는 상태에서 Fusion(Sync -> Summarize -> Judge -> Render)만 실행한다."""
    
    # 파이프라인 기본 설정을 읽어 CLI 인자에 적용한다.
    settings_path = ROOT / "config" / "pipeline" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"pipeline settings file not found: {settings_path}")
    
    output_base_path = Path(output_base)
    safe_video_name = _sanitize_video_name(video_name)
    video_root = output_base_path / safe_video_name
    
    stt_json = video_root / "stt.json"
    # vlm_json과 manifest_json은 배칠별로 다를 수 있으므로 템플릿 로직에 맡기거나 유연하게 처리한다.
    vlm_json = video_root / "vlm.json"
    manifest_json = video_root / "capture.json"
    if not manifest_json.exists():
        manifest_json = video_root / "manifest.json" # try manifest.json if capture.json missing
    
    # 필수 입력 파일 확인 (최소한 STT는 있어야 함)
    if not stt_json.exists():
        # 일부 환경에선 stt.json이 이미 fusion 폴더에 있을 수도 있음
        stt_json_alt = video_root / "fusion" / "stt.json"
        if stt_json_alt.exists():
            stt_json = stt_json_alt
        else:
            print(f"Warning: STT file not found at {stt_json}, but proceeding to see if config.yaml has it.")

    print(f"Starting Fusion-Only pipeline for: {safe_video_name}")
    print(f"Video Root: {video_root}")
    print("-" * 50)

    # Fusion 설정 파일 생성 (없으면 생성)
    fusion_config_path = video_root / "config.yaml"
    template_config = ROOT / "config" / "fusion" / "settings.yaml"
    
    # 설정 파일은 항상 최신 템플릿 기반으로 재생성하는 것이 안전할 수 있으나,
    # 사용자가 수동 수정한 경우를 대비해 존재하지 않을 때만 생성하거나,
    # 명시적으로 덮어쓰기 옵션을 줄 수도 있다. 여기서는 기존 로직대로 없으면 생성.
    if not fusion_config_path.exists():
        print("Generating fusion config from template...")
        generate_fusion_config(
            template_config=template_config,
            output_config=fusion_config_path,
            repo_root=ROOT,
            stt_json=stt_json,
            vlm_json=vlm_json,
            manifest_json=manifest_json,
            output_root=video_root,
        )
    else:
        print(f"Using existing fusion config: {fusion_config_path}")

    # Override prompt versions if specified via CLI
    if summarizer_version:
        config_data = yaml.safe_load(fusion_config_path.read_text(encoding="utf-8"))
        config_data["summarizer"]["prompt_version"] = summarizer_version
        print(f"  📝 Summarizer version override: {summarizer_version}")
        fusion_config_path.write_text(
            yaml.dump(config_data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
    
    if judge_version:
        # Judge config is in a separate file
        judge_settings_path = ROOT / "config" / "judge" / "settings.yaml"
        if judge_settings_path.exists():
            judge_config = yaml.safe_load(judge_settings_path.read_text(encoding="utf-8"))
            judge_config["prompt_version"] = judge_version
            print(f"  📝 Judge version override: {judge_version}")
            judge_settings_path.write_text(
                yaml.dump(judge_config, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )

    timer = BenchmarkTimer()
    timer.start_total()

    # Fusion Pipeline 실행 (Sync -> Summarize -> Judge -> Render)
    print("\n  ⏳ fusion: Starting (Sync/Summarize/Judge/Render)...")
    if batch_mode:
        print("  🔄 Mode: Batch Fusion (Skipping VLM)")
        fusion_stats = run_batch_fusion_pipeline(
            video_root=video_root,
            captures_dir=video_root / "captures",
            manifest_json=manifest_json,
            stt_json=stt_json,
            video_name=safe_video_name,
            batch_size=4, # 기본값, 필요시 인자로 노출
            timer=timer,
            vlm_batch_size=None,
            vlm_concurrency=1,
            vlm_show_progress=False,
            limit=limit,
            repo_root=ROOT,
            skip_vlm=True,
        )
    else:
        print("  🔄 Mode: Monolithic Fusion")
        fusion_stats = run_fusion_pipeline(
            fusion_config_path,
            limit=limit,
            timer=timer,
        )
    timer.end_total()
    
    total_elapsed = timer.get_total_elapsed()
    print(f"  ✅ fusion: {total_elapsed:.1f}s")
    print(f"     - Sync: {fusion_stats['timings'].get('sync_engine_sec', 0):.1f}s")
    print(f"     - Summarizer: {fusion_stats['timings'].get('llm_summarizer_sec', 0):.1f}s")
    print(f"     - Judge: {fusion_stats['timings'].get('judge_sec', 0):.1f}s")
    print(f"     - Render: {fusion_stats['timings'].get('renderer_sec', 0):.1f}s")

    # 벤치마크 리포트 생성
    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    capture_count = len(manifest_payload) if isinstance(manifest_payload, list) else 0

    report_md = print_benchmark_report(
        video_info={"duration_sec": 0, "width": 0, "height": 0, "file_size_mb": 0},
        timer=timer,
        capture_count=capture_count,
        segment_count=fusion_stats.get("segment_count", 0),
        video_path=Path(safe_video_name),
        output_root=video_root,
        parallel=False,
    )
    report_path = video_root / "benchmark_report.md"
    _append_benchmark_report(report_path, report_md, "Fusion-Only")
    
    # 메타데이터 저장 (Fusion 단계만 갱신)
    run_meta_path = video_root / "pipeline_run.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except:
            pass
            
    # 업데이트
    run_meta["last_fusion_run"] = datetime.now(timezone.utc).isoformat()
    if "durations_sec" not in run_meta:
        run_meta["durations_sec"] = {}
    
    # 기존 VLM/STT/Capture 시간은 보존하고 Fusion 관련 시간만 업데이트
    run_meta["durations_sec"].update({
        "fusion.sync_engine_sec": fusion_stats["timings"].get("sync_engine_sec", 0),
        "fusion.summarizer_sec": fusion_stats["timings"].get("llm_summarizer_sec", 0),
        "fusion.judge_sec": fusion_stats["timings"].get("judge_sec", 0),
        "total_sec": total_elapsed, # 주의: Fusion Only 실행 시간만 기록됨
    })
    
    run_meta["processing_stats"] = {
        "segment_count": fusion_stats.get("segment_count", 0),
        # 캡처 카운트 등은 기존 값을 유지하거나 재계산하지 않음 (이 모듈 범위 밖)
    }
    
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nProcessing completed.")
    print(f"Outputs: {video_root}")
    print(f"Benchmark: {report_path}")


def main() -> None:
    # 외부 라이브러리 로그 레벨 조정 (너무 시끄러운 INFO 로그 억제)
    # 직접 이름을 지정해도 안 먹히는 경우가 있어, 전체 로거를 순회하며 설정한다.
    suppress_prefixes = ("httpx", "httpcore", "google_genai", "google.ai", "google.auth")
    for name in logging.root.manager.loggerDict:
        if any(name.startswith(p) for p in suppress_prefixes):
            logging.getLogger(name).setLevel(logging.WARNING)

    # 혹시 모를 메인 로거들도 명시적 설정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run Fusion-Only Pipeline (Skip VLM)")
    parser.add_argument("--video-name", required=True, help="Video name (folder name in outputs)")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    parser.add_argument("--summarizer-version", "-sv", help="Summarizer prompt version (e.g., v1.5, v1.7, v1.8)")
    parser.add_argument("--judge-version", "-jv", help="Judge prompt version (e.g., v2, v3)")
    parser.add_argument("--batch-mode", action="store_true", help="Enable batch mode (requires existing batch artifacts)")
    
    args = parser.parse_args()
    
    run_fusion_only_pipeline(
        video_name=args.video_name,
        output_base=args.output_base,
        limit=args.limit,
        summarizer_version=args.summarizer_version,
        judge_version=args.judge_version,
        batch_mode=args.batch_mode,
    )

if __name__ == "__main__":
    main()
