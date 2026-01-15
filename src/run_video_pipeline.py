"""
비디오 파이프라인 명령줄 진입점.

STT → Capture → VLM → Fusion 전 과정을 실행하고 벤치마크 리포트를 남긴다.
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
from typing import Any, Dict, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

from src.db import sync_pipeline_results_to_db
from src.pipeline.benchmark import (
    BenchmarkTimer,
    format_duration,
    get_video_info,
    print_benchmark_report,
)
from src.pipeline.stages import (
    generate_fusion_config,
    run_batch_fusion_pipeline,
    run_capture,
    run_fusion_pipeline,
    run_stt,
    run_vlm_openrouter,
)


def _sanitize_video_name(stem: str) -> str:
    """비디오 이름을 안전한 디렉토리 이름으로 정규화한다."""
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return "video"
    return value[:80]


def _write_json(path: Path, payload: Any) -> None:
    """JSON 파일을 UTF-8로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="비디오 파이프라인 벤치마크 (STT → Capture → VLM → LLM)"
    )
    parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output-base", default="data/outputs", help="출력 베이스 디렉토리")
    parser.add_argument("--stt-backend", choices=["clova"], default="clova", help="STT 백엔드")
    parser.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction, default=True, help="STT+Capture 병렬 실행"
    )
    parser.add_argument("--capture-threshold", type=float, default=3.0, help="장면 전환 감지 임계값")
    parser.add_argument(
        "--capture-dedupe-threshold", type=float, default=3.0, help="중복 제거 임계값 (2차 정제)"
    )
    parser.add_argument("--capture-min-interval", type=float, default=0.5, help="캡처 최소 간격(초)")
    parser.add_argument("--capture-verbose", action="store_true", help="캡처 상세 로그 출력")
    parser.add_argument("--vlm-batch-size", type=int, default=2, help="VLM 배치 크기(미지정 시 전부 한 번에)")
    parser.add_argument("--vlm-concurrency", type=int, default=3, help="VLM 병렬 요청 수 (기본: 3)")
    parser.add_argument(
        "--vlm-show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VLM 진행 로그 출력 여부 (기본: True)",
    )
    parser.add_argument("--limit", type=int, default=None, help="fusion 단계에서 처리할 segment 수 제한")
    parser.add_argument("--dry-run", action="store_true", help="summarizer LLM 미호출(출력 미생성)")
    parser.add_argument(
        "--batch-mode", action="store_true", default=False, help="배치 모드 활성화 (캡처를 n장씩 분할 처리)"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="배치당 캡처 개수 (기본: 10)")
    return parser.parse_args()


def run_pipeline(
    *,
    video: str,
    output_base: str,
    stt_backend: str,
    parallel: bool,
    capture_threshold: float,
    capture_dedupe_threshold: float,
    capture_min_interval: float,
    capture_verbose: bool,
    vlm_batch_size: Optional[int],
    vlm_concurrency: int,
    vlm_show_progress: bool,
    limit: Optional[int],
    dry_run: bool,
    batch_mode: bool,
    batch_size: int,
) -> None:
    """비디오 1건을 end-to-end로 처리하고 결과/메트릭을 기록한다.

    단계: STT → Capture → VLM → Fusion → (옵션) Judge.
    산출물: 출력 폴더에 stt/vlm/manifest/fusion 결과와 벤치마크 리포트 생성.
    코드 위치: STT(run_stt), Capture(run_capture), VLM(run_vlm_openrouter),
    Fusion(generate_fusion_config + run_fusion_pipeline) 또는 배치 모드(run_batch_fusion_pipeline).
    """
    video_path = Path(video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    repo_root = ROOT
    output_base_path = (repo_root / Path(output_base)).resolve()
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base_path / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    timer = BenchmarkTimer()

    print(f"\n🎬 Analyzing video: {video_path.name}")
    video_info = get_video_info(video_path)
    if video_info["duration_sec"]:
        print(f"   Duration: {format_duration(video_info['duration_sec'])}")

    run_meta_path = video_root / "pipeline_run.json"
    run_args = {
        "video": str(video_path),
        "output_base": str(output_base_path),
        "stt_backend": stt_backend,
        "parallel": parallel,
        "capture_threshold": capture_threshold,
        "capture_dedupe_threshold": capture_dedupe_threshold,
        "capture_min_interval": capture_min_interval,
        "capture_verbose": capture_verbose,
        "vlm_batch_size": vlm_batch_size,
        "vlm_concurrency": vlm_concurrency,
        "vlm_show_progress": vlm_show_progress,
        "limit": limit,
        "dry_run": dry_run,
        "batch_mode": batch_mode,
        "batch_size": batch_size,
    }
    run_meta: Dict[str, Any] = {
        "video_path": str(video_path),
        "video_name": video_name,
        "video_info": video_info,
        "output_base": str(output_base_path),
        "video_root": str(video_root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": run_args,
        "durations_sec": {},
        "benchmark": {},
        "status": "running",
    }
    _write_json(run_meta_path, run_meta)

    timer.start_total()
    capture_count = 0
    segment_count = 0

    try:
        """STT/Capture 입력/출력 경로 준비."""
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "manifest.json"

        print(f"\n🚀 Starting pipeline (parallel={parallel})...")
        print("-" * 50)

        stt_elapsed = 0.0
        capture_elapsed = 0.0

        """STT + Capture 실행."""
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                def run_stt_timed():
                    """STT 단계를 타이밍 포함으로 실행한다."""
                    start = time.perf_counter()
                    run_stt(video_path, stt_json, backend=stt_backend)
                    return time.perf_counter() - start

                def run_capture_timed():
                    """Capture 단계를 타이밍 포함으로 실행한다."""
                    start = time.perf_counter()
                    result = run_capture(
                        video_path,
                        output_base_path,
                        threshold=capture_threshold,
                        dedupe_threshold=capture_dedupe_threshold,
                        min_interval=capture_min_interval,
                        verbose=capture_verbose,
                        video_name=video_name,
                    )
                    elapsed = time.perf_counter() - start
                    return result, elapsed

                stt_future = executor.submit(run_stt_timed)
                capture_future = executor.submit(run_capture_timed)

                stt_elapsed = stt_future.result()
                capture_result, capture_elapsed = capture_future.result()
                capture_count = len(capture_result) if capture_result else 0

            timer.record_stage("stt", stt_elapsed)
            timer.record_stage("capture", capture_elapsed)
            print(f"  ✓ STT done in {format_duration(stt_elapsed)} (parallel)")
            print(f"  ✓ Capture done in {format_duration(capture_elapsed)} (parallel)")
        else:
            """STT + Capture 순차 실행."""
            _, stt_elapsed = timer.time_stage("STT", run_stt, video_path, stt_json, backend=stt_backend)
            capture_result, capture_elapsed = timer.time_stage(
                "Capture",
                run_capture,
                video_path,
                output_base_path,
                threshold=capture_threshold,
                dedupe_threshold=capture_dedupe_threshold,
                min_interval=capture_min_interval,
                verbose=capture_verbose,
                video_name=video_name,
            )
            capture_count = len(capture_result) if capture_result else 0

        """VLM + Fusion 실행."""
        if batch_mode:
            """배치 모드: VLM/Sync/Summarize를 배치 단위로 반복."""
            vlm_elapsed = 0.0
            fusion_info = run_batch_fusion_pipeline(
                video_root=video_root,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                stt_json=stt_json,
                video_name=video_name,
                batch_size=batch_size,
                timer=timer,
                vlm_batch_size=vlm_batch_size,
                vlm_concurrency=vlm_concurrency,
                vlm_show_progress=vlm_show_progress,
                limit=limit,
                dry_run=dry_run,
                repo_root=repo_root,
            )
            segment_count = fusion_info.get("segment_count", 0)
            vlm_image_count = capture_count
        else:
            """VLM 단독 실행."""
            vlm_image_count, vlm_elapsed = timer.time_stage(
                "VLM",
                run_vlm_openrouter,
                captures_dir=captures_dir,
                manifest_json=manifest_json,
                video_name=video_name,
                output_base=output_base_path,
                batch_size=vlm_batch_size,
                concurrency=vlm_concurrency,
                show_progress=vlm_show_progress,
            )

            template_config = repo_root / "config" / "fusion" / "config.yaml"
            if not template_config.exists():
                raise FileNotFoundError(f"fusion config template을 찾을 수 없습니다: {template_config}")

            fusion_config_path = video_root / "config.yaml"
            """Fusion 설정 생성."""
            generate_fusion_config(
                template_config=template_config,
                output_config=fusion_config_path,
                repo_root=repo_root,
                stt_json=stt_json,
                vlm_json=video_root / "vlm.json",
                manifest_json=manifest_json,
                output_root=video_root,
            )

            """Fusion 파이프라인 실행."""
            fusion_info = run_fusion_pipeline(
                fusion_config_path,
                limit=limit,
                dry_run=dry_run,
                timer=timer,
            )
            segment_count = fusion_info.get("segment_count", 0)

        timer.end_total()

        md_report = print_benchmark_report(
            video_info=video_info,
            timer=timer,
            capture_count=capture_count,
            segment_count=segment_count,
            video_path=video_path,
            output_root=video_root,
            parallel=parallel,
        )

        report_path = video_root / "benchmark_report.md"
        report_path.write_text(md_report, encoding="utf-8")

        benchmark_report = timer.get_report(video_info.get("duration_sec"))

        run_meta["durations_sec"] = {
            "stt_sec": round(stt_elapsed, 6),
            "capture_sec": round(capture_elapsed, 6),
            "vlm_sec": round(vlm_elapsed, 6),
            "total_sec": round(timer.get_total_elapsed(), 6),
            **{f"fusion.{k}": round(v, 6) for k, v in fusion_info.get("timings", {}).items()},
        }
        run_meta["benchmark"] = benchmark_report
        run_meta["processing_stats"] = {
            "capture_count": capture_count,
            "vlm_image_count": vlm_image_count,
            "segment_count": segment_count,
        }
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "ok"
        _write_json(run_meta_path, run_meta)

        print("\n✅ Pipeline completed successfully!")
        print(f"   Outputs: {video_root}")
        print(f"   Benchmark: {report_path}")

        print("\n📤 Syncing results to Supabase...")
        db_success = sync_pipeline_results_to_db(
            video_path=video_path,
            video_root=video_root,
            run_meta=run_meta,
            duration_sec=video_info.get("duration_sec"),
            provider=stt_backend,
        )
        if db_success:
            print("✅ Database sync completed!")
        else:
            print("⚠️ Database sync skipped or failed (check logs above)")

    except Exception as exc:
        timer.end_total()
        run_meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        _write_json(run_meta_path, run_meta)
        print(f"\n❌ Pipeline failed: {exc}")
        raise


def main() -> None:
    """CLI 진입점."""
    args = parse_args()
    run_pipeline(
        video=args.video,
        output_base=args.output_base,
        stt_backend=args.stt_backend,
        parallel=args.parallel,
        capture_threshold=args.capture_threshold,
        capture_dedupe_threshold=args.capture_dedupe_threshold,
        capture_min_interval=args.capture_min_interval,
        capture_verbose=args.capture_verbose,
        vlm_batch_size=args.vlm_batch_size,
        vlm_concurrency=args.vlm_concurrency,
        vlm_show_progress=args.vlm_show_progress,
        limit=args.limit,
        dry_run=args.dry_run,
        batch_mode=args.batch_mode,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
