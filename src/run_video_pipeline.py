"""
================================================================================
run_video_pipeline.py - 비디오 파이프라인 벤치마크 도구
================================================================================

[목적]
    비디오 1개 입력 → STT/Capture/VLM → Fusion 요약까지 end-to-end 실행하며
    각 단계별 처리 시간을 정밀하게 측정하여 벤치마크 리포트를 생성합니다.

[사용법]
    python src/run_video_pipeline.py --video <video_path> [옵션...]

[출력]
    - pipeline_run.json: 상세한 벤치마크 메트릭 (JSON)
    - benchmark_report.md: 사람이 읽기 쉬운 벤치마크 리포트
    - 터미널: 실시간 진행 상황 및 최종 벤치마크 요약

================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from src.audio.stt_router import STTRouter
from src.capture.process_content import process_single_video_capture
from src.fusion.config import load_config
from src.fusion.final_summary_composer import compose_final_summaries
from src.fusion.io_utils import ensure_output_root
from src.fusion.renderer import render_segment_summaries_md
from src.fusion.summarizer import run_summarizer
from src.fusion.sync_engine import run_sync_engine
from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm
from src.judge.judge import run_judge


# ============================================================
# 유틸리티 함수
# ============================================================

def _sanitize_video_name(stem: str) -> str:
    """비디오 파일 이름을 안전한 디렉토리 이름으로 변환."""
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return "video"
    return value[:80]


def _write_json(path: Path, payload: Any) -> None:
    """JSON 파일 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_now_iso() -> str:
    """현재 UTC 시간을 ISO 형식으로 반환."""
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    """초 단위를 'Xm Xs' 또는 'Xs' 형식으로 변환."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _get_video_duration(video_path: Path) -> Optional[float]:
    """ffprobe를 사용하여 비디오 길이(초)를 추출."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def _get_video_info(video_path: Path) -> Dict[str, Any]:
    """비디오의 상세 정보를 추출 (ffprobe 사용)."""
    info: Dict[str, Any] = {
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "codec": None,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else None
    }
    
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Format 정보
            if "format" in data and "duration" in data["format"]:
                info["duration_sec"] = float(data["format"]["duration"])
            
            # Stream 정보
            if "streams" in data and data["streams"]:
                stream = data["streams"][0]
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")
                info["codec"] = stream.get("codec_name")
                
                # FPS 계산 (r_frame_rate는 "30/1" 형식)
                fps_str = stream.get("r_frame_rate", "")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    if int(den) > 0:
                        info["fps"] = round(int(num) / int(den), 2)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, json.JSONDecodeError):
        pass
    
    return info


# ============================================================
# 타이밍 측정 함수
# ============================================================

class BenchmarkTimer:
    """벤치마크 타이밍 관리 클래스."""
    
    def __init__(self):
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.total_start: float = 0.0
        self.total_end: float = 0.0
    
    def start_total(self) -> None:
        """전체 타이머 시작."""
        self.total_start = time.perf_counter()
    
    def end_total(self) -> None:
        """전체 타이머 종료."""
        self.total_end = time.perf_counter()
    
    def time_stage(self, stage_name: str, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        특정 스테이지를 실행하고 시간을 측정.
        
        Returns:
            (결과, 소요시간_초)
        """
        # 시작 로그
        print(f"  ⏳ {stage_name}: 시작...", flush=True)
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        self.stages[stage_name] = {
            "elapsed_sec": elapsed,
            "start_time": start,
            "end_time": start + elapsed
        }
        
        # 완료 로그
        print(f"  ✓ {stage_name}: {_format_duration(elapsed)}")
        
        return result, elapsed
    
    def record_stage(self, stage_name: str, elapsed: float) -> None:
        """이미 측정된 스테이지 시간 기록."""
        self.stages[stage_name] = {
            "elapsed_sec": elapsed,
            "start_time": None,
            "end_time": None
        }
    
    def get_total_elapsed(self) -> float:
        """전체 소요 시간 반환."""
        return self.total_end - self.total_start
    
    def get_report(self, video_duration_sec: Optional[float] = None) -> Dict[str, Any]:
        """벤치마크 리포트 생성."""
        total_elapsed = self.get_total_elapsed()
        
        report: Dict[str, Any] = {
            "total_elapsed_sec": round(total_elapsed, 3),
            "total_elapsed_formatted": _format_duration(total_elapsed),
            "stages": {}
        }
        
        # Video duration 기반 메트릭
        if video_duration_sec and video_duration_sec > 0:
            report["video_duration_sec"] = round(video_duration_sec, 2)
            report["speed_ratio"] = round(total_elapsed / video_duration_sec, 2)
            report["realtime_factor"] = f"{report['speed_ratio']:.2f}x"
        
        # 각 스테이지별 상세
        for name, data in self.stages.items():
            elapsed = data["elapsed_sec"]
            pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
            
            report["stages"][name] = {
                "elapsed_sec": round(elapsed, 3),
                "elapsed_formatted": _format_duration(elapsed),
                "percentage": round(pct, 1)
            }
        
        return report


# ============================================================
# 파이프라인 단계 실행 함수
# ============================================================

def _generate_fusion_config(
    *,
    template_config: Path,
    output_config: Path,
    repo_root: Path,
    stt_json: Path,
    vlm_json: Path,
    manifest_json: Path,
    output_root: Path,
) -> None:
    """Fusion 파이프라인용 config.yaml 생성."""
    payload: Dict[str, Any]
    with template_config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            return str(p)

    payload["paths"] = {
        "stt_json": _rel(stt_json),
        "vlm_json": _rel(vlm_json),
        "captures_manifest_json": _rel(manifest_json),
        "output_root": _rel(output_root),
    }

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_stt(video_path: Path, output_stt_json: Path, *, backend: str) -> None:
    """STT(Speech-to-Text) 실행."""
    router = STTRouter(provider=backend)
    audio_output_path = output_stt_json.with_name(f"{video_path.stem}.wav")
    router.transcribe_media(
        video_path,
        provider=backend,
        audio_output_path=audio_output_path,
        mono_method="auto",
        output_path=output_stt_json,
    )


def _run_capture(
    video_path: Path,
    output_base: Path,
    *,
    threshold: float,
    dedupe_threshold: float,
    min_interval: float,
    verbose: bool,
    video_name: str,
) -> List[Dict[str, Any]]:
    """슬라이드 캡처 실행."""
    metadata = process_single_video_capture(
        str(video_path),
        str(output_base),
        scene_threshold=threshold,
        dedupe_threshold=dedupe_threshold,
        min_interval=min_interval
    )
    return metadata


def _run_vlm_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
    concurrency: int,
) -> int:
    """VLM(Vision Language Model) 실행. 처리된 이미지 수 반환."""
    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_base)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_paths: List[str] = []
    for item in sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", 0)), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))

    if not image_paths:
        raise ValueError("VLM 입력 이미지가 없습니다(manifest.json을 확인하세요).")

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=raw_path.with_name("vlm.json"),
    )
    raw_path.unlink(missing_ok=True)
    
    return len(image_paths)


def _run_fusion_pipeline(
    config_path: Path, 
    *, 
    limit: Optional[int], 
    dry_run: bool,
    timer: BenchmarkTimer
) -> Dict[str, Any]:
    """
    Fusion 파이프라인 실행 (sync_engine → LLM summarizer → renderer → final_summary).
    
    Returns:
        fusion 세부 메트릭 딕셔너리
    """
    config = load_config(str(config_path))
    ensure_output_root(config.paths.output_root)

    fusion_info: Dict[str, Any] = {
        "segment_count": 0,
        "timings": {}
    }

    # Sync Engine
    _, sync_elapsed = timer.time_stage(
        "fusion.sync_engine",
        run_sync_engine,
        config,
        limit=limit,
        dry_run=False,
    )
    fusion_info["timings"]["sync_engine_sec"] = sync_elapsed

    # LLM Summarizer
    _, llm_elapsed = timer.time_stage(
        "fusion.llm_summarizer",
        run_summarizer,
        config,
        limit=limit,
        dry_run=dry_run,
    )
    fusion_info["timings"]["llm_summarizer_sec"] = llm_elapsed

    output_dir = config.paths.output_root / "fusion"
    
    if not dry_run:
        # Renderer
        _, render_elapsed = timer.time_stage(
            "fusion.renderer",
            render_segment_summaries_md,
            summaries_jsonl=output_dir / "segment_summaries.jsonl",
            output_md=output_dir / "segment_summaries.md",
            include_sources=config.raw.render.include_sources,
            sources_jsonl=output_dir / "segments_units.jsonl",
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=limit,
        )
        fusion_info["timings"]["renderer_sec"] = render_elapsed

        # Final Summary
        summaries, final_elapsed = timer.time_stage(
            "fusion.final_summary",
            compose_final_summaries,
            summaries_jsonl=output_dir / "segment_summaries.jsonl",
            max_chars=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=limit,
        )
        fusion_info["timings"]["final_summary_sec"] = final_elapsed
        
        # 최종 요약 저장
        outputs_dir = output_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for fmt in config.raw.final_summary.generate_formats:
            if fmt in summaries:
                outputs_dir.joinpath(f"final_summary_{fmt}.md").write_text(
                    summaries[fmt], encoding="utf-8"
                )
        
        # Judge 실행
        judge_output_dir = output_dir / "judge"
        judge_output_dir.mkdir(parents=True, exist_ok=True)
        _, judge_elapsed = timer.time_stage(
            "fusion.judge",
            run_judge,
            config=config,
            segments_units_path=output_dir / "segments_units.jsonl",
            segment_summaries_path=output_dir / "segment_summaries.jsonl",
            output_report_path=judge_output_dir / "judge_report.json",
            output_segments_path=judge_output_dir / "judge_segment_reports.jsonl",
            batch_size=3,
            workers=1,
            json_repair_attempts=1,
            limit=limit,
            return_reasons=True,
            verbose=True,
        )
        fusion_info["timings"]["judge_sec"] = judge_elapsed
    
    # Segment 수 카운트
    segments_file = output_dir / "segment_summaries.jsonl"
    if segments_file.exists():
        fusion_info["segment_count"] = sum(1 for _ in segments_file.open(encoding="utf-8"))
    
    return fusion_info


# ============================================================
# 벤치마크 리포트 생성
# ============================================================

def _print_benchmark_report(
    video_info: Dict[str, Any],
    timer: BenchmarkTimer,
    capture_count: int,
    segment_count: int,
    video_path: Path,
    output_root: Path,
    parallel: bool
) -> str:
    """
    터미널에 벤치마크 결과를 출력하고 마크다운 리포트 반환.
    """
    report = timer.get_report(video_info.get("duration_sec"))
    
    # 터미널 출력
    print("\n" + "=" * 60)
    print("📊 BENCHMARK REPORT")
    print("=" * 60)
    
    # 비디오 정보
    print(f"\n📹 Video: {video_path.name}")
    if video_info["duration_sec"]:
        print(f"   Duration: {_format_duration(video_info['duration_sec'])}")
    if video_info["width"] and video_info["height"]:
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    if video_info["file_size_mb"]:
        print(f"   File Size: {video_info['file_size_mb']} MB")
    
    # 처리 통계
    print(f"\n📈 Processing Stats:")
    print(f"   Captures: {capture_count} frames")
    print(f"   Segments: {segment_count} segments")
    print(f"   Parallel Mode: {'Enabled' if parallel else 'Disabled'}")
    
    # 타이밍 결과
    print(f"\n⏱️  Timing Breakdown:")
    print("-" * 50)
    
    # 주요 스테이지 정렬 출력
    stage_order = ["stt", "capture", "vlm", "fusion.sync_engine", "fusion.llm_summarizer", 
                   "fusion.renderer", "fusion.final_summary", "fusion.judge"]
    
    for stage in stage_order:
        if stage in report["stages"]:
            info = report["stages"][stage]
            bar_len = int(info["percentage"] / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"   {stage:24s} {info['elapsed_formatted']:>10s} ({info['percentage']:5.1f}%)")
    
    print("-" * 50)
    print(f"   {'TOTAL':24s} {report['total_elapsed_formatted']:>10s}")
    
    # 속도 비율
    if "speed_ratio" in report:
        print(f"\n🚀 Speed Ratio: {report['realtime_factor']} (video length)")
        if report["speed_ratio"] < 0.5:
            print("   ✅ 목표 달성! (6분 영상 기준 3분 이내)")
        else:
            print("   ⚠️  경량화 필요 (목표: 0.5x 이하)")
    
    print(f"\n📁 Output: {output_root}")
    print("=" * 60 + "\n")
    
    # 마크다운 리포트 생성
    md_lines = [
        "# Pipeline Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Video Information",
        "",
        f"- **File:** `{video_path.name}`",
    ]
    
    if video_info["duration_sec"]:
        md_lines.append(f"- **Duration:** {_format_duration(video_info['duration_sec'])}")
    if video_info["width"] and video_info["height"]:
        md_lines.append(f"- **Resolution:** {video_info['width']}x{video_info['height']}")
    if video_info["file_size_mb"]:
        md_lines.append(f"- **File Size:** {video_info['file_size_mb']} MB")
    
    md_lines.extend([
        "",
        "## Processing Statistics",
        "",
        f"- **Captured Frames:** {capture_count}",
        f"- **Segments Processed:** {segment_count}",
        f"- **Parallel Mode:** {'Enabled' if parallel else 'Disabled'}",
        "",
        "## Timing Breakdown",
        "",
        "| Stage | Time | Percentage |",
        "|-------|------|------------|",
    ])
    
    for stage in stage_order:
        if stage in report["stages"]:
            info = report["stages"][stage]
            md_lines.append(f"| {stage} | {info['elapsed_formatted']} | {info['percentage']:.1f}% |")
    
    md_lines.extend([
        f"| **TOTAL** | **{report['total_elapsed_formatted']}** | 100% |",
        "",
    ])
    
    if "speed_ratio" in report:
        md_lines.extend([
            "## Performance Analysis",
            "",
            f"- **Speed Ratio:** {report['realtime_factor']} of video duration",
            f"- **Status:** {'✅ Target Achieved' if report['speed_ratio'] < 0.5 else '⚠️ Optimization Required'}",
            "",
        ])
    
    return "\n".join(md_lines)


# ============================================================
# 메인 함수
# ============================================================

def parse_args() -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="비디오 파이프라인 벤치마크 (STT → Capture → VLM → LLM)"
    )
    parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output-base", default="data/outputs", help="출력 베이스 디렉토리")
    parser.add_argument("--stt-backend", choices=["clova"], default="clova", help="STT 백엔드")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True, help="STT+Capture 병렬 실행")
    parser.add_argument("--capture-threshold", type=float, default=3.0, help="장면 전환 감지 임계값")
    parser.add_argument("--capture-dedupe-threshold", type=float, default=3.0, help="중복 제거 임계값 (2차 정제)")
    parser.add_argument("--capture-min-interval", type=float, default=0.5, help="캡처 최소 간격(초)")
    parser.add_argument("--capture-verbose", action="store_true", help="캡처 상세 로그 출력")
    parser.add_argument("--vlm-batch-size", type=int, default=2, help="VLM 배치 크기(미지정 시 전부 한 번에)")
    parser.add_argument("--vlm-concurrency", type=int, default=3, help="VLM 병렬 요청 수 (기본: 3)")
    parser.add_argument("--limit", type=int, default=None, help="fusion 단계에서 처리할 segment 수 제한")
    parser.add_argument("--dry-run", action="store_true", help="summarizer LLM 미호출(출력 미생성)")
    return parser.parse_args()


def main() -> None:
    """메인 실행 함수."""
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    repo_root = ROOT
    output_base = (repo_root / Path(args.output_base)).resolve()
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base / video_name
    video_root.mkdir(parents=True, exist_ok=True)
    
    # 벤치마크 타이머 초기화
    timer = BenchmarkTimer()
    
    # 비디오 정보 추출
    print(f"\n🎬 Analyzing video: {video_path.name}")
    video_info = _get_video_info(video_path)
    if video_info["duration_sec"]:
        print(f"   Duration: {_format_duration(video_info['duration_sec'])}")

    # 메타데이터 초기화
    run_meta_path = video_root / "pipeline_run.json"
    run_meta: Dict[str, Any] = {
        "schema_version": 2,
        "video_path": str(video_path),
        "video_name": video_name,
        "video_info": video_info,
        "output_base": str(output_base),
        "video_root": str(video_root),
        "started_at_utc": _utc_now_iso(),
        "args": vars(args),
        "durations_sec": {},
        "benchmark": {},
        "status": "running",
    }
    _write_json(run_meta_path, run_meta)

    timer.start_total()
    capture_count = 0
    segment_count = 0
    
    try:
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "manifest.json"

        print(f"\n🚀 Starting pipeline (parallel={args.parallel})...")
        print("-" * 50)

        stt_elapsed = 0.0
        capture_elapsed = 0.0

        if args.parallel:
            # 병렬 실행
            with ThreadPoolExecutor(max_workers=2) as executor:
                def run_stt_timed():
                    start = time.perf_counter()
                    _run_stt(video_path, stt_json, backend=args.stt_backend)
                    return time.perf_counter() - start
                
                def run_capture_timed():
                    start = time.perf_counter()
                    result = _run_capture(
                        video_path, output_base,
                        threshold=args.capture_threshold,
                        dedupe_threshold=args.capture_dedupe_threshold,
                        min_interval=args.capture_min_interval,
                        verbose=args.capture_verbose,
                        video_name=video_name,
                    )
                    elapsed = time.perf_counter() - start
                    return result, elapsed
                
                stt_future = executor.submit(run_stt_timed)
                capture_future = executor.submit(run_capture_timed)
                
                stt_elapsed = stt_future.result()
                capture_result, capture_elapsed = capture_future.result()
                capture_count = len(capture_result) if capture_result else 0
            
            # 병렬 실행 결과 기록
            timer.record_stage("stt", stt_elapsed)
            timer.record_stage("capture", capture_elapsed)
            print(f"  ✓ stt: {_format_duration(stt_elapsed)} (parallel)")
            print(f"  ✓ capture: {_format_duration(capture_elapsed)} (parallel)")
        else:
            # 순차 실행
            _, stt_elapsed = timer.time_stage(
                "stt", _run_stt, video_path, stt_json, backend=args.stt_backend
            )
            capture_result, capture_elapsed = timer.time_stage(
                "capture", _run_capture, video_path, output_base,
                threshold=args.capture_threshold,
                dedupe_threshold=args.capture_dedupe_threshold,
                min_interval=args.capture_min_interval,
                verbose=args.capture_verbose,
                video_name=video_name,
            )
            capture_count = len(capture_result) if capture_result else 0

        # VLM 실행
        vlm_image_count, vlm_elapsed = timer.time_stage(
            "vlm",
            _run_vlm_openrouter,
            captures_dir=captures_dir,
            manifest_json=manifest_json,
            video_name=video_name,
            output_base=output_base,
            batch_size=args.vlm_batch_size,
            concurrency=args.vlm_concurrency,
        )

        # Fusion config 생성
        template_config = repo_root / "src" / "fusion" / "config.yaml"
        if not template_config.exists():
            raise FileNotFoundError(f"fusion config template을 찾을 수 없습니다: {template_config}")
        
        fusion_config_path = video_root / "config.yaml"
        _generate_fusion_config(
            template_config=template_config,
            output_config=fusion_config_path,
            repo_root=repo_root,
            stt_json=stt_json,
            vlm_json=video_root / "vlm.json",
            manifest_json=manifest_json,
            output_root=video_root,
        )

        # Fusion 파이프라인 실행
        fusion_info = _run_fusion_pipeline(
            fusion_config_path, 
            limit=args.limit, 
            dry_run=args.dry_run,
            timer=timer
        )
        segment_count = fusion_info.get("segment_count", 0)
        
        timer.end_total()

        # 벤치마크 리포트 생성 및 출력
        md_report = _print_benchmark_report(
            video_info=video_info,
            timer=timer,
            capture_count=capture_count,
            segment_count=segment_count,
            video_path=video_path,
            output_root=video_root,
            parallel=args.parallel
        )
        
        # 마크다운 리포트 저장
        report_path = video_root / "benchmark_report.md"
        report_path.write_text(md_report, encoding="utf-8")

        # 최종 메타데이터 저장
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
        run_meta["ended_at_utc"] = _utc_now_iso()
        run_meta["status"] = "ok"
        _write_json(run_meta_path, run_meta)

        print(f"✅ Pipeline completed successfully!")
        print(f"   Outputs: {video_root}")
        print(f"   Benchmark: {report_path}")
        
    except Exception as exc:
        timer.end_total()
        run_meta["ended_at_utc"] = _utc_now_iso()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        run_meta["durations_sec"]["total_sec"] = round(timer.get_total_elapsed(), 6)
        _write_json(run_meta_path, run_meta)
        print(f"\n❌ Pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
