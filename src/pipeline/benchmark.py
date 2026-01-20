"""파이프라인 실행 시간을 측정하고 리포트를 생성한다."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List


def format_duration(seconds: float) -> str:
    """초 단위를 사람이 읽기 쉬운 문자열로 변환한다."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def get_video_duration(video_path: Path) -> Optional[float]:
    """비디오 길이를 초 단위로 구한다."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """비디오 메타정보를 추출한다."""
    info: Dict[str, Any] = {
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "codec": None,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
        if video_path.exists()
        else None,
    }

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,codec_name",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)

            if "format" in data and "duration" in data["format"]:
                info["duration_sec"] = float(data["format"]["duration"])

            if "streams" in data and data["streams"]:
                stream = data["streams"][0]
                info["width"] = stream.get("width")
                info["height"] = stream.get("height")
                info["codec"] = stream.get("codec_name")

                fps_str = stream.get("r_frame_rate", "")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    if int(den) > 0:
                        info["fps"] = round(int(num) / int(den), 2)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, json.JSONDecodeError):
        pass

    return info


class BenchmarkTimer:
    """스테이지별 실행 시간을 기록하고 리포트를 만든다."""

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.total_start: float = 0.0
        self.total_end: float = 0.0

    def start_total(self) -> None:
        """전체 타이머를 시작한다."""
        self.total_start = time.perf_counter()

    def end_total(self) -> None:
        """전체 타이머를 종료한다."""
        self.total_end = time.perf_counter()

    def time_stage(self, stage_name: str, func, *args, **kwargs) -> Tuple[Any, float]:
        """함수를 실행하고 경과 시간을 기록한다."""
        print(f"\n{'-' * 50}")
        print(f"  ⏳ {stage_name}: Starting...", flush=True)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if stage_name in self.stages:
            self.stages[stage_name]["elapsed_sec"] += elapsed
            self.stages[stage_name]["end_time"] = start + elapsed
        else:
            self.stages[stage_name] = {
                "elapsed_sec": elapsed,
                "start_time": start,
                "end_time": start + elapsed,
            }

        if stage_name != "waiting":
            print(f"  ✅ {stage_name}: {format_duration(elapsed)}")

        return result, elapsed

    def record_stage(self, stage_name: str, elapsed: float) -> None:
        """외부에서 측정한 시간을 기록한다."""
        if stage_name in self.stages:
            self.stages[stage_name]["elapsed_sec"] += elapsed
        else:
            self.stages[stage_name] = {
                "elapsed_sec": elapsed,
                "start_time": None,
                "end_time": None,
            }

    def get_total_elapsed(self) -> float:
        """전체 경과 시간을 반환한다."""
        return self.total_end - self.total_start

    def get_report(self, video_duration_sec: Optional[float] = None) -> Dict[str, Any]:
        """리포트용 요약 데이터를 생성한다."""
        total_elapsed = self.get_total_elapsed()

        report: Dict[str, Any] = {
            "total_elapsed_sec": round(total_elapsed, 3),
            "total_elapsed_formatted": format_duration(total_elapsed),
            "stages": {},
        }

        if video_duration_sec and video_duration_sec > 0:
            report["video_duration_sec"] = round(video_duration_sec, 2)
            report["speed_ratio"] = round(total_elapsed / video_duration_sec, 2)
            report["realtime_factor"] = f"{report['speed_ratio']:.2f}x"

        for name, data in self.stages.items():
            elapsed = data["elapsed_sec"]
            pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0

            report["stages"][name] = {
                "elapsed_sec": round(elapsed, 3),
                "elapsed_formatted": format_duration(elapsed),
                "percentage": round(pct, 1),
            }

        return report


def print_benchmark_report(
    *,
    video_info: Dict[str, Any],
    timer: BenchmarkTimer,
    capture_count: int,
    segment_count: int,
    video_path: Path,
    output_root: Path,
    parallel: bool,
) -> str:
    """벤치마크 요약을 출력하고 마크다운 문자열을 반환한다."""
    report = timer.get_report(video_info.get("duration_sec"))

    print("\n" + "=" * 60)
    print("📊 BENCHMARK REPORT")
    print("=" * 60)

    print(f"\n📹 Video: {video_path.name}")
    if video_info["duration_sec"]:
        print(f"   Duration: {format_duration(video_info['duration_sec'])}")
    if video_info["width"] and video_info["height"]:
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    if video_info["file_size_mb"]:
        print(f"   File Size: {video_info['file_size_mb']} MB")

    print("\n📈 Processing Stats:")
    print(f"   Captures: {capture_count} frames")
    print(f"   Segments: {segment_count} segments")
    print(f"   Parallel Mode: {'Enabled' if parallel else 'Disabled'}")

    print("\n⏱️  Timing Breakdown:")
    print("-" * 50)

    stage_order = [
        "stt",
        "capture",
        "vlm",
        "waiting",
        "fusion.sync_engine",
        "fusion.llm_summarizer",
        "fusion.renderer",
        "fusion.final_summary",
        "fusion.judge",
    ]

    printed_stages = set()
    skip_stages = set()
    display_entries: List[Tuple[str, str, float, Optional[str]]] = []
    accounted_elapsed = 0.0

    has_batch = any(stage.startswith("pipeline_batch_") for stage in report["stages"])
    if has_batch:
        skip_stages.add("vlm")

    if parallel and "stt" in report["stages"] and "capture" in report["stages"]:
        stt_info = report["stages"]["stt"]
        capture_info = report["stages"]["capture"]
        parallel_elapsed = max(stt_info["elapsed_sec"], capture_info["elapsed_sec"])
        parallel_pct = (
            parallel_elapsed / report["total_elapsed_sec"] * 100
            if report["total_elapsed_sec"] > 0
            else 0.0
        )
        details = (
            f"stt={stt_info['elapsed_formatted']}, "
            f"capture={capture_info['elapsed_formatted']}"
        )
        display_entries.append(
            (
                "stt+capture (parallel)",
                format_duration(parallel_elapsed),
                parallel_pct,
                details,
            )
        )
        accounted_elapsed += parallel_elapsed
        skip_stages.update({"stt", "capture"})

    for stage in stage_order:
        if stage in report["stages"] and stage not in skip_stages:
            info = report["stages"][stage]
            display_entries.append(
                (stage, info["elapsed_formatted"], info["percentage"], None)
            )
            printed_stages.add(stage)
            accounted_elapsed += info["elapsed_sec"]

    for stage in report["stages"]:
        if stage in printed_stages or stage in skip_stages:
            continue
        info = report["stages"][stage]
        display_entries.append(
            (stage, info["elapsed_formatted"], info["percentage"], None)
        )
        accounted_elapsed += info["elapsed_sec"]

    total_elapsed_sec = report["total_elapsed_sec"]
    overhead_elapsed = max(0.0, total_elapsed_sec - accounted_elapsed)
    if overhead_elapsed > 0.01:
        overhead_pct = (
            overhead_elapsed / total_elapsed_sec * 100 if total_elapsed_sec > 0 else 0.0
        )
        display_entries.append(
            (
                "overhead",
                format_duration(overhead_elapsed),
                overhead_pct,
                None,
            )
        )

    width = max(24, max((len(stage) for stage, _, _, _ in display_entries), default=0))
    for stage, elapsed_formatted, percentage, details in display_entries:
        line = f"   {stage:<{width}} {elapsed_formatted:>10s} ({percentage:5.1f}%)"
        if details:
            line += f" [{details}]"
        print(line)

    print("-" * 50)
    print(f"   {'TOTAL':<{width}} {report['total_elapsed_formatted']:>10s}")

    if "speed_ratio" in report:
        print(f"\n🚀 Speed Ratio: {report['realtime_factor']} (video length)")
        target_sec = (video_info.get("duration_sec") or 0) * 0.5
        target_str = format_duration(target_sec)
        video_len_str = format_duration(video_info.get("duration_sec") or 0)
        if report["speed_ratio"] < 0.5:
            print(f"   ✅ Target met! (under {target_str} for a {video_len_str} video)")
        else:
            print(f"   ⚠️  Optimization needed (target: <= {target_str} for a {video_len_str} video)")

    print(f"\n📁 Output: {output_root}")
    print("=" * 60 + "\n")

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
        md_lines.append(f"- **Duration:** {format_duration(video_info['duration_sec'])}")
    if video_info["width"] and video_info["height"]:
        md_lines.append(
            f"- **Resolution:** {video_info['width']}x{video_info['height']}"
        )
    if video_info["file_size_mb"]:
        md_lines.append(f"- **File Size:** {video_info['file_size_mb']} MB")

    md_lines.extend(
        [
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
        ]
    )

    md_printed = set()
    md_skip = set(skip_stages)

    if parallel and "stt" in report["stages"] and "capture" in report["stages"]:
        stt_info = report["stages"]["stt"]
        capture_info = report["stages"]["capture"]
        parallel_elapsed = max(stt_info["elapsed_sec"], capture_info["elapsed_sec"])
        parallel_pct = (
            parallel_elapsed / report["total_elapsed_sec"] * 100
            if report["total_elapsed_sec"] > 0
            else 0.0
        )
        md_lines.append(
            f"| stt+capture (parallel) | {format_duration(parallel_elapsed)} | {parallel_pct:.1f}% |"
        )

    for stage in stage_order:
        if stage in report["stages"] and stage not in md_skip:
            info = report["stages"][stage]
            md_lines.append(
                f"| {stage} | {info['elapsed_formatted']} | {info['percentage']:.1f}% |"
            )
            md_printed.add(stage)

    for stage in report["stages"]:
        if stage in md_printed or stage in md_skip:
            continue
        info = report["stages"][stage]
        md_lines.append(
            f"| {stage} | {info['elapsed_formatted']} | {info['percentage']:.1f}% |"
        )

    if overhead_elapsed > 0.01:
        md_lines.append(
            f"| overhead | {format_duration(overhead_elapsed)} | {overhead_pct:.1f}% |"
        )

    md_lines.extend(
        [
            f"| **TOTAL** | **{report['total_elapsed_formatted']}** | 100% |",
            "",
        ]
    )

    if "speed_ratio" in report:
        md_lines.extend(
            [
                "## Performance Analysis",
                "",
                f"- **Speed Ratio:** {report['realtime_factor']} of video duration",
                f"- **Status:** {'✅ Target Achieved' if report['speed_ratio'] < 0.5 else '⚠️ Optimization Required'}",
                "",
            ]
        )

    return "\n".join(md_lines)
