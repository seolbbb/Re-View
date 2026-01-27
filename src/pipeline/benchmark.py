"""
[Intent]
전체 파이프라인의 실행 시간을 측정하고, 하드웨어 성능 및 처리 효율성을 분석하여 
사용자에게 시각적인 벤치마크 리포트(터미널 및 마크다운)를 제공하는 모듈입니다.

[Usage]
- run_preprocess_pipeline.py의 시작과 끝에서 실행 시간을 기록하고 최종 리포트를 생성할 때 사용됩니다.
- 각 처리 단계(stages.py)에서 개별 작업의 소요 시간을 측정하기 위해 활용됩니다.

[Usage Method]
- BenchmarkTimer 인스턴스를 생성하여 .start_total(), .end_total()로 전체 시간을 측정합니다.
- .time_stage() 컨텍스트나 .record_stage()를 통해 세부 단계별 시간을 기록합니다.
- print_benchmark_report()를 호출하여 분석 결과물(터미널 출력 및 MD 파일 내용)을 얻습니다.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def format_duration(seconds: float) -> str:
    """[Purpose] 초 단위 시간을 읽기 쉬운 형식(Ms Ss)으로 변환합니다."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """
    [Usage File] run_preprocess_pipeline.py
    [Purpose] FFprobe를 사용하여 비디오의 메타데이터(해상도, 화질, 길이 등)를 추출합니다.
    [Connection] FFprobe 외부 프로세스 통신
    
    [Args]
    - video_path (Path): 분석할 비디오 파일 경로
    
    [Returns]
    - Dict[str, Any]: duration_sec, width, height, fps, codec 등을 포함한 정보 사전
    """
    info: Dict[str, Any] = {
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "codec": None,
        "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else None,
    }

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-show_entries", "format=duration", "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
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
    except Exception:
        pass
    return info


class BenchmarkTimer:
    """
    [Class Purpose]
    파이프라인의 각 단계별 시작/종료 시간을 기록하고 통계를 산출하는 타이머 클래스입니다.
    """

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.total_start: float = 0.0
        self.total_end: float = 0.0

    def start_total(self) -> None:
        """[Purpose] 전체 프로세스의 시작 시점을 기록합니다."""
        self.total_start = time.perf_counter()

    def end_total(self) -> None:
        """[Purpose] 전체 프로세스의 종료 시점을 기록합니다."""
        self.total_end = time.perf_counter()

    def time_stage(self, stage_name: str, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        [Usage File] stages.py
        [Purpose] 특정 함수 실행 시간을 측정하고 해당 단계의 이름으로 기록합니다.
        
        [Args]
        - stage_name (str): 측정할 단계의 별칭
        - func (Callable): 실행할 함수
        - *args, **kwargs: 함수에 전달할 인자들
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        self.record_stage(stage_name, elapsed)
        return result, elapsed

    def record_stage(self, stage_name: str, elapsed: float) -> None:
        """[Purpose] 외부에서 수동으로 측정한 시간을 특정 단계에 기록합니다."""
        if stage_name in self.stages:
            self.stages[stage_name]["elapsed_sec"] += elapsed
        else:
            self.stages[stage_name] = {"elapsed_sec": elapsed}

    def get_total_elapsed(self) -> float:
        """[Purpose] 전체 소요 시간을 반환합니다. 종료되지 않았다면 현재 시각 기준입니다."""
        if self.total_end == 0.0:
            return time.perf_counter() - self.total_start
        return self.total_end - self.total_start

    def get_report(self, video_duration_sec: Optional[float] = None) -> Dict[str, Any]:
        """
        [Purpose] 기록된 데이터를 바탕으로 리포트용 요약 통계를 생성합니다.
        
        [Returns]
        - Dict[str, Any]: 총 소요시간, 단계별 소요시간 및 비율, 실시간 계수(RTF) 등
        """
        total_elapsed = self.get_total_elapsed()
        report = {
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
    """
    [Usage File] run_preprocess_pipeline.py
    [Purpose] 측정 결과(timer)를 가공하여 터미널에 출력하고, 마크다운 리포트 내용을 생성합니다.
    
    [Args]
    - video_info (Dict): 비디오 메타데이터
    - timer (BenchmarkTimer): 측정된 데이터가 담긴 타이머 객체
    - capture_count (int): 추출된 슬라이드 수
    - segment_count (int): STT 추출 문장 수
    - video_path (Path): 대상 비디오 경로
    - output_root (Path): 결과 저장 루트
    - parallel (bool): 병렬 모드 활성화 여부
    
    [Returns]
    - str: 마크다운 파일에 저장할 리포트 문자열
    
    [Internal Logic]
    1. 병렬 모드인 경우 (Audio+STT vs Capture) 구조를 분석하여 병렬 실행 시간을 산출합니다.
    2. 단계별 소요 시간과 비중(%)을 계산하여 리스트를 만듭니다.
    3. 누락된 시간을 'overhead'로 표시하여 누락 없는 통계를 보장합니다.
    4. 분석 결과를 터미널에 고정 형식으로 출력하고 MD 형식으로 변환하여 반환합니다.
    """
    report = timer.get_report(video_info.get("duration_sec"))
    total_sec = report["total_elapsed_sec"]

    # --- 1. 터미널 출력 섹션 ---
    print("\n" + "=" * 60)
    print("📊 BENCHMARK REPORT")
    print("=" * 60)
    print(f"\n📹 Video: {video_path.name}")
    if video_info["duration_sec"]:
        print(f"   Duration: {format_duration(video_info['duration_sec'])}")
    print(f"\n📈 Processing Stats:")
    print(f"   Captured Slides: {capture_count} images")
    print(f"   STT Transcript: {segment_count} sentences")
    print(f"   Parallel Mode: {'Enabled' if parallel else 'Disabled'}")
    print("\n⏱️  Timing Breakdown:")
    print("-" * 50)

    # 병렬 실행 시간 산출 (Critical Path 분석)
    accounted_elapsed = 0.0
    display_entries = []
    skip_stages = set()

    if parallel and "capture" in report["stages"] and ("stt" in report["stages"] or "audio" in report["stages"]):
        audio_sec = report["stages"].get("audio", {}).get("elapsed_sec", 0.0)
        stt_sec = report["stages"].get("stt", {}).get("elapsed_sec", 0.0)
        cap_sec = report["stages"].get("capture", {}).get("elapsed_sec", 0.0)
        
        peer1 = audio_sec + stt_sec
        peer2 = cap_sec
        parallel_crit_sec = max(peer1, peer2)
        pct = (parallel_crit_sec / total_sec * 100) if total_sec > 0 else 0
        
        details = f"audio={format_duration(audio_sec)}, stt={format_duration(stt_sec)}, capture={format_duration(cap_sec)}"
        display_entries.append(("pipeline (parallel)", format_duration(parallel_crit_sec), pct, details))
        accounted_elapsed += parallel_crit_sec
        skip_stages.update({"stt", "capture", "audio"})

    # 나머지 정의된 단계 출력
    stage_order = ["vlm", "waiting", "fusion.sync_engine", "fusion.llm_summarizer", "fusion.renderer", "fusion.final_summary", "fusion.judge"]
    for s in stage_order:
        if s in report["stages"] and s not in skip_stages:
            info = report["stages"][s]
            display_entries.append((s, info["elapsed_formatted"], info["percentage"], None))
            accounted_elapsed += info["elapsed_sec"]
            skip_stages.add(s)

    # 기타 미분류 단계
    for s, info in report["stages"].items():
        if s not in skip_stages:
            display_entries.append((s, info["elapsed_formatted"], info["percentage"], None))
            accounted_elapsed += info["elapsed_sec"]

    # 오버헤드 산출
    overhead = max(0.0, total_sec - accounted_elapsed)
    if overhead > 0.01:
        display_entries.append(("overhead", format_duration(overhead), (overhead / total_sec * 100) if total_sec > 0 else 0, None))

    # 터미널 포맷팅 출력
    width = max(24, max((len(e[0]) for e in display_entries), default=0))
    for name, time_str, pct, details in display_entries:
        line = f"   {name:<{width}} {time_str:>10s} ({pct:5.1f}%)"
        if details: line += f" [{details}]"
        print(line)
    print("-" * 50)
    print(f"   {'TOTAL':<{width}} {report['total_elapsed_formatted']:>10s}")

    if "speed_ratio" in report:
        print(f"\n🚀 Speed Ratio: {report['realtime_factor']} (video length)")

    # --- 2. 마크다운 생성 섹션 ---
    md = [
        "# Pipeline Benchmark Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Video Information",
        f"- **File:** `{video_path.name}`",
    ]
    if video_info["duration_sec"]:
        md.append(f"- **Duration:** {format_duration(video_info['duration_sec'])}")
    
    md.extend([
        "\n## Processing Statistics",
        f"- **Captured Slides:** {capture_count} images",
        f"- **STT Transcript:** {segment_count} sentences",
        f"- **Parallel Mode:** {'Enabled' if parallel else 'Disabled'}",
        "\n## Timing Breakdown",
        "\n| Stage | Time | Percentage |",
        "|-------|------|------------|"
    ])

    for name, time_str, pct, _ in display_entries:
        md.append(f"| {name} | {time_str} | {pct:.1f}% |")
    
    md.append(f"| **TOTAL** | **{report['total_elapsed_formatted']}** | 100% |")

    if "speed_ratio" in report:
        md.extend([
            "\n## Performance Analysis",
            f"- **Speed Ratio:** {report['realtime_factor']} of video duration",
            f"- **Status:** {'✅ Target Achieved' if report['speed_ratio'] < 0.5 else '⚠️ Optimization Required'}"
        ])

    return "\n".join(md)
