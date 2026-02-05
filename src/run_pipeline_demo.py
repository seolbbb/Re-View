"""
[Re:View] Full Pipeline End-to-End Demo

이 스크립트는 전체 파이프라인의 통합 테스트를 위한 데모 도구입니다.
기본값은 DB 의존 없이 로컬 아티팩트로만 전처리/처리를 수행합니다.

- 기본(Local) 모드:
  1) Preprocessor: 비디오에서 STT/캡처 생성 (로컬 JSON 저장, DB 업로드 안 함)
  2) Processor: 로컬 산출물을 읽어 VLM+Fusion 처리 (DB 업로드 안 함)

- 선택(DB) 모드 (`--use-db`):
  기존처럼 Processor를 continuous로 먼저 띄우고 Preprocessor가 DB로 공급합니다.

Usage:
    python src/run_pipeline_demo.py --video data/inputs/sample.mp4
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).resolve().parents[1]

def log(message: str):
    """현재 시간과 함께 메시지를 출력합니다."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def run_command(command: list[str], label: str) -> int:
    """서브프로세스를 실행하고 종료 코드를 반환한다."""
    log(f"[{label}] Starting: {' '.join(command)}")
    proc = subprocess.Popen(command, cwd=str(ROOT), shell=False)
    try:
        proc.wait()
    except KeyboardInterrupt:
        log(f"[{label}] Interrupted. Terminating subprocess...")
        proc.terminate()
        proc.wait()
        raise
    log(f"[{label}] Finished with exit code {proc.returncode}")
    return proc.returncode

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline demo")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Use legacy DB-backed continuous demo mode",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        log(f"Video not found: {video_path}")
        sys.exit(1)

    # 비디오 이름 추론
    video_name = video_path.stem
    
    print("=" * 60)
    log(f"Re:View Pipeline Demo: {video_name}")
    print("=" * 60)
    print("-" * 60)

    if args.use_db:
        log("Mode: DB continuous (legacy)")
        log("1. Starting Processing first (continuous + force-db + db-sync)")
        log("2. Starting Preprocessing (db-sync)")

        process_cmd = [
            sys.executable, "src/run_process_pipeline.py",
            "--video-name", video_name,
            "--batch-mode",
            "--continuous",
            "--force-db",
            "--db-sync",
        ]
        preprocess_cmd = [
            sys.executable, "src/run_preprocess_pipeline.py",
            "--video", str(video_path),
            "--output-base", args.output_base,
            "--db-sync",
        ]

        proc_process = subprocess.Popen(process_cmd, cwd=str(ROOT), shell=False)
        proc_preprocess = subprocess.Popen(preprocess_cmd, cwd=str(ROOT), shell=False)
        log(f"[Demo] Preprocessor PID: {proc_preprocess.pid}")
        log(f"[Demo] Processor PID:    {proc_process.pid}")

        try:
            proc_preprocess.wait()
            log(f"[Demo] Preprocessing finished with exit code {proc_preprocess.returncode}")
            if proc_preprocess.returncode != 0:
                log("[Demo] Preprocessing failed. Terminating processor.")
                proc_process.terminate()
                sys.exit(1)
            proc_process.wait()
            log(f"[Demo] Processing finished with exit code {proc_process.returncode}")
        except KeyboardInterrupt:
            log("[Demo] Interrupted. Terminating subprocesses...")
            proc_preprocess.terminate()
            proc_process.terminate()
            sys.exit(1)
    else:
        log("Mode: Local only (default)")
        log("1. Running Preprocessing (local-json + no-db-sync)")
        preprocess_cmd = [
            sys.executable, "src/run_preprocess_pipeline.py",
            "--video", str(video_path),
            "--output-base", args.output_base,
            "--local-json",
            "--no-db-sync",
        ]
        pre_code = run_command(preprocess_cmd, "Preprocess")
        if pre_code != 0:
            log("[Demo] Preprocessing failed. Aborting.")
            sys.exit(pre_code)

        log("2. Running Processing (local artifacts + no-db-sync)")
        process_cmd = [
            sys.executable, "src/run_process_pipeline.py",
            "--video-name", video_name,
            "--output-base", args.output_base,
            "--batch-mode",
            "--no-force-db",
            "--no-db-sync",
        ]
        proc_code = run_command(process_cmd, "Process")
        if proc_code != 0:
            log("[Demo] Processing failed.")
            sys.exit(proc_code)

    log("[Demo] All pipelines finished.")

if __name__ == "__main__":
    main()
