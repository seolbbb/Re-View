"""
[Re:View] Full Pipeline End-to-End Demo

이 스크립트는 전체 파이프라인의 통합 테스트를 위한 데모 도구입니다.
단일 비디오 파일에 대해 다음 두 프로세스를 병렬로 실행하여 'Continuous Processing'을 시연합니다.

1. Processor (Consumer): DB/폴더를 모니터링하며 데이터가 들어오는 즉시 처리 (Continuous Mode)
2. Preprocessor (Producer): 비디오에서 오디오/슬라이드를 추출하고 DB/폴더에 업로드

실행이 완료되면 최종 생성된 요약(Summary)을 터미널에 출력합니다.

Usage:
    python src/run_pipeline_demo.py --video data/inputs/sample.mp4
"""

import argparse
import sys
import time
import subprocess
import signal
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
ROOT = Path(__file__).resolve().parents[1]

def log(message: str):
    """현재 시간과 함께 메시지를 출력합니다."""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def run_command_async(command: list[str], log_prefix: str) -> subprocess.Popen:
    """비동기 서브프로세스 실행"""
    log(f"[{log_prefix}] Starting: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        # stdout/stderr를 현재 터미널로 흘려보냄 (실시간 로그 확인용)
        # 별도 파이프로 잡으면 실시간 출력이 어려울 수 있음
        creationflags=0
    )

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline demo")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--output-base", default="data/outputs", help="Output base directory")
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
    log("1. Starting 'Processing Pipeline' in Continuous Mode (Waiting for data...)")
    log("2. Starting 'Preprocessing Pipeline' (Extracting data...)")
    print("-" * 60)

    # 1. Processing Pipeline 실행 (Consumer)
    # --continuous 플래그로 실행하여 데이터가 들어오기를 기다리게 함
    # --force-db: DB에서 메타데이터를 강제로 조회
    # --db-sync: 결과를 DB에 저장
    process_cmd = [
        sys.executable, "src/run_process_pipeline.py",
        "--video-name", video_name,
        "--batch-mode",
        "--continuous",    # 핵심: 데이터가 없어도 죽지 않고 대기
        "--force-db",      # DB 연동 강제
        "--db-sync"
    ]
    
    # 여기서는 'Process'를 먼저 실행해두고 (Background), 'Preprocess'를 실행함.
    proc_process = subprocess.Popen(
        process_cmd,
        cwd=str(ROOT),
        shell=False
    )
    
    # 프로세서가 초기화되고 Loop에 진입할 시간을 줌 (약 3초)
    time.sleep(3)
    
    # 2. Preprocessing Pipeline 실행 (Producer)
    preprocess_cmd = [
        sys.executable, "src/run_preprocess_pipeline.py",
        "--video", str(video_path),
        "--output-base", args.output_base,
        "--db-sync"  # DB에 업로드해야 Process가 가져갈 수 있음
    ]
    
    proc_preprocess = subprocess.Popen(
        preprocess_cmd,
        cwd=str(ROOT),
        shell=False
    )

    log(f"[Demo] Both pipelines are running.")
    log(f"   - Preprocessor PID: {proc_preprocess.pid}")
    log(f"   - Processor PID:    {proc_process.pid}")
    print("=" * 60)

    try:
        # Preprocess가 끝날 때까지 대기
        proc_preprocess.wait()
        log(f"[Demo] Preprocessing finished with exit code {proc_preprocess.returncode}")
        
        if proc_preprocess.returncode != 0:
            log("❌ Preprocessing failed. Terminating processor.")
            proc_process.terminate()
            sys.exit(1)

        log("[Demo] Waiting for Processor to finish (it should auto-terminate)...")
        # Preprocess가 끝나고 DB status가 DONE이 되면 Processor도 Loop를 탈출하고 종료해야 함
        proc_process.wait()
        log(f"[Demo] Processing finished with exit code {proc_process.returncode}")

    except KeyboardInterrupt:
        log("\n[Demo] Interrupted by user. Terminating subprocesses...")
        proc_preprocess.terminate()
        proc_process.terminate()
        sys.exit(1)

    log("[Demo] All pipelines finished.")

if __name__ == "__main__":
    main()
