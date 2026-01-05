"""
Grid Search for Threshold Optimization

SCENE_THRESHOLD와 DEDUPE_THRESHOLD의 다양한 조합을 테스트하여
최적의 임계값 조합을 찾습니다.

측정 항목:
- 실행 시간 (초)
- 1차 정제: 감지된 장면 전환 수
- 2차 정제: 스킵된 프레임 수  
- 최종 저장된 슬라이드 수

결과 저장:
- 각 조합별 이미지 폴더 (grid_search/scene{X}_dedupe{Y}/)
- JSON 결과 파일 (grid_search/grid_search_results.json)
"""

import os
import sys
import time
import json
import glob
from datetime import datetime

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/capture
src_dir = os.path.dirname(current_dir)  # src
project_root = os.path.dirname(src_dir)  # project root

if project_root not in sys.path:
    sys.path.append(project_root)

from src.capture.video_processor import VideoProcessor


def run_grid_search(video_path: str, output_base_dir: str):
    """
    Grid Search 실행 - 모든 결과와 이미지 저장
    """
    # Grid Search 파라미터 설정
    scene_thresholds = [3, 4, 5, 6]         # 1차 정제 임계값
    dedupe_thresholds = [3, 5, 7, 10]       # 2차 정제 임계값
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_processor = VideoProcessor()
    
    # Grid Search 결과 저장 폴더 생성
    grid_search_dir = os.path.join(output_base_dir, "grid_search")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    results = []
    total_combinations = len(scene_thresholds) * len(dedupe_thresholds)
    current_combo = 0
    
    print("="*70)
    print(f"🔬 Grid Search 시작: {video_name}")
    print(f"📊 테스트 조합: {total_combinations}개")
    print(f"   - SCENE_THRESHOLD: {scene_thresholds}")
    print(f"   - DEDUPE_THRESHOLD: {dedupe_thresholds}")
    print(f"📁 결과 저장 위치: {grid_search_dir}")
    print("="*70)
    
    for scene_th in scene_thresholds:
        for dedupe_th in dedupe_thresholds:
            current_combo += 1
            
            # 각 조합별 출력 폴더 생성 (이미지 저장용)
            combo_output_dir = os.path.join(
                grid_search_dir, 
                f"scene{scene_th}_dedupe{dedupe_th}"
            )
            os.makedirs(combo_output_dir, exist_ok=True)
            
            print(f"\n[{current_combo}/{total_combinations}] "
                  f"SCENE={scene_th}, DEDUPE={dedupe_th}")
            
            # 실행 시간 측정 시작
            start_time = time.time()
            
            # 키프레임 추출
            try:
                keyframes_metadata, diff_scores, fps = video_processor.extract_keyframes(
                    video_path,
                    output_dir=combo_output_dir,
                    threshold=scene_th,
                    min_interval=0.5,
                    verbose=False,
                    video_name=video_name,
                    return_analysis_data=True,
                    dedupe_threshold=dedupe_th
                )
                
                # 실행 시간 계산
                execution_time = time.time() - start_time
                
                # 저장된 이미지 파일 수 카운트
                saved_files = glob.glob(os.path.join(combo_output_dir, "*.jpg"))
                saved_files = [f for f in saved_files if "debug" not in f.lower()]
                
                # 결과 기록
                result = {
                    "scene_threshold": scene_th,
                    "dedupe_threshold": dedupe_th,
                    "execution_time_sec": round(execution_time, 2),
                    "detected_scenes": video_processor.last_detected_count,
                    "skipped_frames": video_processor.last_skipped_count,
                    "final_saved": len(saved_files),
                    "metadata_count": len(keyframes_metadata),
                    "output_folder": f"scene{scene_th}_dedupe{dedupe_th}",
                    "files": [os.path.basename(f) for f in saved_files]
                }
                
                results.append(result)
                
                # 조합별 메타데이터 저장
                combo_metadata_path = os.path.join(combo_output_dir, "metadata.json")
                with open(combo_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "scene_threshold": scene_th,
                        "dedupe_threshold": dedupe_th,
                        "execution_time_sec": round(execution_time, 2),
                        "detected_scenes": video_processor.last_detected_count,
                        "skipped_frames": video_processor.last_skipped_count,
                        "keyframes": keyframes_metadata
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"   ⏱️ 실행 시간: {execution_time:.2f}s")
                print(f"   📸 결과: 감지={result['detected_scenes']}, 스킵={result['skipped_frames']}, 최종={result['final_saved']}")
                print(f"   📁 폴더: {result['output_folder']}/")
                
            except Exception as e:
                print(f"   ❌ 오류 발생: {e}")
                results.append({
                    "scene_threshold": scene_th,
                    "dedupe_threshold": dedupe_th,
                    "error": str(e)
                })
    
    # 전체 결과 저장
    result_file = os.path.join(grid_search_dir, "grid_search_results.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "video_name": video_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "scene_thresholds": scene_thresholds,
                "dedupe_thresholds": dedupe_thresholds
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("📊 Grid Search 결과 요약")
    print("="*70)
    
    # 테이블 헤더
    print(f"\n{'SCENE':>6} | {'DEDUPE':>6} | {'TIME(s)':>8} | {'DETECT':>6} | {'SKIP':>6} | {'FINAL':>6}")
    print("-"*60)
    
    for r in results:
        if "error" not in r:
            print(f"{r['scene_threshold']:>6} | {r['dedupe_threshold']:>6} | "
                  f"{r['execution_time_sec']:>8.2f} | {r['detected_scenes']:>6} | "
                  f"{r['skipped_frames']:>6} | {r['final_saved']:>6}")
    
    print("-"*60)
    
    # 최적 조합 추천
    print("\n📈 최적 조합 분석:")
    
    # 저장 파일 수 기준 그룹화
    for target_count in [5, 6, 7, 8, 9, 10]:
        matching = [r for r in results 
                   if "error" not in r and r.get('final_saved', 0) == target_count]
        if matching:
            fastest = min(matching, key=lambda x: x['execution_time_sec'])
            print(f"   {target_count}개 저장: SCENE={fastest['scene_threshold']}, "
                  f"DEDUPE={fastest['dedupe_threshold']} ({fastest['execution_time_sec']}s)")
    
    print(f"\n📁 결과 폴더: {grid_search_dir}")
    print(f"📋 결과 파일: grid_search_results.json")
    
    return results


def main():
    # 경로 설정
    input_dir = os.path.join(src_dir, 'data', 'input')
    output_dir = os.path.join(src_dir, 'data', 'output')
    
    # 비디오 파일 찾기
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not video_files:
        print("❌ 비디오 파일을 찾을 수 없습니다.")
        return
    
    # 첫 번째 비디오로 테스트
    video_path = video_files[0]
    print(f"📹 테스트 비디오: {os.path.basename(video_path)}")
    
    # Grid Search 실행
    results = run_grid_search(video_path, output_dir)
    
    print("\n✅ Grid Search 완료!")


if __name__ == "__main__":
    main()
