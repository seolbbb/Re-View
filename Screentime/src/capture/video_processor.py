import cv2
import os
import numpy as np

class VideoProcessor:
    """
    비디오 처리 클래스: 강의 영상에서 슬라이드 전환을 감지하고 키프레임을 추출
    
    [주요 기능]
    1. Scene Detection: 프레임 간 픽셀 차이를 계산하여 장면 전환 감지
    2. Keyframe Capture: 감지된 시점의 깨끗한 슬라이드 이미지 저장
    3. Mouse Removal: Temporal Median 기법으로 마우스 포인터 제거
    4. Duplicate Removal: dHash 알고리즘으로 중복 프레임 제거
    
    [사용 예시]
    >>> processor = VideoProcessor()
    >>> keyframes = processor.extract_keyframes(
    ...     video_path="lecture.mp4",
    ...     output_dir="output/frames",
    ...     threshold=8,
    ...     min_interval=0.5
    ... )
    
    [핵심 알고리즘]
    - Temporal Median: 시간적으로 분산된 프레임들의 중앙값을 계산하여
                       움직이는 물체(마우스)는 제거하고 고정된 배경(슬라이드)만 추출
    - Multi-point Sampling: 슬라이드 전체 구간에서 무작위로 프레임을 수집하여
                           마우스가 다양한 위치에 있는 순간들을 확보
    """
    def __init__(self):
        pass

    def extract_keyframes(self, video_path, output_dir='captured_frames', threshold=30, min_interval=2.0, verbose=False):
        """
        [핵심 기능] 비디오에서 장면 전환을 감지하여 키프레임을 추출합니다.
        
        Args:
            video_path (str): 입력 비디오 파일 경로
            output_dir (str): 추출된 이미지가 저장될 폴더
            threshold (float): 장면 전환 감지 임계값 (픽셀 차이 평균, 높을수록 둔감)
            min_interval (float): 캡처 간 최소 시간 간격 (초 단위)
            verbose (bool): 디버깅을 위한 상세 로그 출력 여부
            
        Returns:
            list: 캡처된 프레임 정보 리스트
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📂 Created output directory: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video.")
            return []

        # 비디오 정보 출력
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 Video Info: {duration:.2f}s, {fps:.2f} fps, {total_frames} frames")
        print(f"⚙️ Settings: Threshold={threshold}, Min Interval={min_interval}s")

        keyframes = []
        prev_frame_gray = None  # 이전 프레임 (장면 비교용)
        last_capture_time = -min_interval  # 마지막 캡처 시간 (중복 방지)
        
        # === 슬라이드 경계 추적 ===
        # slide_boundaries: 각 슬라이드의 (시작시간, 종료시간) 튜플 리스트
        # 예: [(0, 120), (120, 240), (240, 360)] = 3개 슬라이드
        slide_boundaries = []
        last_scene_change = 0.0  # 마지막 장면 전환 시점
        
        frame_idx = 0
        captured_count = 0

        # === 메인 루프: 모든 프레임을 순회하며 장면 전환 감지 ===
        while True:
            ret, frame = cap.read()
            if not ret:  # 비디오 끝
                break

            current_time = frame_idx / fps  # 현재 프레임의 시간(초)
            
            # ============================================================
            # [Logic 1] 첫 프레임 처리
            # ============================================================
            # 첫 프레임은 무조건 캡처 대상 (첫 번째 슬라이드)
            if frame_idx == 0:
                last_scene_change = 0.0  # 첫 슬라이드 시작 시점 기록
                
                # 다중 시점 샘플링: 0초~6초 구간에서 30개 프레임 랜덤 수집
                # → 마우스가 다양한 위치에 있는 순간들을 확보하여 Median 계산
                clean_frame = self._apply_temporal_median_multipoint(
                    cap, 0.0, min(6.0, duration), fps, num_samples=30
                )
                if clean_frame is not None:
                    self._save_frame(clean_frame, current_time, output_dir, keyframes)
                else:
                    self._save_frame(frame, current_time, output_dir, keyframes)
                
                last_capture_time = current_time
                
                # 다음 프레임과 비교하기 위해 현재 프레임을 흑백으로 변환하여 저장
                # 640x360으로 리사이즈 → 계산 속도 향상
                prev_frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                frame_idx += 1
                continue

            # ============================================================
            # [Logic 2] 최소 간격 체크
            # ============================================================
            # 같은 슬라이드 내에서 너무 자주 캡처하는 것을 방지
            # 예: min_interval=0.5초 → 0.5초 이내에는 재캡처 안 함
            if current_time - last_capture_time < min_interval:
                frame_idx += 1
                continue

            # ============================================================
            # [Logic 3] 장면 전환 감지 (Pixel Difference)
            # ============================================================
            # 이전 프레임과 현재 프레임의 픽셀 차이를 계산하여 장면 전환 판단
            
            # Step 1: 현재 프레임을 작게 리사이즈 & 흑백 변환
            curr_frame_small = cv2.resize(frame, (640, 360))
            curr_frame_gray = cv2.cvtColor(curr_frame_small, cv2.COLOR_BGR2GRAY)

            # Step 2: 이전 프레임과의 절대 차이 계산
            # diff[y, x] = |current[y, x] - previous[y, x]|
            diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
            
            # Step 3: 평균 차이 계산 (0~255 범위)
            # mean_diff가 클수록 → 두 프레임이 많이 다름 → 장면 전환 가능성 높음
            mean_diff = np.mean(diff)

            # [디버깅 로그] verbose=True일 때, 임계값의 절반 이상인 변화 출력
            # → 어떤 시점에서 변화가 감지되는지 확인 가능
            if verbose and mean_diff > (threshold / 2):
                print(f"   [Diff Check] Time: {current_time:.2f}s | Diff: {mean_diff:.2f} (Threshold: {threshold})")

            # ============================================================
            # [Logic 4] 임계값 초과 시 → 장면 전환으로 판단하고 캡처
            # ============================================================
            if mean_diff > threshold:
                print(f"📸 Scene Change Detected at {current_time:.2f}s (Diff: {mean_diff:.2f})")
                
                # --- 디버깅: 원본 프레임 저장 ---
                # 마우스 제거 전의 원본 이미지를 debug 폴더에 저장
                # → 어떤 장면이 감지되었는지, 마우스 제거 전후 비교 가능
                debug_dir = os.path.join(output_dir, "debug_scene_changes")
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                cv2.imwrite(
                    os.path.join(debug_dir, f"scene_change_{current_time:.2f}s_diff_{mean_diff:.1f}.jpg"),
                    frame
                )
                
                # --- 슬라이드 경계 기록 ---
                # 이전 장면 전환 시점 ~ 현재 시점 = 하나의 슬라이드
                slide_boundaries.append((last_scene_change, current_time))
                
                # --- 슬라이드 정보 출력 ---
                slide_start = last_scene_change
                slide_end = current_time
                slide_duration = slide_end - slide_start
                print(f"   📊 Slide boundary: [{slide_start:.1f}s ~ {slide_end:.1f}s] (Duration: {slide_duration:.1f}s)")
                
                # ============================================================
                # [마우스 제거] 다중 시점 샘플링 vs 양방향 수집
                # ============================================================
                # 슬라이드 길이에 따라 다른 전략 사용
                
                if slide_duration >= 3.0:
                    # --- 전략 A: 다중 시점 샘플링 (긴 슬라이드) ---
                    # 슬라이드 전체 구간에서 무작위로 50개 프레임 수집
                    # 장점: 마우스가 다양한 위치에 있는 순간들을 확보
                    #      → Median 계산 시 마우스가 없는 배경만 추출
                    clean_frame = self._apply_temporal_median_multipoint(
                        cap, slide_start, slide_end, fps, num_samples=50
                    )
                else:
                    # --- 전략 B: 양방향 수집 (짧은 슬라이드) ---
                    # 전환 전 2초 + 전환 후 4초 = 총 6초 수집
                    # 짧은 슬라이드는 전체 구간이 부족하므로 전후 구간 활용
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    clean_frame = self._apply_temporal_median_bidirectional(
                        cap, current_pos, before_duration=2.0, after_duration=4.0, fps=fps
                    )
                
                # --- 복원된 프레임 저장 ---
                if clean_frame is not None:
                    self._save_frame(clean_frame, current_time, output_dir, keyframes)
                else:
                    # 복원 실패 시 원본 프레임 저장 (fallback)
                    self._save_frame(frame, current_time, output_dir, keyframes)
                
                # --- 상태 업데이트 ---
                last_capture_time = current_time  # 마지막 캡처 시간 갱신
                last_scene_change = current_time  # 마지막 장면 전환 시점 갱신
                prev_frame_gray = curr_frame_gray  # 다음 비교를 위한 프레임 갱신
                captured_count += 1

            frame_idx += 1

        # ============================================================
        # [마지막 슬라이드 처리]
        # ============================================================
        # 마지막 장면 전환 ~ 비디오 끝 = 마지막 슬라이드
        if last_scene_change < duration:
            slide_boundaries.append((last_scene_change, duration))

        cap.release()
        
        print(f"📋 Total slides detected: {len(slide_boundaries)}")
        
        # [Logic 5] 중복 제거 (Post-processing)
        print(f"🔍 Removing duplicates (Initial: {len(keyframes)} frames)...")
        unique_keyframes = self._remove_duplicates_by_dhash(keyframes)
        
        print(f"✅ Extraction complete. {len(unique_keyframes)} unique frames captured.")
        return unique_keyframes

    # ---------------------------------------------------------
    # [Helper Function] 양방향 Temporal Median
    # ---------------------------------------------------------
    def _apply_temporal_median_bidirectional(self, cap, start_pos, before_duration=2.0, after_duration=4.0, fps=30.0):
        """
        양방향 Temporal Median: 장면 전환 전후의 프레임을 수집하여 배경 복원
        
        Args:
            cap: VideoCapture 객체
            start_pos: 장면 전환 감지 시점의 프레임 위치
            before_duration: 전환 이전 구간 수집 시간 (초)
            after_duration: 전환 이후 구간 수집 시간 (초)
            fps: 프레임레이트
            
        Returns:
            복원된 배경 프레임 (마우스 제거됨)
        """
        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 수집 범위 계산
        before_frames = int(before_duration * fps)
        after_frames = int(after_duration * fps)
        
        # 경계 조건 처리: 시작 위치가 비디오 시작 부분이면 before 생략
        collect_start = max(0, int(start_pos) - before_frames)
        collect_end = min(total_frames, int(start_pos) + after_frames)
        
        # 샘플링 간격 (2프레임마다 1개 수집)
        sample_interval = 2
        
        # 프레임 수집
        cap.set(cv2.CAP_PROP_POS_FRAMES, collect_start)
        
        for frame_pos in range(collect_start, collect_end, sample_interval):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            # 다음 샘플 위치로 이동
            if sample_interval > 1:
                curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
                next_pos = curr + sample_interval - 1
                if next_pos < collect_end:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
        
        # 원래 위치 복구
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        if len(frames) < 3:  # 최소 3개 프레임 필요
            return None
            
        # Temporal Median 계산
        stacked_frames = np.stack(frames, axis=0)
        median_frame = np.median(stacked_frames, axis=0).astype(dtype=np.uint8)
        
        return median_frame

    # ---------------------------------------------------------
    # [Helper Function] 다중 시점 샘플링 Temporal Median
    # ---------------------------------------------------------
    def _apply_temporal_median_multipoint(self, cap, start_time, end_time, fps, num_samples=50):
        """
        [다중 시점 샘플링] 슬라이드 전체 구간에서 무작위로 프레임을 수집하여 배경 복원
        
        [핵심 아이디어]
        - 마우스는 시간에 따라 위치가 변함
        - 슬라이드 전체 구간에서 랜덤하게 프레임을 수집하면,
          각 픽셀 위치에서 "마우스가 없는 프레임"이 과반수가 됨
        - Temporal Median 계산 시 마우스는 사라지고 배경(슬라이드)만 남음
        
        [예시]
        슬라이드 구간: [120초 ~ 240초] (120초 동안)
        마우스 위치:
          - 120~130초: (100, 200)
          - 130~140초: (150, 250)
          - 140~150초: (200, 300)
          ...
        
        랜덤 샘플링 50개 → 각 픽셀에서 마우스가 없는 순간이 대부분
        → Median 결과 = 마우스 없는 깨끗한 슬라이드
        
        Args:
            cap: VideoCapture 객체
            start_time: 슬라이드 시작 시간 (초)
            end_time: 슬라이드 종료 시간 (초)
            fps: 프레임레이트
            num_samples: 수집할 샘플 개수 (기본 50개)
            
        Returns:
            복원된 배경 프레임 (마우스 제거됨) 또는 None (실패 시)
        """
        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 현재 위치 저장 (나중에 복구)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # === Step 1: 시간(초) → 프레임 번호 변환 ===
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # === Step 2: 경계 조건 처리 ===
        start_frame = max(0, start_frame)  # 음수 방지
        end_frame = min(total_frames, end_frame)  # 비디오 끝 초과 방지
        
        if end_frame - start_frame < 10:  # 최소 10프레임 필요
            return None
        
        # === Step 3: 무작위 프레임 위치 생성 ===
        np.random.seed(42)  # 재현성을 위한 시드 (같은 영상은 항상 같은 결과)
        random_frames = np.random.randint(start_frame, end_frame, num_samples)
        random_frames = np.unique(random_frames)  # 중복 제거
        random_frames.sort()  # 정렬 (순차 접근이 빠름)
        
        print(f"   🎲 Random sampling: {len(random_frames)} frames from [{start_time:.1f}s ~ {end_time:.1f}s]")
        
        # === Step 4: 프레임 수집 ===
        for frame_pos in random_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)  # 해당 프레임으로 이동
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
        
        # === Step 5: 원래 위치 복구 ===
        # 메인 루프가 계속 진행될 수 있도록 원래 위치로 되돌림
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        if len(frames) < 3:  # 최소 3개 프레임 필요 (Median 계산 위해)
            return None
            
        # === Step 6: Temporal Median 계산 ===
        # 각 픽셀 위치에서 중간값(Median)을 계산
        # 예: 픽셀 (100, 200)에서 50개 프레임의 값이 [10, 15, 200, 12, 14, ...]
        #     → Median = 14 (마우스 값 200은 이상치로 무시됨)
        stacked_frames = np.stack(frames, axis=0)  # (num_frames, height, width, 3)
        median_frame = np.median(stacked_frames, axis=0).astype(dtype=np.uint8)
        
        return median_frame

    # ---------------------------------------------------------
    # [Helper Function] 중복 프레임 제거 (dHash)
    # ---------------------------------------------------------
    def _remove_duplicates_by_dhash(self, keyframes, hash_threshold=5):
        """
        dHash(Difference Hash)를 사용하여 중복 프레임 제거
        """
        if not keyframes:
            return []

        unique_frames = []
        last_hash = None
        removed_count = 0

        for item in keyframes:
            image_path = item['image_path']
            if not os.path.exists(image_path):
                continue
            
            # 이미지 로드
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            curr_hash = self._calculate_dhash(img)
            
            is_duplicate = False
            if last_hash is not None:
                # Hamming Distance 계산
                dist = bin(last_hash ^ curr_hash).count('1')
                if dist <= hash_threshold:
                    is_duplicate = True
            
            if is_duplicate:
                try:
                    os.remove(image_path)
                    removed_count += 1
                except OSError:
                    pass
            else:
                unique_frames.append(item)
                last_hash = curr_hash
        
        print(f"🗑 Removed {removed_count} duplicate frames.")
        return unique_frames

    # ---------------------------------------------------------
    # [Helper Function] dHash 계산
    # ---------------------------------------------------------
    def _calculate_dhash(self, image):
        """이미지의 dHash (Difference Hash) 계산"""
        resized = cv2.resize(image, (9, 8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hash_val = 0
        for row in range(8):
            for col in range(8):
                if gray[row, col] < gray[row, col+1]:
                    hash_val |= 1 << (row * 8 + col)
        return hash_val

    # ---------------------------------------------------------
    # [Helper Function] 프레임 저장
    # ---------------------------------------------------------
    def _save_frame(self, frame, timestamp, output_dir, keyframes_list):
        """프레임 저장 헬퍼 함수"""
        filename = f"frame_{timestamp:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        
        keyframes_list.append({
            'timestamp': timestamp,
            'image_path': filepath
        })

if __name__ == "__main__":
    # 테스트 실행
    video_file = os.path.join("data", "input", "dirty_ex2_masked.mp4")
    output_folder = os.path.join("data", "output", "captured_frames_masked")
    
    if not os.path.exists(video_file):
        print(f"⚠ Test video not found: {video_file}")
    else:
        processor = VideoProcessor()
        processor.extract_keyframes(video_file, output_dir=output_folder, threshold=10, min_interval=2.0)
