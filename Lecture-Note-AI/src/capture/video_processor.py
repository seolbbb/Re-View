import cv2
import os
import numpy as np
from datetime import timedelta

# MediaPipe Import (Optional)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("⚠ MediaPipe not found. Human removal will be disabled.")

class VideoProcessor:
    def __init__(self):
        if MP_AVAILABLE:
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) # 0: general, 1: landscape(faster)
            print("✅ MediaPipe Selfie Segmentation loaded.")
        else:
            self.segmentation = None

    def extract_keyframes(self, video_path, output_dir='captured_frames', threshold=30, min_interval=2.0, capture_duration=3.0):
        """
        비디오에서 장면 전환을 감지하여 키프레임 추출 (Temporal Reconstruction + Human Removal)
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

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 Video Info: {duration:.2f}s, {fps:.2f} fps, {total_frames} frames")
        print(f"⚙️ Settings: Threshold={threshold}, Min Interval={min_interval}s, Capture Duration={capture_duration}s")

        keyframes = []
        prev_frame_gray = None
        last_capture_time = -min_interval
        
        frame_idx = 0
        captured_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            
            # 1. 첫 프레임 처리
            if frame_idx == 0:
                # 첫 장면도 3초간 분석하여 깨끗하게 추출
                clean_frame = self._collect_and_reconstruct(cap, frame_idx, duration_sec=capture_duration, fps=fps)
                if clean_frame is not None:
                    final_frame = self._remove_human(clean_frame)
                    self._save_frame(final_frame, current_time, output_dir, keyframes)
                    last_capture_time = current_time
                
                prev_frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                frame_idx += 1
                continue

            # 2. 최소 간격 체크
            if current_time - last_capture_time < min_interval:
                frame_idx += 1
                continue

            # 3. 장면 전환 감지
            curr_frame_small = cv2.resize(frame, (640, 360))
            curr_frame_gray = cv2.cvtColor(curr_frame_small, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
            mean_diff = np.mean(diff)

            # 4. 임계값 초과 시 -> Temporal Reconstruction -> Human Removal
            if mean_diff > threshold:
                print(f"📸 Scene Change Detected at {current_time:.2f}s (Diff: {mean_diff:.2f})")
                
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                # 3초간의 데이터를 모아서 배경 복원
                clean_frame = self._collect_and_reconstruct(cap, current_pos, duration_sec=capture_duration, fps=fps)
                
                if clean_frame is not None:
                    final_frame = self._remove_human(clean_frame)
                    self._save_frame(final_frame, current_time, output_dir, keyframes)
                
                last_capture_time = current_time
                prev_frame_gray = curr_frame_gray
                captured_count += 1
                
                # 캡처 분석을 위해 이동했던 위치 복구
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            frame_idx += 1

        cap.release()
        
        # 중복 제거 수행
        print(f"🔍 Removing duplicates (Initial: {len(keyframes)} frames)...")
        unique_keyframes = self._remove_duplicates(keyframes)
        
        print(f"✅ Extraction complete. {len(unique_keyframes)} unique frames captured.")
        return unique_keyframes

    def _remove_duplicates(self, keyframes, hash_threshold=10):
        """
        dHash를 사용하여 중복 프레임 제거
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
            
            # 이미지 로드 (dHash 계산용)
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            curr_hash = self._compute_dhash(img)
            
            is_duplicate = False
            if last_hash is not None:
                # Hamming Distance 계산
                dist = bin(last_hash ^ curr_hash).count('1')
                if dist <= hash_threshold:
                    is_duplicate = True
            
            if is_duplicate:
                # 중복이면 파일 삭제 및 리스트 제외
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

    def _compute_dhash(self, image):
        """
        이미지의 dHash (Difference Hash) 계산
        """
        # 1. 9x8로 리사이즈 (가로 9, 세로 8)
        resized = cv2.resize(image, (9, 8))
        # 2. 흑백 변환
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # 3. 인접 픽셀 비교 (가로 방향)
        # 픽셀[i] < 픽셀[i+1] 이면 1, 아니면 0
        hash_val = 0
        for row in range(8):
            for col in range(8):
                if gray[row, col] < gray[row, col+1]:
                    hash_val |= 1 << (row * 8 + col)
        return hash_val

    def _remove_human(self, frame):
        """
        MediaPipe를 사용하여 사람 영역을 감지하고 Inpainting으로 제거
        """
        if self.segmentation is None:
            return frame

        # MediaPipe는 RGB 입력을 받음
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(frame_rgb)

        if results.segmentation_mask is None:
            return frame

        # 마스크 생성 (사람인 부분: True, 배경: False)
        # threshold 0.5 이상을 사람으로 판단
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
        
        # 사람 영역 마스크 (uint8, 0 or 255)
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # 마스크 팽창 (Dilate) - 경계선 깔끔하게 처리하기 위해
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Inpainting (Telea 알고리즘)
        # radius: 복원 반경 (클수록 뭉개짐 심함, 작으면 덜 지워짐)
        inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        return inpainted_frame

    def _collect_and_reconstruct(self, cap, start_pos, duration_sec=3.0, fps=30.0):
        """
        지정된 위치부터 일정 시간 동안의 프레임을 수집하여 Temporal Median으로 배경 복원
        """
        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # 5초 동안 수집하되, 2프레임 간격으로 샘플링 (밀도 높임)
        # 30fps * 5s = 150 frames -> /2 = 75 frames
        sample_interval = 2 
        max_frames = int(duration_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        
        for i in range(0, max_frames, sample_interval):
            # 현재 위치에서 읽기
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            # 다음 샘플 위치로 점프 (read()가 1프레임 이동했으므로 interval-1 만큼 더 이동)
            if sample_interval > 1:
                curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr + sample_interval - 1)
        
        # 위치 복구
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        if not frames:
            return None
            
        # Temporal Median 계산
        # 메모리 절약을 위해 uint8 유지
        stacked_frames = np.stack(frames, axis=0)
        median_frame = np.median(stacked_frames, axis=0).astype(dtype=np.uint8)
        
        return median_frame

    def _save_frame(self, frame, timestamp, output_dir, keyframes_list):
        """프레임 저장 및 리스트 추가 헬퍼 함수"""
        filename = f"frame_{timestamp:.2f}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        
        keyframes_list.append({
            'timestamp': timestamp,
            'image_path': filepath
        })

if __name__ == "__main__":
    # 테스트 실행 (상대 경로 사용)
    # data/input 폴더에 테스트 영상이 있다고 가정
    video_file = os.path.join("data", "input", "dirty_ex2_masked.mp4")
    output_folder = os.path.join("data", "output", "captured_frames_masked")
    
    if not os.path.exists(video_file):
        print(f"⚠ Test video not found: {video_file}")
        print("Please place a test video in 'data/input/' or update the path.")
    else:
        processor = VideoProcessor()
        # threshold와 min_interval은 영상 특성에 따라 조절 필요
        processor.extract_keyframes(video_file, output_dir=output_folder, threshold=10, min_interval=2.0, capture_duration=5.0)

