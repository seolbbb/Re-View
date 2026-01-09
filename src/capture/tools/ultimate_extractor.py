import cv2
import numpy as np
import os
import time
import logging
from itertools import product
import matplotlib.pyplot as plt

class UltimateSlideExtractor:
    def __init__(self, video_path, output_dir="captured_slides", dedupe_threshold=4.0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.dedupe_threshold = dedupe_threshold
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.logger = logging.getLogger("UltimateExtractor")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler()) 
        self.total_checked = 0
        
        self.last_saved_frame = None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_descriptors(self, frame):
        """특징점 추출 전처리"""
        gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        _, des = self.orb.detectAndCompute(gray, None)
        return des

    def _calculate_match_ratio(self, des1, des2):
        """두 프레임 간의 매칭률 계산"""
        if des1 is None or des2 is None: return 0.0
        matches = self.bf.match(des1, des2)
        # 특징점 개수 대비 좋은 매칭의 비율
        good_matches = [m for m in matches if m.distance < 40]
        return len(good_matches) / max(len(des1), len(des2))

    def analyze_pass1(self, video_name):
        """Pass 1: 전수 조사 기반 유사도 맵 생성"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        descriptors = []
        orb_counts = []
        timestamps = []
        self.logger.info("Scanning for scene transitions (ORB features)...")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % int(fps) == 0:
                des = self._get_descriptors(frame)
                descriptors.append(des)
                orb_counts.append(len(des) if des is not None else 0)
                timestamps.append(frame_idx / fps)
                
            frame_idx += 1
        cap.release()
        
        # 유사도 계산
        similarities = [1.0]
        for i in range(1, len(descriptors)):
            sim = self._calculate_match_ratio(descriptors[i-1], descriptors[i])
            similarities.append(sim)
            
        # 임계값 분석: 로컬 미니마 찾기
        transition_frames = []
        for i in range(1, len(similarities) - 1):
            if similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1]:
                local_avg = np.mean(similarities[max(0, i-3) : min(len(similarities), i+4)])
                if similarities[i] < local_avg * 0.6:
                    actual_frame = i * int(fps)
                    transition_frames.append((actual_frame, similarities[i]))
                    
        return transition_frames

    def _format_time(self, seconds):
        ms = int((seconds % 1) * 1000)
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s{ms:03d}ms"

    def extract_pass2(self, transitions, video_name="video"):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        boundaries = [(0, 0.0)]
        for t in transitions:
            if isinstance(t, tuple): boundaries.append(t)
            else: boundaries.append((t, 0.0))
        boundaries.append((total_frames, 0.0))
            
        extracted_files = []
        for i in range(len(boundaries) - 1):
            start, start_score = boundaries[i]
            end, _ = boundaries[i+1]
            
            if i > 0 and end - start < int(fps * 1.5): continue
            
            sample_frames = []
            num_samples = 30
            step = max(1, (end - start) // num_samples)
            
            for f_idx in range(start, end, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
                    if len(sample_frames) >= num_samples: break
            
            if sample_frames:
                # [핵심] 시간 축(axis=0)에 대해 중앙값 계산
                stacked = np.stack(sample_frames)
                clean_slide = np.median(stacked, axis=0).astype(np.uint8)
                
                # [Deduplication]
                should_save = True
                if self.last_saved_frame is not None:
                    last_gray = cv2.cvtColor(cv2.resize(self.last_saved_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                    curr_gray_med = cv2.cvtColor(cv2.resize(clean_slide, (640, 360)), cv2.COLOR_BGR2GRAY)
                    
                    ddiff = cv2.absdiff(last_gray, curr_gray_med)
                    dedupe_score = np.mean(ddiff)
                    
                    # ORB Check
                    # Use cached KP if available? UltimateExtractor re-reads last_saved_frame but doesn't persist `last_des` object across loop iterations easily unless we add `self.last_des`.
                    # Let's compute it fresh to be safe, or optimize. Computing fresh is safer for Ultimate logic structure.
                    last_kp, last_des = self.orb.detectAndCompute(last_gray, None)
                    curr_kp, curr_des = self.orb.detectAndCompute(curr_gray_med, None)
                    
                    sim_score = 0.0
                    match_count = 0
                    good = []
                    
                    if last_des is not None and curr_des is not None:
                        matches = self.bf.match(last_des, curr_des)
                        if len(matches) > 0:
                            good = [m for m in matches if m.distance < 40]
                            match_count = len(good)
                            if len(last_des) > 0 and len(curr_des) > 0:
                                sim_score = len(good) / max(len(last_des), len(curr_des))
                    
                    # Deduplication Logic:
                    is_duplicate = False
                    ransac_inliers = 0
                    
                    # Basic Check
                    # Only mark as duplicate if BOTH conditions are met:
                    # 1. Low pixel diff (< threshold) OR high similarity (> 0.95)
                    # 2. AND structural similarity is NOT too low (Sim >= 0.5)
                    #    (If Sim < 0.5, it's a different slide structure - popup/menu)
                    if sim_score >= 0.5 and (dedupe_score < self.dedupe_threshold or sim_score > 0.95):
                        is_duplicate = True
                        
                    # Advanced Check (Geometric Consistency)
                    elif match_count > 10:
                        src_pts = np.float32([last_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        
                        if len(src_pts) > 4:
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if mask is not None:
                                ransac_inliers = np.sum(mask)
                                inlier_ratio = ransac_inliers / len(last_des)
                                
                                # [Refined Logic]
                                # Only trust RANSAC if the Pixel Diff isn't catastrophic.
                                # If Diff > 20.0 (Massive change), it might be a new slide with same template. Capture it.
                                # [NEW] Also check Sim Score. If Sim < 0.5 (Massive Structure Change), Capture it.
                                if inlier_ratio > 0.15 and dedupe_score < 20.0 and sim_score >= 0.5: 
                                    is_duplicate = True
                                    self.logger.info(f"Geometric Fix: Low Sim ({sim_score:.2f}) but High Inliers ({ransac_inliers}, {inlier_ratio:.2f}) -> Duplicate")
                                elif inlier_ratio > 0.15 and (dedupe_score >= 20.0 or sim_score < 0.5):
                                    self.logger.info(f"Geometric Bypass: High Inliers ({ransac_inliers}) but Massive Change (Diff {dedupe_score:.1f}, Sim {sim_score:.2f}) -> Capture")
                    
                    if is_duplicate:
                        should_save = False
                        
                        # [Tracking] Save duplicate
                        timestamp = start / fps
                        time_str = self._format_time(timestamp)
                        dup_filename = f"{video_name}_{i+1:03d}_{time_str}_diff{dedupe_score:.1f}_sim{sim_score:.2f}_inliers{ransac_inliers}_dup.jpg"
                        dup_dir = os.path.join(self.output_dir, "duplicates")
                        os.makedirs(dup_dir, exist_ok=True)
                        dup_path = os.path.join(dup_dir, dup_filename)
                        
                        cv2.imwrite(dup_path, clean_slide)
                        
                        self.logger.info(f"Saved (Duplicate): {dup_filename}")
                
                if should_save:
                    # 저장
                    timestamp = start / fps
                time_str = self._format_time(timestamp)
                # Format: {video_name}_{slide_index}_{time}_sim{score}.jpg
                filename = f"{video_name}_{i+1:03d}_{time_str}_sim{start_score:.2f}.jpg"
                save_path = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(save_path, clean_slide)
                extracted_files.append({
                    "timestamp_ms": int(timestamp * 1000),
                    "timestamp_human": time_str,
                    "file_name": filename,
                    "diff_score": round(start_score, 4)
                })
                self.logger.info(f"Saved: {filename} (Segment: {start}-{end})")
                self.last_saved_frame = clean_slide
                    
        cap.release()
        return extracted_files
