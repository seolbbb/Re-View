import cv2
import numpy as np
import os
import time
import logging

class HybridSlideExtractor:
    def __init__(self, video_path, output_dir, sensitivity_diff=2.0, sensitivity_sim=0.8, min_interval=1.0):
        self.video_path = video_path
        self.output_dir = output_dir
        self.sensitivity_diff = sensitivity_diff
        self.sensitivity_sim = sensitivity_sim
        self.min_interval = min_interval
        
        # Tools
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.logger = logging.getLogger("HybridExtractor")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler()) 
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # File Logging Setup
        log_file = os.path.join(self.output_dir, "..", "process_log_hybrid.txt")
        log_file = os.path.abspath(log_file)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        self.file_handler = file_handler 
        self.last_saved_frame = None
        self.current_pending_capture = None

    def _get_frame_signature(self, frame):
        """Calculate frame signature: Grayscale & ORB Descriptors"""
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, des = self.orb.detectAndCompute(gray, None)
        return gray, des

    def _calculate_metrics(self, prev_gray, curr_gray, prev_des, curr_des):
        """Compute both Pixel Difference and ORB Similarity"""
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        diff_score = np.mean(clean_diff)
        
        sim_score = 0.0
        if prev_des is not None and curr_des is not None and len(prev_des) > 0 and len(curr_des) > 0:
            matches = self.bf.match(prev_des, curr_des)
            if len(matches) > 0:
                good_matches = [m for m in matches if m.distance < 50]
                sim_score = len(good_matches) / max(len(prev_des), len(curr_des))
        
        return diff_score, sim_score

    def _format_time(self, seconds):
        ms = int((seconds % 1) * 1000)
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s{ms:03d}ms"

    def process(self, video_name="video"):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # State
        prev_gray = None
        prev_des = None
        last_capture_time = -self.min_interval 
        self.last_saved_frame = None
        
        is_in_transition = False
        transition_start_time = 0.0
        
        frame_idx = 0
        extracted_slides = []
        
        check_step = int(fps * 0.5) 
        if check_step < 1: check_step = 1
        
        # Lookahead Sampling State
        current_pending_capture = None
        
        while cap.isOpened():
            # 1. Video Frame Loop
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            current_time = frame_idx / fps

            # 2. Skip logic (IDLE mode)
            target_skip = 1
            if self.current_pending_capture is None and not is_in_transition:
                if frame_idx % check_step != 0 and frame_idx != 0:
                     target_skip = check_step - (frame_idx % check_step)
            
            if target_skip > 1:
                for _ in range(target_skip - 1):
                    if not cap.grab(): break
                    frame_idx += 1
                continue
                
            # 3. Get Signature (640x360 for better sensitivity)
            try:
                small = cv2.resize(frame, (640, 360))
            except cv2.error:
                break
            curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            _, curr_des = self.orb.detectAndCompute(curr_gray, None)
            
            if prev_gray is not None:
                # 4. Calculate Metrics (Needs 5x5 kernel for 640x360)
                diff = cv2.absdiff(prev_gray, curr_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5,5), np.uint8)
                clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                diff_val = np.mean(clean_diff)
                
                sim_val = 1.0
                if prev_des is not None and curr_des is not None and len(prev_des) > 0 and len(curr_des) > 0:
                    matches = self.bf.match(prev_des, curr_des)
                    if len(matches) > 0:
                        good_matches = [m for m in matches if m.distance < 50]
                        sim_val = len(good_matches) / max(len(prev_des), len(curr_des))
                
                is_significant_change = (diff_val > self.sensitivity_diff or sim_val < self.sensitivity_sim)
                
                if not is_in_transition:
                    # [Logic for Interruption]
                    # If we are buffering but a NEW significant change happens, finalize immediately.
                    if self.current_pending_capture is not None and is_significant_change:
                         self.logger.info(f"New Transition Interrupted Buffer at {current_time:.2f}s. Finalizing previous.")
                         self._finalize_capture(self.current_pending_capture, extracted_slides, video_name)
                         self.current_pending_capture = None

                    if is_significant_change and (current_time - last_capture_time) >= self.min_interval:
                        # Start Transition
                        is_in_transition = True
                        transition_start_time = current_time
                        self.logger.info(f"Transition Started at {current_time:.2f}s (Diff:{diff_val:.1f}, Sim:{sim_val:.2f})")
                        
                else:
                    is_stable = (diff_val < (self.sensitivity_diff / 4.0)) # Stricter stability
                    time_in_transition = current_time - transition_start_time
                    
                    if is_stable or time_in_transition > 2.5:
                        is_in_transition = False
                        # Trigger Capture, but don't finalize yet (Start Buffering)
                        if self.current_pending_capture is None:
                            self.current_pending_capture = {
                                'trigger_time': current_time,
                                'buffer': [frame.copy()], 
                                'buffer_start': current_time
                            }
                            self.logger.info(f"Stable detected at {current_time:.2f}s. Buffering for mouse removal...")

            # 5. Smart Buffering Logic (Post-Capture)
            if self.current_pending_capture is not None:
                # Add current frame to buffer
                self.current_pending_capture['buffer'].append(frame.copy())
                
                # Check if buffer duration (2.5s - Increased) is met
                buffer_duration = current_time - self.current_pending_capture['buffer_start']
                if buffer_duration >= 2.5:
                     self._finalize_capture(self.current_pending_capture, extracted_slides, video_name)
                     last_capture_time = self.current_pending_capture['trigger_time']
                     self.current_pending_capture = None
            
            if not is_in_transition:
                prev_gray = curr_gray
                prev_des = curr_des
            else:
                pass

            # First frame special case
            if frame_idx == 1: # frame_idx is incremented early
                self.current_pending_capture = {
                    'trigger_time': current_time,
                    'buffer': [frame.copy()],
                    'buffer_start': current_time
                }
            
            frame_idx += 1
            
        if self.current_pending_capture:
            self._finalize_capture(self.current_pending_capture, extracted_slides, video_name)
            
        cap.release()
        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            
        return extracted_slides
        
    def _finalize_capture(self, pending, extracted_slides, video_name):
        if not pending['buffer']: return
        
        current_time = pending['trigger_time']
        stack = np.array(pending['buffer'])
        
        # [V2] Best Frame Selection from 2.5s buffer
        # 1. Calculate Median as reference (for noise identification)
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        median_gray = cv2.cvtColor(cv2.resize(median_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        
        # 2. Find the REAL frame closest to the median (cleanest)
        best_frame = pending['buffer'][0]
        best_diff = float('inf')
        
        for frame in pending['buffer']:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(median_gray, frame_gray))
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        self.logger.info(f"Best Frame Selected: Diff to Median = {best_diff:.2f}")
        
        # Use best_frame (real) instead of median_frame (synthetic)
        selected_frame = best_frame

        
        should_save = True
        if self.last_saved_frame is not None:
            last_gray = cv2.cvtColor(cv2.resize(self.last_saved_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            curr_gray_med = cv2.cvtColor(cv2.resize(selected_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            
            # 1. Pixel Difference Analysis
            ddiff = cv2.absdiff(last_gray, curr_gray_med)
            dedupe_score = np.mean(ddiff)

            
            # 2. ORB Similarity Analysis (Structure Check)
            # If cached descriptor exists, use it. Otherwise compute.
            if hasattr(self, 'last_saved_des') and self.last_saved_des is not None and hasattr(self, 'last_saved_kp') and self.last_saved_kp is not None:
                last_des = self.last_saved_des
                last_kp = self.last_saved_kp
            else:
                last_kp, last_des = self.orb.detectAndCompute(last_gray, None)
                
            curr_kp, curr_des = self.orb.detectAndCompute(curr_gray_med, None)
            
            sim_score = 0.0
            matches = []
            good_matches = []
            
            if last_des is not None and curr_des is not None and len(last_des) > 0 and len(curr_des) > 0:
                matches = self.bf.match(last_des, curr_des)
                if len(matches) > 0:
                    good_matches = [m for m in matches if m.distance < 50]
                    sim_score = len(good_matches) / max(len(last_des), len(curr_des))
            
            # Deduplication Logic:
            # 1. Pixel Diff: If very low (< sensitivity), it's a duplicate.
            # 2. Structure (Sim): If very high (> sensitivity), it's a duplicate.
            # 3. [NEW] Geometric Consistency (RANSAC):
            #    If Sim is low (e.g. person covering), check if the *remaining* matches form a rigid slide.
            #    If enough Inliers align with the previous slide, it's the same slide (Foreground Noise).
            
            is_duplicate = False
            ransac_inliers = 0
            
            # Basic Check
            # Only mark as duplicate if structure (Sim) is at least 0.5
            # If Sim < 0.5, it's a different slide - popup/menu, always capture
            if sim_score >= 0.5 and (dedupe_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
                is_duplicate = True
            
            # Advanced Check (If not yet duplicate, but might be 'Slide + Person')
            elif len(good_matches) > 10:
                # Extract location of good matches (Need Keypoints!)
                src_pts = np.float32([last_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # RANSAC (Find rigid background)
                if len(src_pts) > 4:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        ransac_inliers = np.sum(mask)
                        # If > 15% of original structure is rigidly preserved, it's likely the same slide
                        # (Even if 85% is occluded by a person)
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
                
                # [Tracking] Save duplicate to 'duplicates' folder
                time_str = self._format_time(current_time)
                # Include RANSAC info in filename
                dup_filename = f"{video_name}_{len(extracted_slides)+1:03d}_{time_str}_diff{dedupe_score:.1f}_sim{sim_score:.2f}_inliers{ransac_inliers}_dup.jpg"
                dup_dir = os.path.join(self.output_dir, "duplicates")
                os.makedirs(dup_dir, exist_ok=True)
                dup_path = os.path.join(dup_dir, dup_filename)
                
                cv2.imwrite(dup_path, selected_frame)
                
                self.logger.info(f"Saved (Duplicate): {dup_filename}")
            else:
                self.last_saved_des = curr_des # Cache for next comparison
                self.last_saved_kp = curr_kp   # Cache keypoints too
        
        if should_save:
            time_str = self._format_time(current_time)
            filename = f"{video_name}_{len(extracted_slides)+1:03d}_{time_str}_hybrid.jpg"
            save_path = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(save_path, selected_frame)
            extracted_slides.append({
                "file_name": filename,
                "timestamp_ms": int(current_time * 1000),
                "timestamp_human": time_str
            })
            
            self.logger.info(f"Captured/Finalized: {filename} (Samples: {len(pending['buffer'])})")
            if hasattr(self, 'file_handler'):
                self.file_handler.flush()
            
            self.last_saved_frame = selected_frame
