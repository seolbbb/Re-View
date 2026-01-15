"""
강의 영상에서 슬라이드를 캡처하는 HybridSlideExtractor 구현.
"""

import cv2
import numpy as np
import os
import logging


class HybridSlideExtractor:
    """
    픽셀 차이와 ORB 유사도를 결합해 슬라이드 전환을 감지하는 캡처 엔진.

    전환 이후 일정 시간 버퍼를 모아 노이즈가 적은 프레임을 선택하고,
    지연 저장 방식으로 end_ms가 확정된 뒤 한 번만 저장한다.
    """
    
    def __init__(
        self,
        video_path,
        output_dir,
        sensitivity_diff=3.0,
        sensitivity_sim=0.8,
        min_interval=0.5,
        sample_interval_sec=0.5,
        buffer_duration_sec=2.5,
        transition_timeout_sec=2.5,
    ):
        """
        캡처 엔진을 초기화한다.

        video_path: 입력 비디오 파일 경로.
        output_dir: 캡처 이미지를 저장할 디렉터리.
        sensitivity_diff: 픽셀 차이 민감도(낮을수록 민감).
        sensitivity_sim: ORB 유사도 임계값(높을수록 엄격).
        min_interval: 연속 캡처 최소 간격(초).
        sample_interval_sec: 유휴 상태에서 프레임을 샘플링하는 간격(초).
        buffer_duration_sec: 전환 이후 버퍼링 지속 시간(초).
        transition_timeout_sec: 전환 상태 최대 대기 시간(초).
        """
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be > 0")
        if buffer_duration_sec <= 0:
            raise ValueError("buffer_duration_sec must be > 0")
        if transition_timeout_sec <= 0:
            raise ValueError("transition_timeout_sec must be > 0")
        self.video_path = video_path
        self.output_dir = output_dir
        self.sensitivity_diff = sensitivity_diff
        self.sensitivity_sim = sensitivity_sim
        self.min_interval = min_interval
        self.sample_interval_sec = sample_interval_sec
        self.buffer_duration_sec = buffer_duration_sec
        self.transition_timeout_sec = transition_timeout_sec
        
        # ORB 특징점 검출기 (최대 500개 특징점)
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Brute-Force 매처 (Hamming 거리 사용, crossCheck로 양방향 매칭)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 로거 설정
        self.logger = logging.getLogger("HybridExtractor")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())
            
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 파일 로깅 설정 (capture_log.txt)
        log_file = os.path.join(self.output_dir, "..", "capture_log.txt")
        log_file = os.path.abspath(log_file)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        self.file_handler = file_handler
        
        # 상태 변수 초기화
        self.last_saved_frame = None
        self.last_saved_kp = None
        self.last_saved_des = None
        
        # Delayed Save를 위한 버퍼
        self.pending_slide = None  # {"frame": ..., "start_ms": ...}

    def process(self, video_name="video"):
        """
        비디오를 순회하며 슬라이드를 추출한다.

        전환 감지 후 버퍼링한 프레임 중 최적 프레임을 선택하고,
        다음 슬라이드가 감지될 때까지 저장을 지연한다.

        video_name: 출력 파일명 접두사.
        반환: [{"file_name": str, "start_ms": int, "end_ms": int}, ...]
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0
        
        # 상태 변수 초기화
        prev_gray = None
        prev_des = None
        last_capture_time = -self.min_interval
        self.last_saved_frame = None
        self.pending_slide = None
        
        is_in_transition = False
        transition_start_time = 0.0
        
        frame_idx = 0
        extracted_slides = []
        slide_idx = 0  # 슬라이드 인덱스 (1-based)
        
        # 체크 간격 (유휴 상태 샘플링)
        check_step = int(fps * self.sample_interval_sec)
        if check_step < 1:
            check_step = 1
            
        current_pending_capture = None
        
        while cap.isOpened():
            # 1. 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            current_time = frame_idx / fps
            current_ms = int(current_time * 1000)
            
            # 2. 스킵 로직 (IDLE 모드에서 프레임 건너뛰기)
            if current_pending_capture is None and not is_in_transition:
                if frame_idx % check_step != 0 and frame_idx != 1:
                    continue
            
            # 3. 프레임 시그니처 계산
            try:
                small = cv2.resize(frame, (640, 360))
            except cv2.error:
                break
            curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            curr_kp, curr_des = self.orb.detectAndCompute(curr_gray, None)
            
            should_finalize_buffer = False
            
            if prev_gray is not None:
                # 4. 메트릭 계산
                diff = cv2.absdiff(prev_gray, curr_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
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
                    # [인터럽트 로직] 버퍼링 중 새 전환 감지
                    if current_pending_capture is not None and is_significant_change:
                        self.logger.info(
                            f"Buffer interrupted by new transition at {current_time:.2f}s"
                        )
                        should_finalize_buffer = True
                        
                    if is_significant_change and (current_time - last_capture_time) >= self.min_interval:
                        is_in_transition = True
                        transition_start_time = current_time
                        self.logger.info(
                            f"Transition started at {current_time:.2f}s "
                            f"(Diff:{diff_val:.1f}, Sim:{sim_val:.2f})"
                        )
                else:
                    # 안정성 확인
                    is_stable = (diff_val < (self.sensitivity_diff / 4.0))
                    time_in_transition = current_time - transition_start_time
                    
                    if is_stable or time_in_transition > self.transition_timeout_sec:
                        is_in_transition = False
                        if current_pending_capture is None:
                            current_pending_capture = {
                                'trigger_time': current_time,
                                'buffer': [frame.copy()],
                                'buffer_start': current_time
                            }
                            self.logger.info(
                                f"Stable state detected at {current_time:.2f}s. Buffering starts..."
                            )
            
            # 5. 스마트 버퍼링 로직
            if current_pending_capture is not None:
                if not should_finalize_buffer:
                    current_pending_capture['buffer'].append(frame.copy())
                    
                buffer_duration = current_time - current_pending_capture['buffer_start']
                if buffer_duration >= self.buffer_duration_sec or should_finalize_buffer:
                    # 버퍼에서 최적 프레임 선택 및 중복 검사
                    best_frame, should_save = self._select_best_frame_and_check_duplicate(
                        current_pending_capture, curr_kp, curr_des
                    )
                    
                    if should_save:
                        new_start_ms = int(current_pending_capture['trigger_time'] * 1000)
                        
                        # 핵심: Delayed Save로 이전 pending_slide를 저장
                        if self.pending_slide is not None:
                            slide_idx += 1
                            end_ms = new_start_ms
                            self._save_slide(video_name, slide_idx, self.pending_slide, end_ms, extracted_slides)
                        
                        # 현재 슬라이드를 pending으로 저장
                        self.pending_slide = {
                            'frame': best_frame,
                            'start_ms': new_start_ms
                        }
                        self.last_saved_frame = best_frame
                        
                    last_capture_time = current_pending_capture['trigger_time']
                    current_pending_capture = None
            
            # 다음 프레임 비교를 위해 업데이트
            prev_gray = curr_gray
            prev_des = curr_des
            
            # 첫 프레임 특수 처리
            if frame_idx == 1:
                current_pending_capture = {
                    'trigger_time': current_time,
                    'buffer': [frame.copy()],
                    'buffer_start': current_time
                }
        
        # 마지막 버퍼 처리
        if current_pending_capture:
            best_frame, should_save = self._select_best_frame_and_check_duplicate(
                current_pending_capture, None, None
            )
            if should_save:
                new_start_ms = int(current_pending_capture['trigger_time'] * 1000)
                
                if self.pending_slide is not None:
                    slide_idx += 1
                    self._save_slide(video_name, slide_idx, self.pending_slide, new_start_ms, extracted_slides)
                
                self.pending_slide = {
                    'frame': best_frame,
                    'start_ms': new_start_ms
                }
        
        # 마지막 pending_slide 저장 (end_ms = duration)
        if self.pending_slide is not None:
            slide_idx += 1
            self._save_slide(video_name, slide_idx, self.pending_slide, duration_ms, extracted_slides)
        
        cap.release()
        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            
        return extracted_slides

    def _select_best_frame_and_check_duplicate(self, pending, curr_kp, curr_des):
        """
        버퍼에서 최적 프레임을 고르고 중복 여부를 판단한다.

        pending: 버퍼링 데이터({"buffer": [...], ...}).
        반환: (best_frame, should_save) 튜플.
        """
        if not pending['buffer']:
            return None, False
            
        stack = np.array(pending['buffer'])
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        median_gray = cv2.cvtColor(cv2.resize(median_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        
        best_frame = pending['buffer'][0]
        best_diff = float('inf')
        
        for frame in pending['buffer']:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(median_gray, frame_gray))
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        self.logger.info(f"Best frame selected: diff from median = {best_diff:.2f}")
        
        # 중복 검사
        should_save = True
        if self.last_saved_frame is not None:
            last_gray = cv2.cvtColor(cv2.resize(self.last_saved_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            curr_gray_med = cv2.cvtColor(cv2.resize(best_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            
            ddiff = cv2.absdiff(last_gray, curr_gray_med)
            dedupe_score = np.mean(ddiff)
            
            # ORB 유사도
            if self.last_saved_des is not None and self.last_saved_kp is not None:
                last_des = self.last_saved_des
                last_kp = self.last_saved_kp
            else:
                last_kp, last_des = self.orb.detectAndCompute(last_gray, None)
            
            new_kp, new_des = self.orb.detectAndCompute(curr_gray_med, None)
            
            sim_score = 0.0
            good_matches = []
            
            if last_des is not None and new_des is not None and len(last_des) > 0 and len(new_des) > 0:
                matches = self.bf.match(last_des, new_des)
                if len(matches) > 0:
                    good_matches = [m for m in matches if m.distance < 50]
                    sim_score = len(good_matches) / max(len(last_des), len(new_des))
            
            is_duplicate = False
            
            # 기본 검사: Sim >= 0.5이고 (Diff < threshold OR Sim > threshold)
            if sim_score >= 0.5 and (dedupe_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
                is_duplicate = True
            
            # RANSAC 검사 (고급)
            elif len(good_matches) > 10 and last_kp is not None and new_kp is not None:
                src_pts = np.float32([last_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([new_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                if len(src_pts) > 4:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        ransac_inliers = np.sum(mask)
                        inlier_ratio = ransac_inliers / len(last_des) if len(last_des) > 0 else 0
                        
                        if inlier_ratio > 0.15 and dedupe_score < 20.0 and sim_score >= 0.5:
                            is_duplicate = True
                            self.logger.info(
                                f"RANSAC duplicate detected: Inliers={ransac_inliers}, "
                                f"Ratio={inlier_ratio:.2f}"
                            )
            
            if is_duplicate:
                should_save = False
                self.logger.info(f"Duplicate/Dropped: Diff={dedupe_score:.1f}, Sim={sim_score:.2f}")
            else:
                self.last_saved_des = new_des
                self.last_saved_kp = new_kp
        
        return best_frame, should_save

    def _save_slide(self, video_name, idx, slide_data, end_ms, extracted_slides):
        """
        슬라이드 이미지를 저장하고 메타데이터 리스트에 추가한다.

        파일명 형식: {video_name}_{idx:03d}_{start_ms}_{end_ms}.jpg
        """
        start_ms = slide_data['start_ms']
        frame = slide_data['frame']
        
        filename = f"{video_name}_{idx:03d}_{start_ms}_{end_ms}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(save_path, frame)
        
        extracted_slides.append({
            "file_name": filename,
            "start_ms": start_ms,
            "end_ms": end_ms
        })
        
        time_str = f"{start_ms//3600000:02d}h{(start_ms//60000)%60:02d}m{(start_ms//1000)%60:02d}s{start_ms%1000:03d}ms"
        self.logger.info(f"Saved: {filename} ({time_str})")
