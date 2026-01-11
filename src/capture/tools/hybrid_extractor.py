"""
================================================================================
HybridSlideExtractor - 강의 영상 슬라이드 캡처 엔진
================================================================================

[목적]
    강의 영상에서 슬라이드 전환을 감지하고, 마우스/사람 등의 노이즈가 제거된
    깨끗한 슬라이드 이미지를 추출합니다.

[핵심 알고리즘]
    1. 픽셀 차이 분석 (Pixel Difference)
       - 연속 프레임 간 픽셀 변화량 측정
       - 임계값 초과 시 장면 전환으로 판단
    
    2. ORB 구조 유사도 (Structural Similarity)
       - ORB(Oriented FAST and Rotated BRIEF) 특징점 매칭
       - 슬라이드의 구조적 변화 감지
    
    3. 스마트 버퍼링 (2.5초)
       - 장면 전환 후 2.5초간 프레임 수집
       - Median 기반으로 노이즈(마우스, 사람) 제거
       - 가장 깨끗한 프레임 선택
    
    4. RANSAC 중복 제거
       - 기하학적 일관성 검사
       - 사람이 가려도 동일 슬라이드 인식

[파이프라인 흐름]
    입력 비디오
         │
         ▼
    ┌─────────────────────────────────────┐
    │  1. 프레임 읽기 + 스킵 최적화       │
    │     (IDLE 상태에서 0.5초 단위 샘플링) │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  2. 장면 전환 감지                   │
    │     - Pixel Diff > threshold        │
    │     - ORB Similarity < threshold    │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  3. 안정화 대기                      │
    │     - 변화량이 낮아지면 캡처 시작    │
    │     - 최대 2.5초 타임아웃           │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  4. 스마트 버퍼링 (2.5초)           │
    │     - 프레임 수집                   │
    │     - Median으로 노이즈 제거        │
    │     - 최적 프레임 선택              │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  5. 중복 제거 (Deduplication)       │
    │     - Pixel + ORB + RANSAC 검사     │
    │     - 중복이면 Dropped (DEBUG 모드 시 debug 폴더 저장) │
    └─────────────────────────────────────┘
         │
         ▼
    출력: captures/*.jpg + manifest.json

[임계값 설명]
    - sensitivity_diff (기본 3.0)
      픽셀 차이 민감도. 낮을수록 민감하게 감지.
      2.0~5.0 범위 권장.
    
    - sensitivity_sim (기본 0.8)
      ORB 구조 유사도. 높을수록 엄격한 중복 제거.
      0.6~0.9 범위 권장.
    
    - min_interval (기본 0.5초)
      캡처 최소 간격. 너무 빠른 연속 캡처 방지.

[사용 예시]
    from src.capture.tools import HybridSlideExtractor
    
    extractor = HybridSlideExtractor(
        video_path="input.mp4",
        output_dir="captures/",
        sensitivity_diff=3.0,
        sensitivity_sim=0.8,
        min_interval=0.5
    )
    slides = extractor.process(video_name="lecture")
    # slides = [{"file_name": "...", "timestamp_ms": ..., "timestamp_human": "..."}]

[출력 구조]
    output_dir/
    ├── video_001_00h00m05s000ms_hybrid.jpg   # 캡처된 슬라이드
    ├── video_002_00h01m30s000ms_hybrid.jpg
    ├── ...
    ├── debug/                                 # [옵션] 중복 프레임 (코드에서 활성화 필요)
    │   └── video_003_..._dup.jpg
    └── process_log_hybrid.txt                # 처리 로그

[성능]
    - 단일 패스 처리 (1-Pass)
    - IDLE 상태에서 프레임 스킵으로 속도 최적화
    - 5분 영상 기준 약 15~25초 처리 (CPU 의존)
"""

import cv2
import numpy as np
import os
import time
import logging


class HybridSlideExtractor:
    """
    Hybrid 방식 슬라이드 추출기.
    
    픽셀 차이(Pixel Difference)와 ORB 특징점 매칭을 결합하여
    강의 영상에서 슬라이드 전환을 감지하고 깨끗한 이미지를 추출합니다.
    
    Attributes:
        video_path (str): 입력 비디오 파일 경로
        output_dir (str): 캡처 이미지 저장 디렉토리
        sensitivity_diff (float): 픽셀 차이 민감도 (낮을수록 민감)
        sensitivity_sim (float): ORB 유사도 임계값 (높을수록 엄격)
        min_interval (float): 캡처 최소 간격 (초)
        orb: OpenCV ORB 특징점 검출기
        bf: Brute-Force 특징점 매처
        logger: 로깅 객체
    """
    
    def __init__(self, video_path, output_dir, sensitivity_diff=2.0, sensitivity_sim=0.8, min_interval=1.0):
        """
        HybridSlideExtractor 초기화.
        
        Args:
            video_path (str): 처리할 비디오 파일의 경로
            output_dir (str): 캡처된 슬라이드를 저장할 디렉토리
            sensitivity_diff (float): 픽셀 차이 민감도
                - 낮은 값(2.0): 작은 변화도 감지 → 더 많은 캡처
                - 높은 값(5.0): 큰 변화만 감지 → 적은 캡처
            sensitivity_sim (float): ORB 구조 유사도 임계값
                - 높은 값(0.9): 거의 동일해야 중복 판정 → 더 많은 캡처
                - 낮은 값(0.6): 비슷해도 중복 판정 → 적은 캡처
            min_interval (float): 연속 캡처 최소 간격 (초)
                - 너무 빠른 연속 캡처 방지
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.sensitivity_diff = sensitivity_diff
        self.sensitivity_sim = sensitivity_sim
        self.min_interval = min_interval
        
        # ORB 특징점 검출기 (최대 500개 특징점)
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Brute-Force 매처 (Hamming 거리 사용, crossCheck로 양방향 매칭)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 로거 설정
        self.logger = logging.getLogger("HybridExtractor")
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler()) 
            
        # 출력 디렉토리 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
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
        self.last_saved_frame = None  # 마지막으로 저장된 프레임 (중복 비교용)
        self.current_pending_capture = None  # 버퍼링 중인 캡처 데이터

    def _get_frame_signature(self, frame):
        """프레임 시그니처 계산: 그레이스케일 + ORB 디스크립터"""
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, des = self.orb.detectAndCompute(gray, None)
        return gray, des

    def _calculate_metrics(self, prev_gray, curr_gray, prev_des, curr_des):
        """픽셀 차이와 ORB 유사도 계산"""
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
        
        # 상태 변수 초기화
        prev_gray = None
        prev_des = None
        last_capture_time = -self.min_interval 
        self.last_saved_frame = None
        
        is_in_transition = False  # 전환 중 상태
        transition_start_time = 0.0  # 전환 시작 시간
        
        frame_idx = 0
        extracted_slides = []  # 추출된 슬라이드 목록
        
        # 체크 간격 (0.5초 단위 샘플링)
        check_step = int(fps * 0.5) 
        if check_step < 1: check_step = 1
        
        # 버퍼링 상태
        current_pending_capture = None
        
        while cap.isOpened():
            # 1. 프레임 읽기
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            current_time = frame_idx / fps  # 현재 시간 (초)

            # 2. 스킵 로직 (IDLE 모드에서 프레임 건너뛰기)
            target_skip = 1
            if self.current_pending_capture is None and not is_in_transition:
                if frame_idx % check_step != 0 and frame_idx != 0:
                     target_skip = check_step - (frame_idx % check_step)
            
            if target_skip > 1:
                for _ in range(target_skip - 1):
                    if not cap.grab(): break
                    frame_idx += 1
                continue
                
            # 3. 프레임 시그니처 계산 (640x360으로 축소하여 민감도 향상)
            try:
                small = cv2.resize(frame, (640, 360))
            except cv2.error:
                break
            curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            _, curr_des = self.orb.detectAndCompute(curr_gray, None)
            
            if prev_gray is not None:
                # 4. 메트릭 계산 (640x360 해상도에 맞는 5x5 커널 사용)
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
                    # [인터럽트 로직]
                    # 버퍼링 중에 새로운 장면 전환이 감지되면 이전 캡처를 즉시 완료
                    if self.current_pending_capture is not None and is_significant_change:
                         self.logger.info(f"새 전환으로 버퍼 인터럽트 at {current_time:.2f}s. 이전 캡처 완료.")
                         self._finalize_capture(self.current_pending_capture, extracted_slides, video_name)
                         self.current_pending_capture = None

                    if is_significant_change and (current_time - last_capture_time) >= self.min_interval:
                        # 전환 시작
                        is_in_transition = True
                        transition_start_time = current_time
                        self.logger.info(f"전환 시작 at {current_time:.2f}s (Diff:{diff_val:.1f}, Sim:{sim_val:.2f})")
                        
                else:
                    is_stable = (diff_val < (self.sensitivity_diff / 4.0))  # 더 엄격한 안정성 기준
                    time_in_transition = current_time - transition_start_time
                    
                    if is_stable or time_in_transition > 2.5:
                        is_in_transition = False
                        # 캡처 트리거 (아직 완료하지 않고 버퍼링 시작)
                        if self.current_pending_capture is None:
                            self.current_pending_capture = {
                                'trigger_time': current_time,
                                'buffer': [frame.copy()], 
                                'buffer_start': current_time
                            }
                            self.logger.info(f"안정 상태 감지 at {current_time:.2f}s. 마우스 제거를 위한 버퍼링 시작...")

            # 5. 스마트 버퍼링 로직 (캡처 후 처리)
            if self.current_pending_capture is not None:
                # 현재 프레임을 버퍼에 추가
                self.current_pending_capture['buffer'].append(frame.copy())
                
                # 버퍼 지속 시간 확인 (2.5초)
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

            # 첫 프레임 특수 처리
            if frame_idx == 1:  # frame_idx는 이미 증가됨
                self.current_pending_capture = {
                    'trigger_time': current_time,
                    'buffer': [frame.copy()],
                    'buffer_start': current_time
                }
            
            # NOTE: frame_idx는 Line 254에서 이미 증가됨
            # 여기서 다시 증가시키면 안 됨 (2배 타임스탬프 버그 원인이었음)
            
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
        
        # [V2] 2.5초 버퍼에서 최적 프레임 선택
        # 1. Median을 기준으로 계산 (노이즈 식별용)
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        median_gray = cv2.cvtColor(cv2.resize(median_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        
        # 2. Median에 가장 가까운 실제 프레임 찾기 (가장 깨끗한 프레임)
        best_frame = pending['buffer'][0]
        best_diff = float('inf')
        
        for frame in pending['buffer']:
            frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(median_gray, frame_gray))
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        self.logger.info(f"최적 프레임 선택됨: Median과의 차이 = {best_diff:.2f}")
        
        # Median(합성) 대신 best_frame(실제)을 사용
        selected_frame = best_frame

        
        should_save = True
        if self.last_saved_frame is not None:
            last_gray = cv2.cvtColor(cv2.resize(self.last_saved_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            curr_gray_med = cv2.cvtColor(cv2.resize(selected_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            
            # 1. 픽셀 차이 분석
            ddiff = cv2.absdiff(last_gray, curr_gray_med)
            dedupe_score = np.mean(ddiff)

            
            # 2. ORB 유사도 분석 (구조 검사)
            # 캐시된 디스크립터가 있으면 사용, 없으면 계산
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
            
            # 중복 제거 로직:
            # 1. 픽셀 Diff: 매우 낮으면 (< sensitivity) 중복
            # 2. 구조 (Sim): 매우 높으면 (> sensitivity) 중복  
            # 3. [NEW] 기하학적 일관성 (RANSAC):
            #    Sim이 낮아도 (예: 사람이 가림) 남은 매치가 고정된 슬라이드를 형성하는지 확인
            #    충분한 Inlier가 이전 슬라이드와 일치하면 동일 슬라이드 (전경 노이즈)
            
            is_duplicate = False
            ransac_inliers = 0
            
            # 기본 검사
            # Sim이 최소 0.5 이상일 때만 중복으로 판정
            # Sim < 0.5이면 다른 슬라이드 - 팝업/메뉴, 항상 캡처
            if sim_score >= 0.5 and (dedupe_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
                is_duplicate = True
            
            # 고급 검사 (아직 중복이 아니지만 '슬라이드 + 사람'일 수 있음)
            elif len(good_matches) > 10:
                # 좋은 매치의 위치 추출 (키포인트 필요!)
                src_pts = np.float32([last_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # RANSAC (고정된 배경 찾기)
                if len(src_pts) > 4:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        ransac_inliers = np.sum(mask)
                        # 원래 구조의 15% 이상이 고정적으로 보존되면 동일 슬라이드일 가능성 높음
                        # (85%가 사람에 의해 가려져도)
                        inlier_ratio = ransac_inliers / len(last_des)
                        
                        # [정제된 로직]
                        # 픽셀 Diff가 치명적이지 않을 때만 RANSAC 신뢰
                        # Diff > 20.0 (대규모 변화)이면 동일 템플릿의 새 슬라이드일 수 있음. 캡처.
                        # [NEW] Sim Score도 확인. Sim < 0.5 (대규모 구조 변화)면 캡처.
                        if inlier_ratio > 0.15 and dedupe_score < 20.0 and sim_score >= 0.5: 
                            is_duplicate = True
                            self.logger.info(f"Geometric Fix: Low Sim ({sim_score:.2f}) but High Inliers ({ransac_inliers}, {inlier_ratio:.2f}) -> Duplicate")
                        elif inlier_ratio > 0.15 and (dedupe_score >= 20.0 or sim_score < 0.5):
                             self.logger.info(f"Geometric Bypass: High Inliers ({ransac_inliers}) but Massive Change (Diff {dedupe_score:.1f}, Sim {sim_score:.2f}) -> Capture")

            if is_duplicate:
                should_save = False
                
                # [추적용] 중복을 'debug' 폴더에 저장 (필요 시 주석 해제)
                # time_str = self._format_time(current_time)
                # # 파일명에 RANSAC 정보 포함
                # dup_filename = f"{video_name}_{len(extracted_slides)+1:03d}_{time_str}_diff{dedupe_score:.1f}_sim{sim_score:.2f}_inliers{ransac_inliers}_dup.jpg"
                # dup_dir = os.path.join(self.output_dir, "debug")
                # os.makedirs(dup_dir, exist_ok=True)
                # dup_path = os.path.join(dup_dir, dup_filename)
                
                # cv2.imwrite(dup_path, selected_frame)
                
                # self.logger.info(f"저장됨 (중복): {dup_filename}")
                self.logger.info(f"중복 감지됨 (Drop): Diff={dedupe_score:.1f}, Sim={sim_score:.2f}")
            else:
                self.last_saved_des = curr_des  # 다음 비교를 위해 캐시
                self.last_saved_kp = curr_kp    # 키포인트도 캐시
        
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
