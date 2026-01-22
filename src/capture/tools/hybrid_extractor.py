"""
강의 영상에서 슬라이드를 캡처하는 HybridSlideExtractor 구현.

중복 슬라이드 제거 최적화:
- pHash로 O(1) 빠른 중복 검사
- 이전 저장 슬라이드들과 ORB 상세 비교
- time_ranges 병합으로 VLM 호출 최소화
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple


def calculate_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    이미지의 Perceptual Hash(pHash)를 계산한다.

    DCT 기반 해시로 리사이즈/압축에 강건하다.
    
    Args:
        image: BGR 이미지 (numpy array)
        hash_size: 해시 크기 (기본 8 = 64비트 해시)
    
    Returns:
        16진수 해시 문자열
    """
    # 그레이스케일 변환 및 32x32로 리사이즈
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    
    # DCT 계산
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    
    # 중간값 기준 이진화
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    
    # 64비트를 16진수 문자열로 변환
    hash_int = sum(1 << i for i, bit in enumerate(bits) if bit)
    return format(hash_int, '016x')


def hamming_distance(hash1: str, hash2: str) -> int:
    """두 pHash 간의 해밍 거리를 계산한다."""
    if len(hash1) != len(hash2):
        return 64  # 최대 거리 반환
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count('1')


def calculate_info_score(image: np.ndarray, orb_features: int) -> float:
    """
    이미지의 정보량 점수를 계산한다.
    
    ORB 특징점 수 (60%) + 엣지 밀도 (40%) 가중 평균.
    
    Args:
        image: BGR 이미지
        orb_features: ORB 특징점 수
    
    Returns:
        0~1 사이의 정보량 점수
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 엣지 밀도 계산 (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # ORB 특징점 정규화 (500개 기준)
    orb_normalized = min(orb_features / 500.0, 1.0)
    
    # 가중 평균
    return 0.6 * orb_normalized + 0.4 * edge_density



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
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 상태 변수 초기화
        self.last_saved_frame = None
        self.last_saved_kp = None
        self.last_saved_des = None
        
        # Delayed Save를 위한 버퍼
        self.pending_slide = None  # {"frame": ..., "start_ms": ...}
        
        # === 중복 제거 최적화 ===
        # pHash 인덱스: {phash: slide_idx} - O(1) 중복 검사용
        self.phash_index: Dict[str, int] = {}
        
        # 저장된 슬라이드 히스토리: 시간 범위 병합 및 대표 이미지 교체에 사용
        # [{idx, frame, phash, info_score, time_ranges: [{start_ms, end_ms}], kp, des}, ...]
        self.saved_slides_history: List[Dict[str, Any]] = []
        
        # 최근 N장과 ORB 상세 비교 (pHash miss 시)
        self.orb_compare_limit = 5
        
        # pHash 해밍 거리 임계값 (이하면 유사 후보로 간주)
        self.phash_threshold = 10

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

                        should_finalize_buffer = True
                        
                    if is_significant_change and (current_time - last_capture_time) >= self.min_interval:
                        is_in_transition = True
                        transition_start_time = current_time

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

        # === Option A: 첫 슬라이드 재비교 (history 비어있을 때 저장된 경우 대응) ===
        self._merge_first_slide_if_duplicate()

        # === 최종 결과 재구성: saved_slides_history에서 time_ranges 포함 출력 생성 ===
        final_slides = []
        for slide in self.saved_slides_history:
            final_slides.append({
                "file_name": slide['file_name'],
                "time_ranges": slide['time_ranges'],
                "info_score": round(slide['info_score'], 3)
            })
        
        print(f"  [Dedup] Final: {len(final_slides)} unique slides (history: {len(self.saved_slides_history)})")
            
        return final_slides

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

            
            if is_duplicate:
                should_save = False

            else:
                self.last_saved_des = new_des
                self.last_saved_kp = new_kp
        
        return best_frame, should_save

    def _save_slide(self, video_name, idx, slide_data, end_ms, extracted_slides):
        """
        슬라이드 이미지를 저장하고 메타데이터 리스트에 추가한다.

        파일명 형식: {video_name}_{idx:03d}.jpg (time_ranges 사용 시)
        """
        start_ms = slide_data['start_ms']
        frame = slide_data['frame']
        
        # pHash 계산
        phash = calculate_phash(frame)
        
        # ORB 특징점
        gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        orb_count = len(kp) if kp else 0
        
        # 정보량 점수 계산
        info_score = calculate_info_score(frame, orb_count)
        
        # 중복 검사: 기존 저장 슬라이드와 비교
        duplicate_idx = self._find_duplicate_slide(frame, phash, kp, des)
        
        if duplicate_idx is not None:
            # 중복 발견: 시간 범위만 추가
            dup_slide = self.saved_slides_history[duplicate_idx]
            dup_slide['time_ranges'].append({'start_ms': start_ms, 'end_ms': end_ms})
            
            # 정보량이 더 높으면 대표 이미지 교체
            if info_score > dup_slide['info_score']:
                old_path = os.path.join(self.output_dir, dup_slide['file_name'])
                if os.path.exists(old_path):
                    os.remove(old_path)
                
                new_filename = f"{video_name}_{dup_slide['idx']:03d}.jpg"
                new_path = os.path.join(self.output_dir, new_filename)
                cv2.imwrite(new_path, frame)
                
                dup_slide['frame'] = frame
                dup_slide['phash'] = phash
                dup_slide['info_score'] = info_score
                dup_slide['kp'] = kp
                dup_slide['des'] = des
                dup_slide['file_name'] = new_filename
                
                # pHash 인덱스 업데이트
                self.phash_index[phash] = duplicate_idx
            
            print(f"  [Dedup] Slide {idx} merged to #{dup_slide['idx']} (ranges: {len(dup_slide['time_ranges'])})")
            return  # 새로 저장하지 않음
        
        # 새 슬라이드 저장
        filename = f"{video_name}_{idx:03d}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, frame)
        
        # 히스토리에 추가
        new_slide = {
            'idx': idx,
            'frame': frame,
            'phash': phash,
            'info_score': info_score,
            'time_ranges': [{'start_ms': start_ms, 'end_ms': end_ms}],
            'kp': kp,
            'des': des,
            'file_name': filename
        }
        self.saved_slides_history.append(new_slide)
        self.phash_index[phash] = len(self.saved_slides_history) - 1
        
        # extracted_slides에는 최종 결과만 추가 (process 끝에서 재구성)
        extracted_slides.append({
            "file_name": filename,
            "start_ms": start_ms,
            "end_ms": end_ms
        })
        
        # last_saved 업데이트
        self.last_saved_frame = frame
        self.last_saved_kp = kp
        self.last_saved_des = des

    def _find_duplicate_slide(
        self,
        frame: np.ndarray,
        phash: str,
        kp: Any,
        des: Any
    ) -> Optional[int]:
        """
        저장된 슬라이드 중 중복을 찾는다.
        
        1단계: pHash 해밍 거리로 빠른 후보 탐색 (O(n), 실제로는 매우 빠름)
        2단계: 후보에 대해 ORB 상세 비교
        
        Returns:
            중복 슬라이드의 인덱스 (없으면 None)
        """
        if not self.saved_slides_history:
            return None
        
        # 1단계: pHash로 후보 탐색
        candidates = []
        for existing_phash, slide_idx in self.phash_index.items():
            dist = hamming_distance(phash, existing_phash)
            if dist <= self.phash_threshold:
                candidates.append((slide_idx, dist))
        
        # 해밍 거리순 정렬
        candidates.sort(key=lambda x: x[1])
        
        # 2단계: 후보 + 최근 N장에 대해 ORB 상세 비교
        compare_indices = set([c[0] for c in candidates])
        
        # 최근 N장 추가 (pHash miss 대비)
        recent_start = max(0, len(self.saved_slides_history) - self.orb_compare_limit)
        for i in range(recent_start, len(self.saved_slides_history)):
            compare_indices.add(i)
        
        if des is None or len(des) == 0:
            # ORB 비교 불가 → pHash만으로 판단
            if candidates and candidates[0][1] <= 5:  # 매우 유사
                return candidates[0][0]
            return None
        
        gray_new = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
        
        for slide_idx in compare_indices:
            saved = self.saved_slides_history[slide_idx]
            saved_des = saved.get('des')
            saved_kp = saved.get('kp')
            
            if saved_des is None or len(saved_des) == 0:
                continue
            
            # ORB 매칭
            matches = self.bf.match(saved_des, des)
            if not matches:
                continue
            
            good_matches = [m for m in matches if m.distance < 50]
            sim_score = len(good_matches) / max(len(saved_des), len(des))
            
            # 픽셀 차이
            saved_gray = cv2.cvtColor(cv2.resize(saved['frame'], (640, 360)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(saved_gray, gray_new)
            diff_score = np.mean(diff)
            
            # 중복 판정 기준
            if sim_score >= 0.5 and (diff_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
                return slide_idx
            
            # RANSAC 추가 검사
            if len(good_matches) > 10 and saved_kp is not None and kp is not None:
                src_pts = np.float32([saved_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                if len(src_pts) > 4:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inlier_ratio = np.sum(mask) / len(saved_des) if len(saved_des) > 0 else 0
                        if inlier_ratio > 0.15 and diff_score < 20.0 and sim_score >= 0.5:
                            return slide_idx
        
        return None

    def _merge_first_slide_if_duplicate(self) -> None:
        """
        첫 번째 슬라이드를 나머지 슬라이드들과 비교하여 중복 시 병합한다.
        
        첫 슬라이드는 history가 비어있을 때 저장되어 비교 대상이 없었으므로,
        모든 저장 완료 후 재비교하여 중복 제거를 수행한다.
        """
        if len(self.saved_slides_history) < 2:
            return
        
        first = self.saved_slides_history[0]
        first_phash = first['phash']
        first_des = first.get('des')
        first_kp = first.get('kp')
        first_frame = first['frame']
        
        # 나머지 슬라이드들과 비교
        for i in range(1, len(self.saved_slides_history)):
            other = self.saved_slides_history[i]
            
            # pHash 비교
            dist = hamming_distance(first_phash, other['phash'])
            if dist > self.phash_threshold:
                continue
            
            # ORB 비교
            other_des = other.get('des')
            if first_des is None or other_des is None:
                continue
            if len(first_des) == 0 or len(other_des) == 0:
                continue
            
            matches = self.bf.match(first_des, other_des)
            if not matches:
                continue
            
            good_matches = [m for m in matches if m.distance < 50]
            sim_score = len(good_matches) / max(len(first_des), len(other_des))
            
            # 픽셀 차이
            first_gray = cv2.cvtColor(cv2.resize(first_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            other_gray = cv2.cvtColor(cv2.resize(other['frame'], (640, 360)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(first_gray, other_gray)
            diff_score = np.mean(diff)
            
            # 중복 판정
            is_duplicate = False
            if sim_score >= 0.5 and (diff_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
                is_duplicate = True
            
            if is_duplicate:
                # 첫 슬라이드의 time_ranges를 대상 슬라이드에 병합
                for tr in first['time_ranges']:
                    other['time_ranges'].insert(0, tr)  # 시간순으로 앞에 삽입
                
                # 정보량이 더 높으면 대표 이미지 교체
                if first['info_score'] > other['info_score']:
                    other['frame'] = first_frame
                    other['phash'] = first_phash
                    other['info_score'] = first['info_score']
                    other['kp'] = first_kp
                    other['des'] = first_des
                    # 파일은 이미 저장되어 있으므로 교체 로직 필요 시 추가
                
                # 첫 슬라이드 제거
                del self.saved_slides_history[0]
                
                # pHash 인덱스 재정렬
                self.phash_index.clear()
                for idx, slide in enumerate(self.saved_slides_history):
                    self.phash_index[slide['phash']] = idx
                
                # 첫 슬라이드 파일 삭제
                first_path = os.path.join(self.output_dir, first['file_name'])
                if os.path.exists(first_path):
                    os.remove(first_path)
                
                print(f"  [Dedup] First slide merged to #{other['idx']} (post-process)")
                return
