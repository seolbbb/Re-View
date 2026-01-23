"""
강의 영상에서 슬라이드를 캡처하는 HybridSlideExtractor 구현.
"""

import cv2
import numpy as np
import os



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
        sensitivity_sim=0.6,  # Relaxed from 0.8 for better slide merging
        min_interval=0.5,
        sample_interval_sec=0.5,  # Accuracy priority
        buffer_duration_sec=2.5,
        transition_timeout_sec=2.5,
        dedup_enabled=True,
        phash_threshold=15,  # pHash Hamming distance threshold
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
        self.buffer_duration_sec = buffer_duration_sec
        self.transition_timeout_sec = transition_timeout_sec
        self.dedup_enabled = dedup_enabled
        self.phash_threshold = phash_threshold
        
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
        
        # Global Slide Registry (Re-appearance detection)
        # List of {"phash": ..., "orb_des": ..., "file_name": ..., "time_ranges": [...], "frame": ...}
        self.slide_history = []

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
        frame_idx = 0
        self.slide_history = [] # Reset history for each process call
        extracted_slides = [] # Final output list (converted from history)
        slide_idx = 0  # 슬라이드 인덱스 (1-based)
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
                            # slide_idx는 실제 파일 저장 시에만 부여하거나, 내부적으로 관리해야 함
                            # 여기서는 _handle_finished_slide 내부에서 처리하도록 위임
                            end_ms = new_start_ms
                            self._handle_finished_slide(video_name, self.pending_slide, end_ms)
                        
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
                    self._handle_finished_slide(video_name, self.pending_slide, new_start_ms)
                
                self.pending_slide = {
                    'frame': best_frame,
                    'start_ms': new_start_ms
                }
        
        # 마지막 pending_slide 저장 (end_ms = duration)
        if self.pending_slide is not None:
            self._handle_finished_slide(video_name, self.pending_slide, duration_ms)
        
        cap.release()

        # Convert slide_history to output format
        # Sort by first appearance time
        self.slide_history.sort(key=lambda x: x["time_ranges"][0]["start_ms"])
        
        for item in self.slide_history:
            # Merge time ranges before final output (threshold: 200ms)
            merged_ranges = self._merge_time_ranges(item["time_ranges"], gap_threshold=200)

            extracted_slides.append({
                "file_name": item["file_name"],
                "time_ranges": merged_ranges,
                # info_score logic could be added here
            })

        return extracted_slides

    def _select_best_frame_and_check_duplicate(self, pending, curr_kp, curr_des):
        """
        버퍼에서 최적 프레임을 고른다.
        기존의 로컬 중복 검사는 제거됨 (Global Deduplication으로 대체).
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
        

        
        # 리팩토링: 로컬 중복 검사는 제거하고 무조건 True 반환
        # 실제 중복 검사는 _handle_finished_slide에서 전역적으로 수행
        should_save = True
        
        # update last saved vars for potential next frame comparison if needed
        # but for global dedup, we rely on registry. 
        # Updating them here might be useful if we keep local stability check, 
        # but 'self.last_saved_frame' usage in 'process' loop is mainly for this function.
        # So we update them just in case logic depends on it elsewhere, though simplified.
        self.last_saved_frame = best_frame
        
        return best_frame, should_save


    def _compute_phash(self, frame):
        """
        Compute a simple 64-bit perceptual hash (Block Mean Hash).
        Resize to 8x8 -> 64 pixels.
        Hash based on pixel > mean.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            mean_val = np.mean(resized)
            # Create 64-bit boolean array (True if pixel > mean)
            bit_array = resized > mean_val
            # Convert to plain integer (hash)
            # flatten -> packbits could work, but simple iteration is fine for 64 bits
            phash = 0
            for i, val in enumerate(bit_array.flatten()):
                if val:
                    phash |= (1 << i)
            return phash
        except Exception:
            return 0

    def _hamming_distance(self, h1, h2):
        """Compute Hamming distance between two 64-bit integers."""
        x = h1 ^ h2
        # Count set bits
        return bin(x).count('1')

    def _format_time(self, ms):
        """Convert milliseconds to MM:SS format."""
        seconds = ms // 1000
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes:02d}:{sec:02d}"

    def _compute_info_score(self, frame):
        """
        정보량 점수 계산 (Laplacian Variance).
        값이 클수록 이미지가 선명하고 정보량이 많음.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = laplacian.var()
            return score
        except Exception:
            return 0.0

    def _handle_finished_slide(self, video_name, slide_data, end_ms):
        """
        Check global registry for duplicates.
        If found -> merge time range + optionally replace image if new has higher info score.
        If not -> save new file and register.
        """
        start_ms = slide_data['start_ms']
        frame = slide_data['frame']

        # DEDUPLICATION LOGIC
        is_duplicate = False
        duplicate_entry = None
        current_info_score = self._compute_info_score(frame) if self.dedup_enabled else 0.0

        if self.dedup_enabled:
            current_phash = self._compute_phash(frame)
            
            # ORB Descriptors (compute if not already cached)
            current_kp, current_des = self.orb.detectAndCompute(frame, None)
            if current_des is None: 
                current_des = np.array([])
            
            # 1. pHash Filtering with distance scoring
            candidates = []
            for entry in self.slide_history:
                dist = self._hamming_distance(current_phash, entry["phash"])
                if dist < self.phash_threshold:  # Use config-based threshold
                    candidates.append((dist, entry))
            
            # Sort by pHash distance and limit to top 5 candidates for ORB verification
            candidates.sort(key=lambda x: x[0])
            top_candidates = [c[1] for c in candidates[:5]]
            
            # 2. ORB Verification (on top candidates only)
            if len(current_des) > 0:
                for cand in top_candidates:
                    cand_des = cand["orb_des"]
                    if cand_des is None or len(cand_des) == 0:
                        continue
                        
                    matches = self.bf.match(cand_des, current_des)
                    if len(matches) > 0:
                        good_matches = [m for m in matches if m.distance < 50]
                        sim_score = len(good_matches) / max(len(cand_des), len(current_des))
                        
                        if sim_score >= self.sensitivity_sim:
                            is_duplicate = True
                            duplicate_entry = cand
                            break
        
        if is_duplicate and duplicate_entry:
            # Merge time range
            duplicate_entry["time_ranges"].append({"start_ms": start_ms, "end_ms": end_ms})
            
            # Info Score Comparison: replace image if new one is better
            existing_info_score = duplicate_entry.get("info_score", 0.0)
            if current_info_score > existing_info_score:
                # Replace image file
                save_path = os.path.join(self.output_dir, duplicate_entry["file_name"])
                cv2.imwrite(save_path, frame)
                duplicate_entry["info_score"] = current_info_score
                # Update ORB descriptors for future comparisons
                kp, des = self.orb.detectAndCompute(frame, None)
                duplicate_entry["orb_des"] = des
                # Simplified log: filename + time range
                fname = duplicate_entry["file_name"].replace(".jpg", "")
                t_start = self._format_time(start_ms)
                t_end = self._format_time(end_ms)
                print(f"[Dedup] {fname} + {t_start}~{t_end} ({start_ms}~{end_ms}ms) (replaced)")
            else:
                fname = duplicate_entry["file_name"].replace(".jpg", "")
                t_start = self._format_time(start_ms)
                t_end = self._format_time(end_ms)
                print(f"[Dedup] {fname} + {t_start}~{t_end} ({start_ms}~{end_ms}ms)")
        else:
            # Save new slide
            # Determine new index
            idx = len(self.slide_history) + 1
            filename = f"{video_name}_{idx:03d}.jpg"  # Simplified: no timestamp
            save_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(save_path, frame)
            
            # Register with info score
            entry = {
                "phash": self._compute_phash(frame) if self.dedup_enabled else 0,
                "orb_des": None, 
                "file_name": filename,
                "time_ranges": [{"start_ms": start_ms, "end_ms": end_ms}],
                "info_score": current_info_score,
            }
            
            if self.dedup_enabled:
                kp, des = self.orb.detectAndCompute(frame, None)
                entry["orb_des"] = des
            
            self.slide_history.append(entry)


    def _save_slide(self, video_name, idx, slide_data, end_ms, extracted_slides):
        """Deprecated: Internal Use Only if needed, but logic moved to _handle_finished_slide"""
        pass

    def _merge_time_ranges(self, ranges, gap_threshold=200):
        """
        Merge fragmented time ranges if the gap between them is small.
        default gap_threshold = 200ms
        """
        if not ranges:
            return []
        
        # Sort by start time
        sorted_ranges = sorted(ranges, key=lambda x: x["start_ms"])
        
        merged = []
        current = sorted_ranges[0].copy()
        
        for i in range(1, len(sorted_ranges)):
            next_range = sorted_ranges[i]
            # Check gap (next start - current end)
            if next_range["start_ms"] - current["end_ms"] <= gap_threshold:
                # Merge: Extend end_ms
                current["end_ms"] = max(current["end_ms"], next_range["end_ms"])
            else:
                # Push current and start new
                merged.append(current)
                current = next_range.copy()
        
        merged.append(current)
        return merged

