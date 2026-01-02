import cv2
import os
import numpy as np

class VideoProcessor:
    """
    비디오 처리 클래스: 강의 영상에서 슬라이드 전환을 감지하고 키프레임을 추출합니다.
    
    [주요 기능 - 1차 + 2차 정제 통합]
    1. 1차 정제 (Scene Detection): 프레임 간 픽셀 차이를 계산하여 장면 전환 시점 감지
       - threshold: 장면 전환으로 판단하기 위한 최소 diff_score
    2. 2차 정제 (Deduplication): 저장 전 마지막 프레임과 비교하여 중복 제거
       - dedupe_threshold: 저장하기 위한 최소 이미지 차이
    3. 마우스 제거 (Temporal Median): 여러 프레임의 중앙값으로 마우스 포인터 제거
    4. 메타데이터 생성: 각 추출 시점의 점수와 인덱스를 기록
    """
    def __init__(self):
        # 초기화 시 특별한 상태 저장이 필요하지 않음
        pass

    def extract_keyframes(self, video_path, output_dir='captured_frames', threshold=30, min_interval=2.0, verbose=False, video_name=None, return_analysis_data=False, dedupe_threshold=10.0):
        """
        [핵심 기능] 비디오를 분석하여 장면 전환 시점의 깨끗한 키프레임을 추출합니다.
        1차 + 2차 정제를 한 번에 처리하여 중복 이미지를 제거합니다.
        
        Args:
            video_path (str): 입력 비디오 파일 경로
            output_dir (str): 추출된 이미지와 메타데이터가 저장될 폴더 (기본값: 'captured_frames')
            threshold (float): 장면 전환 감지를 위한 diff_score 임계값. 
                               프레임 간 평균 픽셀 차이가 이 값을 초과하면 장면 전환으로 판단.
                               (기본값: 30, 낮을수록 민감하게 감지)
            min_interval (float): 캡처 간 최소 시간 간격, 초 단위 (기본값: 2.0)
            verbose (bool): 상세 분석 로그 출력 여부 (기본값: False)
            return_analysis_data (bool): True면 시각화용 분석 데이터도 함께 반환 (기본값: False)
            dedupe_threshold (float): 2차 정제 임계값. 마지막 저장된 프레임과 픽셀 차이가 이 값 이상이어야 저장.
                                       (기본값: 10.0, 0으로 설정하면 중복 검사 비활성화)
            
        Returns:
            list: 추출된 프레임 정보 리스트 (timestamp_ms, frame_index, file_name, diff_score 포함)
            또는 return_analysis_data=True인 경우: (metadata, diff_scores, fps) 튜플
        """
        if not os.path.exists(video_path):
            print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
            return ([], [], 0) if return_analysis_data else []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📂 출력 디렉토리 생성 완료: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 비디오 파일을 열 수 없습니다.")
            return ([], [], 0) if return_analysis_data else []

        # 비디오 기본 정보 획득
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 비디오 정보: {duration:.2f}초, {fps:.2f} fps, {total_frames} 프레임")
        print(f"⚙️ 설정값: 임계값={threshold}, 최소 간격={min_interval}초, 2차 정제 임계값={dedupe_threshold}")

        keyframes_metadata = [] # 최종 반환할 메타데이터 리스트
        diff_scores_list = []   # 시각화를 위한 diff score 수집 리스트
        prev_frame_gray = None  # 이전 프레임 저장용 (비교 목적)
        last_capture_time = -min_interval # 중복 캡처 방지용 타임스탬프
        
        if video_name is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        slide_idx = 1           # 슬라이드 순번 (1부터 시작)
        debug_idx = 1           # 디버그 이미지 순번
        last_scene_change = 0.0 # 마지막 장면 전환 시점
        last_saved_frame = None # 2차 정제용: 마지막 저장된 프레임 (중복 비교용)
        skipped_count = 0       # 스킵된 프레임 수
        detected_count = 0      # 감지된 장면 전환 수
        
        frame_idx = 0

        # 메인 루프: 비디오의 모든 프레임을 순차적으로 읽음
        while True:
            ret, frame = cap.read()
            if not ret: # 비디오 끝에 도달
                break

            current_time = frame_idx / fps # 현재 시점 (초)
            
            # [Step 1] 첫 번째 프레임 특수 처리 (시작 지점)
            if frame_idx == 0:
                last_scene_change = 0.0
                
                # 첫 슬라이드는 0~6초 구간을 샘플링하여 마우스를 제거함
                clean_frame = self._apply_temporal_median_multipoint(
                    cap, 0.0, min(6.0, duration), fps, num_samples=30
                )
                
                # 저장 및 메타데이터 기록
                save_frame = clean_frame if clean_frame is not None else frame
                self._save_frame_with_meta(
                    save_frame, current_time, frame_idx, output_dir, 
                    keyframes_metadata, slide_idx, diff_score=0.0, prefix=video_name
                )
                
                slide_idx += 1
                last_capture_time = current_time
                last_saved_frame = save_frame  # 2차 정제용 추적
                # 비교를 위해 현재 프레임을 흑백/리사이즈하여 저장
                prev_frame_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                frame_idx += 1
                continue

            # [Step 2] 시간 간격 필터링
            # 너무 짧은 시간에 여러 번 감지되는 현상 방지
            if current_time - last_capture_time < min_interval:
                frame_idx += 1
                continue

            # [Step 3] 픽셀 차이 계산을 통한 장면 전환 감지
            # 연산량 감소를 위해 640x360으로 줄여서 비교
            curr_frame_small = cv2.resize(frame, (640, 360))
            curr_frame_gray = cv2.cvtColor(curr_frame_small, cv2.COLOR_BGR2GRAY)

            # 이전 프레임과 현재 프레임의 절대 차이 합계의 평균 계산
            diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
            mean_diff = np.mean(diff)
            
            # 시각화를 위한 diff score 저장 (샘플링 + 장면 전환 시점 포함)
            if frame_idx % 5 == 0 or mean_diff > threshold:
                diff_scores_list.append((frame_idx, float(mean_diff)))

            # 디버깅 로그 출력 (임계값의 절반 이상 변화 시 표시)
            if verbose and mean_diff > (threshold / 2):
                print(f"   [분석] 시간: {current_time:.2f}s | 차이: {mean_diff:.2f} (임계값: {threshold})")

            # [Step 4] 장면 전환 확정 및 처리
            if mean_diff > threshold:
                detected_count += 1
                print(f"📸 장면 전환 감지: {current_time:.2f}s (차이 점수: {mean_diff:.2f})")
                
                # 디버그 이미지 저장 (감지된 원본 상태 기록)
                debug_dir = os.path.join(output_dir, "debug_scene_changes")
                os.makedirs(debug_dir, exist_ok=True)
                self._save_debug_frame(frame, current_time, debug_idx, mean_diff, debug_dir, prefix=video_name)
                debug_idx += 1
                
                # 슬라이드 구간 정보 계산
                slide_start = last_scene_change
                slide_end = current_time
                slide_duration = slide_end - slide_start
                
                # [마우스 제거 알고리즘 적용]
                if slide_duration >= 3.0:
                    # 긴 슬라이드: 전체 구간에서 50개 프레임 랜덤 샘플링 (가장 깨끗함)
                    clean_frame = self._apply_temporal_median_multipoint(
                        cap, slide_start, slide_end, fps, num_samples=50
                    )
                else:
                    # 짧은 슬라이드: 감지 시점 전후 구간을 집중 수집
                    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    clean_frame = self._apply_temporal_median_bidirectional(
                        cap, current_pos, before_duration=2.0, after_duration=4.0, fps=fps
                    )
                
                save_frame = clean_frame if clean_frame is not None else frame
                
                # [2차 정제] 마지막 저장된 프레임과 비교하여 중복 검사
                should_save = True
                image_diff = 0.0
                
                if last_saved_frame is not None and dedupe_threshold > 0:
                    # 저장된 프레임과 현재 프레임 비교
                    saved_gray = cv2.cvtColor(cv2.resize(last_saved_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                    current_gray = cv2.cvtColor(cv2.resize(save_frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                    diff_img = cv2.absdiff(saved_gray, current_gray)
                    image_diff = np.mean(diff_img)
                    
                    if image_diff < dedupe_threshold:
                        should_save = False
                        skipped_count += 1
                        print(f"   ⏭️ 스킵 (이미지 diff={image_diff:.2f} < {dedupe_threshold})")
                
                if should_save:
                    # 최종 슬라이드 이미지 저장 및 메타데이터 추가
                    self._save_frame_with_meta(
                        save_frame, current_time, frame_idx, output_dir, 
                        keyframes_metadata, slide_idx, diff_score=mean_diff, prefix=video_name
                    )
                    
                    slide_idx += 1
                    last_saved_frame = save_frame  # 2차 정제용 추적
                    print(f"   ✅ 저장됨 (scene{slide_idx-1}, 이미지 diff={image_diff:.2f})")
                
                # 다음 감지를 위한 상태 업데이트
                last_capture_time = current_time
                last_scene_change = current_time
                prev_frame_gray = curr_frame_gray
            
            frame_idx += 1

        cap.release()
        
        # [상태 저장] Grid Search 등 외부에서 접근 가능하도록 저장
        self.last_detected_count = detected_count
        self.last_skipped_count = skipped_count
        
        # [결과 요약]
        print(f"\n🔍 감지된 장면 전환: {detected_count}개")
        print(f"⏭️ 스킵된 프레임 (2차 정제): {skipped_count}개")
        print(f"✅ 최종 저장된 슬라이드: {len(keyframes_metadata)}개")
        
        if return_analysis_data:
            return keyframes_metadata, diff_scores_list, fps
        return keyframes_metadata

    def _format_time(self, seconds):
        """초 단위 시간을 00h00m00s000ms 형식의 문자열로 변환합니다."""
        ms = int((seconds % 1) * 1000)
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s{ms:03d}ms"

    def _save_frame_with_meta(self, frame, seconds, frame_idx, output_dir, meta_list, slide_idx, diff_score, prefix):
        """
        프레임을 파일로 저장하고 확장된 메타데이터 형식을 리스트에 기록합니다.
        파일명 형식: {영상파일명}_{캡처인덱스}_{h_m_s_ms}_{diff:.2f}.jpg
        """
        time_str = self._format_time(seconds)
        file_name = f"{prefix}_{slide_idx:03d}_{time_str}_{diff_score:.2f}.jpg"
        file_path = os.path.join(output_dir, file_name)
        
        cv2.imwrite(file_path, frame)
        
        meta_list.append({
            "timestamp_ms": int(seconds * 1000),
            "timestamp_human": time_str,
            "frame_index": frame_idx,
            "file_name": file_name,
            "diff_score": round(float(diff_score), 2)
        })

    def _save_debug_frame(self, frame, seconds, idx, diff, debug_dir, prefix):
        """디버그용 원본 프레임 저장 (파일명에 점수 포함)"""
        time_str = self._format_time(seconds)
        file_name = f"debug_{prefix}_{idx:03d}_{time_str}_diff{diff:.2f}.jpg"
        cv2.imwrite(os.path.join(debug_dir, file_name), frame)

    def _apply_temporal_median_multipoint(self, cap, start_time, end_time, fps, num_samples=50):
        """
        [Temporal Median 알고리즘] 
        지정된 시간 범위 내에서 무작위로 여러 프레임을 뽑아 중앙값을 계산합니다.
        이 과정을 통해 일시적으로 나타나는 마우스 포인터는 제거되고 고정된 배경만 남게 됩니다.
        """
        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        if end_frame - start_frame < 5:
            return None
        
        # 무작위 샘플링 위치 결정
        np.random.seed(42)
        random_indices = np.random.randint(start_frame, end_frame, num_samples)
        random_indices = np.unique(random_indices)
        random_indices.sort()
        
        for pos in random_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos) # 원래 위치 복구
        
        if len(frames) < 3:
            return None
            
        # 픽셀별 중앙값 계산
        stacked = np.stack(frames, axis=0)
        return np.median(stacked, axis=0).astype(dtype=np.uint8)

    def _apply_temporal_median_bidirectional(self, cap, start_pos, before_duration=2.0, after_duration=4.0, fps=30.0):
        """전환 시점 전후 구간에서 프레임을 수집하여 중앙값 계산"""
        frames = []
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        collect_start = max(0, int(start_pos) - int(before_duration * fps))
        collect_end = min(total_frames, int(start_pos) + int(after_duration * fps))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, collect_start)
        for i in range(collect_start, collect_end, 2): # 2프레임 간격 수집
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        
        if len(frames) < 3: return None
        stacked = np.stack(frames, axis=0)
        return np.median(stacked, axis=0).astype(dtype=np.uint8)

    def _remove_duplicates_by_dhash(self, metadata_list, hash_threshold=5):
        """dHash 기술을 사용하여 시각적으로 중복된 슬라이드 제거"""
        if not metadata_list: return []

        unique_list = []
        last_hash = None
        
        # 각 결과 폴더에서 파일을 다시 읽어 해시 비교
        # (앞선 과정에서 저장된 파일 경로 필요)
        for item in metadata_list:
            # output_dir 정보를 전달받지 않으므로 파일명으로 재구성 필요 
            # (이 함수는 extract_keyframes 내부에서만 쓰이므로 로컬 변수 활용 가능)
            pass 
        
        # 실제 구현에서는 extract_keyframes 내부의 output_dir 활용
        return metadata_list # (구조 유지를 위해 우선 리턴, 실제 중복제거 로직은 추출 단계에서 이미 충분함)

    def _calculate_dhash(self, image):
        """이미지의 시각적 특징을 나타내는 64비트 해시 생성"""
        resized = cv2.resize(image, (9, 8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hash_val = 0
        for row in range(8):
            for col in range(8):
                if gray[row, col] < gray[row, col+1]:
                    hash_val |= 1 << (row * 8 + col)
        return hash_val
