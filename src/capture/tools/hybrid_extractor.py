"""
[Intent]
비디오 프레임 내의 특징점(ORB) 지속성을 분석하여 정적인 슬라이드 구간을 추출하고,
pHash 및 ORB 매칭을 통한 실시간 중복 제거(Deduplication)를 수행하여 최적의 키프레임 목록을 생성합니다.

[Usage]
- process_content.py: 비디오 처리의 핵심 엔진으로 인스턴스화되어 사용됩니다.

[Usage Method]
- HybridSlideExtractor 클래스에 비디오 경로 및 파라미터를 주입하여 초기화합니다.
- process() 메서드를 실행하면 내부적으로 프레임 루프를 돌며 슬라이드를 감지하고 저장합니다.
- ROI 설정이 켜져 있다면 RoiDetector를 통해 콘텐츠 영역만 크롭하여 분석합니다.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.pipeline.logger import pipeline_logger
from .roi_detector import RoiDetector
from .resizer import resize_frame_adaptive
from .phash_util import compute_phash
from .deduplicator import SlideDeduplicator


class HybridSlideExtractor:
    """
    [Class Purpose]
    특징점 밀도 맵(Feature Density Map)을 시간 축으로 누적하여 지속성(Persistence)을 측정하는 알고리즘(Streak Detection)과,
    1-Stage Online Deduplication 로직을 결합한 하이브리드 슬라이드 추출기입니다.
    """
    
    # [Constants] Smart ROI
    ROI_WARMUP_FRAMES = 30  # 30프레임(약 30초 @1fps) 동안 수집 후 Median Lock

    def __init__(
        self, 
        video_path: str, 
        output_dir: str, 
        persistence_drop_ratio: float = 0.4, 
        sample_interval_sec: float = 0.5,
        persistence_threshold: int = 6,
        min_orb_features: int = 50,
        dedup_phash_threshold: int = 12,
        dedup_orb_distance: int = 60,
        dedup_sim_threshold: float = 0.5,
        enable_roi_detection: bool = True,
        roi_padding: int = 10,
        enable_smart_roi: bool = False,
        enable_adaptive_resize: bool = True,
        callback: Optional[Any] = None
    ):
        """
        [Usage File] src/capture/process_content.py
        
        [Purpose]
        - 추출기 인스턴스를 초기화하고, 특징점 추출기(ORB), ROI 감지기, 중복 검사 히스토리 저장소를 준비합니다.
        
        [Args]
        - video_path (str): 분석할 비디오 파일의 절대 경로
        - output_dir (str): 감지된 슬라이드 이미지를 저장할 로컬 디렉토리 경로
        - persistence_drop_ratio (float): 슬라이드 전환으로 판단할 특징점 감소 비율 (0.0~1.0)
        - sample_interval_sec (float): 프레임 샘플링 간격 (초 단위)
        - persistence_threshold (int): Streak Map 유지를 위한 최소 연속 프레임 수
        - min_orb_features (int): 유효 프레임으로 인정하기 위한 최소 특징점 개수
        - dedup_phash_threshold (int): 중복 판단을 위한 pHash 거리 임계값 (낮을수록 엄격)
        - dedup_orb_distance (int): 중복 판단을 위한 ORB 매칭 거리 평균 임계값 (낮을수록 엄격)
        - dedup_sim_threshold (float): 중복 판단을 위한 유사도 점수 (SSIM/Hist) 임계값 (높을수록 엄격)
        - enable_roi_detection (bool): ROI(강의 화면 영역) 자동 감지 사용 여부
        - roi_padding (int): ROI 감지 시 추가할 여백 (픽셀)
        - enable_smart_roi (bool): Smart ROI (Median Lock) 사용 여부
        - enable_adaptive_resize (bool): 계층형 리사이징(Tiered Resize) 사용 여부
        - callback (func, optional): 진행 상황(Progress) 보고를 위한 콜백 함수
        """
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Parameters
        self.persistence_drop_ratio = persistence_drop_ratio
        self.sample_interval_sec = sample_interval_sec
        self.persistence_threshold = persistence_threshold
        self.min_orb_features = min_orb_features
        
        self.enable_roi_detection = enable_roi_detection
        self.enable_smart_roi = enable_smart_roi
        self.enable_adaptive_resize = enable_adaptive_resize
        self.callback = callback
        
        # 특징점 추출 엔진 설정 (ORB: Oriented FAST and Rotated BRIEF)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # ROI 감지기 초기화
        self.roi_detector = RoiDetector(padding=roi_padding) if enable_roi_detection else None
        self.roi_rect = None # (x, y, w, h) - None이면 전체 프레임 사용
        
        # [NEW] Deduplicator 초기화
        self.deduplicator = SlideDeduplicator(
            phash_threshold=dedup_phash_threshold,
            orb_distance=dedup_orb_distance,
            sim_threshold=dedup_sim_threshold
        )
        
        # Smart ROI State
        self.roi_warmup_samples = []
        self.roi_warmup_frames = self.ROI_WARMUP_FRAMES
        
        # 내부 상태 변수 (Persistence Logic)
        self.persistence_streak_map = None  # 특징점별 지속 횟수를 기록하는 히트맵 (32x32 Grid)
        self.pending_slide = None           # 현재 추적 중인 잠재적 슬라이드 이미지
        self.pending_features = 0           # 현재 슬라이드의 기준 특징점 수
        self.current_slide_start_time = 0.0 # 현재 슬라이드의 시작 시간(초)
        
        # 중복 제거 및 이력 관리
        self.slide_count = 0
        self.saved_slides_history: List[Dict[str, Any]] = [] # 저장된 슬라이드들의 메타데이터 리스트
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process(self, video_name: str = "video") -> List[dict]:
        """
        [Usage File]
        - process_content.py
        
        [Purpose]
        - 메인 루프: 비디오를 프레임 단위로 읽고 분석합니다.
        - ROI 감지 -> 특징점 추출 -> Persistence Map 갱신 -> 슬라이드 전환 판단 과정을 수행합니다.
        
        [Returns]
        - List[dict]: 최종 처리된 슬라이드 목록 (time_ranges가 병합된 상태)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        check_step = max(1, int(fps * self.sample_interval_sec))
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.current_slide_start_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # [Step 1] ROI Detection & Locking
            current_roi = self._update_smart_roi(frame)

            # [Step 2] ROI Crop
            if current_roi:
                x, y, w, h = current_roi
                if w > 0 and h > 0:
                    frame = frame[y:y+h, x:x+w]
            
            current_time = frame_idx / fps

            # 정해진 샘플링 주기에만 분석 수행
            if frame_idx % check_step == 0:
                # [Step 3] Resize for Processing (Speed & Accuracy) - [REF] Using resizer module
                processing_frame = resize_frame_adaptive(frame, enable=self.enable_adaptive_resize)

                gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
                kp = self.orb.detect(gray, None)
                
                # [Step 4] Feature Persistence Analysis (Streak Detection)
                current_map = np.zeros((32, 32), dtype=np.int32)
                h_img, w_img = gray.shape
                for k in kp:
                    kx, ky = k.pt
                    ix, iy = int(kx * 32 / w_img), int(ky * 32 / h_img)
                    if 0 <= ix < 32 and 0 <= iy < 32:
                        current_map[iy, ix] = 1

                if self.persistence_streak_map is None:
                    self.persistence_streak_map = current_map
                else:
                    self.persistence_streak_map = (self.persistence_streak_map + 1) * current_map

                confirmed_mask = (self.persistence_streak_map >= self.persistence_threshold)
                current_text_count = np.sum(confirmed_mask)

                # [Step 5] Slide Transition Logic
                if self.pending_slide is None and current_text_count > self.min_orb_features:
                    # Start of Slide
                    self.pending_slide = frame.copy()
                    self.pending_features = current_text_count
                    self.current_slide_start_time = current_time

                elif self.pending_slide is not None:
                    if current_text_count < (self.pending_features * (1 - self.persistence_drop_ratio)):
                        # End of Slide (Drop 감지) -> Save Trigger
                        self._save_slide(self.pending_slide, self.current_slide_start_time, current_time)
                        
                        # 즉시 재탐색
                        if current_text_count > self.min_orb_features:
                            self.pending_slide = frame.copy()
                            self.pending_features = current_text_count
                            self.current_slide_start_time = current_time
                        else:
                            self.pending_slide = None
                            self.pending_features = 0
                    
                    elif current_text_count > self.pending_features:
                        # Update Best Frame (더 선명한 프레임으로 교체)
                        self.pending_slide = frame.copy()
                        self.pending_features = current_text_count

            frame_idx += 1

        # [Step 6] Finalize remaining slide
        if self.pending_slide is not None:
            self._save_slide(self.pending_slide, self.current_slide_start_time, total_frames / fps)

        cap.release()
        
        # [Step 7] Post-processing (Time Range Merge & Result Formatting)
        if self.saved_slides_history:
            # 0초 보정
            first_slide = self.saved_slides_history[0]
            if first_slide['time_ranges']:
                first_slide['time_ranges'][0]['start_ms'] = 0

        final_manifest = []
        for slide in self.saved_slides_history:
            merged_ranges = self._merge_time_ranges(slide['time_ranges'])
            final_manifest.append({
                "id": slide['id'],
                "file_name": slide['file_name'],
                "time_ranges": merged_ranges
            })
            
        timestamp = datetime.now().strftime('%Y-%m-%d | %H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [Capture] Completed. Total Unique Slides: {len(final_manifest)}")
        return final_manifest

    def _update_smart_roi(self, frame: np.ndarray) -> Optional[tuple]:
        """
        [Usage File] Internal Use (process 메서드 초반)
        
        [Purpose]
        - 현재 프레임에 대한 ROI(관심 영역)를 계산하거나 업데이트합니다.
        - Smart ROI(Median Lock) 또는 Legacy(First Frame Lock) 로직을 분기 처리합니다.
        
        [Args]
        - frame: 입력 프레임 (BGR)
        
        [Returns]
        - current_roi: (x, y, w, h) 튜플 또는 None
        """
        if self.roi_detector is None:
            return None

        # 1. 이미 고정된 ROI가 있는 경우
        if self.roi_rect is not None:
            return self.roi_rect

        # 2. Smart ROI: Warm-up & Median Lock
        if self.enable_smart_roi:
            temp_roi = self.roi_detector.detect(frame)
            
            # 수집 중에는 현재 프레임의 감지 결과를 임시로 사용
            current_roi = temp_roi
            self.roi_warmup_samples.append(temp_roi)
            
            if len(self.roi_warmup_samples) >= self.roi_warmup_frames:
                # Warm-up 완료: Median 계산 및 Lock
                samples = np.array(self.roi_warmup_samples)
                median_roi = np.median(samples, axis=0).astype(int)
                self.roi_rect = tuple(median_roi)
                
                x, y, w, h = self.roi_rect
                pipeline_logger.log("Capture", f"ROI Locked (Smart): x={x}, y={y}, w={w}, h={h}")
                
                # 이후부터는 Lock된 값 사용
                current_roi = self.roi_rect
            
            return current_roi

        # 3. Legacy: 첫 프레임 즉시 고정 (Baseline)
        else:
            self.roi_rect = self.roi_detector.detect(frame)
            x, y, w, h = self.roi_rect
            pipeline_logger.log("Capture", f"ROI Detected: x={x}, y={y}, w={w}, h={h}")
            return self.roi_rect

    def _save_slide(self, image: np.ndarray, start_time: float, end_time: float) -> None:
        """
        [Usage File] Internal Use (process 메서드에서 슬라이드 종료 시 호출)
        
        [Purpose]
        - 슬라이드 종료가 감지되었을 때 호출되며, 중복 검사를 수행합니다.
        - 중복이면 기존 슬라이드의 time_ranges에 구간을 추가하고, 신규면 파일을 저장합니다.
        
        [Connection]
        - pipeline_logger: 진행 상황 로깅
        - compute_phash: 해시 계산 (from phash_util)
        - deduplicator.find_duplicate: 중복 검사 (from deduplicator)
        """
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # [REF] Use phash_util
        phash = compute_phash(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # [REF] Use deduplicator
        dup_idx = self.deduplicator.find_duplicate(phash, des, self.saved_slides_history)
        
        if dup_idx is not None:
            # Case A: 중복 발견 (Grouping)
            dup_slide = self.saved_slides_history[dup_idx]
            dup_slide['time_ranges'].append({'start_ms': start_ms, 'end_ms': end_ms})
            
            fname_stem = Path(dup_slide['file_name']).stem
            pipeline_logger.log("Capture", f"Grouped: {fname_stem}")
            
            if self.callback:
                self.callback("update", dup_slide)
            return

        # Case B: 신규 슬라이드저장
        self.slide_count += 1
        video_name = Path(self.video_path).stem
        filename = f"{video_name}_{self.slide_count:03d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        new_slide = {
            "id": f"cap_{self.slide_count:03d}",
            "idx": self.slide_count,
            "file_name": filename,
            "phash": phash,
            "des": des,
            "time_ranges": [{'start_ms': start_ms, 'end_ms': end_ms}]
        }
        self.saved_slides_history.append(new_slide)
        
        fname_stem = Path(filename).stem
        pipeline_logger.log("Capture", f"Saved: {fname_stem}")

        if self.callback:
            self.callback("new", new_slide)

    def _merge_time_ranges(self, ranges: List[Dict[str, int]], gap_threshold_ms: int = 200) -> List[Dict[str, int]]:
        """
        [Usage File] Internal Use (process 메서드 종료 시 후처리)
        [Purpose] 파편화된 시간 구간들을 병합하여 연속적인 타임라인을 생성합니다.
        [Args] ranges(구간 리스트), gap_threshold_ms(병합할 최대 공백 시간)
        """
        if not ranges:
            return []
        
        sorted_ranges = sorted(ranges, key=lambda x: x["start_ms"])
        merged = []
        current = sorted_ranges[0].copy()
        
        for i in range(1, len(sorted_ranges)):
            next_range = sorted_ranges[i]
            if next_range["start_ms"] - current["end_ms"] <= gap_threshold_ms:
                current["end_ms"] = max(current["end_ms"], next_range["end_ms"])
            else:
                merged.append(current)
                current = next_range.copy()
        
        merged.append(current)
        return merged
