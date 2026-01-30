"""
[Intent]
비디오 프레임 내의 특징점(ORB) 지속성을 분석하여 슬라이드를 추출하고, 
동일한 슬라이드가 반복될 경우 재저장하지 않고 시간 구간(Time Ranges)을 병합하는 엔진입니다.
화면의 변화량(Pixel Difference)감지가 아닌, '특징점의 공간적 지속성(Persistence)'을 기반으로 
동적 애니메이션과 정적 슬라이드를 정밀하게 구분합니다.

[Usage]
- src/capture/process_content.py 에서 캡처 작업을 수행할 때 메인 유틸리티로 활용됩니다.
- 대규모 비디오 강좌나 발표 자료 등 화자가 영상을 포함하면서도 정적인 슬라이드가 교체되는 상황에 최적화되어 있습니다.

[Usage Method]
- HybridSlideExtractor를 생성할 때 비디오 경로와 설정 파라미터를 전달합니다.
- .process() 메서드를 호출하여 프레임을 순차 분석하고, 중복이 제거되고 구간이 병합된 슬라이드 결과를 반환받습니다.
- 반환된 List[dict]는 manifest.json 생성에 직접 사용될 수 있는 구조를 가집니다.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from src.pipeline.logger import pipeline_logger


class HybridSlideExtractor:
    """
    [Class Purpose]
    특징점(ORB) 기반의 Streak Detection 알고리즘과 1-Stage Online Deduplication을 결합하여
    최적의 슬라이드 캡처 및 그룹화를 수행합니다.

    [Connection]
    - OpenCV (cv2): 영상 프레임 디코딩, 특징점 추출(ORB), 이미지 저장, 이미지 해싱
    - NumPy: 특징점 격자 계산 및 지속성 맵 연산
    """

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
        dedup_sim_threshold: float = 0.5
    ):
        """
        [Usage File] src/capture/process_content.py
        [Purpose] 슬라이드 추출기를 초기화하고 내부 상태 및 중복 제거를 위한 히스토리 저장소를 설정합니다.
        
        [Args]
        - video_path (str): 입력 비디오 파일 경로
        - output_dir (str): 슬라이드 이미지를 저장할 디렉토리
        - persistence_drop_ratio (float): 현재 슬라이드가 종료되었다고 판단할 특징점 감소 비율 (0.4 = 40% 감소 시 종료)
        - sample_interval_sec (float): 프레임 분석 간격 (기본 0.5초)
        - persistence_threshold (int): 특징점이 유효한 요소로 인정받기 위한 최소 지속 횟수
        - min_orb_features (int): 유의미한 슬라이드로 판단하기 위한 최소 특징점 개수
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.persistence_drop_ratio = persistence_drop_ratio
        self.sample_interval_sec = sample_interval_sec
        self.persistence_threshold = persistence_threshold
        self.min_orb_features = min_orb_features
        self.dedup_phash_threshold = dedup_phash_threshold
        self.dedup_orb_distance = dedup_orb_distance
        self.dedup_sim_threshold = dedup_sim_threshold
        
        # 특징점 추출 엔진 설정 (ORB: 빠른 속도와 적절한 성능 제공)
        self.orb = cv2.ORB_create(nfeatures=1500)
        
        # 내부 상태 변수
        self.persistence_streak_map = None  # 특징점별 지속 횟수를 기록하는 히트맵
        self.pending_slide = None           # 현재 추적 중인 잠재적 슬라이드(최대 정보량 프레임)
        self.pending_features = 0           # 현재 슬라이드의 기준 특징점 수
        self.current_slide_start_time = 0.0 # 현재 슬라이드의 시작 시간
        
        # 중복 제거 및 이력 관리
        self.slide_count = 0                # 유니크 슬라이드 ID 발급용 카운터
        self.saved_slides_history: List[Dict[str, Any]] = [] # 저장된 슬라이드들의 메타데이터 및 특징점 리스트
        
        # pHash lookup table (Key: pHash string, Value: List of history indices)
        # 여러 슬라이드가 비슷한 해시를 가질 수 있으므로 리스트로 관리
        self.phash_index: Dict[str, List[int]] = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _compute_phash(self, image: np.ndarray) -> str:
        """
        [Usage File] Internal Use
        [Purpose] 이미지의 Perceptual Hash(pHash)를 계산하여 이미지 유사도 비교의 1차 필터로 사용합니다.
        
        [Args]
        - image (np.ndarray): BGR 이미지 배열

        [Returns]
        - str: 16진수 형태의 64비트 해시 문자열
        """
        # 1. Grayscale 변환 및 32x32 리사이즈
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        # 2. DCT (Discrete Cosine Transform) 수행, float32 변환 필요
        dct = cv2.dct(np.float32(resized))
        
        # 3. 저주파 영역(8x8) 추출 (가장 왼쪽 위 DC 성분 제외)
        dct_low_freq = dct[0:8, 0:8]
        
        # 4. 평균값 계산 (DC 성분 제외하고 평균 내는 것이 정석이나, 간단하게 전체 평균 사용하기도 함)
        avg = dct_low_freq.mean()
        
        # 5. 평균보다 크면 1, 작으면 0으로 비트 생성
        diff = dct_low_freq > avg
        
        # 6. 비트를 16진수 문자열로 변환
        # flatten 후 비트 합치기
        bits = diff.flatten()
        decimal_val = 0
        for i, bit in enumerate(bits):
            if bit:
                decimal_val += 2 ** i
                
        return hex(decimal_val)[2:]

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        [Usage File] Internal Use
        [Purpose] 두 해시 문자열 간의 해밍 거리(다른 비트 수)를 계산합니다.
        """
        # 정수로 변환 후 XOR 연산
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            xor_val = val1 ^ val2
            return bin(xor_val).count('1')
        except ValueError:
            return 64 # 에러 시 최대 거리 반환

    def _merge_time_ranges(self, ranges: List[Dict[str, int]], gap_threshold_ms: int = 200) -> List[Dict[str, int]]:
        """
        [Usage File] Internal Use
        [Purpose] 파편화된 시간 구간들을 병합하여 깔끔한 타임라인을 생성합니다.
        
        [Args]
        - ranges: [{'start_ms': 100, 'end_ms': 2000}, ...] 형태의 리스트
        - gap_threshold_ms: 이 시간(ms) 이하의 공백은 하나의 구간으로 병합

        [Returns]
        - 병합된 시간 구간 리스트
        """
        if not ranges:
            return []
        
        # 시작 시간 기준 정렬
        sorted_ranges = sorted(ranges, key=lambda x: x["start_ms"])
        merged = []
        current = sorted_ranges[0].copy()
        
        for i in range(1, len(sorted_ranges)):
            next_range = sorted_ranges[i]
            # 갭 확인 (다음 시작 - 현재 끝)
            if next_range["start_ms"] - current["end_ms"] <= gap_threshold_ms:
                # 병합: 끝 시간 연장
                current["end_ms"] = max(current["end_ms"], next_range["end_ms"])
            else:
                # 확정 및 새 구간 시작
                merged.append(current)
                current = next_range.copy()
        
        merged.append(current)
        return merged

    def _find_duplicate_slide(self, phash: str, des: Optional[np.ndarray]) -> Optional[int]:
        """
        [Usage File] Internal Use (_save_slide)
        [Purpose] 현재 프레임과 과거 저장된 슬라이드들을 비교하여 중복 여부를 판단합니다.
        
        [Args]
        - phash (str): 현재 프레임의 pHash
        - des (np.ndarray): 현재 프레임의 ORB descriptors

        [Returns]
        - Optional[int]: 중복된 슬라이드의 saved_slides_history 인덱스. 없으면 None.
        """
        if not self.saved_slides_history:
            return None
        
        # 1차 필터: 저장된 모든 슬라이드와 pHash 비교 (O(N))
        # 해시 인덱스를 쓰지 않고 전체 순회하는 것이 안전함 (해시는 근사값이므로)
        candidates = []
        PHASH_THRESHOLD = self.dedup_phash_threshold
        
        for idx, slide in enumerate(self.saved_slides_history):
            dist = self._hamming_distance(phash, slide['phash'])
            if dist <= PHASH_THRESHOLD:
                candidates.append((idx, dist))
        
        # 거리순 정렬
        candidates.sort(key=lambda x: x[1])
        
        # 2차 정밀 검사: ORB 매칭
        # 후보가 없더라도 최근 3장은 무조건 비교 (히스토리 로컬성 고려)
        target_indices = [c[0] for c in candidates]
        recent_indices = range(max(0, len(self.saved_slides_history) - 3), len(self.saved_slides_history))
        for ri in recent_indices:
            if ri not in target_indices:
                target_indices.append(ri)
                
        if des is None or len(des) == 0:
            # ORB가 없으면 가장 가까운 pHash 후보(매우 유사한 경우)만 리턴
            if candidates and candidates[0][1] <= 5:
                return candidates[0][0]
            return None
            
        # matcher 생성 (Hamming distance for binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        for idx in target_indices:
            existing_slide = self.saved_slides_history[idx]
            existing_des = existing_slide.get('des')
            
            if existing_des is None or len(existing_des) == 0:
                continue
                
            # ORB 매칭
            try:
                matches = bf.match(existing_des, des)
                if not matches:
                    continue
                
                # 좋은 매칭점 선별
                # 거리가 가까운 상위 매칭만 사용 or 거리 threshold
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < self.dedup_orb_distance]
                
                # 유사도 점수 산출
                # 기준: 두 이미지 중 feature 수가 적은 쪽 대비 매칭 비율
                min_features = min(len(existing_des), len(des))
                sim_score = len(good_matches) / min_features if min_features > 0 else 0
                
                # 판정 기준 (pHash가 가깝고 ORB 유사도가 설정값 이상이면 중복으로 간주)
                if sim_score >= self.dedup_sim_threshold:
                    return idx
            except cv2.error:
                continue
                
        return None

    def _save_slide(self, image: np.ndarray, start_time: float, end_time: float) -> None:
        """
        [Usage File] Internal Use
        [Purpose] 슬라이드 발생 시 중복 검사를 수행하고, 신규 슬라이드면 저장, 중복이면 이력만 업데이트합니다.
        
        [Args]
        - image (np.ndarray): 저장할 슬라이드 이미지 (BGR)
        - start_time (float): 슬라이드 시작 시간 (초)
        - end_time (float): 슬라이드 종료 시간 (초)
        """
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # 1. 서명(Signature) 생성
        phash = self._compute_phash(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # 2. 중복 검사 (Online Deduplication)
        dup_idx = self._find_duplicate_slide(phash, des)
        
        if dup_idx is not None:
            # === Case A: 중복 발견 (Grouping) ===
            dup_slide = self.saved_slides_history[dup_idx]
            dup_slide['time_ranges'].append({'start_ms': start_ms, 'end_ms': end_ms})
            
            # 로깅
            # 로깅 - [요청 변경 사항] 파일명(확장자 제거)만 심플하게 출력
            # 예: DIFFUSION_6_002
            fname_stem = Path(dup_slide['file_name']).stem
            pipeline_logger.log("Capture", f"Grouped: {fname_stem}")
            return

        # === Case B: 신규 슬라이드 ===
        self.slide_count += 1
        video_name = Path(self.video_path).stem
        filename = f"{video_name}_{self.slide_count:03d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # 파일 저장
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # 이력 등록
        new_slide = {
            "id": f"cap_{self.slide_count:03d}", # Unique ID for internal logic
            "idx": self.slide_count,
            "file_name": filename,
            "phash": phash,
            "des": des, # ORB descriptors 저장 (for future matching)
            # time_ranges 초기화
            "time_ranges": [{'start_ms': start_ms, 'end_ms': end_ms}]
        }
        self.saved_slides_history.append(new_slide)
        
        # 로깅 - [요청 변경 사항] 파일명(확장자 제거)만 심플하게 출력
        # 예: DIFFUSION_6_003
        fname_stem = Path(filename).stem
        pipeline_logger.log("Capture", f"Saved: {fname_stem}")

    def process(self, video_name: str = "video") -> List[dict]:
        """
        [Usage File] src/capture/process_content.py
        [Purpose] 비디오를 프레임 단위로 순차 분석하여 슬라이드 이벤트를 감지하고 처리 결과를 반환합니다.
        
        [Returns]
        - List[dict]: 최종 처리된 슬라이드 목록 (파일명, time_ranges 포함)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        check_step = max(1, int(fps * self.sample_interval_sec))
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 첫 슬라이드 시작 시간
        self.current_slide_start_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps

            # 정해진 샘플링 주기에만 분석 수행
            if frame_idx % check_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp = self.orb.detect(gray, None)
                
                # 특징점 밀도 맵 생성 (32x32)
                current_map = np.zeros((32, 32), dtype=np.int32)
                h, w = gray.shape
                for k in kp:
                    x, y = k.pt
                    ix, iy = int(x * 32 / w), int(y * 32 / h)
                    if 0 <= ix < 32 and 0 <= iy < 32:
                        current_map[iy, ix] = 1

                # 지속성(Persistence) 업데이트
                if self.persistence_streak_map is None:
                    self.persistence_streak_map = current_map
                else:
                    self.persistence_streak_map = (self.persistence_streak_map + 1) * current_map

                # 유효 특징점 카운트
                confirmed_mask = (self.persistence_streak_map >= self.persistence_threshold)
                current_text_count = np.sum(confirmed_mask)

                # --- 슬라이드 전환 감지 로직 ---
                # 1. Start of Slide
                if self.pending_slide is None and current_text_count > self.min_orb_features:
                    self.pending_slide = frame.copy()
                    self.pending_features = current_text_count
                    self.current_slide_start_time = current_time # 시작 시간 기록

                # 2. End of Slide (Drop Detection)
                elif self.pending_slide is not None:
                    if current_text_count < (self.pending_features * (1 - self.persistence_drop_ratio)):
                        # 슬라이드 종료 확정 -> 저장 및 중복 검사
                        # 종료 시간은 현재 프레임 시간
                        self._save_slide(self.pending_slide, self.current_slide_start_time, current_time)
                        
                        # 상태 리셋 및 즉시 재탐색 준비
                        if current_text_count > self.min_orb_features:
                            # 바로 다음 슬라이드가 이어지는 경우
                            self.pending_slide = frame.copy()
                            self.pending_features = current_text_count
                            self.current_slide_start_time = current_time
                        else:
                            self.pending_slide = None
                            self.pending_features = 0
                    
                    # 3. Update Best Frame (정보량이 더 많아지면 갱신)
                    elif current_text_count > self.pending_features:
                        self.pending_slide = frame.copy()
                        self.pending_features = current_text_count

            frame_idx += 1

        # 마지막에 남은 슬라이드 처리
        if self.pending_slide is not None:
            self._save_slide(self.pending_slide, self.current_slide_start_time, total_frames / fps)

        cap.release()
        
        # === 최종 결과 정리 (Formatting) ===
        # 1. 첫 번째 슬라이드(시간상 가장 먼저 시작한)의 시작 시간을 0으로 보정 (Persistence Delay 보정)
        if self.saved_slides_history:
            # 가장 먼저 저장된 슬라이드(cap_001)의 첫 번째 등장 구간을 0초부터 시작하도록 수정
            first_slide = self.saved_slides_history[0]
            if first_slide['time_ranges']:
                # 리스트의 첫 번째 원소가 가장 이른 시간일 것임 (append 순서)
                first_slide['time_ranges'][0]['start_ms'] = 0

        # 2. 내부용 키(phash, des 등) 제거 및 time_ranges 병합
        final_manifest = []
        for slide in self.saved_slides_history:
            # 시간 구간 병합 (미세한 끊김 제거)
            merged_ranges = self._merge_time_ranges(slide['time_ranges'])
            
            final_manifest.append({
                "id": slide['id'],
                "file_name": slide['file_name'],
                "time_ranges": merged_ranges
            })
            
        print(f"[Capture] Completed. Total Unique Slides: {len(final_manifest)}")
        return final_manifest
