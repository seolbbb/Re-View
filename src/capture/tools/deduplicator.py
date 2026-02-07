"""
[Intent]
추출된 슬라이드 후보군에 대해 기존에 저장된 슬라이드들과 비교하여 중복 여부를 판단하는 모듈입니다.
pHash와 ORB 매칭을 결합한 2단계 검증(Two-Stage Verification)을 수행합니다.

[Usage]
- hybrid_extractor.py: 슬라이드 저장 전 중복 검사를 위해 사용됩니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .phash_util import hamming_distance

class SlideDeduplicator:
    """
    [Class Purpose]
    슬라이드 중복 제거 로직과 관련 설정값(임계값)을 캡슐화한 클래스입니다.
    """
    
    def __init__(
        self,
        phash_threshold: int = 12,
        orb_distance: int = 60,
        sim_threshold: float = 0.5
    ):
        """
        [Usage File] src/capture/tools/hybrid_extractor.py
        
        [Args]
        - phash_threshold (int): 1차 필터용 pHash거리 임계값
        - orb_distance (int): 2차 검증용 ORB 매칭 거리
        - sim_threshold (float): 2차 검증용 유사도 점수 (낮을수록 엄격)
        """
        self.phash_threshold = phash_threshold
        self.orb_distance = orb_distance
        self.sim_threshold = sim_threshold
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def find_duplicate(
        self, 
        current_phash: str, 
        current_des: Optional[np.ndarray], 
        saved_slides: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        [Usage File] src/capture/tools/hybrid_extractor.py
        
        [Purpose]
        - 현재 저장하려는 슬라이드가 이전에 저장된 슬라이드 리스트(saved_slides)에 존재하는지 확인합니다.
        
        [Logic]
        1. 1차 필터: pHash 거리가 임계값 이하인 후보군 선정
        2. 2차 검증: 후보군 + 최근 3장 슬라이드에 대해 ORB 특징점 정밀 매칭 수행
        
        [Args]
        - current_phash (str): 현재 이미지의 pHash
        - current_des (np.ndarray): 현재 이미지의 ORB Descriptor
        - saved_slides (List[Dict]): 기존 저장된 슬라이드 메타데이터 목록

        [Returns]
        - Optional[int]: 중복된 슬라이드의 인덱스. 중복이 없으면 None.
        """
        if not saved_slides:
            return None
            
        # 1차 필터: pHash 비교 (Fast)
        candidates: List[Tuple[int, int]] = []
        for idx, slide in enumerate(saved_slides):
            dist = hamming_distance(current_phash, slide['phash'])
            if dist <= self.phash_threshold:
                candidates.append((idx, dist))
        
        # 거리순 정렬
        candidates.sort(key=lambda x: x[1])
        
        # 2차 검증 대상 추리기 (pHash 후보군 + 최근 3장)
        # 최근 3장은 pHash가 조금 다르더라도 시간적 인접성 때문에 검증 가치가 있음
        target_indices = [c[0] for c in candidates]
        recent_indices = range(max(0, len(saved_slides) - 3), len(saved_slides))
        
        for ri in recent_indices:
            if ri not in target_indices:
                target_indices.append(ri)
        
        # Descriptor가 없으면 pHash만으로 판단 (아주 가까운 경우만)
        if current_des is None or len(current_des) == 0:
            if candidates and candidates[0][1] <= 5: # 매우 가까움
                return candidates[0][0]
            return None
            
        # 2차 정밀 검사: ORB 매칭 (Detail)
        for idx in target_indices:
            existing_slide = saved_slides[idx]
            existing_des = existing_slide.get('des')
            
            if existing_des is None or len(existing_des) == 0:
                continue
                
            try:
                matches = self.bf_matcher.match(existing_des, current_des)
                if not matches:
                    continue
                
                # 좋은 매칭점만 필터링
                good_matches = [m for m in matches if m.distance < self.orb_distance]
                
                # 유사도 점수 계산
                min_features = min(len(existing_des), len(current_des))
                sim_score = len(good_matches) / min_features if min_features > 0 else 0
                
                if sim_score >= self.sim_threshold:
                    return idx  # 중복 발견!
            except cv2.error:
                continue
                
        return None
