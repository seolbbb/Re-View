"""
================================================================================
dedup_detector.py - 중복 슬라이드 감지 모듈 (RANSAC 기반)
================================================================================

[역할]
    연속으로 캡처된 슬라이드가 동일한 슬라이드인지 감지합니다.
    픽셀 차이, ORB 유사도, RANSAC 기하학적 검증을 조합하여
    사람이나 마우스가 가려도 정확하게 중복을 판별합니다.

[중복 판정 알고리즘]
    1단계: 기본 검사
        - 픽셀 Diff < threshold → 거의 동일 → 중복
        - ORB Sim > threshold → 구조적으로 동일 → 중복
        - 단, Sim < 0.5면 구조적 변화가 크므로 캡처
    
    2단계: RANSAC 검사 (고급)
        - 좋은 매치(distance < 50)가 10개 이상일 때 적용
        - Homography 변환 계산 → Inlier 비율 확인
        - Inlier > 15% → 고정된 배경(슬라이드) 존재 → 중복
        - 단, Diff > 20.0이면 새 슬라이드일 수 있음 → 캡처

[사용 예시]
    from src.capture.tools.modules import DedupDetector
    
    detector = DedupDetector(
        sensitivity_diff=3.0,
        sensitivity_sim=0.8
    )
    
    is_dup = detector.is_duplicate(last_gray, curr_gray, last_kp, curr_kp, last_des, curr_des)
    if is_dup:
        print("중복 감지됨 - 캡처 스킵")

[출력]
    - is_duplicate: 중복 여부 (bool)
    - dedup_info: 판정에 사용된 메트릭 정보 (dict)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DedupDetector:
    """
    RANSAC 기반 중복 슬라이드 감지기.
    
    픽셀 차이, ORB 유사도, 기하학적 일관성(RANSAC)을 조합하여
    사람이 가린 슬라이드도 정확하게 중복 판별합니다.
    
    Attributes:
        sensitivity_diff (float): 픽셀 차이 민감도
        sensitivity_sim (float): ORB 유사도 임계값
        bf: Brute-Force 특징점 매처
    """
    
    def __init__(self, sensitivity_diff: float = 3.0, sensitivity_sim: float = 0.8):
        """
        DedupDetector 초기화.
        
        Args:
            sensitivity_diff (float): 픽셀 차이 민감도 (기본 3.0)
                - 낮은 값: 작은 변화도 다른 슬라이드로 판정
                - 높은 값: 큰 변화만 다른 슬라이드로 판정
            sensitivity_sim (float): ORB 유사도 임계값 (기본 0.8)
                - 높은 값: 매우 유사해야 중복으로 판정
                - 낮은 값: 조금만 유사해도 중복으로 판정
        """
        self.sensitivity_diff = sensitivity_diff
        self.sensitivity_sim = sensitivity_sim
        
        # Brute-Force 매처 (Hamming 거리, 양방향 매칭)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def is_duplicate(
        self,
        last_gray: np.ndarray,
        curr_gray: np.ndarray,
        last_kp,
        curr_kp,
        last_des: np.ndarray,
        curr_des: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        두 프레임이 중복(동일 슬라이드)인지 판정합니다.
        
        [판정 로직]
        1. 픽셀 차이(Diff) 계산
        2. ORB 유사도(Sim) 계산
        3. 기본 조건 검사: Sim >= 0.5이고 (Diff < threshold OR Sim > threshold)
        4. RANSAC 검사: 좋은 매치가 10개 이상일 때 기하학적 일관성 확인
        
        Args:
            last_gray (np.ndarray): 이전 저장된 프레임 (그레이스케일)
            curr_gray (np.ndarray): 현재 프레임 (그레이스케일)
            last_kp: 이전 프레임의 ORB 키포인트
            curr_kp: 현재 프레임의 ORB 키포인트
            last_des (np.ndarray): 이전 프레임의 ORB 디스크립터
            curr_des (np.ndarray): 현재 프레임의 ORB 디스크립터
            
        Returns:
            Tuple[bool, Dict]: (중복 여부, 판정 메트릭 정보)
                - 메트릭: {"diff": float, "sim": float, "inliers": int, "reason": str}
        """
        info = {
            "diff": 0.0,
            "sim": 0.0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reason": "new_slide"
        }
        
        # 1. 픽셀 차이 계산
        #    absdiff는 픽셀별 절대 차이, mean은 평균
        ddiff = cv2.absdiff(last_gray, curr_gray)
        dedupe_score = np.mean(ddiff)
        info["diff"] = round(dedupe_score, 2)
        
        # 2. ORB 유사도 계산
        sim_score = 0.0
        good_matches = []
        
        if (last_des is not None and curr_des is not None 
            and len(last_des) > 0 and len(curr_des) > 0):
            
            # 특징점 매칭 (Brute-Force)
            matches = self.bf.match(last_des, curr_des)
            
            if len(matches) > 0:
                # 좋은 매치: distance < 50 (낮을수록 유사)
                good_matches = [m for m in matches if m.distance < 50]
                
                # 유사도: 좋은 매치 수 / 전체 디스크립터 수
                sim_score = len(good_matches) / max(len(last_des), len(curr_des))
        
        info["sim"] = round(sim_score, 2)
        
        # 3. 기본 중복 검사
        #    - Sim < 0.5: 구조적으로 매우 다름 → 무조건 캡처
        #    - Sim >= 0.5이고 (Diff 낮음 OR Sim 높음) → 중복
        is_duplicate = False
        
        if sim_score >= 0.5 and (dedupe_score < self.sensitivity_diff or sim_score > self.sensitivity_sim):
            is_duplicate = True
            info["reason"] = "basic_check"
        
        # 4. RANSAC 검사 (고급)
        #    기본 검사를 통과했지만 '슬라이드 + 사람'일 수 있는 경우
        elif len(good_matches) > 10 and last_kp is not None and curr_kp is not None:
            # 매칭된 키포인트 좌표 추출
            src_pts = np.float32(
                [last_kp[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            
            dst_pts = np.float32(
                [curr_kp[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            
            # RANSAC으로 Homography 계산
            #    - 고정된 배경(슬라이드)은 동일한 변환 행렬을 공유
            #    - 전경 노이즈(사람)는 다른 변환 → Outlier로 제거
            if len(src_pts) > 4:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    # Inlier: 고정된 배경에 속하는 매치 수
                    ransac_inliers = np.sum(mask)
                    inlier_ratio = ransac_inliers / len(last_des) if len(last_des) > 0 else 0
                    
                    info["inliers"] = int(ransac_inliers)
                    info["inlier_ratio"] = round(inlier_ratio, 3)
                    
                    # 판정 조건:
                    #    - Inlier > 15%: 고정된 배경 존재
                    #    - Diff < 20.0: 대규모 변화 아님
                    #    - Sim >= 0.5: 구조적으로 유사
                    if inlier_ratio > 0.15 and dedupe_score < 20.0 and sim_score >= 0.5:
                        is_duplicate = True
                        info["reason"] = "ransac_fix"
        
        if not is_duplicate:
            info["reason"] = "new_slide"
        
        return is_duplicate, info
    
    def compute_similarity(
        self,
        last_des: np.ndarray,
        curr_des: np.ndarray
    ) -> Tuple[float, list]:
        """
        두 디스크립터 간의 ORB 유사도를 계산합니다.
        
        Args:
            last_des (np.ndarray): 이전 프레임의 디스크립터
            curr_des (np.ndarray): 현재 프레임의 디스크립터
            
        Returns:
            Tuple[float, list]: (유사도 점수, 좋은 매치 리스트)
        """
        if (last_des is None or curr_des is None 
            or len(last_des) == 0 or len(curr_des) == 0):
            return 0.0, []
        
        matches = self.bf.match(last_des, curr_des)
        
        if len(matches) == 0:
            return 0.0, []
        
        good_matches = [m for m in matches if m.distance < 50]
        sim_score = len(good_matches) / max(len(last_des), len(curr_des))
        
        return sim_score, good_matches
