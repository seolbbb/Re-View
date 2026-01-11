"""
================================================================================
pixel_diff.py - 픽셀 차이 분석 모듈
================================================================================

[역할]
    연속 프레임 간의 픽셀 차이를 분석하여 장면 전환을 감지합니다.
    모폴로지 연산으로 노이즈를 제거하고 의미 있는 변화만 측정합니다.

[사용 예시]
    from src.capture.tools.modules import PixelDiffAnalyzer
    
    analyzer = PixelDiffAnalyzer(threshold=30, kernel_size=5)
    diff_score = analyzer.compute(prev_gray, curr_gray)
    is_change = analyzer.is_significant_change(diff_score, sensitivity=3.0)
    
[출력]
    - diff_score: 평균 픽셀 차이 값 (0.0 ~ 255.0)
"""

import cv2
import numpy as np


class PixelDiffAnalyzer:
    """
    픽셀 차이 기반 장면 변화 감지기.
    
    두 그레이스케일 프레임 간의 차이를 계산하고,
    모폴로지 연산으로 노이즈를 제거한 후 평균 diff 값을 반환합니다.
    
    Attributes:
        threshold (int): 이진화 임계값
        kernel (np.ndarray): 모폴로지 연산용 커널
    """
    
    def __init__(self, threshold: int = 30, kernel_size: int = 5):
        """
        PixelDiffAnalyzer 초기화.
        
        Args:
            threshold (int): 이진화 임계값 (기본 30)
            kernel_size (int): 모폴로지 커널 크기 (기본 5x5)
        """
        self.threshold = threshold
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        두 프레임 간 픽셀 차이를 계산합니다.
        
        Args:
            prev_gray (np.ndarray): 이전 프레임 (그레이스케일)
            curr_gray (np.ndarray): 현재 프레임 (그레이스케일)
            
        Returns:
            float: 평균 픽셀 차이 값 (0.0 ~ 255.0)
        """
        if prev_gray is None or curr_gray is None:
            return 0.0
            
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        
        return np.mean(clean_diff)
    
    def is_significant_change(self, diff_score: float, sensitivity: float = 3.0) -> bool:
        """
        픽셀 차이가 유의미한 변화인지 판단합니다.
        
        Args:
            diff_score (float): compute()의 반환값
            sensitivity (float): 민감도 임계값 (낮을수록 민감)
            
        Returns:
            bool: 유의미한 변화 여부
        """
        return diff_score > sensitivity
    
    def compute_raw_diff(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        노이즈 제거 없이 순수 평균 픽셀 차이를 계산합니다.
        
        중복 감지 등 간단한 비교에 사용합니다.
        
        Args:
            prev_gray (np.ndarray): 이전 프레임 (그레이스케일)
            curr_gray (np.ndarray): 현재 프레임 (그레이스케일)
            
        Returns:
            float: 평균 픽셀 차이 값
        """
        if prev_gray is None or curr_gray is None:
            return 0.0
            
        diff = cv2.absdiff(prev_gray, curr_gray)
        return np.mean(diff)
