"""
================================================================================
frame_signature.py - 프레임 시그니처 계산 모듈
================================================================================

[역할]
    비디오 프레임에서 ORB(Oriented FAST and Rotated BRIEF) 특징점을 추출하여
    프레임의 구조적 시그니처를 생성합니다.

[사용 예시]
    from src.capture.tools.modules import FrameSignature
    
    fs = FrameSignature(nfeatures=500)
    gray, keypoints, descriptors = fs.compute(frame)
    
[출력]
    - gray: 그레이스케일 이미지 (640x360 리사이즈됨)
    - keypoints: ORB 키포인트 리스트
    - descriptors: ORB 디스크립터 배열
"""

import cv2
import numpy as np


class FrameSignature:
    """
    ORB 기반 프레임 시그니처 계산기.
    
    프레임을 리사이즈하고 ORB 특징점을 추출하여
    구조적 비교에 사용할 수 있는 시그니처를 생성합니다.
    
    Attributes:
        orb: OpenCV ORB 특징점 검출기
        resize_dim (tuple): 리사이즈 크기 (width, height)
    """
    
    def __init__(self, nfeatures: int = 500, resize_dim: tuple = (640, 360)):
        """
        FrameSignature 초기화.
        
        Args:
            nfeatures (int): 추출할 최대 특징점 개수 (기본 500)
            resize_dim (tuple): 리사이즈 크기 (width, height)
        """
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.resize_dim = resize_dim
    
    def compute(self, frame: np.ndarray) -> tuple:
        """
        프레임에서 ORB 시그니처를 계산합니다.
        
        Args:
            frame (np.ndarray): BGR 컬러 프레임 (OpenCV 형식)
            
        Returns:
            tuple: (gray, keypoints, descriptors)
                - gray: 그레이스케일 이미지
                - keypoints: ORB 키포인트 리스트
                - descriptors: ORB 디스크립터 배열 (None일 수 있음)
        """
        try:
            small = cv2.resize(frame, self.resize_dim)
        except cv2.error:
            return None, None, None
            
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return gray, keypoints, descriptors
    
    def compute_gray_only(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임에서 그레이스케일 이미지만 계산합니다.
        
        Args:
            frame (np.ndarray): BGR 컬러 프레임
            
        Returns:
            np.ndarray: 그레이스케일 이미지 (640x360)
        """
        try:
            small = cv2.resize(frame, self.resize_dim)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return None
