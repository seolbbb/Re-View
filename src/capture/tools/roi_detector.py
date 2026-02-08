"""
[Intent]
영상 프레임의 테두리(상하좌우)에 존재하는 검은색 레터박스(Letterbox) 및 필러박스(Pillarbox)를 감지하고,
실제 콘텐츠가 담긴 유효 영역(Region of Interest, ROI)의 좌표를 계산하는 도구입니다.

[Usage]
- hybrid_extractor.py: 영상 초기 프레임 분석 시 콘텐츠 영역을 확정하기 위해 사용

[Usage Method]
- RoiDetector 인스턴스 생성 시 padding, threshold 설정
- detect(frame) 메서드에 이미지를 넘겨주면 (x, y, w, h) 튜플 반환
"""

import cv2
import numpy as np
from typing import Tuple

class RoiDetector:
    """
    [Class Purpose]
    이미지의 픽셀 밝기(Intensity)를 행/열 단위로 분석하여 콘텐츠 영역과 배경(검은색 여백)을 구분합니다.
    """

    def __init__(self, padding: int = 5, threshold: float = 10.0):
        """
        [Usage File] hybrid_extractor.py

        [Purpose]
        - 감지기 초기화 및 민감도 설정

        [Args]
        - padding (int): 감지된 ROI 경계에 추가할 안전 여백 (픽셀, 기본값 5)
        - threshold (float): 콘텐츠로 간주할 최소 픽셀 밝기 평균값 (0~255, 기본값 10.0)
        """
        self.padding = padding
        self.threshold = threshold

    def detect(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        [Usage File] hybrid_extractor.py (process 메서드)
        
        [Purpose]
        - 입력 프레임을 분석하여 상하좌우 레터박스를 제외한 ROI 좌표를 계산합니다.
        
        [Connection]
        - OpenCV: BGR -> Grayscale 변환
        - NumPy: 행/열 단위 평균 연산
        
        [Args]
        - frame (np.ndarray): 분석할 비디오 프레임 (BGR 포맷)

        [Returns]
        - Tuple[int, int, int, int]: (x, y, w, h) 형태의 ROI 좌표. 
          감지에 실패하거나 유효하지 않은 경우 전체 프레임 영역을 반환합니다.
        """
        if frame is None or frame.size == 0:
            return (0, 0, 0, 0)

        height, width = frame.shape[:2]
        
        # 1. Grayscale 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 행(Row) 단위 평균 밝기 계산 (상하 감지)
        row_means = np.mean(gray, axis=1)
        
        # 3. 열(Col) 단위 평균 밝기 계산 (좌우 감지)
        col_means = np.mean(gray, axis=0)
        
        # 4. 상하 감지 (Top/Bottom)
        top = 0
        for y in range(height // 2):
            if row_means[y] > self.threshold:
                top = y
                break

        bottom = height
        for y in range(height - 1, height // 2, -1):
            if row_means[y] > self.threshold:
                bottom = y + 1
                break
                
        # 5. 좌우 감지 (Left/Right)
        left = 0
        for x in range(width // 2):
            if col_means[x] > self.threshold:
                left = x
                break
                
        right = width
        for x in range(width - 1, width // 2, -1):
            if col_means[x] > self.threshold:
                right = x + 1
                break
        
        # 6. Padding 적용
        top = max(0, top - self.padding)
        bottom = min(height, bottom + self.padding)
        left = max(0, left - self.padding)
        right = min(width, right + self.padding)
        
        # 유효성 검사 (역전 현상 방지)
        if bottom <= top or right <= left:
            return (0, 0, width, height)
            
        return (left, top, right - left, bottom - top)
