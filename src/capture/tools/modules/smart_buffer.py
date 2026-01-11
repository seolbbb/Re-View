"""
================================================================================
smart_buffer.py - 스마트 버퍼링 모듈 (노이즈 제거)
================================================================================

[역할]
    장면 전환 후 일정 시간(기본 2.5초) 동안 프레임을 수집하고,
    Median 기반 분석으로 마우스 포인터, 사람 등의 노이즈가 가장 적은
    깨끗한 프레임을 선택합니다.

[원리]
    1. 버퍼에 연속 프레임들을 저장
    2. 모든 프레임의 Median(중앙값) 이미지 계산
       - Median은 일시적인 노이즈(마우스, 사람)가 제거된 "기준" 이미지
    3. 각 프레임과 Median의 차이를 계산
    4. Median에 가장 가까운 프레임 = 노이즈가 가장 적은 프레임

[사용 예시]
    from src.capture.tools.modules import SmartBuffer
    
    buffer = SmartBuffer(duration=2.5, fps=30)
    
    # 프레임 추가
    buffer.add_frame(frame, current_time)
    
    # 버퍼가 채워졌는지 확인
    if buffer.is_ready():
        best_frame = buffer.select_best_frame()
        buffer.reset()

[출력]
    - best_frame: 노이즈가 가장 적은 프레임 (np.ndarray)
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class SmartBuffer:
    """
    스마트 버퍼링을 통한 노이즈 제거 모듈.
    
    연속된 프레임을 버퍼에 저장하고, Median 기반 분석으로
    마우스/사람 등의 노이즈가 가장 적은 프레임을 선택합니다.
    
    Attributes:
        duration (float): 버퍼링 지속 시간 (초)
        buffer (List[np.ndarray]): 수집된 프레임 리스트
        start_time (float): 버퍼링 시작 시간
        trigger_time (float): 캡처 트리거 시간
    """
    
    def __init__(self, duration: float = 2.5):
        """
        SmartBuffer 초기화.
        
        Args:
            duration (float): 버퍼링 지속 시간 (초, 기본 2.5초)
                - 짧은 시간(1.0초): 빠른 처리, 노이즈 제거 효과 감소
                - 긴 시간(3.0초): 더 나은 노이즈 제거, 처리 시간 증가
        """
        self.duration = duration
        self.buffer: List[np.ndarray] = []
        self.start_time: float = 0.0
        self.trigger_time: float = 0.0
    
    def start(self, trigger_time: float, initial_frame: np.ndarray) -> None:
        """
        새로운 버퍼링 세션을 시작합니다.
        
        Args:
            trigger_time (float): 캡처가 트리거된 시간 (초)
            initial_frame (np.ndarray): 첫 번째 프레임
        """
        self.buffer = [initial_frame.copy()]
        self.start_time = trigger_time
        self.trigger_time = trigger_time
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        버퍼에 프레임을 추가합니다.
        
        Args:
            frame (np.ndarray): 추가할 프레임 (BGR 컬러)
        """
        self.buffer.append(frame.copy())
    
    def is_ready(self, current_time: float) -> bool:
        """
        버퍼링이 완료되었는지 확인합니다.
        
        Args:
            current_time (float): 현재 시간 (초)
            
        Returns:
            bool: duration 시간이 경과했으면 True
        """
        return (current_time - self.start_time) >= self.duration
    
    def is_active(self) -> bool:
        """
        버퍼링이 진행 중인지 확인합니다.
        
        Returns:
            bool: 버퍼에 프레임이 있으면 True
        """
        return len(self.buffer) > 0
    
    def select_best_frame(self, resize_dim: Tuple[int, int] = (640, 360)) -> Optional[np.ndarray]:
        """
        버퍼에서 노이즈가 가장 적은 최적 프레임을 선택합니다.
        
        [알고리즘]
        1. 모든 프레임의 픽셀별 Median 계산 → "기준 이미지" 생성
        2. 각 프레임과 기준 이미지의 차이 계산
        3. 차이가 가장 작은 프레임 = 노이즈가 가장 적은 프레임
        
        Args:
            resize_dim (Tuple[int, int]): 비교용 리사이즈 크기 (width, height)
            
        Returns:
            np.ndarray: 최적 프레임 (원본 해상도)
            None: 버퍼가 비어있는 경우
        """
        if not self.buffer:
            return None
        
        # 1. 모든 프레임을 스택으로 변환
        #    shape: (num_frames, height, width, channels)
        stack = np.array(self.buffer)
        
        # 2. 픽셀별 Median 계산 (axis=0: 프레임 축)
        #    일시적인 노이즈(마우스, 사람)는 Median에서 제거됨
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        
        # 3. Median을 그레이스케일로 변환 (비교용)
        median_gray = cv2.cvtColor(
            cv2.resize(median_frame, resize_dim),
            cv2.COLOR_BGR2GRAY
        )
        
        # 4. 각 프레임과 Median의 차이 계산
        best_frame = self.buffer[0]
        best_diff = float('inf')
        
        for frame in self.buffer:
            # 리사이즈 및 그레이스케일 변환
            frame_gray = cv2.cvtColor(
                cv2.resize(frame, resize_dim),
                cv2.COLOR_BGR2GRAY
            )
            
            # 픽셀 차이의 평균 계산
            diff = np.mean(cv2.absdiff(median_gray, frame_gray))
            
            # 가장 작은 차이를 가진 프레임 선택
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        return best_frame
    
    def get_best_frame_info(self, resize_dim: Tuple[int, int] = (640, 360)) -> Tuple[Optional[np.ndarray], float]:
        """
        최적 프레임과 Median과의 차이값을 함께 반환합니다.
        
        Args:
            resize_dim (Tuple[int, int]): 비교용 리사이즈 크기
            
        Returns:
            Tuple[np.ndarray, float]: (최적 프레임, Median과의 차이)
        """
        if not self.buffer:
            return None, 0.0
        
        stack = np.array(self.buffer)
        median_frame = np.median(stack, axis=0).astype(np.uint8)
        median_gray = cv2.cvtColor(cv2.resize(median_frame, resize_dim), cv2.COLOR_BGR2GRAY)
        
        best_frame = self.buffer[0]
        best_diff = float('inf')
        
        for frame in self.buffer:
            frame_gray = cv2.cvtColor(cv2.resize(frame, resize_dim), cv2.COLOR_BGR2GRAY)
            diff = np.mean(cv2.absdiff(median_gray, frame_gray))
            if diff < best_diff:
                best_diff = diff
                best_frame = frame
        
        return best_frame, best_diff
    
    def reset(self) -> None:
        """
        버퍼를 초기화합니다.
        """
        self.buffer = []
        self.start_time = 0.0
        self.trigger_time = 0.0
    
    def get_trigger_time(self) -> float:
        """
        캡처 트리거 시간을 반환합니다.
        
        Returns:
            float: 트리거 시간 (초)
        """
        return self.trigger_time
    
    def get_frame_count(self) -> int:
        """
        버퍼에 저장된 프레임 수를 반환합니다.
        
        Returns:
            int: 프레임 수
        """
        return len(self.buffer)
