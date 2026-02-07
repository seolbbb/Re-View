"""
[Intent]
영상 프레임의 크기를 조정하는 리사이징 로직을 담당하는 모듈입니다.
특히 입력 해상도에 따라 최적의 크기로 다운스케일링하는 'Adaptive Resize' 기능을 제공합니다.

[Usage]
- hybrid_extractor.py: 처리 속도와 정확도를 높이기 위해 프레임을 적절한 크기로 줄이는 데 사용됩니다.
"""

import cv2
import numpy as np

def calculate_adaptive_size(w: int, h: int) -> tuple[int, int]:
    """
    [Usage File] Internal use (resize_frame_adaptive)
    
    [Purpose]
    - 입력 해상도(w)에 따라 적절한 Target 해상도를 결정합니다.
    - 다운스케일링만 수행하며, 업스케일링은 하지 않습니다.
    
    [Tiered Logic]
    - > 1920px -> 1280px
    - > 1280px -> 1024px
    - > 960px  -> 960px
    - <= 960px -> Original (No resize)
    
    [Args]
    - w (int): 원본 너비
    - h (int): 원본 높이

    [Returns]
    - (target_w, target_h): 계산된 목표 해상도 튜플
    """
    if w > 1920:
        target_w = 1280
    elif w > 1280:
        target_w = 1024
    elif w > 960:
        target_w = 960
    else:
        return w, h  # 960 이하는 원본 유지
        
    # 비율 유지하며 높이 계산
    scale = target_w / w
    target_h = int(h * scale)
    return target_w, target_h

def resize_frame_adaptive(frame: np.ndarray, enable: bool = True) -> np.ndarray:
    """
    [Usage File] src/capture/tools/hybrid_extractor.py
    
    [Purpose]
    - 탐지(Detection)를 위한 최적 해상도의 프레임을 생성합니다.
    - enable=True일 때만 계층형 Adaptive Resize를 적용하여 다운스케일링을 수행합니다.
    
    [Args]
    - frame (np.ndarray): 원본(또는 Crop된) 프레임
    - enable (bool): 리사이징 활성화 여부 (기본값 True)

    [Returns]
    - np.ndarray: 리사이징된 프레임 (또는 원본)
    """
    if not enable:
        return frame
        
    h, w = frame.shape[:2]
    target_w, target_h = calculate_adaptive_size(w, h)
    
    if target_w != w or target_h != h:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return frame
