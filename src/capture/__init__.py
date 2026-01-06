"""
Video capture and slide extraction utilities.

이 패키지는 강의 영상에서 키프레임을 추출하고 시각화하는 기능을 제공합니다.
"""

from src.capture.video_processor import VideoProcessor
from src.capture.scene_visualizer import SceneVisualizer

__all__ = ['VideoProcessor', 'SceneVisualizer']
