"""
[Intent]
비디오 프레임 내의 특징점(ORB) 지속성을 분석하여 슬라이드를 추출하는 엔진입니다.
단순히 화면의 변화(Pixel Difference)를 감지하는 대신, 영상 내의 텍스트나 도형이 
특정 위치에 얼마나 오래 머무는지를 계산하여 동적 애니메이션과 정적 슬라이드를 정밀하게 구분합니다.

[Usage]
- process_content.py에서 캡처 작업을 수행할 때 메인 유틸리티로 활용됩니다.
- 대규모 비디오 강좌나 발표 자료 등 화자가 영상을 포함하면서도 정적인 슬라이드가 교체되는 상황에 최적화되어 있습니다.

[Usage Method]
- HybridSlideExtractor를 생성할 때 비디오 경로와 설정 파라미터를 전달합니다.
- .process() 메서드를 호출하여 프레임을 순차 분석하고 슬라이드를 이미지 파일로 저장합니다.
"""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.pipeline.logger import pipeline_logger


class HybridSlideExtractor:
    """
    [Class Purpose]
    특징점(ORB) 기반의 Streak Detection 알고리즘을 사용하여 비디오에서 슬라이드를 추출합니다.
    
    [Connection]
    - OpenCV (cv2): 영상 프레임 디코딩 및 특징점 추출, 이미지 저장
    - NumPy: 특징점 격자 계산 및 지속성 맵(Persistence Map) 연산
    """

    def __init__(
        self, 
        video_path: str, 
        output_dir: str, 
        persistence_drop_ratio: float = 0.4, 
        sample_interval_sec: float = 0.5,
        persistence_threshold: int = 6,
        min_orb_features: int = 50
    ):
        """
        [Usage File] process_content.py
        [Purpose] 슬라이드 추출기를 초기화하고 내부 상태를 설정합니다.
        
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
        
        # 특징점 추출 엔진 설정 (ORB: 빠른 속도와 적절한 성능 제공)
        self.orb = cv2.ORB_create(nfeatures=1500)
        
        # 내부 상태 변수
        self.persistence_streak_map = None  # 특징점별 지속 횟수를 기록하는 히트맵
        self.pending_slide = None           # 현재 추적 중인 잠재적 슬라이드 이미지
        self.pending_features = 0           # 현재 슬라이드의 기준 특징점 수
        self.slide_count = 0                # 추출된 슬라이드 누적 개수
        self.MANIFEST = []                  # 전체 캡처 이력 (타임스탬프, 파일명 등)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_slide(self, image: np.ndarray, timestamp: float) -> None:
        """[Internal] 현재 슬라이드 프레임을 파일로 저장하고 매니페스트를 업데이트합니다."""
        self.slide_count += 1
        video_name = Path(self.video_path).stem
        filename = f"{video_name}_{self.slide_count:03d}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        self.MANIFEST.append({
            "id": self.slide_count,
            "timestamp": round(timestamp, 2),
            "image_path": filename
        })
        pipeline_logger.log("Capture", f"Saved: {video_name}_{self.slide_count:03d}")

    def process(self, video_name: str = "video") -> List[dict]:
        """
        [Usage File] process_content.py
        [Purpose] 비디오를 순차적으로 읽으며 슬라이드 전환점을 포착하고 이미지를 저장합니다.
        [Connection] FFmpeg 기반의 OpenCV VideoCapture 활용
        
        [Args]
        - video_name (str): 로깅 시 표시할 비디오 별칭
        
        [Returns]
        - List[dict]: 저장된 슬라이드들의 메타데이터 리스트 (MANIFEST)
        
        [Internal Logic]
        1. 비디오의 FPS를 기반으로 샘플링 주기(check_step)를 계산합니다.
        2. 프레임을 읽어 ORB 특징점을 추출하고 32x32 격자에 매핑합니다.
        3. 이전 프레임과 매치되는 격자의 지속(Streak) 카운트를 1씩 증가시킵니다.
        4. 지속 카운트가 persistence_threshold를 넘으면 '고정된 텍스트/도형'으로 간주합니다.
        5. 유효 특징점이 persistence_drop_ratio 이상 줄어들면 슬라이드 전환으로 판단하여 이전 것을 저장합니다.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        check_step = max(1, int(fps * self.sample_interval_sec))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 정해진 샘플링 주기에만 분석 수행
            if frame_idx % check_step == 0:
                current_time = frame_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp = self.orb.detect(gray, None)
                
                # 특징점 밀도 맵 생성 (32x32 그리드 활용)
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
                    # 특징점이 유지되는 곳은 +1, 사라진 곳은 0으로 초기화
                    self.persistence_streak_map = (self.persistence_streak_map + 1) * current_map

                # 임계값(persistence_threshold) 이상 유지된 특징점만 슬라이드 요소로 인정
                confirmed_mask = (self.persistence_streak_map >= self.persistence_threshold)
                current_text_count = np.sum(confirmed_mask)

                # --- 슬라이드 전환 감지 로직 ---
                # 1. 새로운 슬라이드 시작 감지
                if self.pending_slide is None and current_text_count > self.min_orb_features:
                    self.pending_slide = frame.copy()
                    self.pending_features = current_text_count

                # 2. 현재 슬라이드 종료 및 다음 슬라이드 전이 감지 (drop_ratio 기반)
                # 이전 고정 요소 대비 상당수(drop_ratio)가 사라졌다면 장면 전환으로 판단
                elif self.pending_slide is not None:
                    if current_text_count < (self.pending_features * (1 - self.persistence_drop_ratio)):
                        # 현재 슬라이드 저장
                        self._save_slide(self.pending_slide, current_time)
                        
                        # 새로운 잠재 슬라이드 후보 설정
                        if current_text_count > self.min_orb_features:
                            self.pending_slide = frame.copy()
                            self.pending_features = current_text_count
                        else:
                            self.pending_slide = None
                            self.pending_features = 0
                    
                    # 더 많은 텍스트가 나타나면 가장 정보가 많은 화면으로 프레임 갱신
                    elif current_text_count > self.pending_features:
                        self.pending_slide = frame.copy()
                        self.pending_features = current_text_count

            frame_idx += 1

        # 종료 시점에 남은 슬라이드가 있다면 저장
        if self.pending_slide is not None:
            self._save_slide(self.pending_slide, frame_idx / fps)

        cap.release()
        return self.MANIFEST
