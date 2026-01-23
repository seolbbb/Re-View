import cv2
import numpy as np
import os

class HybridSlideExtractor:
    def __init__(self, video_path, output_dir, persistence_drop_ratio=0.4):
        """
        Args:
            video_path (str): 비디오 파일 경로
            output_dir (str): 슬라이드 저장 경로
            persistence_drop_ratio (float): 글자수 감소 민감도 (0.4 = 40% 감소해야 분리)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        
        # 1. 설정값 (동영상 포함 슬라이드 대응)
        self.persistence_drop_ratio = persistence_drop_ratio
        self.persistence_time_threshold = 6  # 0.5초당 1회 샘플링 시 2초 생존 시 인정
        
        # 2. 상태 관리 변수
        self.persistence_streak_map = np.zeros((360, 640), dtype=np.int16)
        self.persistence_max_text_count = 0
        self.pending_slide = None
        
        # 3. 도구 설정
        self.orb = cv2.ORB_create(nfeatures=2000)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, video_name="video"):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0
        
        check_step = max(1, int(fps * 0.5))
        frame_idx = 0
        slide_idx = 0
        extracted_slides = []

        #print(f"🚀 분석 시작: {video_name} (총 {total_frames} 프레임)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            if frame_idx % check_step == 0:
                current_ms = int((frame_idx / fps) * 1000)
                
                small = cv2.resize(frame, (640, 360))
                curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                
                # 1. 특징점 추출
                kp = self.orb.detect(curr_gray, None)
                curr_orb_map = np.zeros((360, 640), dtype=bool)
                if kp:
                    for k in kp:
                        x, y = map(int, k.pt)
                        if x < 640 and y < 360:
                            curr_orb_map[y, x] = True
                
                # 2. 생존 업데이트
                self.persistence_streak_map[curr_orb_map] += 1
                self.persistence_streak_map[~curr_orb_map] = 0
                
                # 3. 2초 이상 고정된 특징점만 카운트
                confirmed_mask = (self.persistence_streak_map >= self.persistence_time_threshold)
                current_text_count = np.sum(confirmed_mask)
                
                # 4. 슬라이드 판단
                if self.pending_slide is not None:
                    # 동영상 재생 등으로 인한 일시적 감소를 견디기 위해 drop_limit 계산
                    drop_limit = self.persistence_max_text_count * (1 - self.persistence_drop_ratio)
                    
                    if current_text_count < drop_limit:
                        # 정말로 슬라이드가 끝났다고 판단될 때 저장
                        slide_idx += 1
                        self._save_slide(video_name, slide_idx, self.pending_slide, current_ms, extracted_slides)
                        self.pending_slide = None
                        self.persistence_max_text_count = 0
                    else:
                        # 특징점이 더 많이 나타나는 순간(동영상이 멈춘 순간 등)의 프레임으로 계속 교체
                        if current_text_count >= self.persistence_max_text_count:
                            self.pending_slide['frame'] = frame.copy()
                            self.persistence_max_text_count = current_text_count
                
                # 5. 새 슬라이드 포착 (최소 50개 이상 특징점)
                if self.pending_slide is None and current_text_count > 50:
                    self.pending_slide = {'frame': frame.copy(), 'start_ms': current_ms}
                    self.persistence_max_text_count = current_text_count

        # 마지막 슬라이드
        if self.pending_slide is not None:
            slide_idx += 1
            self._save_slide(video_name, slide_idx, self.pending_slide, duration_ms, extracted_slides)

        cap.release()
        #print(f"✅ 분석 완료: 총 {len(extracted_slides)}개 추출됨.")
        return extracted_slides

    def _save_slide(self, video_name, idx, slide_data, end_ms, extracted_slides):
        start_ms = slide_data['start_ms']
        filename = f"{video_name}_{idx:03d}_{start_ms}_{end_ms}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, slide_data['frame'])
        
        meta = {"file_name": filename, "start_ms": start_ms, "end_ms": end_ms}
        extracted_slides.append(meta)
        #print(f"💾 Saved: {filename} (Text points: {self.persistence_max_text_count})")
