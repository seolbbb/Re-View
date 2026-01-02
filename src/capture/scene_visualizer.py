"""
Scene Change Analysis 시각화 모듈

비디오 분석 결과를 시각적 그래프로 표현합니다.
- Diff Score: 연속 프레임 간 픽셀 차이 (1차 정제 기준)
- Threshold 라인: 장면 전환 감지 임계값
- Captured Slides: 최종 저장된 슬라이드 위치 (2차 정제 후)

Note: 2차 정제(dedupe_threshold)는 '저장된 이미지 간' 비교로,
      프레임 간 diff_score와는 다른 측정값입니다.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


class SceneVisualizer:
    """Scene Change Analysis 그래프 생성 클래스"""
    
    def __init__(self):
        # 기본 스타일 설정
        self.style_config = {
            'figure_size': (16, 6),
            'dpi': 150,
            'bg_color': '#0d1b2a',
            'grid_color': '#1b3a4b',
            'diff_line_color': '#40e0d0',       # 터콰이즈 - Diff Score
            'threshold_color': '#ff6b6b',        # 코랄 레드 - 1차 정제 임계값
            'saved_marker_color': '#32cd32',     # 라임 그린 - 저장된 슬라이드
            'text_color': '#ffffff',
            'font_size': 10
        }
    
    def create_scene_change_graph(
        self,
        diff_scores: list,
        keyframes_metadata: list,
        threshold: float,
        fps: float,
        video_name: str,
        output_path: str,
        dedupe_threshold: float = 10.0,
        skipped_metadata: list = None
    ) -> str:
        """
        Scene Change Analysis 그래프를 생성하고 PNG 파일로 저장합니다.
        
        Args:
            diff_scores: 각 프레임의 diff score 리스트 [(frame_idx, diff_score), ...]
            keyframes_metadata: 최종 저장된 키프레임 메타데이터 리스트
            threshold: 1차 정제 임계값 (장면 전환 감지)
            fps: 비디오 FPS
            video_name: 비디오 이름 (그래프 제목용)
            output_path: 출력 PNG 파일 경로
            dedupe_threshold: 2차 정제 임계값 (범례 표시용)
            skipped_metadata: 스킵된 프레임 메타데이터 리스트 (선택)
            
        Returns:
            저장된 파일 경로
        """
        config = self.style_config
        
        # Figure 생성 및 스타일 설정
        fig, ax = plt.subplots(figsize=config['figure_size'], dpi=config['dpi'])
        fig.patch.set_facecolor(config['bg_color'])
        ax.set_facecolor(config['bg_color'])
        
        # 데이터 준비
        if not diff_scores:
            print("⚠ diff_scores가 비어있습니다. 그래프를 생성할 수 없습니다.")
            return None
            
        # diff_scores를 시간(초) 기준으로 변환
        times = [frame_idx / fps for frame_idx, _ in diff_scores]
        scores = [score for _, score in diff_scores]
        
        # Diff Score 라인 그리기
        ax.plot(times, scores, 
                color=config['diff_line_color'], 
                linewidth=0.8, 
                alpha=0.9,
                label='Frame Diff Score')
        
        # 1차 정제 Threshold 수평선 (장면 전환 감지)
        ax.axhline(y=threshold, 
                   color=config['threshold_color'], 
                   linestyle='--', 
                   linewidth=1.5, 
                   alpha=0.8,
                   label=f'Scene Threshold ({threshold})')
        
        # Y축 최대값 계산 (전체 diff score 범위 표시)
        y_max = max(scores) * 1.1 if scores else 15
        
        # 최종 저장된 슬라이드 마커 (▼ 초록색)
        for meta in keyframes_metadata:
            capture_time = meta['timestamp_ms'] / 1000.0
            diff_score = meta.get('diff_score', 0)
            
            # 마커를 해당 시점의 diff_score 위치에 표시
            marker_y = y_max * 0.95
            ax.scatter(capture_time, marker_y, 
                      marker='v',
                      s=100, 
                      color=config['saved_marker_color'],
                      edgecolors='white',
                      linewidths=0.5,
                      zorder=5)
            
            # diff_score 위치에 수직선 추가
            ax.axvline(x=capture_time, 
                      color=config['saved_marker_color'], 
                      linestyle=':', 
                      linewidth=0.5, 
                      alpha=0.5)
        
        # 축 스타일 설정
        ax.set_xlabel('Video Time (seconds)', 
                      color=config['text_color'], 
                      fontsize=config['font_size'])
        ax.set_ylabel('Diff Score (pixel change)', 
                      color=config['text_color'], 
                      fontsize=config['font_size'])
        ax.set_title(f'Scene Change Analysis: {video_name}', 
                     color=config['text_color'], 
                     fontsize=config['font_size'] + 2,
                     fontweight='bold')
        
        # 그리드 설정
        ax.grid(True, color=config['grid_color'], alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 축 색상 설정
        ax.tick_params(colors=config['text_color'], labelsize=config['font_size'] - 1)
        for spine in ax.spines.values():
            spine.set_color(config['grid_color'])
        
        # Y축 범위 설정
        ax.set_ylim(0, y_max)
        
        # X축 범위 설정
        if times:
            ax.set_xlim(0, max(times) * 1.02)
        
        # 범례 생성
        diff_patch = mpatches.Patch(color=config['diff_line_color'], label='Frame Diff Score')
        threshold_patch = mpatches.Patch(color=config['threshold_color'], 
                                         label=f'Scene Threshold (>={threshold})')
        saved_patch = plt.Line2D([0], [0], marker='v', color='w', 
                                  markerfacecolor=config['saved_marker_color'],
                                  markersize=8, 
                                  label=f'Saved ({len(keyframes_metadata)}, dedupe>={dedupe_threshold})', 
                                  linestyle='None')
        
        legend = ax.legend(handles=[diff_patch, threshold_patch, saved_patch],
                          loc='upper right',
                          facecolor=config['bg_color'],
                          edgecolor=config['grid_color'],
                          fontsize=config['font_size'] - 1)
        for text in legend.get_texts():
            text.set_color(config['text_color'])
        
        # 레이아웃 조정 및 저장
        plt.tight_layout()
        
        # 출력 디렉토리 확인
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(output_path, 
                    facecolor=config['bg_color'], 
                    edgecolor='none',
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.close(fig)
        
        return output_path
