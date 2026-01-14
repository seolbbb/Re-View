# Capture 모듈 안내

이 문서는 `src/capture` 하위 파일들의 역할과 동작을 요약합니다.

## process_content.py

**역할**
- 강의 영상에서 슬라이드 전환을 감지하고 캡처 이미지를 저장하는 메인 파이프라인입니다.
- 내부적으로 `HybridSlideExtractor`를 사용하며 단일 패스로 캡처를 수행합니다.

**V2 지연 저장 방식**
- 슬라이드를 즉시 저장하지 않고 다음 슬라이드 감지 시점까지 버퍼링합니다.
- 다음 슬라이드의 시작 시점을 이전 슬라이드의 종료 시점으로 확정합니다.
- 확정된 시점에 최종 파일명으로 한 번만 저장합니다.
- 후처리 파일명 변경 루프가 없어져 입출력 오버헤드가 줄어듭니다.

**입출력**
- 입력: MP4 비디오 파일
- 출력:
  - `data/output/{video}/captures/*.jpg`
  - `data/output/{video}/manifest.json`

**파이프라인 흐름(요약)**
1. 프레임 읽기 + 유휴 구간 샘플링
2. 장면 전환 감지(픽셀 차이 + ORB 유사도)
3. 안정화 대기 후 버퍼링
4. 버퍼에서 최적 프레임 선택
5. 지연 저장으로 end_ms 확정 후 저장
6. 중복 슬라이드 제거

**임계값**
- `SENSITIVITY_DIFF`: 픽셀 차이 민감도(낮을수록 민감, 권장 2.0~5.0)
- `SENSITIVITY_SIM`: ORB 유사도 임계값(높을수록 엄격, 권장 0.7~0.9)
- `MIN_INTERVAL`: 최소 캡처 간격(초)

**연동**
- 단독 실행: `python src/capture/process_content.py`
- 파이프라인 호출: `run_video_pipeline.py` → `process_single_video_capture`

## tools/hybrid_extractor.py

**역할**
- 픽셀 차이와 ORB 유사도를 결합해 슬라이드 전환을 감지하는 캡처 엔진입니다.
- 전환 이후 일정 시간 버퍼를 모아 노이즈가 적은 프레임을 선택합니다.
- 지연 저장 방식으로 end_ms가 확정된 후 한 번만 저장합니다.

**핵심 알고리즘**
1. 픽셀 차이 분석: 연속 프레임 변화량으로 전환 감지
2. ORB 구조 유사도: 슬라이드의 구조적 변화 감지
3. 스마트 버퍼링(약 2.5초): 중앙값 기반 노이즈 제거 후 최적 프레임 선택
4. RANSAC 기반 중복 제거: 기하학적 일관성으로 중복 판별

**출력 형식**
- 파일명: `{video}_{idx}_{start_ms}_{end_ms}.jpg`
- 로그: `capture_log.txt`

**임계값**
- `sensitivity_diff`(기본 3.0): 픽셀 차이 민감도
- `sensitivity_sim`(기본 0.8): ORB 유사도 임계값
- `min_interval`(기본 0.5초): 연속 캡처 최소 간격

**사용 예시**
```python
from src.capture.tools import HybridSlideExtractor

extractor = HybridSlideExtractor(
    video_path="input.mp4",
    output_dir="captures/",
    sensitivity_diff=3.0,
    sensitivity_sim=0.8,
    min_interval=0.5,
)
slides = extractor.process(video_name="lecture")
```
