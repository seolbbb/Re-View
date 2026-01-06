# Grid Search: 임계값 최적화 결과

1차 정제(장면 전환 감지)와 2차 정제(중복 제거)의 임계값 조합별 성능을 분석하였습니다.

## 최적 조합 요약

| 저장 개수 | SCENE_THRESHOLD | DEDUPE_THRESHOLD | 실행 시간 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **7개 (추천)** | **6** | **3** | **17.30s** | 빠른 속도와 정확한 전환 탐지 |
| 6개 | 6 | 5 | 17.39s | |
| 8개 | 4 | 3 | 20.54s | |
| 9개 | 3 | 3 | 24.92s | |

## 전체 테스트 결과

| SCENE | DEDUPE | TIME(s) | DETECT | SKIP | FINAL | FOLDER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | 3 | 24.92 | 20 | 12 | 9 | scene3_dedupe3/ |
| 3 | 5 | 24.24 | 20 | 14 | 7 | scene3_dedupe5/ |
| **6** | **3** | **17.30** | **9** | **3** | **7** | **scene6_dedupe3/** |
| 6 | 10 | 19.08 | 9 | 6 | 4 | scene6_dedupe10/ |

> [!NOTE]
>
> - **DETECT**: 1차 정제에서 감지된 장면 전환 횟수
> - **SKIP**: 2차 정제에서 이미지 중복으로 스킵된 횟수
> - **FINAL**: 최종 저장된 슬라이드 이미지 수

## 저장된 결과물 확인

결과는 `src/data/output/grid_search/` 폴더 하위에 각 조합별로 저장되어 있습니다.

- `scene{X}_dedupe{Y}/`: 캡처된 이미지 및 개별 메타데이터
- `grid_search_results.json`: 전체 테스트 결과 요약

![Grid Search Console Output](file:///c:/Users/irubw/geminiProject/Screentime-MVP/src/data/output/grid_search_results_summary.png)
*(참고: 콘솔에 출력된 요약 테이블입니다)*
