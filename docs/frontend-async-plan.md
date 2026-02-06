# 프론트 Async 전환 계획

마지막 업데이트: 2026-02-06

## 목표
- 프론트 코드는 유지한 채, 백엔드 업로드 파이프라인을 async 엔진으로 전환
- 첫 요약(summarizer) 결과가 나오는 시점에만 분석 페이지로 이동
- STT 게이트 유지 (STT 완료 이후 요약 시작)

## 계획

1. **프론트 이동 트리거 (SSE summaries)**
- `PREPROCESS_DONE` / `PROCESSING`에서의 즉시 이동 제거
- SSE `summaries.items.length > 0`일 때 `/analysis/:id`로 이동
- `DONE`은 안전장치로 유지
- 파일: `frontend/src/pages/LoadingPage.jsx`

2. **Async 엔진 DB 업로드**
- `FusionWorker`가 배치 결과(요약/평가/세그먼트)를 DB에 업로드하도록 허용
- 처리 중 SSE `summaries`가 배치 단위로 누적되도록 지원
- 파일:
  - `src/pipeline/fusion_worker_async.py`
  - `src/run_pipeline_demo_async.py`

3. **API 업로드 경로 -> async 엔진**
- 업로드 완료 후 async 파이프라인으로 실행
- 엔진 선택 우선순위:
  - `PIPELINE_ENGINE` 환경변수
  - `config/pipeline/settings.yaml` (`api_pipeline_engine`)
  - 기본값: `async`
- 파일: `src/process_api.py`

4. **Status/SSE 호환성**
- `video_status`, `processing_job` 스키마 유지
- SSE 이벤트(`status`, `summaries`, `done`) 포맷 유지
- 프론트 변경은 1번만 적용

5. **검증 체크리스트**
- 업로드 후 Loading에서 대기
- 첫 요약 생성 시 분석 페이지 이동
- 요약이 배치 단위로 계속 누적 표시
- 요약이 늦어도 `DONE`이면 정상 이동

## 참고
- SSE summaries 동작을 위해 DB 연결이 필요함
- Supabase Storage에 `videos` 버킷이 없으면 signed upload URL 생성에서 실패함
