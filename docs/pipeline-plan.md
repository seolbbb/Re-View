# Pipeline Async Plan

Last updated: 2026-02-05

## Goal
Current batch-serial execution을 스테이지 병렬 파이프라인으로 전환해서 VLM idle을 줄이고, 중간 결과를 프론트에 빠르게 노출한다.

## Task Checklist
- [ ] 1. 요구사항 확정 (`x` 장 기준, Fusion 증분 반영 방식, 프론트 중간결과 정책)  
Status: In Progress
- [ ] 2. 코드 점검 (`src/run_pipeline_demo.py`, `src/pipeline/stages.py`)에서 블로킹/비동기 구간 분류  
Status: Pending
- [ ] 3. 파이프라인 데이터 계약 정의 (`job_id`, `batch_id`, `timestamp`, queue payload)  
Status: Pending
- [ ] 4. 전처리 병렬화 (Audio Split + Capture + STT를 asyncio task로 분리)  
Status: Pending
- [ ] 5. VLM 배치 트리거 구현 (캡처 `x`장 누적마다 요청, `Semaphore`로 동시성 제한)  
Status: Pending
- [ ] 6. Fusion 오케스트레이션 구현 (`STT 완료` 게이트 후, `각 VLM 배치 완료 이벤트`마다 즉시 증분 병합)  
Status: Pending
- [ ] 7. Summary/Judge 스트리밍 구현 (Fusion과 독립 실행, 프론트 중간 출력 연결)  
Status: Pending
- [ ] 8. 안정화 (재시도, 백프레셔, 종료 시그널, 통합 테스트)  
Status: Pending

## Notes
- 구현 후보: `asyncio + Queue + Semaphore`
- 동기 SDK 사용 시: `asyncio.to_thread()` 또는 `run_in_executor()`로 이벤트 루프 블로킹 방지
- `STT 완료 전`에 끝난 VLM 배치는 큐에 버퍼링하고, `STT 완료 후`부터는 배치 완료 즉시 Fusion으로 전달

## 블로킹/비동기 분류
| 구간 | 현재 코드 | 성격 | 네가 결정할 것 |
|---|---|---|---|
| Audio Extract + STT | `src/run_preprocess_pipeline.py:350`, `src/run_preprocess_pipeline.py:380`, `src/audio/clova_stt.py:122` | 블로킹(파일 I/O + 외부 API) | STT를 별도 워커로 둘지, `to_thread`로 감쌀지 |
| Capture | `src/run_preprocess_pipeline.py:387`, `src/run_preprocess_pipeline.py:392` | 블로킹(CPU/디스크) | 캡처 이벤트를 몇 장 단위로 VLM 큐에 넣을지 |
| VLM 요청 | `src/pipeline/stages.py:970`, `src/vlm/vlm_engine.py:245`, `src/vlm/vlm_engine.py:451` | 블로킹(외부 API), 내부 ThreadPool 이미 있음 | 파이프라인 레벨 동시성과 VLM 내부 `concurrency`를 어떻게 배분할지 |
| Batch Sync(Fusion 입력 생성) | `src/pipeline/stages.py:1031`, `src/fusion/sync_engine.py:484` | 블로킹(CPU/파일) | Fusion worker를 1개로 직렬 처리할지 병렬 허용할지 |
| Summarizer | `src/pipeline/stages.py:1089`, `src/fusion/summarizer.py:746`, `src/fusion/gemini.py:192` | 블로킹(외부 API) | VLM과 모델 quota 공유 시 동시성 제한값 |
| Judge | `src/pipeline/stages.py:1122`, `src/judge/judge.py:395` | 블로킹(외부 API), 내부 ThreadPool 있음 | Judge `workers`와 파이프라인 동시성 충돌 방지 |
| Sleep/Poll | `src/pipeline/stages.py:915`, `src/run_process_pipeline.py:745` | 하드 대기(병목) | `asyncio.sleep` + 이벤트 기반 트리거로 바꿀지 |