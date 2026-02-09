# 파이프라인 아키텍처

## 개요

Re:View 파이프라인은 **전처리(Preprocess)**와 **처리(Process)** 두 단계로 구성되며, 비동기(Async) 모드와 순차(Sequential) 모드를 지원합니다.

---

## 실행 모드

| 모드 | 진입점 | 동작 |
|---|---|---|
| **Async (기본)** | `run_pipeline_demo_async.py` | Producer → VLM Worker → Fusion Worker를 `asyncio.gather()`로 동시 실행 |
| **Sequential** | `run_pipeline_demo.py` | Preprocess 완료 후 Process 순차 실행 |
| **API** | `process_api.py` | 프론트엔드에서 업로드 → BackgroundTasks로 파이프라인 실행 |

### Async 모드 흐름

```
Video Upload
    │
    ├─ Producer (asyncio task)
    │   ├─ Audio Extract → STT (Clova / Whisper)
    │   └─ Slide Capture (ORB + pHash) → 캡처 이벤트 발행
    │
    ├─ VLM Worker (asyncio task)
    │   └─ 캡처 이벤트 수신 → Qwen VLM 분석 → Fusion 큐 전달
    │
    └─ Fusion Worker (asyncio task)
        └─ STT 완료 대기 → VLM 결과 수신 → Sync → Summarize → Judge
            → 증분 결과를 프론트엔드로 SSE 스트리밍
```

### Sequential 모드 흐름

```
Video
  ├─ [Preprocess] ─── STT + Capture (ThreadPoolExecutor 병렬)
  └─ [Process] ────── VLM → Sync → Summarize → Judge (배치 순차)
```

---

## 전처리 파이프라인

**진입점**: `src/run_preprocess_pipeline.py`

1. **초기화**: `preprocessing_jobs` 테이블에 레코드 생성 (`RUNNING`)
2. **병렬 처리**: `ThreadPoolExecutor`로 STT + Capture 동시 실행
   - **STT**: 오디오 추출 → 음성→텍스트 변환 → `stt_results` 테이블 저장
   - **Capture**: 슬라이드 이미지 추출 → Storage 업로드 → `captures` 테이블 저장
3. **상태 업데이트**:

| 조건 | Job 상태 | Video 상태 |
|---|---|---|
| 모든 단계 성공 | `DONE` | `PREPROCESS_DONE` |
| 실패/부분 에러 | `FAILED` | `FAILED` |

---

## 처리 파이프라인

**진입점**: `src/run_process_pipeline.py`

### 입력 해석
- **로컬 우선**: `stt.json`, `captures/` 로컬 파일 사용
- **DB 폴백**: 로컬 부재 시 Supabase에서 자동 다운로드

### 배치 처리
캡처 이미지를 `batch_size`(기본 10) 단위로 분할하여 순차 처리:

```
각 배치:
  VLM 분석 → Sync Engine → Summarizer → Judge (& Retry)
  ↓
  결과를 segment_summaries.jsonl에 누적 저장
  ↓
  다음 배치 (5초 rate limit 대기)
```

- **Judge Retry**: 품질 평가 실패 시 피드백 반영 후 최대 2회 재시도
- **Progress**: `processing_jobs` 테이블에 `current_batch/total_batch` 기록

### 결과 저장

| 테이블 | 저장 내용 |
|---|---|
| `vlm_results` | VLM 분석 결과 |
| `fs_segments` | 분석 단위 (STT + VLM 매칭) |
| `fs_summaries` | 최종 요약 + Judge 평가 |
| `processing_jobs` | 실행 이력 및 통계 |

---

## API 엔드포인트

**서버**: `uvicorn src.process_api:app --host 0.0.0.0 --port 8080`

### 파이프라인/상태

| Method | Endpoint | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| POST | `/process` | 처리 파이프라인 실행 요청 |
| POST | `/stt/process` | Storage 기반 STT 실행 |
| GET | `/videos/{id}/status` | 비디오 + Job 통합 상태 조회 |
| GET | `/videos/{id}/status/stream` | SSE 실시간 상태 + 요약 스트리밍 |
| GET | `/videos/{id}/progress` | 처리 진행률 조회 (Polling) |
| GET | `/videos/{id}/summary` | 최신 요약 결과 조회 |
| GET | `/videos/{id}/summaries` | 세그먼트별 요약 리스트 |
| GET | `/videos/{id}/evidence` | 근거 데이터 조회 (챗봇용) |
| GET | `/runs/{video_name}` | 마지막 파이프라인 실행 메타데이터 |

### 비디오 관리

| Method | Endpoint | 설명 |
|---|---|---|
| POST | `/api/videos/upload` | 단일 파일 업로드 |
| POST | `/api/videos/upload/init` | 멀티파트 업로드 초기화 |
| POST | `/api/videos/upload/complete` | 멀티파트 업로드 완료 |
| GET | `/api/videos` | 사용자 비디오 목록 |
| DELETE | `/api/videos/{id}` | 비디오 삭제 (soft/hard) |
| GET | `/api/videos/{id}/stream` | 비디오 미디어 스트리밍 |
| GET | `/api/videos/{id}/thumbnail` | 썸네일 조회 |
| POST | `/api/media/ticket` | 미디어 스트리밍 티켓 발급 |

### 챗봇

| Method | Endpoint | 설명 |
|---|---|---|
| POST | `/api/chat` | 챗봇 응답 (일반) |
| POST | `/api/chat/stream` | 챗봇 응답 (SSE 스트리밍) |

---

## SSE 이벤트 타입

`/videos/{id}/status/stream` 엔드포인트에서 전송되는 이벤트:

| 필드 | 설명 |
|---|---|
| `video_status` | 비디오 전체 상태 |
| `preprocess_job` | 전처리 작업 상태 |
| `processing_job` | 처리 작업 상태 (진행률 포함) |

---

## DB 테이블 요약

| 테이블 | 역할 | 기록 시점 |
|---|---|---|
| `videos` | 비디오 메타데이터 + 상태 | 업로드 시 생성, 각 단계 완료 시 상태 업데이트 |
| `preprocessing_jobs` | 전처리 실행 이력 | `RUNNING` → `DONE` / `FAILED` |
| `processing_jobs` | 처리 실행 이력 + 진행률 | `VLM_RUNNING` → `DONE` / `FAILED` |
| `stt_results` | STT 세그먼트 데이터 | 전처리 완료 시 |
| `captures` | 캡처 이미지 메타데이터 | 전처리 완료 시 |
| `vlm_results` | VLM 분석 결과 | 처리 완료 시 |
| `fs_segments` | STT+VLM 매칭 분석 단위 | 처리 완료 시 |
| `fs_summaries` | 최종 요약 + 평가 | 처리 완료 시 |
