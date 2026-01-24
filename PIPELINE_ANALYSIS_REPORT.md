# ReViewFeature 파이프라인 상세 분석 보고서

이 문서는 `run_preprocess_pipeline.py` (전처리), `run_process_pipeline.py` (메인 처리), 그리고 `process_api.py` (API 서버)의 실행 흐름, 데이터 처리, 배치 로직, 그리고 **상태(Status) 변경 처리 로직**을 상세하게 분석하여 기술합니다.

---

## 1. 전처리 파이프라인 (`run_preprocess_pipeline.py`)

### 1.1 개요
- **목적**: 원본 비디오 파일에서 오디오(STT)와 시각 정보(Capture)를 추출하여 `preprocessing_jobs`, `stt_results`, `captures` 테이블에 저장합니다.
- **특징**: 병렬 처리 지원, `preprocessing_jobs` 테이블을 통한 작업 상태 관리.

### 1.2 실행 및 데이터 처리 흐름

1.  **초기화 (Initialization)**
    *   `pipeline_run.json` 생성 (Status: `running`).
    *   **DB 연결**: `preprocessing_jobs` 테이블에 새로운 실행 레코드 생성 (`RUNNING`).
    *   **Video 레코드 확보**: 파일명을 기준으로 `videos` 테이블을 조회하거나 새로 생성합니다.

2.  **병렬 처리 (Parallel Execution)**
    *   `ThreadPoolExecutor`를 통해 **STT**와 **Capture** 작업을 동시에 수행합니다.
    *   **[STT 단계]**:
        *   음성을 텍스트로 변환 (`stt.json` 생성).
        *   완료 즉시 `stt_results` 테이블에 세그먼트 데이터 업로드.
    *   **[Capture 단계]**:
        *   주요 장면 이미지 추출 및 스토리지 업로드.
        *   완료 즉시 `captures` 테이블에 파일 메타데이터(경로, 시간 등) 업로드.

3.  **상태 변경 및 데이터 동기화 (Status & Sync Logic)**
    *   전처리가 완료되면 `finalize_preprocess_db_sync` 함수를 통해 최종 상태를 결정하고 DB를 업데이트합니다.

### 1.3 상세 상태(Status) 변경 로직

전처리 파이프라인은 실행 결과 및 에러 발생 여부에 따라 `videos` 및 `preprocessing_jobs` 테이블의 **status** 컬럼을 다음과 같이 업데이트합니다.

| 조건 (Condition) | Job 상태 (preprocessing_jobs) | Video 상태 (videos) | 설명 |
| :--- | :--- | :--- | :--- |
| **모든 단계 성공** | `DONE` | `PREPROCESS_DONE` | STT 및 Capture, DB 업로드가 모두 에러 없이 완료됨. (**주의**: 코드상 `completed`가 아닌 `PREPROCESS_DONE` 사용) |
| **실패 (또는 부분 에러)** | `FAILED` | `FAILED` | 파이프라인 실행 도중 예외가 발생하거나, DB 업로드 중 하나라도 에러가 발생한 경우. |

> **구현 로직 (Code Logic):**
> ```python
> if run_status == "error" or errors:
>     # 파이프라인 실행 중 치명적 예외 발생 또는 DB 업로드 에러
>     adapter.update_preprocessing_job_status(job_id, "FAILED", error=...)
>     adapter.update_video_status(video_id, "FAILED", error=...)
> else:
>     # 모든 과정 정상 완료
>     adapter.update_preprocessing_job_status(job_id, "DONE")
>     adapter.update_video_status(video_id, "PREPROCESS_DONE")
> ```

---

## 2. 메인 처리 파이프라인 (`run_process_pipeline.py`)

### 2.1 개요
- **목적**: 전처리된 데이터를 바탕으로 VLM 분석, 문맥 병합(Fusion), 요약(Summarizer), 평가(Judge)를 수행하고 최종 리포트를 생성합니다.
- **특징**: **로컬/DB 하이브리드 입력**, 대용량 처리를 위한 **배치(Batch) 모드**, `processing_jobs` 테이블을 통한 상태 추적.

### 2.2 입력 데이터 처리 (Input Resolution)

1.  **로컬 우선 (Local First)**: `stt.json`, `captures/` 등이 로컬에 있으면 즉시 사용.
2.  **DB 폴백 (DB Fallback)**: 로컬 파일 부재 시 DB에서 자동 다운로드 (`stt_results`, `captures` 테이블 조회 -> 로컬 복원).
    *   *Storage*: 캡처 이미지가 없으면 Supabase Storage에서 다운로드.

### 2.3 배치 처리 로직 (Batch Processing)

`--batch-mode` 사용 시, 대용량 비디오 처리를 위해 데이터를 청크(Chunk)로 나누어 순차 처리합니다.

1.  **배치 분할**: 전체 캡처 이미지를 `batch_size`(기본 10) 단위로 그룹화.
2.  **순차 실행 루프**:
    *   **VLM 분석**: 해당 배치의 이미지만 VLM API로 분석.
    *   **Sync Engine**: 해당 시간대의 STT와 VLM 데이터를 매칭하여 분석 단위 생성.
    *   **Summarizer**: 매칭된 컨텍스트를 요약.
    *   **Judge & Retry**:
        *   요약 품질 평가 (Pass/Fail).
        *   **Fail 시**: 피드백을 반영하여 요약 재생성 (최대 2회 Retry).
    *   **누적 저장**: 배치 결과(`segment_summaries.jsonl`)를 즉시 파일에 추가(Append).
3.  **Rate Limiting**: 배치 사이 5초 대기하여 외부 API 과부하 방지.

### 2.4 파이프라인 상태 추적 (Pipeline Status)

메인 파이프라인은 `processing_jobs` 테이블을 통해 상세 상태를 기록합니다.

*   **실행 시작**: `processing_jobs` 생성 및 `VLM_RUNNING` 상태 기록.
*   **배치 진행**: 배치 실행 중 `status_callback`을 통해 진행률(`current_batch`/`total_batch`) 업데이트.
*   **실행 완료**: `DONE` 상태 기록.
*   **실행 실패**: `FAILED` 상태 및 에러 메시지 기록.

### 2.5 DB 동기화 및 결과 저장

모든 처리가 완료되면 `sync_processing_results_to_db` 등을 통해 결과물을 저장합니다.

*   **VLM Results**: `vlm_results` 테이블.
*   **Segments**: `fs_segments` 테이블 (분석 단위).
*   **Summaries**: `fs_summaries` 테이블 (최종 요약 및 Judge 평가).
*   **Processing Job**: `processing_jobs` 테이블 (실행 이력 및 통계).

---

## 3. API 인터페이스 (`process_api.py`)

### 3.1 개요
- **목적**: FastAPI 기반의 REST API 서버로, 외부(프론트엔드 등)에서 파이프라인을 실행하고 상태를 조회할 수 있게 합니다.
- **실행**: `uvicorn src.process_api:app --host 0.0.0.0 --port 8000`

### 3.2 주요 엔드포인트

| Method | Endpoint | 설명 |
| :--- | :--- | :--- |
| **POST** | `/process` | 메인 처리 파이프라인(`run_process_pipeline`) 비동기 실행 요청. |
| **POST** | `/stt/process` | Storage에 업로드된 오디오 파일을 기반으로 STT 실행. |
| **GET** | `/videos/{id}/status` | 비디오 상태, 전처리(`preprocessing_jobs`), 메인처리(`processing_jobs`) 상태 통합 조회. |
| **GET** | `/videos/{id}/progress` | 메인 파이프라인의 진행률(퍼센트) 조회 (Polling용). |
| **GET** | `/videos/{id}/summary` | 최신 요약 결과(JSON) 조회 (완료 여부 `is_complete` 포함). |
| **GET** | `/videos/{id}/summaries` | 채팅봇 등을 위한 세부 세그먼트별 요약 리스트 조회. |

---

## 4. 데이터베이스 및 스토리지 연동 요약

| 구분 | 테이블/버킷 | 역할 및 저장 시점 |
| :--- | :--- | :--- |
| **Video Status** | `videos.status` | **전처리 단계** 완료 시 `PREPROCESS_DONE` 또는 `FAILED`로 업데이트. |
| **Preprocess** | `preprocessing_jobs` | 전처리 실행 이력 및 상태 (`RUNNING` -> `DONE`/`FAILED`). |
| **Process** | `processing_jobs` | 메인 처리 실행 이력, 진행률, 상태 (`VLM_RUNNING` -> `DONE`/`FAILED`). |
| **Artifacts** | `stt_results` | **전처리 단계**에서 STT 완료 즉시 저장. |
| **Images** | `storage/captures` | **전처리 단계**에서 캡처 완료 즉시 업로드. |
| **Analysis** | `vlm_results`, `fs_segments` | **메인 파이프라인** 완료 후 일괄 저장 (VLM 및 분석 단위). |
| **Reports** | `fs_summaries` | **메인 파이프라인** 완료 후 최종 요약 일괄 저장. |
