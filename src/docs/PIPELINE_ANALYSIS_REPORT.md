# ReViewFeature 파이프라인 상세 분석 및 검증 보고서

이 문서는 최신 스키마(v2.0)가 적용된 `run_preprocess_pipeline.py` (전처리)와 `run_process_pipeline.py` (메인 처리)의 연동 로직 및 실제 DB 데이터 검증 결과를 기술합니다.

---

## 1. 파이프라인 통합 아키텍처

파이프라인은 비디오 파일로부터 최종 요약 리포트를 생성하기 위해 2단계 구조를 가집니다. 모든 단계는 Supabase와 실시간으로 동기화됩니다.

### 1.1 전처리 단계 (`preprocessing_jobs`)

- **목적**: 시각/청각 에셋 추출 및 DB 적재
- **데이터 흐름**: `Local Video` -> `Audio Extraction` & `Scene Capture` -> `Supabase Storage` & `DB`
- **주요 테이블**:

  - `videos`: 비디오 통합 관리 (Status: `UPLOADED` -> `PREPROCESSING` -> `PREPROCESS_DONE`)
  - `preprocessing_jobs`: 전처리 작업 이력 관리
  - `stt_results`: 전체 타임라인 STT 텍스트 (Clova/Whisper)
  - `captures`: 주요 프레임 이미지 경로 및 **`time_ranges` (1:N 매핑)** 정보

### 1.2 메인 처리 단계 (`processing_jobs`)

- **목적**: AI 모델을 통한 컨텍스트 병합, 요약 및 평가
- **데이터 흐름**: `DB Artifacts` -> `VLM (Vision)` -> `Sync Engine` -> `LLM Summarizer` -> `Judge` -> `Final Results`
- **주요 테이블**:

  - `processing_jobs`: 메인 처리 작업 이력 및 배치 진행률 (`current_batch`/`total_batch`)
  - `segments`: Sync Engine에 의해 생성된 시간 기반 분석 단위
  - `summaries`: 개별 세그먼트에 대한 생성형 요약 결과
  - `judge`: 배치 단위 요약 품질 점수 및 피드백 (Self-Correction 루프 작동)
  - `summary_results`: 최종 렌더링된 통합 요약 리포트 (Format: `tldr_timeline` 등)

---

## 2. 핵심 로직 검증 결과

### 2.1 Captures 1:N 매핑 (`time_ranges`)

기존의 `start_ms`/`end_ms` 단일 필드 구조에서 `time_ranges` JSONB 배열 구조로 변경되었습니다.
- **검증**: 동일 이미지가 여러 시간대에 걸쳐 사용되는 "슬라이드 재사용" 케이스를 완벽하게 지원합니다.
- **로직**: VLM 데이터 병합 시 `time_ranges`의 모든 구간을 Flattening하여 중복 없는 컨텍스트를 구성합니다.

### 2.2 배치 모드 및 상태 관리

`--batch-mode` 실행 시 `processing_jobs` 테이블을 통해 실시간 진행 상태가 업데이트됩니다.
- **상태 전이**: `UPLOADED` -> `PROCESSING` -> `DONE` (성공 시)
- **배치 추적**: `current_batch`가 증가하며 대용량 영상의 처리 현황을 외부(API/App)에서 감시할 수 있습니다.

### 2.3 Judge & Self-Correction

요약 품질이 기준점에 미달할 경우 자동으로 재시도를 수행합니다.
- **검증**: 실제 실행 과정에서 `Batch 1` 점수 미달 시 Judge 피드백을 반영한 `summarize_retry_1` 단계가 실행되어 품질이 개선됨을 확인했습니다.

---

## 3. 실제 실행 검증 데이터 (`sample4` 기준)

| 항목 | 검증 지표 | 결과 |
| :--- | :--- | :--- |
| **Video ID** | `df1e1d59-6a66-4a35-8214-2d41d5ac49fc` | 등록 확인 |
| **Video Status** | `DONE` (최종 완료) | **PASS** |
| **Captures** | 5개 레코드 (`time_ranges` 스키마 포함) | **PASS** |
| **STT Units** | 42개 세그먼트 저장 완료 | **PASS** |
| **Processing Job** | `3c433162-465c-43a2-9597-08939f5ece73` | 생성 확인 |
| **Job Progress** | Batch 2/2 완료 및 Status `DONE` | **PASS** |
| **Analytic Segments** | 총 8개 세그먼트 분석 및 요약 완료 | **PASS** |
| **Judge Scores** | Batch 1: 8.26 / Batch 2: 8.96 | **PASS** |
| **Final Results** | `summary_results` (Format: `tldr_timeline`) 생성됨 | **PASS** |

---

## 4. 결론

파이프라인의 모든 코드는 새로운 데이터베이스 스키마와 완벽히 호환되며, 전처리부터 메인 처리, 리포트 생성에 이르는 전 과정이 Supabase 클라우드 DB와 정상적으로 연동되고 있음을 확인하였습니다. (최종 검증일: 2026-01-24)
