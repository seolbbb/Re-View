# DB 스키마 + 파이프라인 회귀 명세서 (post-ec7bdf0)

## 범위
- Repo: Screentime-MVP
- Branch: feature/db_base
- 마지막 커밋: ec7bdf0 (feat: diagram.md flow alignment)
- 목표: 스키마 변경 이후 왜 기존 동작이 깨졌는지와 맞춰야 할 지점을 설명

## 의도된 동작 (diagram.md)
- 클라이언트 전처리에서 audio + captures + manifest를 DB/Storage에 업로드
- 백엔드 전처리가 stt_results 저장 및 preprocessing_jobs를 DONE으로 갱신
- summarize 호출 시 processing_jobs 생성 후 VLM -> sync -> summarize -> judge 수행
- 배치 모드에서 배치 단위 산출물을 DB에 실시간 업로드하고 progress/status를 갱신
- 요약 상태 및 챗봇 API가 summary_results와 summaries + segments를 조회

## 관측된 회귀

### 1) 스키마 vs 어댑터 불일치 (치명적 실패)
- videos 스키마는 storage_path를 제거하고 status를 ('UPLOADED','PREPROCESSING','PREPROCESS_DONE','FAILED')로 제한하지만,
  create_video는 storage_path와 status "uploaded"를 기록함.
  근거: src/db/adapters/video_adapter.py:52,72-74 및 src/db/supabase_schema.sql:14-28.
  영향: video insert가 missing column + check constraint 에러로 실패.

- pipeline_runs 테이블이 스키마에 없는데 코드가 계속 insert/update함.
  근거: src/db/adapters/video_adapter.py:146 및 src/db/pipeline_sync.py:91.
  영향: sync_pipeline_results_to_db가 다른 insert가 성공해도 중단됨.

- captures 테이블은 preprocess_job_id를 사용하지만 save_captures는 pipeline_run_id를 저장함.
  근거: src/db/adapters/capture_adapter.py:34-47 및 src/db/supabase_schema.sql:138-146.
  영향: unknown column pipeline_run_id로 insert 실패.

- save_all_pipeline_results가 pipeline_run_id를 save_captures_with_upload에 전달하지만 해당 함수는 인자를 받지 않음.
  근거: src/db/supabase_adapter.py:82 및 src/db/adapters/capture_adapter.py:181-241.
  영향: sync 경로에서 TypeError 발생.

- stt_results 스키마는 transcript만 있고 provider가 없는데 save_stt_result는 provider + text를 저장함.
  근거: src/db/adapters/content_adapter.py:52-54 및 src/db/supabase_schema.sql:167-174.
  영향: stt_results insert 실패 혹은 transcript 비어 있음.

- segments 스키마에 embedding 컬럼이 없는데 save_segments는 embedding을 insert함.
  근거: src/db/adapters/content_adapter.py:134-136 및 src/db/supabase_schema.sql:223-233.
  영향: segments insert 실패.

- summaries 스키마에 summary_text가 없는데 save_summaries는 summary_text를 insert함.
  근거: src/db/adapters/content_adapter.py:196-204 및 src/db/supabase_schema.sql:253-263.
  영향: summaries insert 실패; semantic search 테스트는 summary_text를 기대함.

- save_all_pipeline_results가 videos.status를 "completed" 또는 "completed_with_errors"로 갱신하는데
  이는 새 status 제약과 충돌함.
  근거: src/db/supabase_adapter.py:131-133 및 src/db/supabase_schema.sql:20-28.
  영향: sync 후 status update 실패.

### 2) 전처리 동기화 실패
- prepare_preprocess_db_sync가 create_video를 호출하는데 스키마 불일치로 실패하여
  db_context가 None이 되고 video_id가 반환되지 않음.
  근거: src/db/pipeline_sync.py:155-175 및 src/db/adapters/video_adapter.py:52-74.
  영향: pipeline_service가 video_id를 세팅하지 못해 이후 DB 흐름이 깨짐.

- sync_preprocess_artifacts_to_db가 text/provider 컬럼 기준의 save_stt_result를 사용하여
  stt_results가 정상 삽입되지 않음.
  근거: src/db/pipeline_sync.py:235-265 및 src/db/adapters/content_adapter.py:52-54.
  영향: processing 단계가 DB에서 STT를 로드하지 못함.

### 3) 처리 입력 실패 (DB -> 로컬 아티팩트)
- run_processing_pipeline이 stt_results.segments 또는 row.text를 기대하지만 스키마는 transcript를 저장함.
  근거: src/run_process_pipeline.py:250-260.
  영향: stt.json이 빈 텍스트로 생성되어 sync_engine에서 세그먼트가 비어짐.

- captures.storage_path가 상위 업로드 실패로 null일 수 있는데,
  storage_path가 없으면 다운로드가 실패함.
  근거: src/run_process_pipeline.py:294-307.
  영향: DB 모드에서 VLM 단계가 진행 불가.

### 4) 배치 출력 + 실시간 DB 업로드 공백
- batch_mode 기본값이 true인데, sync_processing_results_to_db는 루트 vlm.json과
  fusion/segments_units.jsonl만 읽어서 배치 모드에서 파일을 찾지 못함.
  근거: config/pipeline/settings.yaml:2 및 src/db/pipeline_sync.py:322-360.
  영향: 배치 모드에서 VLM/segments/summaries가 DB에 업로드되지 않음.

- run_batch_fusion_pipeline은 batches/batch_N/ 아래에 배치 산출물을 쓰고,
  fusion/에는 segment_summaries.jsonl만 누적하며 segments_units.jsonl은 만들지 않음.
  근거: src/pipeline/stages.py:566,649-650.
  영향: sync_processing_results_to_db가 summaries와 segments를 매핑하지 못함.

- 배치별 DB 쓰기가 없고 processing_jobs 상태도 VLM_RUNNING과 DONE만 갱신됨.
  근거: src/run_process_pipeline.py:401-526.
  영향: 배치 실시간 업데이트가 누락됨.

### 5) summary_results 및 judge 갱신 누락
- summary_results와 judge 테이블은 있지만 파이프라인에서 쓰지 않음.
  근거: src/db/adapters/job_adapter.py:308-358 및 src/process_api.py:207-237.
  영향: 요약 상태 API가 빈 결과를 반환하고 챗봇 업데이트가 불가.

## 근본 원인 요약
- docs/diagram.md의 ERD로 스키마는 바뀌었지만 어댑터/동기화 코드는 이전 스키마를 전제로 동작함
  (pipeline_runs, storage_path, text/provider, summary_text, status 값 등).
- 배치 모드 산출물이 DB 업로드 흐름에 연결되지 않았고 summary/judge 집계도 누락됨.

## 필요한 변경 (명세 수준)
1) 코드와 스키마 정합화 (단일 소스 결정).
   - create_video/update_video_status가 새 status 값을 사용하도록 변경.
   - storage_path 사용을 제거하거나 스키마에 컬럼을 재추가.
   - pipeline_runs 사용을 제거하거나 테이블을 재도입.
   - stt_results를 transcript에 매핑하고 provider는 스키마에 있을 때만 저장.
   - segments.embedding insert 제거 혹은 스키마에 추가.
   - summaries.summary_text 필요 시 스키마 추가 또는 insert 제거.
   - save_all_pipeline_results/sync_pipeline_results_to_db는 pipeline_run_id 대신
     preprocess_job_id/processing_job_id를 사용.

2) 처리 입력(DB -> 로컬) 매핑 수정.
   - stt_results.transcript로 stt.json을 구성 (start_ms 기준 정렬 포함).
   - stt_results.segments 의존 제거 (스키마에 없음).

3) 배치 인지형 DB 업로드 + 실시간 업데이트 구현.
   - 배치별 vlm_results, segments, summaries insert (batch_index 사용).
   - 배치마다 summary_results를 IN_PROGRESS로 upsert, 종료 시 DONE으로 갱신.
   - judge 결과를 종료 시 insert (필요하면 배치별도 가능).
   - processing_jobs.status를 단계별 갱신하고 batch마다 progress_current 갱신.

4) 검증 훅 추가.
   - 전처리: videos, preprocessing_jobs, captures, stt_results rows 확인.
   - 처리: 배치별 summaries + progress 갱신, summary_results가 API에서 보이는지 확인.

## 수용 기준
- 전처리가 video_id를 반환하고 DB에 captures + stt_results가 non-empty transcript로 저장됨.
- 배치 모드 처리에서 vlm_results, segments, summaries가 점진적으로 저장됨.
- summary_results가 처리 중에도 조회되고 종료 시 DONE 상태가 됨.
- processing_jobs.progress_current가 배치마다 증가하고 최종 status가 DONE이 됨.
- Supabase에서 missing column/status constraint 관련 에러가 발생하지 않음.