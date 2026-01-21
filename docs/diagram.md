# ERD
```mermaid
erDiagram
    users ||--o{ videos : owns

    videos ||--o{ preprocessing_jobs : has
    videos ||--o{ processing_jobs : has
    videos ||--o{ captures : has
    videos ||--o{ stt_results : has
    videos ||--o{ vlm_results : has
    videos ||--o{ segments : has
    videos ||--o{ summaries : has
    videos ||--o{ summary_results : has
    videos ||--o{ judge : has

    preprocessing_jobs ||--o{ captures : produces
    preprocessing_jobs ||--o{ stt_results : produces

    processing_jobs ||--o{ vlm_results : produces
    processing_jobs ||--o{ segments : produces
    processing_jobs ||--o{ summaries : produces
    processing_jobs ||--o{ summary_results : produces
    processing_jobs ||--o{ judge : produces

    captures ||--o{ vlm_results : analyzed_as
    segments ||--o{ summaries : summarized_as

    users {
      uuid id PK
      text email
      timestamptz created_at
    }

    videos {
      uuid id PK
      uuid user_id FK
      text name
      text original_filename
      int duration_sec
      text status "UPLOADED|PREPROCESSING|PREPROCESS_DONE|FAILED"
      text error_message
      uuid current_preprocess_job_id FK
      uuid current_processing_job_id FK
      uuid current_summary_result_id FK
      timestamptz created_at
      timestamptz updated_at
    }

    preprocessing_jobs {
      uuid id PK
      uuid video_id FK
      text source "CLIENT|SERVER"
      text status "QUEUED|RUNNING|DONE|FAILED"
      text stt_backend
      text audio_storage_key
      text manifest_storage_key
      text config_hash
      text error_message
      timestamptz started_at
      timestamptz ended_at
      timestamptz created_at
      timestamptz updated_at
    }

    processing_jobs {
      uuid id PK
      uuid video_id FK
      text status "QUEUED|VLM_RUNNING|SUMMARY_RUNNING|JUDGE_RUNNING|DONE|FAILED"
      text triggered_by "CHAT_OPEN|MANUAL|SCHEDULE"
      int run_no
      text config_hash
      int progress_current
      int progress_total
      text error_message
      timestamptz started_at
      timestamptz ended_at
      timestamptz created_at
      timestamptz updated_at
    }

    captures {
      uuid id PK
      uuid video_id FK
      uuid preprocess_job_id FK
      text file_name
      int start_ms
      int end_ms
      text storage_path
      timestamptz created_at
    }

    stt_results {
      uuid id PK
      uuid video_id FK
      uuid preprocess_job_id FK
      int start_ms
      int end_ms
      text transcript
      float confidence
      timestamptz created_at
    }

    vlm_results {
      uuid id PK
      uuid video_id FK
      uuid processing_job_id FK
      uuid capture_id FK
      jsonb payload
      timestamptz created_at
    }

    segments {
      uuid id PK
      uuid video_id FK
      uuid processing_job_id FK
      int segment_index
      int start_ms
      int end_ms
      text transcript_units
      jsonb visual_units
      timestamptz created_at
    }

    summaries {
      uuid id PK
      uuid video_id FK
      uuid processing_job_id FK
      uuid segment_id FK
      int batch_index
      text summary
      jsonb version
      timestamptz created_at
    }

    summary_results {
      uuid id PK
      uuid video_id FK
      uuid processing_job_id FK
      text format "timeline|tldr"
      text status "PARTIAL|DONE"
      jsonb payload
      timestamptz created_at
    }

    judge {
      uuid id PK
      uuid video_id FK
      uuid processing_job_id FK
      text status "DONE|FAILED"
      float score
      jsonb report
      timestamptz created_at
    }
```

# Chatbot 조회 기준 (DB)

- 최신 결과 기준: `videos.current_summary_result_id` 또는 `summary_results`(status=PARTIAL|DONE, video_id)
- 부분 요약 응답: `summaries` + `segments`(transcript_units, visual_units)
- 원문/타임라인 근거: `stt_results`(start_ms, end_ms, transcript)
- 이미지 컨텍스트: `captures.storage_path` (필요 시 렌더)
- 진행 상태: `processing_jobs.status`, `processing_jobs.progress_current/total`

# Sequence Diagram (Future Async - Client Preprocess + Backend STT)
```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Frontend
    participant ST as Storage
    participant BE as Backend API
    participant DB as Postgres DB

    U->>FE: Select video
    FE->>BE: Request signed URLs (audio, captures, manifest)
    BE-->>FE: signed URLs

    Note over FE: Local preprocess (extract audio + captures + manifest)
    FE->>ST: Upload audio/captures/manifest
    ST-->>FE: upload success (artifact_keys)

    FE->>BE: Start preprocessing(artifact_keys, metadata)
    BE->>DB: INSERT videos(status=UPLOADED, user_id)
    BE->>DB: INSERT preprocessing_jobs(status=QUEUED, video_id, config_hash)
    BE->>DB: INSERT captures(...) from manifest
    BE-->>FE: 202 Accepted (video_id)

    Note over BE,DB: preprocessing async pipeline (STT only)
    BE->>DB: UPDATE preprocessing_jobs.status=RUNNING
    BE->>ST: Fetch audio
    BE->>DB: INSERT stt_results(...)
    BE->>DB: UPDATE preprocessing_jobs.status=DONE
    BE->>DB: UPDATE videos.status=PREPROCESS_DONE

    FE->>BE: Open chatbot (ensure summary)
    BE->>DB: SELECT latest summary_results for video_id
    alt summary already exists
        BE-->>FE: 200 OK (summary available)
    else no summary yet
        BE->>DB: INSERT processing_jobs(status=QUEUED, triggered_by=CHAT_OPEN, run_no=+1, config_hash)
        BE-->>FE: 202 Accepted (processing_job_id)

        Note over BE,DB: processing async pipeline (VLM -> segments -> summaries -> judge)
        par Summary worker
            BE->>DB: UPDATE processing_jobs.status=VLM_RUNNING
            BE->>DB: INSERT segments(...)
            BE->>DB: UPDATE processing_jobs.status=SUMMARY_RUNNING
            loop each batch
                BE->>DB: INSERT summaries(batch_index)
                BE->>DB: UPSERT summary_results(format, status=PARTIAL, payload=...)
                BE->>DB: UPDATE processing_jobs.progress_current
            end
            BE->>DB: UPDATE processing_jobs.status=JUDGE_RUNNING
            BE->>DB: INSERT judge(...)
            BE->>DB: UPDATE processing_jobs.status=DONE
            BE->>DB: UPDATE summary_results(status=DONE)
            BE->>DB: UPDATE videos.current_summary_result_id=(summary_results.id)
        and FE polling/WebSocket
            FE->>BE: Poll/subscribe summary status
            BE->>DB: SELECT latest summary_results(status=PARTIAL|DONE)
            BE-->>FE: progress + partial summary
        end
    end

    U->>FE: Ask chatbot
    FE->>BE: chat(video_id, message)
    BE->>DB: SELECT latest summary_results(status=PARTIAL|DONE)
    alt partial or done summary available
        BE-->>FE: Answer using latest summaries (partial ok)
    else not ready
        BE-->>FE: "아직 요약되지 않은 부분입니다."
    end
```

# Sequence Diagram (Current Logic)
```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Frontend
    participant BE as Backend API
    participant FS as Local Filesystem
    participant DB as Supabase (optional)

    U->>FE: Upload video / choose video_name
    FE->>BE: (Optional) Run preprocess
    BE->>FS: run_preprocess_pipeline (STT + Capture)
    BE->>FS: Write outputs (stt.json, captures/, manifest.json)
    opt sync_to_db_preprocess
        BE->>DB: sync_pipeline_results_to_db (videos, pipeline_runs, captures, stt_results)
    end

    U->>FE: Start summary / open chatbot
    FE->>BE: POST /process (video_name)
    BE-->>FE: 202 started
    BE->>FS: run_processing_pipeline (VLM + Fusion)
    BE->>FS: Write outputs (segments_units.jsonl, segment_summaries.jsonl, etc.)
    opt sync_to_db_process
        BE->>DB: sync_pipeline_results_to_db (segments, summaries, etc.)
    end

    FE->>BE: GET /runs/{video_name}
    BE->>FS: Read pipeline_run.json
    BE-->>FE: status + stats

    FE->>BE: get_summary_updates / get_summary_context
    BE->>FS: Read segment_summaries.jsonl
    BE-->>FE: Answer using local summaries
```

# Sequence Diagram (Client-side Preprocess Proposal)
```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FE as Frontend (Browser)
    participant LP as Local Preprocess (FFmpeg/WebCodecs)
    participant ST as Storage
    participant BE as Backend API
    participant DB as Postgres DB

    U->>FE: Select video
    FE->>BE: Request upload session
    BE->>DB: INSERT videos(status=UPLOADED, user_id)
    BE->>DB: INSERT preprocessing_jobs(status=QUEUED, video_id, source=CLIENT)
    BE-->>FE: video_id + signed URLs (audio, captures)

    FE->>LP: Extract audio (.mp3) + capture frames
    LP-->>FE: audio + captures + manifest

    FE->>ST: Upload audio (.mp3)
    FE->>ST: Upload captures (images)
    FE->>ST: Upload manifest.json (optional)
    FE->>BE: Notify uploads complete + storage keys
    BE->>DB: INSERT captures(storage_path, start_ms, end_ms)
    BE->>DB: UPDATE preprocessing_jobs.status=DONE
    BE->>DB: UPDATE videos.status=PREPROCESS_DONE

    FE->>BE: Start STT / summary
    BE->>ST: Download audio/captures from Storage
    BE->>DB: INSERT stt_results / segments / summaries ...
    BE-->>FE: status + results
```
