-- ============================================================================
-- ReView Supabase 테이블 스키마 (ERD 기반 - 2026-01-21)
-- ============================================================================
-- docs/diagram.md의 ERD를 기반으로 생성된 스키마입니다.
-- ============================================================================

-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS moddatetime;

-- ============================================================================
-- 2. videos (비디오 메타데이터)
-- ============================================================================
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    original_filename TEXT,
    duration_sec INTEGER,
    status TEXT DEFAULT 'UPLOADED',
    error_message TEXT,
    current_preprocess_job_id UUID,  -- FK added after preprocessing_jobs created
    current_processing_job_id UUID,  -- FK added after processing_jobs created
    current_summary_result_id UUID,  -- FK added after summary_results created
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    delete_requested_at TIMESTAMPTZ,
    
    CONSTRAINT check_video_status CHECK (status IN ('UPLOADED', 'PREPROCESSING', 'PREPROCESS_DONE', 'PROCESSING', 'DONE', 'FAILED'))
);

-- Auto-update updated_at trigger
DROP TRIGGER IF EXISTS handle_videos_updated_at ON videos;
CREATE TRIGGER handle_videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE PROCEDURE moddatetime(updated_at);

-- RLS
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view own videos" ON videos;
CREATE POLICY "Users can view own videos" ON videos FOR SELECT USING (auth.uid() = user_id);
DROP POLICY IF EXISTS "Users can insert own videos" ON videos;
CREATE POLICY "Users can insert own videos" ON videos FOR INSERT WITH CHECK (auth.uid() = user_id);
DROP POLICY IF EXISTS "Users can update own videos" ON videos;
CREATE POLICY "Users can update own videos" ON videos FOR UPDATE USING (auth.uid() = user_id);
DROP POLICY IF EXISTS "Users can delete own videos" ON videos;
CREATE POLICY "Users can delete own videos" ON videos FOR DELETE USING (auth.uid() = user_id);

-- ============================================================================
-- 3. preprocessing_jobs (전처리 작업)
-- ============================================================================
CREATE TABLE IF NOT EXISTS preprocessing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    source TEXT DEFAULT 'SERVER',
    status TEXT DEFAULT 'QUEUED',
    stt_backend TEXT,
    audio_storage_key TEXT,
    config_hash TEXT,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT check_preprocess_source CHECK (source IN ('CLIENT', 'SERVER')),
    CONSTRAINT check_preprocess_status CHECK (status IN ('QUEUED', 'RUNNING', 'DONE', 'FAILED'))
);

DROP TRIGGER IF EXISTS handle_preprocessing_jobs_updated_at ON preprocessing_jobs;
CREATE TRIGGER handle_preprocessing_jobs_updated_at
    BEFORE UPDATE ON preprocessing_jobs
    FOR EACH ROW
    EXECUTE PROCEDURE moddatetime(updated_at);

-- RLS
ALTER TABLE preprocessing_jobs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view preprocessing_jobs of own videos" ON preprocessing_jobs;
CREATE POLICY "Users can view preprocessing_jobs of own videos" ON preprocessing_jobs FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = preprocessing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert preprocessing_jobs of own videos" ON preprocessing_jobs;
CREATE POLICY "Users can insert preprocessing_jobs of own videos" ON preprocessing_jobs FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = preprocessing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update preprocessing_jobs of own videos" ON preprocessing_jobs;
CREATE POLICY "Users can update preprocessing_jobs of own videos" ON preprocessing_jobs FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = preprocessing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete preprocessing_jobs of own videos" ON preprocessing_jobs;
CREATE POLICY "Users can delete preprocessing_jobs of own videos" ON preprocessing_jobs FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = preprocessing_jobs.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 4. processing_jobs (처리 작업)
-- ============================================================================
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'QUEUED',
    triggered_by TEXT DEFAULT 'MANUAL',
    run_no INTEGER DEFAULT 1,
    config_hash TEXT,
    current_batch INTEGER DEFAULT 0,
    total_batch INTEGER DEFAULT 0, 
    error_message TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT check_processing_status CHECK (status IN ('QUEUED', 'VLM_RUNNING', 'SUMMARY_RUNNING', 'JUDGE_RUNNING', 'DONE', 'FAILED')),
    CONSTRAINT check_triggered_by CHECK (triggered_by IN ('CHAT_OPEN', 'MANUAL', 'SCHEDULE'))
);

DROP TRIGGER IF EXISTS handle_processing_jobs_updated_at ON processing_jobs;
CREATE TRIGGER handle_processing_jobs_updated_at
    BEFORE UPDATE ON processing_jobs
    FOR EACH ROW
    EXECUTE PROCEDURE moddatetime(updated_at);

-- RLS
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view processing_jobs of own videos" ON processing_jobs;
CREATE POLICY "Users can view processing_jobs of own videos" ON processing_jobs FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = processing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert processing_jobs of own videos" ON processing_jobs;
CREATE POLICY "Users can insert processing_jobs of own videos" ON processing_jobs FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = processing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update processing_jobs of own videos" ON processing_jobs;
CREATE POLICY "Users can update processing_jobs of own videos" ON processing_jobs FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = processing_jobs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete processing_jobs of own videos" ON processing_jobs;
CREATE POLICY "Users can delete processing_jobs of own videos" ON processing_jobs FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = processing_jobs.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 5. captures (화면 캡처)
-- ============================================================================
CREATE TABLE IF NOT EXISTS captures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    preprocess_job_id UUID REFERENCES preprocessing_jobs(id) ON DELETE SET NULL,
    file_name TEXT NOT NULL,
    storage_path TEXT,
    time_ranges JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE captures ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view captures of own videos" ON captures;
CREATE POLICY "Users can view captures of own videos" ON captures FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert captures of own videos" ON captures;
CREATE POLICY "Users can insert captures of own videos" ON captures FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update captures of own videos" ON captures;
CREATE POLICY "Users can update captures of own videos" ON captures FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete captures of own videos" ON captures;
CREATE POLICY "Users can delete captures of own videos" ON captures FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 6. stt_results (음성 인식 결과)
-- ============================================================================
CREATE TABLE IF NOT EXISTS stt_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    preprocess_job_id UUID REFERENCES preprocessing_jobs(id) ON DELETE SET NULL,
    stt_id TEXT,  -- stt.json의 id 필드 (e.g., "stt_001")
    start_ms INTEGER,
    end_ms INTEGER,
    transcript TEXT,
    confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE stt_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view stt_results of own videos" ON stt_results;
CREATE POLICY "Users can view stt_results of own videos" ON stt_results FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert stt_results of own videos" ON stt_results;
CREATE POLICY "Users can insert stt_results of own videos" ON stt_results FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update stt_results of own videos" ON stt_results;
CREATE POLICY "Users can update stt_results of own videos" ON stt_results FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete stt_results of own videos" ON stt_results;
CREATE POLICY "Users can delete stt_results of own videos" ON stt_results FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 7. vlm_results (VLM 분석 결과)
-- ============================================================================
CREATE TABLE IF NOT EXISTS vlm_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    capture_id UUID REFERENCES captures(id) ON DELETE SET NULL,
    cap_id TEXT,  -- vlm.json의 id 필드 (e.g., "cap_00001")
    time_ranges JSONB,  -- captures과 동일한 시간 범위 스키마 [{start_ms, end_ms}, ...]
    extracted_text TEXT,  -- vlm.json의 extracted_text 필드
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE vlm_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view vlm_results of own videos" ON vlm_results;
CREATE POLICY "Users can view vlm_results of own videos" ON vlm_results FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = vlm_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert vlm_results of own videos" ON vlm_results;
CREATE POLICY "Users can insert vlm_results of own videos" ON vlm_results FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = vlm_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update vlm_results of own videos" ON vlm_results;
CREATE POLICY "Users can update vlm_results of own videos" ON vlm_results FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = vlm_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete vlm_results of own videos" ON vlm_results;
CREATE POLICY "Users can delete vlm_results of own videos" ON vlm_results FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = vlm_results.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 8. segments (통합 세그먼트)
-- ============================================================================
CREATE TABLE IF NOT EXISTS segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    segment_index INTEGER,
    start_ms INTEGER,
    end_ms INTEGER,
    transcript_units TEXT,
    visual_units JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE segments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view segments of own videos" ON segments;
CREATE POLICY "Users can view segments of own videos" ON segments FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert segments of own videos" ON segments;
CREATE POLICY "Users can insert segments of own videos" ON segments FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update segments of own videos" ON segments;
CREATE POLICY "Users can update segments of own videos" ON segments FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete segments of own videos" ON segments;
CREATE POLICY "Users can delete segments of own videos" ON segments FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 9. summaries (요약 결과)
-- ============================================================================
CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
    batch_index INTEGER,
    summary JSONB,
    version JSONB,
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view summaries of own videos" ON summaries;
CREATE POLICY "Users can view summaries of own videos" ON summaries FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert summaries of own videos" ON summaries;
CREATE POLICY "Users can insert summaries of own videos" ON summaries FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update summaries of own videos" ON summaries;
CREATE POLICY "Users can update summaries of own videos" ON summaries FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete summaries of own videos" ON summaries;
CREATE POLICY "Users can delete summaries of own videos" ON summaries FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));

-- HNSW index for vector search
CREATE INDEX IF NOT EXISTS idx_summaries_embedding_hnsw 
ON summaries USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 10. summary_results (요약 결과 집계)
-- ============================================================================
CREATE TABLE IF NOT EXISTS summary_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    format TEXT DEFAULT 'timeline',
    status TEXT DEFAULT 'IN_PROGRESS',
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT check_summary_format CHECK (format IN ('timeline', 'tldr', 'tldr_timeline')),
    CONSTRAINT check_summary_status CHECK (status IN ('IN_PROGRESS', 'DONE'))
);

-- RLS
ALTER TABLE summary_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view summary_results of own videos" ON summary_results;
CREATE POLICY "Users can view summary_results of own videos" ON summary_results FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summary_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert summary_results of own videos" ON summary_results;
CREATE POLICY "Users can insert summary_results of own videos" ON summary_results FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = summary_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update summary_results of own videos" ON summary_results;
CREATE POLICY "Users can update summary_results of own videos" ON summary_results FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summary_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete summary_results of own videos" ON summary_results;
CREATE POLICY "Users can delete summary_results of own videos" ON summary_results FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summary_results.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 11. judge (품질 평가)
-- ============================================================================
CREATE TABLE IF NOT EXISTS judge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    batch_index INTEGER,  -- 배치 인덱스 (0-indexed)
    status TEXT DEFAULT 'DONE',
    score FLOAT,
    report JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT check_judge_status CHECK (status IN ('DONE', 'FAILED'))
);

-- RLS
ALTER TABLE judge ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view judge of own videos" ON judge;
CREATE POLICY "Users can view judge of own videos" ON judge FOR SELECT 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = judge.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can insert judge of own videos" ON judge;
CREATE POLICY "Users can insert judge of own videos" ON judge FOR INSERT 
    WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = judge.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update judge of own videos" ON judge;
CREATE POLICY "Users can update judge of own videos" ON judge FOR UPDATE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = judge.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete judge of own videos" ON judge;
CREATE POLICY "Users can delete judge of own videos" ON judge FOR DELETE 
    USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = judge.video_id AND videos.user_id = auth.uid()));

-- ============================================================================
-- 12. 순환 참조 외래 키 추가
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_videos_current_preprocess_job') THEN
        ALTER TABLE videos ADD CONSTRAINT fk_videos_current_preprocess_job 
            FOREIGN KEY (current_preprocess_job_id) REFERENCES preprocessing_jobs(id) ON DELETE SET NULL;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_videos_current_processing_job') THEN
        ALTER TABLE videos ADD CONSTRAINT fk_videos_current_processing_job 
            FOREIGN KEY (current_processing_job_id) REFERENCES processing_jobs(id) ON DELETE SET NULL;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_videos_current_summary_result') THEN
        ALTER TABLE videos ADD CONSTRAINT fk_videos_current_summary_result 
            FOREIGN KEY (current_summary_result_id) REFERENCES summary_results(id) ON DELETE SET NULL;
    END IF;
END $$;

-- ============================================================================
-- 13. 인덱스 생성
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_videos_user_status ON videos(user_id, status);
CREATE INDEX IF NOT EXISTS idx_videos_name ON videos(name);

CREATE INDEX IF NOT EXISTS idx_preprocessing_jobs_video ON preprocessing_jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_preprocessing_jobs_status ON preprocessing_jobs(status);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_video ON processing_jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status);

CREATE INDEX IF NOT EXISTS idx_captures_video_ts ON captures(video_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_stt_results_video_ts ON stt_results(video_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_vlm_results_video ON vlm_results(video_id);

CREATE INDEX IF NOT EXISTS idx_segments_video_idx ON segments(video_id, segment_index);
CREATE INDEX IF NOT EXISTS idx_summaries_video ON summaries(video_id);
CREATE INDEX IF NOT EXISTS idx_summaries_segment ON summaries(segment_id);

CREATE INDEX IF NOT EXISTS idx_summary_results_video ON summary_results(video_id);
CREATE INDEX IF NOT EXISTS idx_judge_video ON judge(video_id);

-- ============================================================================
-- 14. 기존 테이블 마이그레이션 (2026-01-22 추가)
-- ============================================================================
-- 이미 테이블이 존재하는 경우 새 컬럼 추가
DO $$
BEGIN
    -- videos.status 제약조건 업데이트 (PROCESSING, DONE 추가)
    ALTER TABLE videos DROP CONSTRAINT IF EXISTS check_video_status;
    ALTER TABLE videos ADD CONSTRAINT check_video_status
        CHECK (status IN ('UPLOADED', 'PREPROCESSING', 'PREPROCESS_DONE', 'PROCESSING', 'DONE', 'FAILED'));

    -- videos.delete_requested_at 컬럼 추가 (처리 중 삭제/취소 신호용)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'videos' AND column_name = 'delete_requested_at') THEN
        ALTER TABLE videos ADD COLUMN delete_requested_at TIMESTAMPTZ;
    END IF;

    -- stt_results.stt_id 컬럼 추가
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'stt_results' AND column_name = 'stt_id') THEN
        ALTER TABLE stt_results ADD COLUMN stt_id TEXT;
    END IF;

    -- vlm_results 새 컬럼 추가 및 payload 컬럼 삭제
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'vlm_results' AND column_name = 'cap_id') THEN
        ALTER TABLE vlm_results ADD COLUMN cap_id TEXT;
    END IF;
    -- timestamp_ms -> time_ranges 마이그레이션 (Issue #141)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'vlm_results' AND column_name = 'time_ranges') THEN
        ALTER TABLE vlm_results ADD COLUMN time_ranges JSONB;
    END IF;
    -- 기존 timestamp_ms 컬럼 삭제
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'vlm_results' AND column_name = 'timestamp_ms') THEN
        ALTER TABLE vlm_results DROP COLUMN timestamp_ms;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'vlm_results' AND column_name = 'extracted_text') THEN
        ALTER TABLE vlm_results ADD COLUMN extracted_text TEXT;
    END IF;
    -- payload 컬럼 삭제 (더 이상 사용하지 않음)
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'vlm_results' AND column_name = 'payload') THEN
        ALTER TABLE vlm_results DROP COLUMN payload;
    END IF;

    -- processing_jobs 컬럼명 변경 (progress_current → current_batch, progress_total → total_batch)
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'progress_current') THEN
        ALTER TABLE processing_jobs RENAME COLUMN progress_current TO current_batch;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'processing_jobs' AND column_name = 'progress_total') THEN
        ALTER TABLE processing_jobs RENAME COLUMN progress_total TO total_batch;
    END IF;

    -- summary_results.format 제약조건 업데이트 (tldr_timeline 추가)
    ALTER TABLE summary_results DROP CONSTRAINT IF EXISTS check_summary_format;
    ALTER TABLE summary_results ADD CONSTRAINT check_summary_format
        CHECK (format IN ('timeline', 'tldr', 'tldr_timeline'));

    -- judge.batch_index 컬럼 추가
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'judge' AND column_name = 'batch_index') THEN
        ALTER TABLE judge ADD COLUMN batch_index INTEGER;
    END IF;

    -- captures 테이블 컬럼 추가 (time_ranges)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'captures' AND column_name = 'time_ranges') THEN
        ALTER TABLE captures ADD COLUMN time_ranges JSONB;
    END IF;

    -- captures 테이블 레거시 컬럼 삭제 (start_ms, end_ms, info_score)
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'captures' AND column_name = 'start_ms') THEN
        ALTER TABLE captures DROP COLUMN start_ms;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'captures' AND column_name = 'end_ms') THEN
        ALTER TABLE captures DROP COLUMN end_ms;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'captures' AND column_name = 'info_score') THEN
        ALTER TABLE captures DROP COLUMN info_score;
    END IF;
END $$;

-- ============================================================================
-- 완료 메시지
-- ============================================================================
SELECT 'Screentime-MVP ERD 기반 테이블 생성 완료!' as message;
