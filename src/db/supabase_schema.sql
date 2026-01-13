-- ============================================================================
-- ReView Supabase 테이블 스키마
-- ============================================================================
-- 1. Vector 익스텐션 활성화 (AI 검색용)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS moddatetime;

-- 2. videos (비디오 메타데이터)
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT DEFAULT 'uploaded',
    error_message TEXT,
    name TEXT NOT NULL,
    original_filename TEXT,
    duration_sec INTEGER,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    storage_path TEXT,
    
    -- [개선] 상태값 무결성 체크
    CONSTRAINT check_status CHECK (status IN ('uploaded', 'processing', 'completed', 'completed_with_errors', 'failed'))
);

-- [개선] updated_at 자동 갱신 트리거
DROP TRIGGER IF EXISTS handle_updated_at ON videos;
CREATE TRIGGER handle_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE PROCEDURE moddatetime(updated_at);

-- RLS 활성화
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

-- [기존] 조회/추가/수정 정책
DROP POLICY IF EXISTS "Users can view own videos" ON videos;
CREATE POLICY "Users can view own videos" ON videos FOR SELECT USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own videos" ON videos;
CREATE POLICY "Users can insert own videos" ON videos FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update own videos" ON videos;
CREATE POLICY "Users can update own videos" ON videos FOR UPDATE USING (auth.uid() = user_id);

-- [개선] 삭제 정책 추가 (누락되었던 부분)
DROP POLICY IF EXISTS "Users can delete own videos" ON videos;
CREATE POLICY "Users can delete own videos" ON videos FOR DELETE USING (auth.uid() = user_id);


-- 3. pipeline_runs (파이프라인 실행 메타데이터) - Moved up for FK reference
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT DEFAULT 'running',
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    run_meta JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view runs of own videos" ON pipeline_runs;
CREATE POLICY "Users can view runs of own videos" ON pipeline_runs
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM videos WHERE videos.id = pipeline_runs.video_id AND videos.user_id = auth.uid())
    );


-- 4. captures (화면 캡처)
CREATE TABLE IF NOT EXISTS captures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name TEXT NOT NULL,
    start_ms INTEGER,
    end_ms INTEGER,
    storage_path TEXT,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_captures_video_ts ON captures(video_id, start_ms);

-- RLS
ALTER TABLE captures ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view captures of own videos" ON captures;
CREATE POLICY "Users can view captures of own videos" ON captures
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid())
    );


-- 4. stt_results (음성 인식 결과)
CREATE TABLE IF NOT EXISTS stt_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT DEFAULT 'clova',
    segments JSONB,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE stt_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view stt of own videos" ON stt_results;
CREATE POLICY "Users can view stt of own videos" ON stt_results
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid())
    );


-- 5. segments (통합 세그먼트)
CREATE TABLE IF NOT EXISTS segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_index INTEGER,
    start_ms INTEGER,
    end_ms INTEGER,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    transcript_units JSONB,
    visual_units JSONB,
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    -- [개선] Vector Embedding (OpenAI text-embedding-3-small 기준 1536차원)
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_segments_video_idx ON segments(video_id, segment_index);

-- [개선] HNSW 인덱스 (벡터 검색 속도 최적화)
-- 데이터가 쌓인 후 생성하는 것이 좋으나, 스키마 정의 차원에서 포함
CREATE INDEX IF NOT EXISTS idx_segments_embedding_hnsw ON segments USING hnsw (embedding vector_cosine_ops);

-- RLS
ALTER TABLE segments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view segments of own videos" ON segments;
CREATE POLICY "Users can view segments of own videos" ON segments
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid())
    );


-- 6. summaries (요약 결과)
CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    summary JSONB,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    segment_id UUID REFERENCES segments(id) ON DELETE CASCADE,
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    version JSONB,
    -- [개선] 요약 내용에 대한 임베딩도 필요할 수 있음
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can view summaries of own videos" ON summaries;
CREATE POLICY "Users can view summaries of own videos" ON summaries
    FOR SELECT USING (
        EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid())
    );





-- ============================================================================
-- 부가적인 인덱스
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_videos_user_status ON videos(user_id, status);
CREATE INDEX IF NOT EXISTS idx_videos_name ON videos(name);

-- ============================================================================
-- 완료 메시지
-- ============================================================================
SELECT 'Screentime-MVP 개선된 테이블 생성 완료!' as message;

-- 7. Additional CRUD Policies for Child Tables
-- Users should be able to manage data related to their own videos.

-- pipeline_runs
DROP POLICY IF EXISTS "Users can insert runs of own videos" ON pipeline_runs;
CREATE POLICY "Users can insert runs of own videos" ON pipeline_runs FOR INSERT WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = pipeline_runs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update runs of own videos" ON pipeline_runs;
CREATE POLICY "Users can update runs of own videos" ON pipeline_runs FOR UPDATE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = pipeline_runs.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete runs of own videos" ON pipeline_runs;
CREATE POLICY "Users can delete runs of own videos" ON pipeline_runs FOR DELETE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = pipeline_runs.video_id AND videos.user_id = auth.uid()));

-- captures
DROP POLICY IF EXISTS "Users can insert captures of own videos" ON captures;
CREATE POLICY "Users can insert captures of own videos" ON captures FOR INSERT WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update captures of own videos" ON captures;
CREATE POLICY "Users can update captures of own videos" ON captures FOR UPDATE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete captures of own videos" ON captures;
CREATE POLICY "Users can delete captures of own videos" ON captures FOR DELETE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = captures.video_id AND videos.user_id = auth.uid()));

-- stt_results
DROP POLICY IF EXISTS "Users can insert stt of own videos" ON stt_results;
CREATE POLICY "Users can insert stt of own videos" ON stt_results FOR INSERT WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update stt of own videos" ON stt_results;
CREATE POLICY "Users can update stt of own videos" ON stt_results FOR UPDATE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete stt of own videos" ON stt_results;
CREATE POLICY "Users can delete stt of own videos" ON stt_results FOR DELETE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = stt_results.video_id AND videos.user_id = auth.uid()));

-- segments
DROP POLICY IF EXISTS "Users can insert segments of own videos" ON segments;
CREATE POLICY "Users can insert segments of own videos" ON segments FOR INSERT WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update segments of own videos" ON segments;
CREATE POLICY "Users can update segments of own videos" ON segments FOR UPDATE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete segments of own videos" ON segments;
CREATE POLICY "Users can delete segments of own videos" ON segments FOR DELETE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = segments.video_id AND videos.user_id = auth.uid()));

-- summaries
DROP POLICY IF EXISTS "Users can insert summaries of own videos" ON summaries;
CREATE POLICY "Users can insert summaries of own videos" ON summaries FOR INSERT WITH CHECK (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can update summaries of own videos" ON summaries;
CREATE POLICY "Users can update summaries of own videos" ON summaries FOR UPDATE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));
DROP POLICY IF EXISTS "Users can delete summaries of own videos" ON summaries;
CREATE POLICY "Users can delete summaries of own videos" ON summaries FOR DELETE USING (EXISTS (SELECT 1 FROM videos WHERE videos.id = summaries.video_id AND videos.user_id = auth.uid()));


-- ============================================================================
-- [Migration 2026-01-13] Convert transcript_units to TEXT
-- ============================================================================

-- ============================================================================
-- [Migration 2026-01-13] Convert transcript_units to TEXT
-- ============================================================================

-- ============================================================================
-- [Migration 2026-01-13] Convert transcript_units to TEXT
-- ============================================================================

-- 1. Helper function (reusing extract_transcript_text for one-time migration)
CREATE OR REPLACE FUNCTION extract_transcript_text_v2(units jsonb) 
RETURNS TEXT IMMUTABLE LANGUAGE sql AS $$
    SELECT string_agg(elem->>'text', E'\n' ORDER BY (elem->>'start_ms')::int)
    FROM jsonb_array_elements(units) elem;
$$;

-- 2. Add temporary text column
ALTER TABLE segments ADD COLUMN IF NOT EXISTS transcript_units_text TEXT;

-- 3. Populate new column
UPDATE segments SET transcript_units_text = extract_transcript_text_v2(transcript_units);

-- 4. Swap columns (Drop dependencies first)
-- transcript_text generated column depends on transcript_units, so we drop it.
ALTER TABLE segments DROP COLUMN IF EXISTS transcript_text; 
ALTER TABLE segments DROP COLUMN transcript_units;
ALTER TABLE segments RENAME COLUMN transcript_units_text TO transcript_units;

-- 5. Cleanup
DROP FUNCTION IF EXISTS extract_transcript_text;
DROP FUNCTION IF EXISTS extract_transcript_text_v2;


-- ============================================================================
-- [Migration 2026-01-13] Normalize stt_results (Single Table Flattening)
-- ============================================================================

-- 1. Add new columns for segment data
ALTER TABLE stt_results 
ADD COLUMN IF NOT EXISTS text TEXT,
ADD COLUMN IF NOT EXISTS start_ms INTEGER,
ADD COLUMN IF NOT EXISTS end_ms INTEGER,
ADD COLUMN IF NOT EXISTS confidence FLOAT,
ADD COLUMN IF NOT EXISTS segment_index INTEGER;

-- 2. Migrate Data: Expand JSONB array into individual rows
-- Strategy: Insert new rows for each segment, then delete the original "container" rows.
-- Note: 'segments' column still exists at this point.
INSERT INTO stt_results (
    id, provider, video_id, pipeline_run_id, embedding, created_at,
    text, start_ms, end_ms, confidence, segment_index
)
SELECT 
    gen_random_uuid(), -- New ID for each segment row
    provider, 
    video_id, 
    pipeline_run_id, 
    embedding, 
    created_at,
    elem->>'text', 
    (elem->>'start_ms')::INTEGER, 
    (elem->>'end_ms')::INTEGER, 
    (elem->>'confidence')::FLOAT,
    idx::INTEGER
FROM stt_results, jsonb_array_elements(segments) WITH ORDINALITY AS arr(elem, idx)
WHERE segments IS NOT NULL AND jsonb_array_length(segments) > 0;

-- 3. Delete original container rows (rows where 'segments' is not null)
-- Caution: We must ensure we don't delete the rows we just inserted (which have null 'segments')
DELETE FROM stt_results WHERE segments IS NOT NULL;

-- ============================================================================
-- [Migration 2026-01-13] Remove segment_index from stt_results
-- ============================================================================
ALTER TABLE stt_results DROP COLUMN IF EXISTS segment_index;
