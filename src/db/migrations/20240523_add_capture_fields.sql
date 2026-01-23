-- Migration: Add time_ranges and info_score columns to captures table

-- 1. Add new columns
ALTER TABLE captures 
ADD COLUMN IF NOT EXISTS time_ranges JSONB,
ADD COLUMN IF NOT EXISTS info_score FLOAT;

-- 2. Migrate existing data (start_ms, end_ms) to time_ranges format
-- only if time_ranges is null and we have legacy data
UPDATE captures
SET time_ranges = jsonb_build_array(
    jsonb_build_object('start_ms', start_ms, 'end_ms', end_ms)
)
WHERE time_ranges IS NULL AND start_ms IS NOT NULL;

-- 3. Update comments/documentation if necessary
COMMENT ON COLUMN captures.time_ranges IS 'List of time ranges [{start_ms, end_ms}] for this slide';
COMMENT ON COLUMN captures.info_score IS 'Information score of the slide image';
