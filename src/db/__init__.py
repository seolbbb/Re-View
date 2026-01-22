"""DB 모듈 패키지."""
from .supabase_adapter import SupabaseAdapter, get_supabase_adapter
from .pipeline_sync import sync_pipeline_results_to_db, sync_processing_results_to_db
from .stage_uploader import (
    get_vlm_results_with_fallback,
    get_segments_with_fallback,
    get_summaries_with_fallback,
    upload_vlm_results_for_batch,
    upload_segments_for_batch,
    upload_summaries_for_batch,
    upload_judge_result,
    upsert_final_summary_results,
    accumulate_segments_to_fusion,
)

__all__ = [
    "SupabaseAdapter",
    "get_supabase_adapter",
    "sync_pipeline_results_to_db",
    "sync_processing_results_to_db",
    # Stage uploader functions
    "get_vlm_results_with_fallback",
    "get_segments_with_fallback",
    "get_summaries_with_fallback",
    "upload_vlm_results_for_batch",
    "upload_segments_for_batch",
    "upload_summaries_for_batch",
    "upload_judge_result",
    "upsert_final_summary_results",
    "accumulate_segments_to_fusion",
]
