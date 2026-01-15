"""Supabase 연결 및 데이터 삽입 테스트 스크립트."""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# .env 로드
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print("[WARN] .env not found. Set environment variables manually.")


def test_connection():
    """Supabase 연결 테스트."""
    print("=" * 60)
    print("1. Supabase connection test")
    print("=" * 60)
    
    from src.db.supabase_adapter import get_supabase_adapter
    
    db = get_supabase_adapter()
    if db is None:
        print("[FAIL] Supabase connection failed!")
        print("  - Check SUPABASE_URL and SUPABASE_KEY environment variables.")
        return None
    
    print("[OK] Supabase connection successful!")
    print(f"  - URL: {db.url[:50]}...")
    return db


def test_video_insert(db):
    """비디오 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("2. videos table insert test")
    print("=" * 60)
    
    try:
        video = db.create_video(
            name="test_video",
            original_filename="test.mp4",
            storage_path="/local/path/test.mp4",
        )
        print("[OK] Video created successfully!")
        print(f"  - ID: {video['id']}")
        print(f"  - Name: {video['name']}")
        print(f"  - Status: {video['status']}")
        return video["id"]
    except Exception as e:
        print(f"[FAIL] Video creation failed: {e}")
        return None


def test_stt_insert(db, video_id):
    """STT 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("3. stt_results table insert test")
    print("=" * 60)
    
    try:
        test_segments = [
            {"start_ms": 0, "end_ms": 5000, "text": "테스트 문장 1", "confidence": 0.95},
            {"start_ms": 5000, "end_ms": 10000, "text": "테스트 문장 2", "confidence": 0.92},
        ]
        result = db.save_stt_result(video_id, test_segments, provider="test")
        print("[OK] STT results saved successfully!")
        print(f"  - ID: {result.get('id')}")
        return True
    except Exception as e:
        print(f"[FAIL] STT results save failed: {e}")
        return False


def test_captures_insert(db, video_id):
    """Captures 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("4. captures table insert test")
    print("=" * 60)
    
    try:
        test_captures = [
            {"file_name": "frame_001.jpg", "timestamp_ms": 1000, "timestamp_human": "00h00m01s"},
            {"file_name": "frame_002.jpg", "timestamp_ms": 2000, "timestamp_human": "00h00m02s"},
        ]
        results = db.save_captures(video_id, test_captures)
        print("[OK] Captures saved successfully!")
        print(f"  - Saved count: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] Captures save failed: {e}")
        return False


def test_segments_insert(db, video_id):
    """Segments 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("5. segments table insert test")
    print("=" * 60)
    
    try:
        test_segments = [
            {
                "segment_id": 1,
                "start_ms": 0,
                "end_ms": 30000,
                "transcript_units": [{"unit_id": "t1", "text": "테스트"}],
                "visual_units": [{"unit_id": "v1", "text": "슬라이드"}],
            }
        ]
        results = db.save_segments(video_id, test_segments)
        print("[OK] Segments saved successfully!")
        print(f"  - Saved count: {len(results)}")
        
        # UUID 매핑 생성해서 반환
        segment_map = {}
        for r in results:
            idx = r.get("segment_index")
            uid = r.get("id")
            if idx is not None and uid:
                segment_map[idx] = uid
        return segment_map
    except Exception as e:
        print(f"[FAIL] Segments save failed: {e}")
        return False


def test_summaries_insert(db, video_id, segment_map):
    """Summaries 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("6. summaries table insert test")
    print("=" * 60)
    
    try:
        test_summaries = [
            {
                "segment_id": 1,
                # "start_ms": 0, # Removed from schema
                # "end_ms": 30000,
                "summary": {
                    "bullets": [{"claim": "테스트 요약 포인트"}],
                    "definitions": [{"term": "테스트", "definition": "검증"}],
                },
                "version": {"llm_model_id": "test-model"},
            }
        ]
        results = db.save_summaries(video_id, test_summaries, segment_map)
        print("[OK] Summaries saved successfully!")
        print(f"  - Saved count: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] Summaries save failed: {e}")
        return False


def cleanup_test_data(db, video_id):
    """테스트 데이터 정리."""
    print("\n" + "=" * 60)
    print("7. Test data cleanup")
    print("=" * 60)
    
    try:
        # CASCADE 설정으로 videos 삭제 시 연관 데이터도 삭제됨
        db.client.table("videos").delete().eq("id", video_id).execute()
        print(f"[OK] Test data deleted (video_id: {video_id})")
        return True
    except Exception as e:
        print(f"[WARN] Test data deletion failed: {e}")
        return False


def main():
    print("\n[START] Supabase integration test\n")
    
    # 1. 연결 테스트
    db = test_connection()
    if not db:
        return
    
    # 2. Video insert
    video_id = test_video_insert(db)
    if not video_id:
        return
    
    # 3. Child table tests
    test_stt_insert(db, video_id)
    test_captures_insert(db, video_id)
    segment_map = test_segments_insert(db, video_id)
    if not segment_map:
        return
        
    test_summaries_insert(db, video_id, segment_map)
    
    # 4. Cleanup prompt
    print("\n" + "=" * 60)
    cleanup = input("Delete test data? (y/n): ").strip().lower()
    if cleanup == "y":
        cleanup_test_data(db, video_id)
    else:
        print(f"[INFO] Test data kept (video_id: {video_id})")
    
    print("\n[DONE] Test complete!")


if __name__ == "__main__":
    main()
