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
    print("[WARN] .env 파일이 없습니다. 환경변수를 직접 설정하세요.")


def test_connection():
    """Supabase 연결 테스트."""
    print("=" * 60)
    print("1. Supabase 연결 테스트")
    print("=" * 60)
    
    from src.db.supabase_adapter import get_supabase_adapter
    
    db = get_supabase_adapter()
    if db is None:
        print("[FAIL] Supabase 연결 실패!")
        print("  - SUPABASE_URL과 SUPABASE_KEY 환경변수를 확인하세요.")
        return None
    
    print(f"[OK] Supabase 연결 성공!")
    print(f"  - URL: {db.url[:50]}...")
    return db


def test_video_insert(db):
    """비디오 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("2. videos 테이블 삽입 테스트")
    print("=" * 60)
    
    try:
        video = db.create_video(
            name="test_video",
            original_filename="test.mp4",
            storage_path="/local/path/test.mp4",
        )
        print(f"[OK] 비디오 생성 성공!")
        print(f"  - ID: {video['id']}")
        print(f"  - Name: {video['name']}")
        print(f"  - Status: {video['status']}")
        return video["id"]
    except Exception as e:
        print(f"[FAIL] 비디오 생성 실패: {e}")
        return None


def test_stt_insert(db, video_id):
    """STT 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("3. stt_results 테이블 삽입 테스트")
    print("=" * 60)
    
    try:
        test_segments = [
            {"start_ms": 0, "end_ms": 5000, "text": "테스트 문장 1", "confidence": 0.95},
            {"start_ms": 5000, "end_ms": 10000, "text": "테스트 문장 2", "confidence": 0.92},
        ]
        result = db.save_stt_result(video_id, test_segments, provider="test")
        print(f"[OK] STT 결과 저장 성공!")
        print(f"  - ID: {result.get('id')}")
        return True
    except Exception as e:
        print(f"[FAIL] STT 결과 저장 실패: {e}")
        return False


def test_captures_insert(db, video_id):
    """Captures 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("4. captures 테이블 삽입 테스트")
    print("=" * 60)
    
    try:
        test_captures = [
            {"file_name": "frame_001.jpg", "timestamp_ms": 1000, "timestamp_human": "00h00m01s"},
            {"file_name": "frame_002.jpg", "timestamp_ms": 2000, "timestamp_human": "00h00m02s"},
        ]
        results = db.save_captures(video_id, test_captures)
        print(f"[OK] 캡처 저장 성공!")
        print(f"  - 저장된 개수: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] 캡처 저장 실패: {e}")
        return False


def test_segments_insert(db, video_id):
    """Segments 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("5. segments 테이블 삽입 테스트")
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
        print(f"[OK] 세그먼트 저장 성공!")
        print(f"  - 저장된 개수: {len(results)}")
        
        # UUID 매핑 생성해서 반환
        segment_map = {}
        for r in results:
            idx = r.get("segment_index")
            uid = r.get("id")
            if idx is not None and uid:
                segment_map[idx] = uid
        return segment_map
    except Exception as e:
        print(f"[FAIL] 세그먼트 저장 실패: {e}")
        return False


def test_summaries_insert(db, video_id, segment_map):
    """Summaries 테이블 삽입 테스트."""
    print("\n" + "=" * 60)
    print("6. summaries 테이블 삽입 테스트")
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
        print(f"[OK] 요약 저장 성공!")
        print(f"  - 저장된 개수: {len(results)}")
        return True
    except Exception as e:
        print(f"[FAIL] 요약 저장 실패: {e}")
        return False


def cleanup_test_data(db, video_id):
    """테스트 데이터 정리."""
    print("\n" + "=" * 60)
    print("7. 테스트 데이터 정리")
    print("=" * 60)
    
    try:
        # CASCADE 설정으로 videos 삭제 시 연관 데이터도 삭제됨
        db.client.table("videos").delete().eq("id", video_id).execute()
        print(f"[OK] 테스트 데이터 삭제 완료 (video_id: {video_id})")
        return True
    except Exception as e:
        print(f"[WARN] 테스트 데이터 삭제 실패: {e}")
        return False


def main():
    print("\n[START] Supabase 통합 테스트 시작\n")
    
    # 1. 연결 테스트
    db = test_connection()
    if not db:
        return
    
    # 2. 비디오 삽입
    video_id = test_video_insert(db)
    if not video_id:
        return
    
    # 3. 하위 테이블 테스트
    test_stt_insert(db, video_id)
    test_captures_insert(db, video_id)
    segment_map = test_segments_insert(db, video_id)
    if not segment_map:
        return
        
    test_summaries_insert(db, video_id, segment_map)
    
    # 4. 정리 여부 확인
    print("\n" + "=" * 60)
    cleanup = input("테스트 데이터를 삭제할까요? (y/n): ").strip().lower()
    if cleanup == "y":
        cleanup_test_data(db, video_id)
    else:
        print(f"[INFO] 테스트 데이터 유지됨 (video_id: {video_id})")
    
    print("\n[DONE] 테스트 완료!")


if __name__ == "__main__":
    main()
