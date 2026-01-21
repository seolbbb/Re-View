"""시맨틱 검색 테스트 스크립트.

OpenRouter Qwen3-Embedding-8B 모델을 사용하여 임베딩을 생성하고,
Supabase의 vector similarity search (match_summaries)를 수행합니다.

사용법:
    python src/db/test_semantic_search.py "검색어"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client

from src.db.embedding import generate_embedding

# .env 로드
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def get_supabase_client():
    """Supabase 클라이언트 생성."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL 및 SUPABASE_KEY 환경변수가 필요합니다.")
    return create_client(url, key)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Supabase 시맨틱 검색 테스트")
    parser.add_argument("query", help="검색어")
    parser.add_argument("--user-id", help="필터링할 사용자 ID (UUID)", default=None)
    parser.add_argument("--threshold", type=float, default=0.4, help="유사도 임계값")
    parser.add_argument("--count", type=int, default=5, help="검색 결과 개수")
    
    args = parser.parse_args()
    
    search_semantic(
        args.query, 
        match_count=args.count, 
        threshold=args.threshold,
        filter_user_id=args.user_id
    )


def search_semantic(
    query: str, 
    match_count: int = 5, 
    threshold: float = 0.4,
    filter_user_id: str | None = None
):
    """시맨틱 검색 수행."""
    print(f"\n{'='*60}")
    print(f"[Semantic Search] '{query}'")
    if filter_user_id:
        print(f"👉 User Filter: {filter_user_id}")
    print(f"{'='*60}")
    
    # 1. 질문을 임베딩 벡터로 변환
    print("[1/2] Generating embedding...")
    query_embedding = generate_embedding(query)
    print(f"      OK - embedding generated (dim: {len(query_embedding)})")
    
    # 2. Supabase RPC 호출
    print(f"[2/2] Searching (threshold: {threshold})...")
    client = get_supabase_client()
    
    params = {
        'query_embedding': query_embedding,
        'match_threshold': threshold,
        'match_count': match_count,
    }
    if filter_user_id:
        params['filter_user_id'] = filter_user_id
        
    result = client.rpc('match_summaries', params).execute()
    
    if not result.data:
        print("      No results found")
        return []
    
    print(f"\n[Results] {len(result.data)} items\n")
    for i, row in enumerate(result.data, 1):
        similarity = row.get('similarity', 0)
        text = row.get('summary_text', '')[:150]
        # 줄바꿈 제거하여 깔끔하게 출력
        text = text.replace('\n', ' ')
        print(f"[{i}] Similarity: {similarity:.3f}")
        print(f"    {text}...")
        print()
    
    return result.data


if __name__ == "__main__":
    main()
