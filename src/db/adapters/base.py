"""Base adapter with initialization and shared utilities."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None


class BaseAdapter:
    """Base class for Supabase database adapters.
    
    Handles client initialization and provides shared utility methods.
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client.
        
        Args:
            url: Supabase project URL (optional, defaults to SUPABASE_URL env var)
            key: Supabase anon/service key (optional, defaults to SUPABASE_KEY env var)
            
        Raises:
            ImportError: If supabase package is not installed
            ValueError: If connection credentials are not provided
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase 패키지가 설치되지 않았습니다. "
                "'pip install supabase' 명령어로 설치하세요."
            )
        
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL과 SUPABASE_KEY 환경변수가 필요합니다. "
                ".env 파일에 설정하거나 환경변수로 지정하세요."
            )
        
        # Supabase 클라이언트 생성 (Lazy connection: 실제 요청 시 접속)
        self.client: Client = create_client(self.url, self.key)
        self._current_video_id: Optional[str] = None
    
    def _utc_now_iso(self) -> str:
        """현재 UTC 시간을 ISO 형식으로 반환."""
        return datetime.now(timezone.utc).isoformat()
