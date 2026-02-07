"""Supabase 및 Cloudflare R2 스토리지 어댑터 기본 클래스.

=============================================================================
모듈 목적 (Purpose)
=============================================================================
이 모듈은 데이터베이스(Supabase) 및 오브젝트 스토리지(Cloudflare R2) 클라이언트의
초기화 로직을 담당하는 기본 어댑터 클래스를 정의합니다.

=============================================================================
활용처 (Usage Context)
=============================================================================
- src/db/supabase_adapter.py → SupabaseAdapter가 이 클래스를 상속
- src/db/adapters/video_adapter.py → 비디오 업로드 시 R2 클라이언트 활용
- src/db/adapters/capture_adapter.py → 캡처/오디오 업로드 시 R2 클라이언트 활용
- src/pipeline/stages.py → 처리 파이프라인에서 signed URL 생성 시 활용

=============================================================================
환경 변수 (Required Environment Variables)
=============================================================================
[Supabase - 필수]
- SUPABASE_URL: Supabase 프로젝트 URL
- SUPABASE_KEY: Supabase anon 또는 service role key

[Cloudflare R2 - 선택]
- R2_ENDPOINT_URL: R2 S3 호환 엔드포인트 (format: https://<ACCOUNT_ID>.r2.cloudflarestorage.com)
- R2_ACCESS_KEY_ID: R2 API 토큰 Access Key
- R2_SECRET_ACCESS_KEY: R2 API 토큰 Secret Key
- R2_BUCKET: R2 버킷 이름 (default: review-storage, fallback: R2_BUCKET_VIDEOS)
- R2_PREFIX_VIDEOS: 비디오 파일 prefix (default: videos)
- R2_PREFIX_CAPTURES: 캡처 이미지 prefix (default: captures)
- R2_PREFIX_AUDIOS: 오디오 파일 prefix (default: audios)

=============================================================================
R2 스토리지 경로 구조 (Storage Path Structure)
=============================================================================
{video_id}/{R2_PREFIX_VIDEOS}/{filename}   → 비디오 원본
{video_id}/{R2_PREFIX_CAPTURES}/{filename} → 캡처 이미지
{video_id}/{R2_PREFIX_AUDIOS}/{filename}   → 추출된 오디오

=============================================================================
의존성 (Dependencies)
=============================================================================
- supabase (필수): pip install supabase
- boto3 (선택, R2용): pip install boto3
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Supabase Client Import
# -----------------------------------------------------------------------------
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

# -----------------------------------------------------------------------------
# Cloudflare R2 (boto3 S3-compatible) Import
# -----------------------------------------------------------------------------
try:
    import boto3
    from botocore.client import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

class BaseAdapter:
    """Supabase 데이터베이스 어댑터의 기본 클래스.
    
    클라이언트 초기화 및 공유 유틸리티 메서드를 제공합니다.
    Cloudflare R2 (S3 호환) 스토리지를 지원합니다.
    
    ==========================================================================
    사용 파일 (Used By)
    ==========================================================================
    - src/db/supabase_adapter.py: SupabaseAdapter 클래스가 이 클래스를 상속
    - 모든 Mixin 어댑터: VideoAdapterMixin, CaptureAdapterMixin 등
    
    ==========================================================================
    인스턴스 속성 (Instance Attributes)
    ==========================================================================
    [Supabase]
    client (Client): Supabase 클라이언트 인스턴스
    url (str): Supabase 프로젝트 URL
    key (str): Supabase API 키
    
    [Cloudflare R2]
    s3_client: boto3 S3 클라이언트 (R2 연결) 또는 None
    r2_bucket (str): R2 버킷 이름
    r2_prefix_videos (str): 비디오 파일 경로 prefix
    r2_prefix_captures (str): 캡처 이미지 경로 prefix
    r2_prefix_audios (str): 오디오 파일 경로 prefix
    
    NOTE: R2 자격 증명(endpoint, access_key, secret_key)은 보안상
          인스턴스 속성으로 저장하지 않습니다.
    
    ==========================================================================
    사용 예시 (Usage Example)
    ==========================================================================
    >>> from src.db.adapters.base import BaseAdapter
    >>> adapter = BaseAdapter()
    >>> if adapter.s3_client:
    ...     print("R2 연결됨")
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Supabase 및 R2 클라이언트를 초기화합니다.
        
        ======================================================================
        사용 파일 (Called By)
        ======================================================================
        - src/db/supabase_adapter.py → get_supabase_adapter() 팩토리 함수
        - src/db/__init__.py → 어댑터 싱글톤 생성
        
        ======================================================================
        연결 방식 (Connection)
        ======================================================================
        - Supabase: HTTPS REST API (supabase-py 라이브러리)
        - R2: S3 호환 API (boto3 라이브러리)
        
        Args:
            url (Optional[str]): Supabase 프로젝트 URL.
                지정하지 않으면 SUPABASE_URL 환경변수 사용.
            key (Optional[str]): Supabase anon/service key.
                지정하지 않으면 SUPABASE_KEY 환경변수 사용.
            
        Raises:
            ImportError: supabase 패키지가 설치되지 않은 경우
            ValueError: Supabase 연결 정보가 없는 경우
            
        Note:
            R2 설정이 없어도 에러를 발생시키지 않습니다.
            s3_client가 None이면 Supabase Storage fallback을 사용합니다.
        """
        # ---------------------------------------------------------------------
        # Supabase 패키지 확인
        # ---------------------------------------------------------------------
        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase 패키지가 설치되지 않았습니다. "
                "'pip install supabase' 명령어로 설치하세요."
            )
        
        # ---------------------------------------------------------------------
        # 1. Supabase 클라이언트 초기화
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 2. Cloudflare R2 클라이언트 초기화 (선택적)
        # ---------------------------------------------------------------------
        # R2 전용 모드: Supabase Storage fallback을 비활성화한다.
        # - "1", "true", "yes" => True
        self.r2_only = str(os.getenv("R2_ONLY_STORAGE", "1")).lower() in ("1", "true", "yes")
        self._init_r2_client()

    def _init_r2_client(self) -> None:
        """Cloudflare R2 S3 호환 클라이언트를 초기화합니다.
        
        ======================================================================
        내부 메서드 (Internal)
        ======================================================================
        __init__에서 호출되며, R2 환경변수가 설정된 경우에만
        boto3 S3 클라이언트를 생성합니다.
        
        설정되는 인스턴스 속성:
            s3_client: boto3 S3 클라이언트 또는 None
            r2_bucket: R2 버킷 이름
            r2_prefix_videos: 비디오 경로 prefix
            r2_prefix_captures: 캡처 경로 prefix
            r2_prefix_audios: 오디오 경로 prefix
        """
        if not BOTO3_AVAILABLE:
            self.s3_client = None
            print("Warning: 'boto3' not installed. R2 storage features will be unavailable.")
            return
            
        # R2 환경변수 로드 (로컬 변수로만 사용 - 보안상 인스턴스 속성 저장 X)
        r2_endpoint = os.getenv("R2_ENDPOINT_URL")
        r2_access_key = os.getenv("R2_ACCESS_KEY_ID")
        r2_secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        
        # R2 설정이 완전한 경우에만 클라이언트 초기화
        if r2_endpoint and r2_access_key and r2_secret_key:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                config=Config(signature_version='s3v4'),
                region_name='auto'  # R2는 region을 요구하지만 'auto' 사용 가능
            )
            
            # 버킷 및 경로 prefix 설정 (R2_BUCKET 우선, R2_BUCKET_VIDEOS fallback)
            self.r2_bucket = os.getenv("R2_BUCKET") or os.getenv("R2_BUCKET_VIDEOS", "review-storage")
            self.r2_prefix_videos = os.getenv("R2_PREFIX_VIDEOS", "videos")
            self.r2_prefix_captures = os.getenv("R2_PREFIX_CAPTURES", "captures")
            self.r2_prefix_audios = os.getenv("R2_PREFIX_AUDIOS", "audios")
        else:
            self.s3_client = None

    def _utc_now_iso(self) -> str:
        """현재 UTC 시간을 ISO 8601 형식 문자열로 반환합니다.
        
        Returns:
            str: ISO 8601 형식의 UTC 시간 (예: "2026-02-05T12:00:00+00:00")
        """
        return datetime.now(timezone.utc).isoformat()

    # ---------------------------------------------------------------------
    # Cloudflare R2 helpers
    # ---------------------------------------------------------------------
    def r2_delete_prefix(self, prefix: str) -> Dict[str, Any]:
        """Delete every object under a given key prefix in the R2 bucket.

        This is intentionally implemented as a prefix delete (list + batch delete)
        so callers can reliably remove all artifacts for a video_id.

        Returns:
            Dict with keys:
              - prefix: str
              - total: int (objects found)
              - deleted: int (objects deleted, best-effort)
              - errors: list (delete_objects errors)
        """
        if not getattr(self, "s3_client", None):
            if getattr(self, "r2_only", False):
                raise RuntimeError("R2 storage is required (check R2_* env vars)")
            return {"prefix": prefix, "total": 0, "deleted": 0, "errors": []}

        bucket = getattr(self, "r2_bucket", None)
        if not bucket:
            raise RuntimeError("R2 bucket is not configured (check R2_BUCKET/R2_BUCKET_VIDEOS)")

        keys: List[str] = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj.get("Key")
                if key:
                    keys.append(key)

        total = len(keys)
        deleted = 0
        errors: List[Dict[str, Any]] = []

        # S3-compatible delete_objects supports up to 1000 objects per call.
        for i in range(0, total, 1000):
            chunk = keys[i : i + 1000]
            objs = [{"Key": key} for key in chunk]
            try:
                resp = self.s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objs})
            except Exception as exc:
                errors.append({"Key": prefix, "Code": "DeleteObjectsFailed", "Message": str(exc)})
                continue

            deleted_items = resp.get("Deleted") or []
            resp_errors = resp.get("Errors") or []
            deleted += len(deleted_items) if deleted_items else max(0, len(objs) - len(resp_errors))
            errors.extend(resp_errors)

        return {"prefix": prefix, "total": total, "deleted": deleted, "errors": errors}
