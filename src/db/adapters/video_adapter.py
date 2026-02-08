"""비디오 및 파이프라인 실행 관리 어댑터 모듈.

=============================================================================
모듈 목적 (Purpose)
=============================================================================
이 모듈은 videos 테이블에 대한 CRUD 작업과 비디오 파일의 스토리지 업로드를
담당하는 Mixin 클래스를 정의합니다. Cloudflare R2와 Supabase Storage 모두 지원합니다.

=============================================================================
활용처 (Usage Context)
=============================================================================
- src/db/supabase_adapter.py → SupabaseAdapter가 이 Mixin을 상속
- src/process_api.py → /api/videos/upload 엔드포인트에서 upload_video() 호출
- src/run_preprocess_pipeline.py → 전처리 파이프라인에서 비디오 레코드 생성/업데이트

=============================================================================
API 엔드포인트 연결 (Connected API Endpoints)
=============================================================================
- POST /api/videos/upload → upload_video() 호출
- GET /api/videos → list_videos() 호출
- GET /api/videos/{video_id}/status → get_video() 호출

=============================================================================
R2 스토리지 경로 구조 (Storage Path Structure)
=============================================================================
R2 사용 시: {video_id}/{R2_PREFIX_VIDEOS}/{timestamp}_{filename}
Supabase Fallback: {video_id}/{timestamp}_{filename}

예시:
  - R2: 92f1750b-08df-47e3/videos/20260205120000_sample.mp4
  - Supabase: 92f1750b-08df-47e3/20260205120000_sample.mp4

=============================================================================
의존성 (Dependencies)
=============================================================================
- src/db/adapters/base.py: BaseAdapter (s3_client, r2_bucket, r2_prefix_videos 제공)
- mimetypes: MIME 타입 추론
- pathlib: 파일 경로 처리
"""

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional


class VideoAdapterMixin:
    """Videos 테이블 작업을 위한 Mixin 클래스.
    
    이 클래스는 Video 모델과 관련된 CRUD(생성, 조회, 상태 업데이트) 기능을 제공합니다.
    SupabaseAdapter에 상속되어 사용됩니다.
    
    NOTE: 파이프라인 실행 이력은 preprocessing_jobs/processing_jobs 테이블에서 관리됩니다.
    """
    
    def get_video_by_filename(
        self,
        user_id: str,
        original_filename: str
    ) -> Optional[Dict[str, Any]]:
        """사용자 ID와 원본 파일명으로 기존 비디오를 조회합니다.
        
        파이프라인 재실행 등의 시나리오에서 중복 비디오 생성을 방지하기 위해 사용됩니다.
        
        Args:
            user_id: 비디오 소유자 ID (UUID)
            original_filename: 업로드된 원본 파일명 (예: sample.mp4)
            
        Returns:
            Dict: 비디오 레코드 정보 (찾은 경우)
            None: 해당 조건의 비디오가 없는 경우
        """
        try:
            # supabase-py 클라이언트를 통해 videos 테이블 조회
            query = self.client.table("videos").select("*").eq("original_filename", original_filename)
            
            if user_id:
                query = query.eq("user_id", user_id)
                
            result = query.execute()
            
            # 조회 결과가 있으면 첫 번째 레코드 반환
            if result.data:
                return result.data[0]
            return None
        except Exception:
            # 조회 중 에러 발생 시(예: 테이블 없음, 권한 문제 등) None 반환하여 안전하게 처리
            return None

    def create_video(
        self,
        name: str,
        original_filename: str,
        duration_sec: Optional[int] = None,
        user_id: Optional[str] = None,
        video_storage_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """새 비디오 레코드를 생성합니다.

        Args:
            name: 비디오 표시 이름 (보통 파일명에서 확장자 제거)
            original_filename: 원본 파일명
            duration_sec: 비디오 길이 (초)
            user_id: 소유자 ID (Optional)
            video_storage_key: Supabase Storage 내 경로 (Optional)

        Returns:
            Dict: 생성된 비디오 레코드 (자동 생성된 id 포함)
        """
        # 1. DB 삽입을 위한 데이터 객체 구성
        data = {
            "name": name,
            "original_filename": original_filename,
            "duration_sec": duration_sec,
            "status": "UPLOADED",  # 초기 상태 (유효값: UPLOADED, PREPROCESSING, PREPROCESS_DONE, FAILED)
        }
        if user_id:
            data["user_id"] = user_id
        if video_storage_key:
            data["video_storage_key"] = video_storage_key

        # 2. videos 테이블에 insert 실행
        result = self.client.table("videos").insert(data).execute()
        video = result.data[0]

        # 3. 현재 작업 중인 비디오 ID 캐싱 (인스턴스 상태 관리)
        self._current_video_id = video["id"]

        return video
    
    def update_video_status(
        self,
        video_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """비디오의 처리 상태를 업데이트합니다.
        
        Args:
            video_id: 대상 비디오 ID
            status: 변경할 상태값 ('processing', 'completed', 'completed_with_errors' 등)
            error: 에러 메시지 (status가 실패 관련일 경우 상세 내용 기록)
            
        Returns:
            Dict: 업데이트된 비디오 레코드 정보를 반환
        """
        # 1. 업데이트할 데이터 준비
        #
        # UX/streaming consistency: FAILED 상태를 벗어나는 순간 기존 error_message를
        # 지워야 프론트가 "실패 배너"를 계속 유지하지 않습니다.
        status_upper = (status or "").upper()
        data: Dict[str, Any] = {"status": status}
        if status_upper == "FAILED":
            # 명시적으로 에러가 주어진 경우에만 overwrite (그 외에는 기존 메시지 유지)
            if error is not None:
                data["error_message"] = error
        else:
            data["error_message"] = None
        
        # 2. 업데이트 실행 (ID 기준)
        result = self.client.table("videos").update(data).eq("id", video_id).execute()
        return result.data[0] if result.data else {}
    
    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """ID로 비디오 정보를 단건 조회합니다.
        
        Args:
            video_id: 조회할 비디오의 UUID
            
        Returns:
            Dict: 비디오 레코드 정보 혹은 None
        """
        result = self.client.table("videos").select("*").eq("id", video_id).execute()
        return result.data[0] if result.data else None
    
    def list_videos(self) -> list:
        """전체 비디오 목록을 최신순으로 조회합니다.

        Returns:
            List[Dict]: 비디오 레코드 리스트 (created_at 내림차순)
        """
        try:
            result = (
                self.client.table("videos")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception:
            return []

    def list_videos_for_user(self, user_id: str) -> list:
        """List videos for a specific user (newest first)."""
        if not user_id:
            return []
        try:
            result = (
                self.client.table("videos")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception:
            return []

    def update_video_storage_key(
        self,
        video_id: str,
        storage_key: str,
    ) -> Dict[str, Any]:
        """비디오의 Storage 경로를 업데이트합니다.

        Args:
            video_id: 대상 비디오 ID
            storage_key: Supabase Storage 내 경로

        Returns:
            Dict: 업데이트된 비디오 레코드
        """
        result = (
            self.client.table("videos")
            .update({"video_storage_key": storage_key})
            .eq("id", video_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    def delete_video(self, video_id: str, user_id: str) -> bool:
        """Delete a video row if it belongs to the given user.

        This deletes only the DB row. Storage cleanup must be handled separately.
        """
        if not video_id or not user_id:
            return False
        try:
            (
                self.client.table("videos")
                .delete()
                .eq("id", video_id)
                .eq("user_id", user_id)
                .execute()
            )
            # Some PostgREST configurations may return no representation for DELETE.
            # Treat "no exception" as success; callers already verify ownership.
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Deletion cancellation marker
    # ---------------------------------------------------------------------

    def mark_video_delete_requested(self, video_id: str, *, when_iso: Optional[str] = None) -> bool:
        """Mark a video as delete-requested (best-effort).

        This is used as a cross-instance cancellation signal for long-running pipelines.
        The DB schema must include videos.delete_requested_at.
        """
        if not video_id:
            return False
        if not when_iso:
            from datetime import datetime, timezone

            when_iso = datetime.now(timezone.utc).isoformat()
        try:
            (
                self.client.table("videos")
                .update({"delete_requested_at": when_iso})
                .eq("id", video_id)
                .execute()
            )
            return True
        except Exception:
            return False

    def clear_video_delete_requested(self, video_id: str) -> bool:
        """Clear delete-requested marker (best-effort)."""
        if not video_id:
            return False
        try:
            (
                self.client.table("videos")
                .update({"delete_requested_at": None})
                .eq("id", video_id)
                .execute()
            )
            return True
        except Exception:
            return False

    def upload_video(
        self,
        video_id: str,
        video_path: Path,
        bucket: str = "videos",
    ) -> Dict[str, Any]:
        """비디오 원본 파일을 스토리지에 업로드하고 DB를 업데이트합니다.
        
        ======================================================================
        사용 파일 (Called By)
        ======================================================================
        - src/process_api.py → upload_video() 엔드포인트 (POST /api/videos/upload)
        - src/run_preprocess_pipeline.py → 전처리 파이프라인에서 호출
        
        ======================================================================
        연결 방식 (Connection)
        ======================================================================
        - R2 활성화 시: boto3 S3 클라이언트로 Cloudflare R2에 업로드
        - R2 비활성화 시: Supabase Storage API로 업로드 (Fallback)
        
        ======================================================================
        스토리지 경로 (Storage Path)
        ======================================================================
        R2 사용:     {video_id}/{R2_PREFIX_VIDEOS}/{filename}
                     예: 92f1750b/videos/20260205_sample.mp4
        
        Supabase:    {video_id}/{filename}
                     예: 92f1750b/20260205_sample.mp4

        Args:
            video_id (str): 대상 비디오 UUID.
                videos 테이블의 id 컬럼 값.
            video_path (Path): 업로드할 로컬 비디오 파일 경로.
                파일이 존재하지 않으면 FileNotFoundError 발생.
            bucket (str, optional): Supabase Storage 버킷 이름.
                R2 사용 시 무시됨. Defaults to "videos".

        Returns:
            Dict[str, Any]: 업로드 결과
                - video_id: 비디오 UUID
                - storage_path: 저장된 경로
                - updated_video: DB 업데이트 결과
                
        Raises:
            FileNotFoundError: video_path 파일이 존재하지 않는 경우
            Exception: R2 또는 Supabase 업로드 실패 시
            
        Note:
            업로드 성공 시 videos.video_storage_key 컬럼이 자동 업데이트됩니다.
        """
        # =====================================================================
        # 1. 파일 존재 확인 및 MIME 타입 추론
        # =====================================================================
        if not video_path.exists():
             raise FileNotFoundError(f"Video file not found: {video_path}")

        file_name = video_path.name
        
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = "video/mp4"

        # =====================================================================
        # 2. 스토리지 업로드 (R2 우선, Supabase Fallback)
        # =====================================================================
        if self.s3_client:
            # R2 Upload: {video_id}/{prefix}/{filename}
            storage_path = f"{video_id}/{self.r2_prefix_videos}/{file_name}"
            try:
                self.s3_client.upload_file(
                    str(video_path),
                    self.r2_bucket,
                    storage_path,
                    ExtraArgs={"ContentType": mime_type}
                )
                print(f"[R2] Uploaded video to {self.r2_bucket}/{storage_path}")
            except Exception as e:
                print(f"[R2] Upload failed: {e}")
                raise e
        else:
            if getattr(self, "r2_only", False):
                raise RuntimeError("R2 storage is required (set R2_* env vars).")
            # Supabase Fallback: {video_id}/{filename}
            storage_path = f"{video_id}/{file_name}"
            with open(video_path, "rb") as f:
                self.client.storage.from_(bucket).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": mime_type, "upsert": "true"}
                )
        
        # =====================================================================
        # 3. DB 업데이트 (videos.video_storage_key)
        # =====================================================================
        updated_video = self.update_video_storage_key(video_id, storage_path)
        
        return {
            "video_id": video_id,
            "storage_path": storage_path,
            "updated_video": updated_video
        }

    # NOTE: pipeline_runs 테이블은 더 이상 사용되지 않습니다.
    # 대신 preprocessing_jobs/processing_jobs 테이블을 사용하세요.
    # 관련 함수는 job_adapter.py에서 제공됩니다.
