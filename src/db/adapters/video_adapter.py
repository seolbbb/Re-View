"""비디오 및 파이프라인 실행 관리 어댑터 모듈."""

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
        data = {"status": status}
        if error:
            data["error_message"] = error
        
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

    def upload_video(
        self,
        video_id: str,
        video_path: Path,
        bucket: str = "videos",
    ) -> Dict[str, Any]:
        """비디오 원본 파일을 Supabase Storage에 업로드하고 DB를 업데이트합니다.

        Args:
            video_id: 대상 비디오 UUID
            video_path: 업로드할 로컬 비디오 파일 경로
            bucket: 타겟 Storage 버킷 이름 (기본값: 'videos')

        Returns:
            Dict: 업로드 결과 및 업데이트된 스토리지 키
        """
        if not video_path.exists():
             raise FileNotFoundError(f"Video file not found: {video_path}")

        file_name = video_path.name
        # 스토리지 경로 구조: {video_id}/{filename}
        storage_path = f"{video_id}/{file_name}"
        
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = "video/mp4"

        # 1. Storage에 파일 업로드
        with open(video_path, "rb") as f:
            # 큰 파일일 수 있으므로 그대로 넘김 (supabase-py가 내부에서 처리)
            self.client.storage.from_(bucket).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": mime_type, "upsert": "true"}
            )
        
        # 2. DB 업데이트
        updated_video = self.update_video_storage_key(video_id, storage_path)
        
        return {
            "video_id": video_id,
            "storage_path": storage_path,
            "updated_video": updated_video
        }

    def delete_video(self, video_id: str) -> bool:
        """비디오 레코드를 삭제합니다 (CASCADE로 관련 데이터 자동 삭제).

        Args:
            video_id: 삭제할 비디오의 UUID

        Returns:
            bool: 삭제 성공 여부
        """
        result = self.client.table("videos").delete().eq("id", video_id).execute()
        return len(result.data) > 0

    # NOTE: pipeline_runs 테이블은 더 이상 사용되지 않습니다.
    # 대신 preprocessing_jobs/processing_jobs 테이블을 사용하세요.
    # 관련 함수는 job_adapter.py에서 제공됩니다.
