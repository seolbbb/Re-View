"""비디오 및 파이프라인 실행 관리 어댑터 모듈."""

from __future__ import annotations

from typing import Any, Dict, Optional


class VideoAdapterMixin:
    """Videos 및 pipeline_runs 테이블 작업을 위한 Mixin 클래스.
    
    이 클래스는 Video 모델과 관련된 CRUD(생성, 조회, 상태 업데이트) 기능과
    파이프라인 실행 이력(Pipeline Run)을 저장하는 기능을 제공합니다.
    SupabaseAdapter에 상속되어 사용됩니다.
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
            result = self.client.table("videos").select("*") \
                .eq("user_id", user_id) \
                .eq("original_filename", original_filename) \
                .execute()
            
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
        storage_path: Optional[str] = None,
        duration_sec: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """새 비디오 레코드를 생성합니다.
        
        Args:
            name: 비디오 표시 이름 (보통 파일명에서 확장자 제거)
            original_filename: 원본 파일명
            storage_path: 파일이 저장된 경로 (로컬 또는 원격)
            duration_sec: 비디오 길이 (초)
            user_id: 소유자 ID (Optional)
            
        Returns:
            Dict: 생성된 비디오 레코드 (자동 생성된 id 포함)
        """
        # 1. DB 삽입을 위한 데이터 객체 구성
        data = {
            "name": name,
            "original_filename": original_filename,
            "storage_path": storage_path,
            "duration_sec": duration_sec,
            "status": "uploaded",  # 초기 상태 (유효값: uploaded, processing, completed, completed_with_errors, failed)
        }
        if user_id:
            data["user_id"] = user_id
        
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
    
    def save_pipeline_run(
        self,
        video_id: str,
        run_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """파이프라인 실행 메타데이터를 저장합니다.
        
        분석 파이프라인이 실행될 때마다 그 실행 정보(시간, 버전, 설정값 등)를 기록하여
        데이터의 버전 관리 및 디버깅을 돕습니다.
        
        Args:
            video_id: 연관된 비디오 ID (FK)
            run_meta: 실행 메타데이터 JSON 객체 (timings, status, config 등 포함)
            
        Returns:
            Dict: 생성된 pipeline_run 레코드
        """
        data = {
            "video_id": video_id,
            "run_meta": run_meta,
        }
        result = self.client.table("pipeline_runs").insert(data).execute()
        return result.data[0] if result.data else {}

    def update_pipeline_run(
        self,
        pipeline_run_id: str,
        run_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """기존 pipeline_runs 레코드의 메타데이터를 갱신합니다."""
        data: Dict[str, Any] = {"run_meta": run_meta}
        status = run_meta.get("status") if isinstance(run_meta, dict) else None
        if status:
            data["status"] = status
        result = (
            self.client.table("pipeline_runs")
            .update(data)
            .eq("id", pipeline_run_id)
            .execute()
        )
        return result.data[0] if result.data else {}
