"""Job 관리 어댑터 모듈 (preprocessing_jobs, processing_jobs)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_timestamp() -> str:
    """[YYYY-MM-DD | HH:MM:SS.mmm] 형식의 타임스탬프를 반환한다."""
    from datetime import datetime
    now = datetime.now()
    return f"[{now.strftime('%Y-%m-%d | %H:%M:%S')}.{now.strftime('%f')[:3]}]"


def compute_config_hash(config_paths: List[Path]) -> str:
    """설정 파일들의 해시를 계산합니다.

    Args:
        config_paths: 설정 파일 경로 리스트

    Returns:
        str: 설정 내용의 SHA256 해시 (처음 16자)
    """
    combined_content = ""
    for path in sorted(config_paths):  # 일관된 순서를 위해 정렬
        if path.exists():
            combined_content += f"---{path.name}---\n"
            combined_content += path.read_text(encoding="utf-8")
            combined_content += "\n"

    if not combined_content:
        return ""

    return hashlib.sha256(combined_content.encode("utf-8")).hexdigest()[:16]


class JobAdapterMixin:
    """preprocessing_jobs, processing_jobs 테이블 작업을 위한 Mixin 클래스.
    
    ERD 시퀀스 다이어그램에 따른 Job 기반 상태 관리를 지원합니다.
    """
    
    # =========================================================================
    # Preprocessing Jobs
    # =========================================================================
    
    def create_preprocessing_job(
        self,
        video_id: str,
        *,
        source: str = "SERVER",
        stt_backend: Optional[str] = None,
        config_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """새 전처리 작업 레코드를 생성합니다.
        
        Args:
            video_id: 대상 비디오 ID
            source: 전처리 소스 ('CLIENT' 또는 'SERVER')
            stt_backend: STT 백엔드 (예: 'clova', 'whisper')
            config_hash: 설정 해시 (중복 실행 방지용)
            
        Returns:
            Dict: 생성된 preprocessing_job 레코드
        """
        data = {
            "video_id": video_id,
            "source": source,
            "status": "QUEUED",
            "stt_backend": stt_backend,
            "config_hash": config_hash,
        }
        result = self.client.table("preprocessing_jobs").insert(data).execute()
        job = result.data[0] if result.data else {}
        
        # videos.current_preprocess_job_id 업데이트
        if job.get("id"):
            self.client.table("videos").update({
                "current_preprocess_job_id": job["id"],
                "status": "PREPROCESSING",
            }).eq("id", video_id).execute()
        
        return job
    
    def update_preprocessing_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """전처리 작업 상태를 업데이트합니다.
        
        Args:
            job_id: 작업 ID
            status: 새 상태 ('QUEUED', 'RUNNING', 'DONE', 'FAILED')
            error_message: 에러 메시지 (FAILED 시)
            
        Returns:
            Dict: 업데이트된 레코드
        """
        data: Dict[str, Any] = {"status": status}
        
        if status == "RUNNING":
            data["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in ("DONE", "FAILED"):
            data["ended_at"] = datetime.now(timezone.utc).isoformat()
            
        if error_message:
            data["error_message"] = error_message
            
        result = self.client.table("preprocessing_jobs").update(data).eq("id", job_id).execute()
        job = result.data[0] if result.data else {}
        
        # DONE/FAILED 시 videos 상태도 업데이트
        if job and status == "DONE":
            self.client.table("videos").update({
                "status": "PREPROCESS_DONE"
            }).eq("current_preprocess_job_id", job_id).execute()
        elif job and status == "FAILED":
            self.client.table("videos").update({
                "status": "FAILED",
                "error_message": error_message,
            }).eq("current_preprocess_job_id", job_id).execute()
        
        return job
    
    def get_preprocessing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """전처리 작업을 조회합니다."""
        result = self.client.table("preprocessing_jobs").select("*").eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    # =========================================================================
    # Processing Jobs
    # =========================================================================
    
    def create_processing_job(
        self,
        video_id: str,
        *,
        triggered_by: str = "MANUAL",
        config_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """새 처리 작업 레코드를 생성합니다.
        
        Args:
            video_id: 대상 비디오 ID
            triggered_by: 트리거 ('CHAT_OPEN', 'MANUAL', 'SCHEDULE')
            config_hash: 설정 해시
            
        Returns:
            Dict: 생성된 processing_job 레코드
        """
        # run_no 계산: 해당 video의 기존 job 수 + 1
        existing = self.client.table("processing_jobs").select("id").eq("video_id", video_id).execute()
        run_no = len(existing.data) + 1 if existing.data else 1
        
        data = {
            "video_id": video_id,
            "status": "QUEUED",
            "triggered_by": triggered_by,
            "run_no": run_no,
            "config_hash": config_hash,
            "current_batch": 0,
            "total_batch": 0,
        }
        result = self.client.table("processing_jobs").insert(data).execute()
        job = result.data[0] if result.data else {}

        # videos.current_processing_job_id 및 status 업데이트
        if job.get("id"):
            update_result = self.client.table("videos").update({
                "current_processing_job_id": job["id"],
                "status": "PROCESSING",  # PREPROCESS_DONE -> PROCESSING
            }).eq("id", video_id).execute()
            print(f"{_get_timestamp()} [DB] Updated videos.status to PROCESSING for video_id: {video_id}")

        return job
    
    def update_processing_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """처리 작업 상태를 업데이트합니다.
        
        Args:
            job_id: 작업 ID
            status: 새 상태 ('QUEUED', 'VLM_RUNNING', 'SUMMARY_RUNNING', 'JUDGE_RUNNING', 'DONE', 'FAILED')
            error_message: 에러 메시지 (FAILED 시)
            
        Returns:
            Dict: 업데이트된 레코드
        """
        data: Dict[str, Any] = {"status": status}
        
        if status == "VLM_RUNNING":
            data["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in ("DONE", "FAILED"):
            data["ended_at"] = datetime.now(timezone.utc).isoformat()
            
        if error_message:
            data["error_message"] = error_message
            
        result = self.client.table("processing_jobs").update(data).eq("id", job_id).execute()
        return result.data[0] if result.data else {}
    
    def update_processing_job_progress(
        self,
        job_id: str,
        current: int,
        total: Optional[int] = None,
    ) -> Dict[str, Any]:
        """처리 작업 진행률을 업데이트합니다.

        Args:
            job_id: 작업 ID
            current: 현재 배치 번호
            total: 전체 배치 수 (설정 시에만 업데이트)

        Returns:
            Dict: 업데이트된 레코드
        """
        data: Dict[str, Any] = {"current_batch": current}
        if total is not None:
            data["total_batch"] = total

        result = self.client.table("processing_jobs").update(data).eq("id", job_id).execute()
        return result.data[0] if result.data else {}

    def update_processing_job_batch_progress(
        self,
        job_id: str,
        current_batch: int,
        total_batches: int,
        current_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """배치+단계별 진행률을 업데이트합니다.

        Args:
            job_id: 작업 ID
            current_batch: 현재 배치 번호 (1-indexed)
            total_batches: 전체 배치 수
            current_stage: 현재 단계 (예: 'VLM', 'SYNC', 'SUMMARY', 'JUDGE')

        Returns:
            Dict: 업데이트된 레코드
        """
        data: Dict[str, Any] = {
            "current_batch": current_batch,
            "total_batch": total_batches,
        }

        # 상태도 함께 업데이트 (단계에 따라)
        if current_stage:
            stage_status_map = {
                "VLM": "VLM_RUNNING",
                "SYNC": "VLM_RUNNING",  # SYNC는 VLM과 함께 처리
                "SUMMARY": "SUMMARY_RUNNING",
                "JUDGE": "JUDGE_RUNNING",
            }
            if current_stage in stage_status_map:
                data["status"] = stage_status_map[current_stage]

        result = self.client.table("processing_jobs").update(data).eq("id", job_id).execute()
        return result.data[0] if result.data else {}
    
    def get_processing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """처리 작업을 조회합니다."""
        result = self.client.table("processing_jobs").select("*").eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    # =========================================================================
    # STT Results (개별 segment 단위)
    # =========================================================================
    
    def insert_stt_results_batch(
        self,
        video_id: str,
        stt_segments: List[Dict[str, Any]],
        *,
        preprocess_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """STT 결과를 배치로 저장합니다.
        
        Args:
            video_id: 비디오 ID
            stt_segments: STT 세그먼트 리스트
                [{"start_ms": 0, "end_ms": 1000, "transcript": "...", "confidence": 0.9}, ...]
            preprocess_job_id: 전처리 작업 ID
            
        Returns:
            List[Dict]: 저장된 레코드들
        """
        if not stt_segments:
            return []
            
        records = []
        for seg in stt_segments:
            records.append({
                "video_id": video_id,
                "preprocess_job_id": preprocess_job_id,
                "stt_id": seg.get("id") or seg.get("stt_id"),
                "start_ms": seg.get("start_ms"),
                "end_ms": seg.get("end_ms"),
                "transcript": seg.get("transcript") or seg.get("text"),
                "confidence": seg.get("confidence"),
            })
        
        result = self.client.table("stt_results").insert(records).execute()
        return result.data if result.data else []
    
    # =========================================================================
    # VLM Results
    # =========================================================================
    
    def insert_vlm_results(
        self,
        video_id: str,
        vlm_results: List[Dict[str, Any]],
        *,
        processing_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """VLM 분석 결과를 저장합니다.

        Args:
            video_id: 비디오 ID
            vlm_results: VLM 결과 리스트
                [{"id": "cap_001", "time_ranges": [...], "extracted_text": "..."}, ...]
            processing_job_id: 처리 작업 ID

        Returns:
            List[Dict]: 저장된 레코드들
        """
        if not vlm_results:
            return []

        if not vlm_results:
            return []

        # DB captures 조회하여 매핑 테이블 생성 (cap_id <-> uuid)
        # 1. capture_id(UUID)를 찾기 위한 매핑 (Source ID -> UUID)
        # 2. cap_id(Text)를 찾기 위한 매핑 (UUID -> Source ID)
        capture_map_to_uuid = {}
        capture_map_to_text = {}

        captures = self.client.table("captures").select("id,cap_id").eq("video_id", video_id).execute()
        if captures.data:
            for cap in captures.data:
                uuid = cap.get("id")
                text_id = cap.get("cap_id")
                if uuid:
                    capture_map_to_text[uuid] = text_id
                if text_id:
                    capture_map_to_uuid[text_id] = uuid

        records = []
        for vlm in vlm_results:
            # vlm.json 구조: {"id": "vlm_001", "cap_id": "...", "time_ranges": [...], "extracted_text": ...}
            
            source_val = vlm.get("cap_id")
            db_capture_id = vlm.get("capture_id")
            final_cap_id_text = None

            # Case A: 입력된 cap_id가 UUID 형태인 경우 (예: DIFFUSION_6)
            if source_val and len(str(source_val)) == 36 and "-" in str(source_val):
                if source_val in capture_map_to_text:
                    # 유효한(현재 DB에 존재하는) UUID인 경우
                    db_capture_id = source_val
                    final_cap_id_text = capture_map_to_text.get(source_val)
                else:
                    # DB에 없는(오래된) UUID인 경우 -> vlm_id로 추론 시도
                    # vlm_001 -> cap_001
                    vlm_item_id = vlm.get("id")
                    if vlm_item_id and vlm_item_id.startswith("vlm_"):
                        inferred_cap_id = vlm_item_id.replace("vlm_", "cap_")
                        if inferred_cap_id in capture_map_to_uuid:
                            db_capture_id = capture_map_to_uuid[inferred_cap_id]
                            final_cap_id_text = inferred_cap_id
            
            # Case B: 입력된 cap_id가 cap_001 형태인 경우 (예: sample2)
            elif source_val:
                final_cap_id_text = source_val
                db_capture_id = capture_map_to_uuid.get(source_val)
            
            # Case C: 이미 capture_id가 명시된 경우
            if vlm.get("capture_id"):
                 db_capture_id = vlm.get("capture_id")
                 if db_capture_id in capture_map_to_text:
                     final_cap_id_text = capture_map_to_text[db_capture_id]

            # 2. time_ranges 처리
            time_ranges = vlm.get("time_ranges")
            
            # 레거시 timestamp_ms 지원 (time_ranges가 없으면 생성)
            timestamp_ms = vlm.get("timestamp_ms")
            if not time_ranges and timestamp_ms is not None:
                time_ranges = [{"start_ms": timestamp_ms, "end_ms": timestamp_ms + 1000}]

            records.append({
                "video_id": video_id,
                "processing_job_id": processing_job_id,
                "capture_id": db_capture_id, # UUID (Foreign Key)
                "cap_id": final_cap_id_text or source_val, # Logical ID (Text)
                "time_ranges": time_ranges, 
                "extracted_text": vlm.get("extracted_text"),
            })

        result = self.client.table("vlm_results").insert(records).execute()
        return result.data if result.data else []
    
    # =========================================================================
    # Summary Results
    # =========================================================================
    
    def upsert_summary_results(
        self,
        video_id: str,
        format: str,
        payload: Dict[str, Any],
        *,
        processing_job_id: Optional[str] = None,
        status: str = "IN_PROGRESS",
    ) -> Dict[str, Any]:
        """요약 결과를 UPSERT합니다 (없으면 생성, 있으면 업데이트).
        
        Args:
            video_id: 비디오 ID
            format: 포맷 ('timeline' 또는 'tldr')
            payload: 요약 페이로드
            processing_job_id: 처리 작업 ID
            status: 상태 ('IN_PROGRESS' 또는 'DONE')
            
        Returns:
            Dict: UPSERT된 레코드
        """
        # 기존 레코드 확인
        existing = self.client.table("summary_results").select("id").eq("video_id", video_id).eq("format", format).execute()
        
        data = {
            "video_id": video_id,
            "format": format,
            "payload": payload,
            "status": status,
        }
        if processing_job_id:
            data["processing_job_id"] = processing_job_id
        
        if existing.data:
            # 업데이트
            result = self.client.table("summary_results").update(data).eq("id", existing.data[0]["id"]).execute()
        else:
            # 신규 생성
            result = self.client.table("summary_results").insert(data).execute()
        
        summary_result = result.data[0] if result.data else {}
        
        # DONE 상태일 때 videos.current_summary_result_id 업데이트
        if status == "DONE" and summary_result.get("id"):
            self.client.table("videos").update({
                "current_summary_result_id": summary_result["id"],
            }).eq("id", video_id).execute()
        
        return summary_result
    
    def get_latest_summary_results(
        self,
        video_id: str,
        format: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """최신 요약 결과를 조회합니다.
        
        Args:
            video_id: 비디오 ID
            format: 포맷 필터 (선택)
            
        Returns:
            Dict: 최신 summary_results 레코드 or None
        """
        query = self.client.table("summary_results").select("*").eq("video_id", video_id)
        if format:
            query = query.eq("format", format)
        query = query.order("created_at", desc=True).limit(1)
        
        result = query.execute()
        result = query.execute()
        return result.data[0] if result.data else None
    
    def get_summaries(
        self,
        video_id: str,
        *,
        processing_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """비디오의 모든 요약 결과(세그먼트 포함)를 조회합니다.
        
        Args:
            video_id: 비디오 ID
            processing_job_id: 특정 작업 ID (선택)
            
        Returns:
            List[Dict]: 요약 리스트 (segments 정보 포함/평탄화 필요 시 호출처 처리)
        """
        query = self.client.table("summaries").select(
            "*, segments(*)"
        ).eq("video_id", video_id)
        
        if processing_job_id:
            query = query.eq("processing_job_id", processing_job_id)
            
        # 세그먼트 순서대로 정렬 (segment_index 기준)
        # Note: segments가 nested object라 직접 정렬이 어려울 수 있으므로
        # created_at 또는 fetch 후 정렬 권장. 여기서는 일단 summaries 생성순.
        query = query.order("created_at", desc=False)
        
        result = query.execute()
        return result.data if result.data else []
    
    # =========================================================================
    # Judge
    # =========================================================================
    
    def insert_judge(
        self,
        video_id: str,
        score: float,
        report: Dict[str, Any],
        *,
        processing_job_id: Optional[str] = None,
        status: str = "DONE",
        batch_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """품질 평가 결과를 저장합니다.

        Args:
            video_id: 비디오 ID
            score: 평가 점수
            report: 평가 리포트 (JSONB)
            processing_job_id: 처리 작업 ID
            status: 상태 ('DONE' 또는 'FAILED')
            batch_index: 배치 인덱스 (1-indexed)

        Returns:
            Dict: 저장된 레코드
        """
        data = {
            "video_id": video_id,
            "processing_job_id": processing_job_id,
            "status": status,
            "score": score,
            "report": report,
        }

        if batch_index is not None:
            data["batch_index"] = batch_index

        result = self.client.table("judge").insert(data).execute()
        return result.data[0] if result.data else {}
