"""Supabase 클라이언트와 DB 어댑터 모듈 (Refactored with Mixins)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import (
    BaseAdapter,
    VideoAdapterMixin,
    CaptureAdapterMixin,
    ContentAdapterMixin,
)


class SupabaseAdapter(BaseAdapter, VideoAdapterMixin, CaptureAdapterMixin, ContentAdapterMixin):
    """Supabase 데이터베이스 통합 어댑터 클래스.
    
    이 클래스는 여러 Mixin을 상속받아 Supabase와의 모든 상호작용을 단일 인터페이스로 제공합니다.
    
    Backend Architecture:
        - 로컬 파이프라인과 원격 Supabase DB(PostgreSQL) 사이의 데이터 동기화를 담당합니다.
        - Singleton 패턴 사용을 권장하며, `get_supabase_adapter()` 팩토리 함수를 제공합니다.
    
    Modules (Mixins):
        1. BaseAdapter: 클라이언트 초기화 및 공통 유틸리티
        2. VideoAdapterMixin: Videos 테이블 및 Pipeline Runs 관리
        3. CaptureAdapterMixin: Captures 테이블 및 Storage 이미지 업로드
        4. ContentAdapterMixin: STT, Segments, Summaries 등 분석 콘텐츠 저장
    """
    
    def save_all_pipeline_results(
        self,
        video_id: str,
        video_root: Path,
        provider: str = "clova",
        pipeline_run_id: Optional[str] = None,
        include_preprocess: bool = True,
    ) -> Dict[str, Any]:
        """파이프라인 실행 완료 후 생성된 모든 결과물(JSON/JSONL)을 DB에 저장합니다.
        
        이 메서드는 파이프라인의 최종 단계에서 호출되어야 하며, 로컬 파일시스템에 생성된 
        분석 결과물을 관계형 DB 구조로 변환하여 순차적으로 업로드합니다.
        
        Process Flow:
            1. Captures: capture.json 읽어 'captures' 테이블 저장 + 이미지 Storage 업로드
            2. STT: stt.json 읽어 'stt_results' 테이블 저장
            3. Segments: segments_units.jsonl 읽어 'segments' 테이블 저장 (Fusion 결과)
            4. Summaries: segment_summaries.jsonl 읽어 'summaries' 테이블 저장 (LLM 요약)
            5. Status: 모든 작업 완료 후 성공/실패 여부에 따라 videos 테이블 상태 갱신
        
        Args:
            video_id: 대상 비디오의 UUID
            video_root: 분석 결과 파일들이 위치한 디렉터리 경로 (예: ./captures/video_name)
            provider: STT 엔진 이름 (기본값: clova)
            pipeline_run_id: 파이프라인 실행 이력 ID (선택)
            include_preprocess: 캡처/음성 결과까지 업로드할지 여부
            
        Returns:
            Dict: 저장된 레코드 수 및 발생한 에러 목록을 포함하는 요약 리포트
                {
                    "video_id": "...",
                    "saved": { "captures": 10, "stt_results": 5, ... },
                    "errors": [ "error message 1", ... ]
                }
        """
        results = {
            "video_id": video_id,
            "saved": {},
            "errors": [],
        }
        
        # 1. Captures (capture.json) - 화면 캡처 정보 및 이미지 업로드
        if include_preprocess:
            manifest_path = video_root / "capture.json"
            if manifest_path.exists():
                try:
                    captures_dir = video_root / "captures"
                    # 이미지 업로드와 메타데이터 저장을 동시에 수행
                    captures_result = self.save_captures_with_upload(video_id, manifest_path, captures_dir, pipeline_run_id=pipeline_run_id)
                    results["saved"]["captures"] = captures_result["db_saved"]
                    if captures_result["errors"]:
                         results["errors"].extend(captures_result["errors"])
                except Exception as e:
                    results["errors"].append(f"captures: {str(e)}")
        
        # 2. STT Results (stt.json) - 음성 인식 결과
        if include_preprocess:
            stt_path = video_root / "stt.json"
            if stt_path.exists():
                try:
                    stt = self.save_stt_from_file(video_id, stt_path, provider, pipeline_run_id)
                    results["saved"]["stt_results"] = len(stt) if stt else 0
                except Exception as e:
                    results["errors"].append(f"stt_results: {str(e)}")
        
        # 3. Segments (fusion/segments_units.jsonl) - 시각/음성 통합 세그먼트
        # Summaries 저장 시 FK 연결을 위해 로컬 인덱스와 DB UUID 매핑이 필요함
        segment_map = {}
        segments_path = video_root / "fusion" / "segments_units.jsonl"
        if segments_path.exists():
            try:
                segments = self.save_segments_from_file(video_id, segments_path, pipeline_run_id)
                results["saved"]["segments"] = len(segments)
                
                # Build ID Map (index -> uuid)
                for seg in segments:
                    idx = seg.get("segment_index")
                    uuid = seg.get("id")
                    if idx is not None and uuid:
                        segment_map[idx] = uuid
            except Exception as e:
                results["errors"].append(f"segments: {str(e)}")
        
        # 4. Summaries (fusion/segment_summaries.jsonl) - LLM 요약 결과
        # 위에서 생성한 segment_map을 사용하여 정확한 세그먼트와 연결
        summaries_path = video_root / "fusion" / "segment_summaries.jsonl"
        if summaries_path.exists():
            try:
                summaries = self.save_summaries_from_file(video_id, summaries_path, segment_map, pipeline_run_id)
                results["saved"]["summaries"] = len(summaries)
            except Exception as e:
                results["errors"].append(f"summaries: {str(e)}")
        
        # 5. 비디오 상태 업데이트 (Status Update)
        # 에러가 하나라도 있으면 'completed_with_errors', 아니면 'completed'로 마킹
        if results["errors"]:
            error_msg = "; ".join(results["errors"])
            self.update_video_status(video_id, "completed_with_errors", error=error_msg)
        else:
            self.update_video_status(video_id, "completed")
        
        return results


def get_supabase_adapter() -> Optional[SupabaseAdapter]:
    """Supabase 어댑터 인스턴스를 생성하는 팩토리 함수.
    
    환경변수(SUPABASE_URL, SUPABASE_KEY)를 확인하여 유효한 경우에만 인스턴스를 반환합니다.
    
    Returns:
        SupabaseAdapter: 초기화된 어댑터 인스턴스
        None: 환경변수가 없거나 패키지가 설치되지 않은 경우
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        return None
    
    try:
        return SupabaseAdapter(url, key)
    except ImportError:
        print("[WARN] supabase package not installed; DB features are disabled.")
        return None
