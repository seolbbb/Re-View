"""콘텐츠 관리 어댑터 모듈 (STT, Segments, Summaries)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.fusion.renderer import render_summary_to_text
from src.db.embedding import generate_embedding, generate_embeddings_batch

logger = logging.getLogger(__name__)

class ContentAdapterMixin:
    """STT, Segments, Summaries 테이블 작업을 위한 Mixin 클래스.
    
    비디오 분석 결과의 핵심 콘텐츠를 DB에 저장합니다.
    1. STT Results: 음성 인식 결과 (평문 텍스트)
    2. Segments: 시각/청각적 의미 단위로 통합된 비디오 구간
    3. Summaries: 각 세그먼트 또는 전체 비디오에 대한 LLM 요약
    """
    
    # =========================================================================
    # STT Results
    # =========================================================================
    
    def save_stt_result(
        self,
        video_id: str,
        segments: List[Dict[str, Any]],
        preprocess_job_id: Optional[str] = None,
        provider: str = "clova"
    ) -> List[Dict[str, Any]]:
        """STT 실행 결과를 DB에 저장합니다.

        JSON 계층 구조를 Flattening하여 관계형 데이터베이스 테이블(stt_results)에 저장합니다.

        Args:
            video_id: 비디오 ID (FK)
            segments: STT 엔진 출력 세그먼트 리스트 (text, start, end, id 등 포함)
            preprocess_job_id: 전처리 작업 ID (ERD 기준)
            provider: STT 엔진 이름 (예: 'openai', 'clova')

        Returns:
            List[Dict]: DB에 저장된 레코드 리스트
        """
        rows = []
        for idx, seg in enumerate(segments):
            rows.append({
                "video_id": video_id,
                "preprocess_job_id": preprocess_job_id,
                "stt_id": seg.get("id"),  # stt.json의 id 필드 (e.g., "stt_001")
                "transcript": seg.get("text", ""),  # 스키마는 transcript 컬럼 사용
                "start_ms": seg.get("start_ms"),
                "end_ms": seg.get("end_ms"),
                "confidence": seg.get("confidence"),
            })

        if rows:
            # 대량 삽입 (Batch Insert)
            result = self.client.table("stt_results").insert(rows).execute()
            return result.data
        return []
    
    def save_stt_from_file(
        self,
        video_id: str,
        stt_json_path: Path,
        provider: str = "clova",
        preprocess_job_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """로컬 stt.json 파일에서 데이터를 읽어 저장합니다.
        
        Args:
            video_id: 비디오 ID
            stt_json_path: stt.json 파일 경로
            provider: STT 엔진 이름
            preprocess_job_id: 전처리 작업 ID (ERD 기준)
        """
        with open(stt_json_path, "r", encoding="utf-8") as f:
            stt_data = json.load(f)
        
        # 구조 유연성 처리: 최상위가 dict이고 내부에 segments 키가 있을 수도, 바로 list일 수도 있음
        segments = stt_data.get("segments", stt_data)
        if isinstance(segments, dict):
            segments = segments.get("segments", [])
        
        return self.save_stt_result(video_id, segments, preprocess_job_id, provider)

    def get_stt_results_by_ids(self, video_id: str, stt_ids: List[str]) -> List[Dict[str, Any]]:
        if not stt_ids:
            return []
        cleaned = [str(item).strip() for item in stt_ids if str(item).strip()]
        if not cleaned:
            return []

        query = (
            self.client.table("stt_results")
            .select("stt_id, transcript, start_ms, end_ms, confidence")
            .eq("video_id", video_id)
        )
        if len(cleaned) == 1:
            query = query.eq("stt_id", cleaned[0])
        else:
            in_method = getattr(query, "in_", None)
            if callable(in_method):
                query = in_method("stt_id", cleaned)
            else:
                query = query.filter("stt_id", "in", f"({','.join(cleaned)})")

        result = query.execute()
        return result.data if result.data else []
    
    # =========================================================================
    # Segments (통합 세그먼트)
    # =========================================================================
    
    def save_segments(
        self,
        video_id: str,
        segments: List[Dict[str, Any]],
        processing_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """멀티모달 통합 세그먼트(Fusion Segments) 결과를 저장합니다.
        
        각 세그먼트는 시각적 장면 변화와 음성적 대화 단위를 고려하여 나누어진 구간입니다.
        
        Args:
            video_id: 비디오 ID
            segments: segments_units.jsonl에서 파싱된 세그먼트 리스트
                - transcript_units: 해당 구간의 대화 내용 (저장 시 텍스트 병합)
                - visual_units: 해당 구간의 시각적 특징 (JSONB 등으로 저장 가능)
                - embedding: 검색용 벡터 (선택)
            processing_job_id: 처리 작업 ID (ERD 기준)
            
        Returns:
            List[Dict]: DB 저장 결과
        """
        rows = []

        for seg in segments:
            # DB 스키마 최적화: transcript_units(JSON) 대신 평문 텍스트(TEXT)로 병합하여 저장
            # 사유: 검색 및 RAG(Retrieval-Augmented Generation) 활용 용이성
            t_units = seg.get("transcript_units", [])
            transcript_text = ""
            if isinstance(t_units, list):
                # 각 단위의 text 필드만 추출하여 개행으로 연결
                transcript_text = "\n".join(u.get("text", "") for u in t_units if u.get("text"))
            
            rows.append({
                "video_id": video_id,
                "segment_index": seg.get("segment_id"), # 로컬 파일 내의 인덱스 보존
                "start_ms": seg.get("start_ms"),
                "end_ms": seg.get("end_ms"),
                "transcript_units": transcript_text,  # 병합된 텍스트 저장
                "visual_units": seg.get("visual_units"), # JSONB 타입
                "source_refs": seg.get("source_refs"),   # JSONB (stt_ids, vlm_ids)
                "processing_job_id": processing_job_id,
            })
        
        if rows:
            result = self.client.table("segments").insert(rows).execute()
            return result.data
        return []
    
    def save_segments_from_file(
        self,
        video_id: str,
        jsonl_path: Path,
        processing_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """segments_units.jsonl (Line delimited JSON) 파일을 읽어 저장합니다.
        
        JSONL 포맷은 대용량 데이터를 한 줄씩 스트리밍 처리하기 적합합니다.
        """
        segments = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append(json.loads(line))
        return self.save_segments(video_id, segments, processing_job_id)

    def get_vlm_results_by_ids(self, video_id: str, cap_ids: List[str]) -> List[Dict[str, Any]]:
        if not cap_ids:
            return []
        cleaned = [str(item).strip() for item in cap_ids if str(item).strip()]
        if not cleaned:
            return []

        query = (
            self.client.table("vlm_results")
            .select("cap_id, extracted_text, timestamp_ms")
            .eq("video_id", video_id)
        )
        if len(cleaned) == 1:
            query = query.eq("cap_id", cleaned[0])
        else:
            in_method = getattr(query, "in_", None)
            if callable(in_method):
                query = in_method("cap_id", cleaned)
            else:
                query = query.filter("cap_id", "in", f"({','.join(cleaned)})")

        result = query.execute()
        return result.data if result.data else []
    
    # =========================================================================
    # Summaries (요약 결과)
    # =========================================================================
    
    def save_summaries(
        self,
        video_id: str,
        summaries: List[Dict[str, Any]],
        segment_map: Optional[Dict[int, str]] = None,
        processing_job_id: Optional[str] = None,
        generate_embedding_flag: bool = True,
        batch_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """LLM 분석 기반 요약 결과를 저장합니다.

        segments 테이블과의 관계(FK)를 설정할 수 있습니다.

        Args:
            video_id: 비디오 ID
            summaries: 요약 데이터 리스트
            segment_map: 로컬 segment_index(int) -> DB segment_id(uuid) 매핑 테이블
            processing_job_id: 처리 작업 ID (ERD 기준)
            generate_embedding_flag: True이면 임베딩 벡터 자동 생성
            batch_index: 배치 인덱스 (0-indexed)

        Returns:
            List[Dict]: 저장된 요약 레코드
        """
        rows = []
        texts_for_embedding = []  # 배치 임베딩 생성용

        for summ in summaries:
            summary_data = summ.get("summary", {})

            # JSONB -> 시맨틱 검색용 텍스트 렌더링
            summary_text = render_summary_to_text(summary_data) if summary_data else ""
            texts_for_embedding.append(summary_text)

            data = {
                "video_id": video_id,
                "summary": summary_data,  # JSONB로 저장
                "version": summ.get("version"),
                "embedding": None,  # 나중에 채움
                "processing_job_id": processing_job_id,
                # NOTE: summary_text 컬럼은 스키마에 없음 (summary JSONB 내에 포함)
            }

            # batch_index 추가
            if batch_index is not None:
                data["batch_index"] = batch_index

            # Segment FK 연결
            seg_idx = summ.get("segment_id")
            if segment_map and seg_idx in segment_map:
                data["segment_id"] = segment_map[seg_idx]

            rows.append(data)

        # 임베딩 생성 (배치 처리)
        if generate_embedding_flag and texts_for_embedding:
            try:
                embeddings = generate_embeddings_batch(texts_for_embedding)
                for i, emb in enumerate(embeddings):
                    rows[i]["embedding"] = emb
            except Exception as e:
                logger.warning(f"임베딩 생성 실패, 요약 저장은 계속 진행: {e}")

        if rows:
            result = self.client.table("summaries").insert(rows).execute()
            return result.data
        return []
    
    def save_summaries_from_file(
        self,
        video_id: str,
        jsonl_path: Path,
        segment_map: Optional[Dict[int, str]] = None,
        processing_job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """segment_summaries.jsonl 파일을 읽어 저장합니다."""
        summaries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(json.loads(line))
        return self.save_summaries(video_id, summaries, segment_map, processing_job_id)
