"""캡처 및 스토리지 관리 어댑터 모듈."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class CaptureAdapterMixin:
    """Captures 테이블 및 Supabase Storage 작업을 위한 Mixin 클래스.
    
    이 클래스는 비디오 분석 과정에서 생성된 비디오 캡처(이미지) 정보를 관리합니다.
    주요 역할:
    1. `captures` 테이블에 메타데이터(시간, 파일명 등) 저장
    2. Supabase Storage(보통 Private 버킷)에 실제 이미지 파일 업로드
    3. Private 파일 접근을 위한 Signed URL 생성
    """
    
    def save_captures(
        self,
        video_id: str,
        captures: List[Dict[str, Any]],
        table_name: str = "captures",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """메모리 상의 캡처 메타데이터 리스트를 DB에 일괄 저장합니다.
        
        Args:
            video_id: 연관된 비디오 ID (FK)
            captures: 저장할 캡처 정보 리스트 (manifest.json 형식)
                - file_name: 파일명
                - start_ms, end_ms: 타임스탬프
                - storage_path: (선택) 스토리지 경로
            **kwargs: 추가 필드 (예: pipeline_run_id)
            
        Returns:
            List[Dict]: DB에 저장된 레코드 리스트
        """
        rows = []
        for cap in captures:
            rows.append({
                "video_id": video_id,
                "cap_id": cap.get("id"),
                "file_name": cap.get("file_name"),
                "storage_path": cap.get("storage_path"),
                "preprocess_job_id": kwargs.get("preprocess_job_id"),
                "time_ranges": cap.get("time_ranges"),
            })
        
        if rows:
            # bulk insert 실행
            result = self.client.table(table_name).insert(rows).execute()
            return result.data
        return []
    
    def save_captures_from_file(
        self,
        video_id: str,
        manifest_json_path: Path,
        table_name: str = "captures",
    ) -> List[Dict[str, Any]]:
        """로컬 manifest.json 파일을 읽어 캡처 정보를 DB에 저장합니다.
        
        이 메서드는 이미지가 이미 업로드되어 있거나, 메타데이터만 저장할 때 유용합니다.
        """
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            captures = json.load(f)
        return self.save_captures(video_id, captures, table_name=table_name)
    
    def upload_capture_image(
        self,
        video_id: str,
        image_path: Path,
        bucket: str = "captures",
        signed_url_expires: int = 3600,
    ) -> Dict[str, Any]:
        """단일 캡처 이미지를 Supabase Storage에 업로드하고 접근 URL을 반환합니다.
        
        보안을 위해 Private 버킷 사용을 권장하며, 이 경우 Signed URL을 발급받아야 합니다.
        
        Args:
            video_id: 비디오 ID (경로 구분용)
            image_path: 업로드할 로컬 이미지 파일 경로
            bucket: 타겟 Storage 버킷 이름
            signed_url_expires: Signed URL 유효 기간 (초단위, 기본 1시간)
            
        Returns:
            Dict: 업로드 결과 정보
                - file_name: 파일명
                - storage_path: 버킷 내 저장 경로 ({video_id}/{filename})
                - signed_url: 접근 가능한 임시 URL
        """
        file_name = image_path.name
        # 스토리지 경로 구조: video_id/filename.jpg
        storage_path = f"{video_id}/{file_name}"
        
        with open(image_path, "rb") as f:
            file_data = f.read()
        
        # 1. Storage에 파일 업로드
        # upsert=False가 기본이므로 중복 시 에러 발생 가능 (필요 시 file_options에 upsert=True 추가)
        self.client.storage.from_(bucket).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "image/jpeg"}
        )
        
        # 2. Private 버킷 접근을 위한 Signed URL 생성
        signed_result = self.client.storage.from_(bucket).create_signed_url(
            path=storage_path,
            expires_in=signed_url_expires
        )
        signed_url = signed_result.get("signedURL", "")
        
        return {
            "file_name": file_name,
            "storage_path": storage_path,
            "signed_url": signed_url,
            "expires_in": signed_url_expires,
        }
    
    def upload_audio(
        self,
        video_id: str,
        audio_path: Path,
        bucket: str = "audio",
    ) -> Dict[str, Any]:
        """오디오 파일을 Supabase Storage에 업로드합니다.
        
        Args:
            video_id: 비디오 ID (경로 구분용)
            audio_path: 업로드할 로컬 오디오 파일 경로 (.wav, .mp3 등)
            bucket: 타겟 Storage 버킷 이름 (기본: audio)
            
        Returns:
            Dict: 업로드 결과 정보
                - file_name: 파일명
                - storage_path: 버킷 내 저장 경로 ({video_id}/{filename})
        """
        file_name = audio_path.name
        storage_path = f"{video_id}/{file_name}"
        
        # Content-Type 결정
        suffix = audio_path.suffix.lower()
        content_type = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }.get(suffix, "audio/wav")
        
        with open(audio_path, "rb") as f:
            file_data = f.read()
        
        # Storage에 업로드 (upsert=True로 덮어쓰기 허용)
        self.client.storage.from_(bucket).upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        return {
            "file_name": file_name,
            "storage_path": storage_path,
        }
    
    def download_audio(
        self,
        storage_path: str,
        local_path: Path,
        bucket: str = "audio",
    ) -> Path:
        """Storage에서 오디오 파일을 다운로드합니다.
        
        Args:
            storage_path: Storage 내 경로 (예: {video_id}/audio.wav)
            local_path: 다운로드할 로컬 경로
            bucket: Storage 버킷 이름 (기본: audio)
            
        Returns:
            Path: 다운로드된 파일의 로컬 경로
        """
        # Storage에서 파일 다운로드
        file_data = self.client.storage.from_(bucket).download(storage_path)
        
        # 로컬에 저장
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(file_data)
        
        return local_path
    
    def get_signed_url(
        self,
        storage_path: str,
        bucket: str = "captures",
        expires_in: int = 3600,
    ) -> str:
        """이미 업로드된 파일에 대한 새로운 Signed URL을 발급합니다.
        
        Args:
            storage_path: Storage 내부 경로 (예: uuid/image.jpg)
            bucket: 버킷 이름
            expires_in: 유효 기간 (초)
            
        Returns:
            str: 생성된 Signed URL
        """
        result = self.client.storage.from_(bucket).create_signed_url(
            path=storage_path,
            expires_in=expires_in
        )
        return result.get("signedURL", "")
    
    def upload_all_captures(
        self,
        video_id: str,
        captures_dir: Path,
        bucket: str = "captures",
    ) -> List[Dict[str, Any]]:
        """지정된 디렉토리의 모든 JPG 이미지를 Storage에 일괄 업로드합니다.
        
        Args:
            video_id: 비디오 ID
            captures_dir: 이미지가 들어있는 로컬 폴더
            bucket: 타겟 버킷
            
        Returns:
            List[Dict]: 각 파일별 업로드 성공/실패 결과
        """
        results = []
        
        # JPG, JPEG 확장자 파일 탐색
        image_files = list(captures_dir.glob("*.jpg")) + list(captures_dir.glob("*.jpeg"))
        
        for image_path in sorted(image_files):
            try:
                upload_result = self.upload_capture_image(video_id, image_path, bucket)
                results.append({
                    "status": "success",
                    **upload_result
                })
            except Exception as e:
                # 개별 파일 실패 시에도 전체 프로세스는 중단하지 않고 로그 기록
                results.append({
                    "status": "error",
                    "file_name": image_path.name,
                    "error": str(e)
                })
        
        return results
    
    def save_captures_with_upload(
        self,
        video_id: str,
        manifest_json_path: Path,
        captures_dir: Path,
        bucket: str = "captures",
        preprocess_job_id: Optional[str] = None,
        table_name: str = "captures",
    ) -> Dict[str, Any]:
        """캡처 이미지 업로드와 DB 메타데이터 저장을 통합 수행하는 편의 메소드입니다.
        
        Process:
        1. manifest.json 파싱
        2. 각 항목별 실제 이미지 파일 확인 및 업로드
        3. 업로드된 경로(storage_path)를 포함하여 DB에 메타데이터 저장
        
        Args:
            video_id: 비디오 ID
            manifest_json_path: 메타데이터 파일 경로
            captures_dir: 이미지 파일이 있는 디렉토리
            bucket: 업로드할 버킷
            preprocess_job_id: 전처리 작업 ID (ERD 기준)
            
        Returns:
            Dict: 전체 처리 결과 요약 (저장된 DB 수, 업로드된 파일 수, 에러 목록)
        """
        results = {
            "db_saved": 0,
            "storage_uploaded": 0,
            "errors": []
        }
        
        # 1. manifest.json 로드
        with open(manifest_json_path, "r", encoding="utf-8") as f:
            captures = json.load(f)
        
        rows = []
        
        # 2. 이미지 업로드 및 Row 데이터 구성
        for cap in captures:
            file_name = cap.get("file_name")
            image_path = captures_dir / file_name
            
            storage_path = None
            
            # 실제 파일이 존재하면 업로드 시도
            if image_path.exists():
                try:
                    upload_result = self.upload_capture_image(video_id, image_path, bucket)
                    storage_path = upload_result["storage_path"]
                    results["storage_uploaded"] += 1
                except Exception as e:
                    # 업로드 실패시에도 DB 저장은 시도 (storage_path는 None)
                    results["errors"].append(f"upload error ({file_name}): {str(e)}")
            
            rows.append({
                "video_id": video_id,
                "cap_id": cap.get("id"),
                "file_name": file_name,
                "storage_path": storage_path, # 업로드 성공 시 경로, 아니면 None
                "time_ranges": cap.get("time_ranges"),
                "preprocess_job_id": preprocess_job_id,
                "time_ranges": cap.get("time_ranges"),
            })
        
        # 3. DB에 일괄 저장
        if rows:
            try:
                db_result = self.client.table(table_name).insert(rows).execute()
                results["db_saved"] = len(db_result.data)
            except Exception as e:
                results["errors"].append(f"db insert error: {str(e)}")
        
        return results

    def save_captures_with_upload_payload(
        self,
        video_id: str,
        captures: List[Dict[str, Any]],
        captures_dir: Path,
        bucket: str = "captures",
        preprocess_job_id: Optional[str] = None,
        table_name: str = "captures",
    ) -> Dict[str, Any]:
        """메타데이터 리스트로 캡처 업로드/저장을 수행한다."""
        results = {
            "db_saved": 0,
            "storage_uploaded": 0,
            "errors": []
        }

        rows = []
        for cap in captures:
            file_name = cap.get("file_name")
            if not file_name:
                continue
            image_path = captures_dir / file_name

            storage_path = None

            if image_path.exists():
                try:
                    upload_result = self.upload_capture_image(video_id, image_path, bucket)
                    storage_path = upload_result["storage_path"]
                    results["storage_uploaded"] += 1
                except Exception as e:
                    results["errors"].append(f"upload error ({file_name}): {str(e)}")

            rows.append({
                "video_id": video_id,
                "cap_id": cap.get("id"),
                "file_name": file_name,
                "storage_path": storage_path,
                "time_ranges": cap.get("time_ranges"),
                "preprocess_job_id": preprocess_job_id,
                "time_ranges": cap.get("time_ranges"),
            })

        if rows:
            try:
                db_result = self.client.table(table_name).insert(rows).execute()
                results["db_saved"] = len(db_result.data)
            except Exception as e:
                results["errors"].append(f"db insert error: {str(e)}")

        return results
