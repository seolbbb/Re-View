"""퓨전 파이프라인 설정 loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    """입력 데이터(STT, VLM)와 출력 경로를 설정한다."""
    model_config = ConfigDict(extra="forbid")

    stt_json: str
    vlm_json: str
    captures_manifest_json: Optional[str] = None
    output_root: str


class SyncEngineConfig(BaseModel):
    """STT와 VLM 데이터를 동기화하고 세그먼트를 나누는 규칙을 정의한다."""
    model_config = ConfigDict(extra="forbid")

    boundary_source: Literal["vlm_timestamps_visual_first"]
    min_segment_sec: int = Field(..., ge=1)
    max_segment_sec: int = Field(..., ge=1)
    max_transcript_chars: int = Field(..., ge=1)
    silence_gap_ms: int = Field(..., ge=0)
    max_visual_items: int = Field(..., ge=1)
    max_visual_chars: int = Field(..., ge=0)
    dedup_similarity_threshold: float = Field(..., ge=0.0, le=1.0)


class SummarizerConfig(BaseModel):
    """각 세그먼트(문단)를 요약하는 방식과 포맷을 설정한다."""
    model_config = ConfigDict(extra="forbid")

    bullets_per_segment_min: int = Field(..., ge=0)
    bullets_per_segment_max: int = Field(..., ge=0)
    claim_max_chars: int = Field(..., ge=0)
    temperature: float = Field(..., ge=0.0, le=1.0)
    json_repair_attempts: int = Field(..., ge=0)
    prompt_version: Optional[str] = None


class JudgeConfig(BaseModel):
    """생성된 요약문의 품질을 평가(Self-Correction)하는 기준을 설정한다."""
    model_config = ConfigDict(extra="forbid")

    min_score: float = Field(7.0, ge=0.0, le=10.0)
    include_segments: bool = False
    batch_size: int = Field(3, ge=1)
    workers: int = Field(1, ge=1)
    json_repair_attempts: int = Field(1, ge=0)
    verbose: bool = False
    prompt_version: Optional[str] = None


class DeveloperApiConfig(BaseModel):
    """Google Developer API(AI Studio) 사용 시 필요한 키 설정을 담는다."""
    model_config = ConfigDict(extra="forbid")

    api_key_env_candidates: List[str]


class VertexAiConfig(BaseModel):
    """Google Vertex AI 플랫폼 사용 시 필요한 인증 및 리전 설정을 담는다."""
    model_config = ConfigDict(extra="forbid")

    auth_mode: Literal["adc", "express_api_key"]
    project: Optional[str] = None
    location: Optional[str] = None
    api_key_env: str = "GOOGLE_API_KEY"


class LlmGeminiConfig(BaseModel):
    """Gemini 모델 호출을 위한 백엔드, 모델명, 재시도 정책 등을 설정한다."""
    model_config = ConfigDict(extra="forbid")

    backend: Literal["developer_api", "vertex_ai"]
    model: str
    timeout_sec: int = Field(..., ge=1)
    max_retries: int = Field(..., ge=0)
    backoff_sec: List[int]
    response_mime_type: str
    developer_api: DeveloperApiConfig
    vertex_ai: VertexAiConfig


class RenderConfig(BaseModel):
    """최종 마크다운 리포트 생성 시 포맷팅 옵션을 설정한다."""
    model_config = ConfigDict(extra="forbid")

    include_sources: bool = False
    md_wrap_width: int = Field(..., ge=0)


class FinalSummaryStyleConfig(BaseModel):
    """최종 요약문의 스타일(타임스탬프 포함 여부, 언어 등)을 설정한다."""
    model_config = ConfigDict(extra="forbid")

    include_timestamps: bool = True
    language: Literal["ko"] = "ko"


class FinalSummaryConfig(BaseModel):
    """전체 영상을 아우르는 최종 타임라인 요약 생성 설정을 담는다."""
    model_config = ConfigDict(extra="forbid")

    generate_formats: List[Literal["timeline", "tldr_timeline"]]
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_chars_per_format: int = Field(..., ge=0)
    style: FinalSummaryStyleConfig


class FusionConfig(BaseModel):
    """Fusion 파이프라인 전체를 관장하는 최상위 설정 클래스."""
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    sync_engine: SyncEngineConfig
    summarizer: SummarizerConfig
    llm_gemini: LlmGeminiConfig
    render: RenderConfig
    final_summary: FinalSummaryConfig


@dataclass(frozen=True)
class ResolvedPaths:
    """설정 파일 내의 상대 경로들을 절대 경로로 변환하여 담는 컨테이너."""
    stt_json: Path
    vlm_json: Path
    captures_manifest_json: Optional[Path]
    output_root: Path


@dataclass(frozen=True)
class ConfigBundle:
    """로드된 설정 원본과 파생된 정보(경로, Judge 설정 등)를 묶어서 전달하는 객체."""
    raw: FusionConfig
    config_path: Path
    repo_root: Path
    paths: ResolvedPaths
    judge: JudgeConfig


def _find_repo_root(config_path: Path) -> Path:
    """설정 파일 위치를 기준으로 상위 디렉토리를 탐색해 프로젝트 루트(rep_root)를 찾는다."""
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return config_path.parent


def _resolve_path(path_str: Optional[str], repo_root: Path) -> Optional[Path]:
    """경로 문자열을 프로젝트 루트 기준 절대 경로로 변환한다 (이미 절대 경로면 유지)."""
    if path_str is None:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def load_config(config_path: str) -> ConfigBundle:
    """YAML 설정 파일을 읽고 검증하여 ConfigBundle 객체를 생성한다.
    
    이 함수는 환경변수(.env) 로딩과 Judge 설정 로딩도 함께 수행한다.
    """
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    config = FusionConfig.model_validate(payload)
    repo_root = _find_repo_root(config_file)
    load_dotenv(dotenv_path=repo_root / ".env")

    judge_config_path = repo_root / "config" / "judge" / "settings.yaml"
    if not judge_config_path.exists():
        raise FileNotFoundError(f"judge settings file not found: {judge_config_path}")
    with judge_config_path.open("r", encoding="utf-8") as handle:
        judge_payload = yaml.safe_load(handle) or {}
    judge_config = JudgeConfig.model_validate(judge_payload)

    resolved_paths = ResolvedPaths(
        stt_json=_resolve_path(config.paths.stt_json, repo_root),
        vlm_json=_resolve_path(config.paths.vlm_json, repo_root),
        captures_manifest_json=_resolve_path(config.paths.captures_manifest_json, repo_root),
        output_root=_resolve_path(config.paths.output_root, repo_root),
    )
    return ConfigBundle(
        raw=config,
        config_path=config_file,
        repo_root=repo_root,
        paths=resolved_paths,
        judge=judge_config,
    )
