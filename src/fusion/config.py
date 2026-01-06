"""퓨전 파이프라인 설정 로더."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stt_json: str
    vlm_json: str
    captures_manifest_json: Optional[str] = None
    qwen3_detect_json: Optional[str] = None
    output_root: str


class SyncEngineConfig(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    bullets_per_segment_min: int = Field(..., ge=0)
    bullets_per_segment_max: int = Field(..., ge=0)
    claim_max_chars: int = Field(..., ge=0)
    temperature: float = Field(..., ge=0.0, le=1.0)
    json_repair_attempts: int = Field(..., ge=0)


class DeveloperApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key_env_candidates: List[str]


class VertexAiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_mode: Literal["adc", "express_api_key"]
    project: Optional[str] = None
    location: Optional[str] = None
    api_key_env: str = "GOOGLE_API_KEY"


class LlmGeminiConfig(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    include_sources: bool = False
    md_wrap_width: int = Field(..., ge=0)


class FinalSummaryStyleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_timestamps: bool = True
    language: Literal["ko"] = "ko"


class FinalSummaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generate_formats: List[Literal["timeline", "tldr_timeline"]]
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_chars_per_format: int = Field(..., ge=0)
    style: FinalSummaryStyleConfig


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    sync_engine: SyncEngineConfig
    summarizer: SummarizerConfig
    llm_gemini: LlmGeminiConfig
    render: RenderConfig
    final_summary: FinalSummaryConfig


@dataclass(frozen=True)
class ResolvedPaths:
    stt_json: Path
    vlm_json: Path
    captures_manifest_json: Optional[Path]
    qwen3_detect_json: Optional[Path]
    output_root: Path


@dataclass(frozen=True)
class ConfigBundle:
    raw: FusionConfig
    config_path: Path
    repo_root: Path
    paths: ResolvedPaths


def _find_repo_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return config_path.parent


def _resolve_path(path_str: Optional[str], repo_root: Path) -> Optional[Path]:
    if path_str is None:
        return None
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def load_config(config_path: str) -> ConfigBundle:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    config = FusionConfig.model_validate(payload)
    repo_root = _find_repo_root(config_file)
    load_dotenv(dotenv_path=repo_root / ".env")

    resolved_paths = ResolvedPaths(
        stt_json=_resolve_path(config.paths.stt_json, repo_root),
        vlm_json=_resolve_path(config.paths.vlm_json, repo_root),
        captures_manifest_json=_resolve_path(config.paths.captures_manifest_json, repo_root),
        qwen3_detect_json=_resolve_path(config.paths.qwen3_detect_json, repo_root),
        output_root=_resolve_path(config.paths.output_root, repo_root),
    )
    return ConfigBundle(raw=config, config_path=config_file, repo_root=repo_root, paths=resolved_paths)
