"""ADK 파이프라인 경로 규약.

- 입력은 `data/inputs` (로컬 업로드 대체)
- 출력은 `data/outputs/{video_name}` (DB 대체)

기존 `src/run_video_pipeline.py`의 산출물 구조를 가능한 그대로 따른다.
"""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/inputs")
DEFAULT_OUTPUT_BASE = Path("data/outputs")
DEFAULT_FUSION_TEMPLATE_CONFIG = Path("src/fusion/config.yaml")


def sanitize_video_name(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return "video"
    return value[:80]


def resolve_video_path(input_dir: Path, video_name_or_file: str) -> Path:
    """`data/inputs`에서 mp4 경로를 resolve 한다.

    - `video_name_or_file`이 파일명(확장자 포함)이면 그대로 찾는다.
    - 확장자 없이 주어지면 `.mp4`를 붙여 찾는다.
    """

    candidate = (input_dir / video_name_or_file).expanduser()
    if candidate.exists():
        return candidate.resolve()

    mp4_candidate = candidate.with_suffix(".mp4")
    if mp4_candidate.exists():
        return mp4_candidate.resolve()

    raise FileNotFoundError(
        f"입력 비디오를 찾을 수 없습니다: {candidate} (또는 {mp4_candidate})"
    )

