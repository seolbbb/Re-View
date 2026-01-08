"""심판하기 tool (Gemini-only) - 현재는 껍데기만 제공.

나중에 Gemini로 품질 판단 로직을 넣을 예정.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.fusion.io_utils import write_json


def judge_stub_gemini(*, fusion_dir: Path) -> Dict[str, Any]:
    """현재는 항상 pass=True를 반환하고 judge.json을 쓴다."""

    result: Dict[str, Any] = {
        "schema_version": 1,
        "model": "gemini",
        "pass": True,
        "reason": "judge 미구현 - 항상 통과",
    }
    write_json(fusion_dir / "judge.json", result)
    return result
