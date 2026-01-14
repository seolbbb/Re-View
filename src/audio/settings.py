from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "audio" / "settings.yaml"


def load_audio_settings(*, settings_path: Optional[Path] = None) -> Dict[str, Any]:
    path = settings_path or SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"audio settings file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audio settings must be a mapping.")
    return payload
