"""ADK 기반 비디오 파이프라인 패키지.

이 파일은 adk web에서 agent를 로드할 때 가장 먼저 실행됩니다.
sys.path에 프로젝트 루트를 추가하여 src.* import가 가능하게 합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 계산 (src/adk_pipeline/__init__.py -> 프로젝트 루트)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# sys.path에 프로젝트 루트 추가
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# .env 로드 (API 키 등)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# ADK discovery용 agent 모듈 import
from . import agent
