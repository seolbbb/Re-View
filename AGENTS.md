# AGENTS.md - Coding Agent Guidelines

## Project Overview

비디오 강의 처리 파이프라인: 슬라이드 캡처, 음성 텍스트 변환(STT), VLM 시각정보 추출, LLM 기반 요약 노트 생성.  
**Google ADK(Agent Development Kit) 기반 멀티 에이전트 아키텍처**로 구성됨.

**Tech Stack**: Python 3.10+, Google ADK, OpenCV, Pydantic v2, Supabase (Postgres)

---

## 📢 Project Direction & Context (Must Read)

**`docs/PROJECT_DIRECTION.md`** 파일은 본 프로젝트의 **네비게이션** 역할을 하는 핵심 문서입니다.
에이전트 및 개발자는 작업 시작 전 반드시 이 파일을 참고하여 **프로젝트의 최신 상태와 방향성**을 일치시켜야 합니다.

- **프로젝트 개요 및 목표**: Screentime 파이프라인의 핵심 가치와 목표.
- **AS-IS 파이프라인 분석**: 각 Phase (Capture, VLM, Fusion, Judge) 별 상세 로직과 현재 병목 지점 분석.
- **회의 노트 & Action Items**: 최근 회의에서 결정된 최적화 전략 (Async VLM, Streaming, DB 도입 등) 및 우선순위.

> **Note**: 작업 중 방향성이 모호하거나 성능 최적화의 맥락이 궁금하다면 `docs/PROJECT_DIRECTION.md`를 가장 먼저 확인하세요.

---

## Build / Run / Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# === Pre-ADK Pipeline (CLI) ===
# STT + Capture 전처리 실행
python src/pre_adk_pipeline.py --video "path/to/video.mp4"

# === ADK Pipeline (Interactive Web) ===
# ADK 웹 서버 실행
adk web src

# === Individual Components ===
python src/capture/process_content.py                             # Capture only
python src/audio/stt_router.py --video "path/to/video.mp4"        # STT only

# === Database ===
python src/utils/postgres_ingest.py --stt stt.json --manifest manifest.json

# === Testing (dry-run) ===
python src/fusion/run_summarizer.py --config src/fusion/config.yaml --dry_run
```

**Linter (권장)**: `ruff format`, `ruff check`, `mypy --strict`

---

## Code Style Guidelines

### Imports (in order)

```python
from __future__ import annotations          # 1. Future annotations
import json, os                             # 2. Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2                                  # 3. Third-party
from google.adk.agents import Agent
from pydantic import BaseModel, ConfigDict
from .config import ConfigBundle            # 4. Local (relative in package)
```

### Type Hints (mandatory)

```python
def process_video(path: str, threshold: float = 8.0) -> List[Dict[str, Any]]:
    ...
```

- Use `Optional[X]` for nullable (not `X | None`)
- Import from `typing`: `List`, `Dict`, `Tuple`, `Any`, `Optional`

### Naming Conventions

| Element           | Convention  | Example                  |
| ----------------- | ----------- | ------------------------ |
| Functions/methods | snake_case  | `extract_keyframes()`    |
| Classes           | PascalCase  | `VideoProcessor`         |
| Constants         | UPPER_SNAKE | `DEFAULT_REQUEST_PARAMS` |
| Private           | `_` prefix  | `_load_stt_segments()`   |

### Data Models

```python
# Pydantic v2 with strict validation
class AudioSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(..., ge=0)
    text: str = Field(..., min_length=1)

# Immutable dataclasses
@dataclass(frozen=True)
class SegmentWindow:
    start_ms: int
    end_ms: int
```

### Error Handling

- Raise specific exceptions with descriptive messages (Korean OK)
- Never use bare `except:` or suppress silently

```python
if not path.exists():
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
```

### File I/O

- Always specify `encoding="utf-8"`
- Use `pathlib.Path`, not string paths

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

### Environment Variables

```python
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

def load_env() -> None:
    load_dotenv(ENV_PATH) if ENV_PATH.exists() else load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set.")
```

---

## Project Structure

```
src/
├── pre_adk_pipeline.py         # Pre-ADK CLI (STT + Capture)
├── adk_pipeline/               # ADK 멀티에이전트 파이프라인
│   ├── agent.py                # ⭐ Agent 정의 (Root + Sub-agents)
│   ├── store.py                # ⭐ VideoStore (DB 추상화 레이어)
│   ├── paths.py                # 경로 상수/유틸
│   └── tools/
│       ├── root_tools.py       # Root Agent 도구
│       ├── preprocessing_tools.py  # Preprocessing Agent 도구
│       ├── summarize_tools.py  # Summarize Agent 도구
│       ├── judge_tools.py      # Judge Agent 도구
│       └── internal/           # 내부 구현 모듈
├── audio/                      # STT (Clova, Whisper)
├── capture/                    # Video frame extraction
├── common/                     # Shared Pydantic schemas
├── fusion/                     # STT+VLM sync, summarization
├── vlm/                        # Vision-Language Model
└── utils/                      # Utilities (DB ingest 등)
```

**Key Files:**

- `src/adk_pipeline/agent.py` - 멀티에이전트 정의 (Root, Preprocessing, Summarize, Judge)
- `src/adk_pipeline/store.py` - VideoStore (파일시스템/DB 추상화)
- `src/pre_adk_pipeline.py` - Pre-ADK CLI 진입점
- `docs/DEVELOPER_GUIDE.md` - 상세 개발 가이드

---

## Configuration

Required `.env` variables:

```
GOOGLE_API_KEY=...
CLOVA_SPEECH_URL=...
CLOVA_SPEECH_API_KEY=...
OPENROUTER_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
```

Output artifacts under `data/outputs/{video_name}/`:

- `stt.json` - STT 결과
- `manifest.json` - 캡처 메타데이터
- `captures/*.png` - 캡처 이미지
- `vlm.json` - VLM 추출 결과
- `fusion/segments_units.jsonl` - 동기화된 세그먼트
- `fusion/segment_summaries.jsonl` - 세그먼트별 요약
- `fusion/outputs/final_summary_*.md` - 최종 요약 마크다운
- `fusion/judge.json` - Judge 평가 결과

---

## Common Patterns

### CLI Entry Points

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--video", required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    ...

if __name__ == "__main__":
    main()
```

### ADK Tool Pattern

```python
def my_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """ADK Tool - ToolContext로 상태 접근"""
    video_name = tool_context.state.get("video_name")
    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 내부 로직 실행
    result = do_something(store.some_path())

    return {"success": True, "result": result}
```

### Retry Logic

```python
def _run_with_retries(func, max_retries: int, backoff_sec: List[int]) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(backoff_sec[min(attempt, len(backoff_sec) - 1)])
```

---

## Anti-Patterns to Avoid

1. **No type suppression** - No `# type: ignore`, `as Any` without reason
2. **No bare except** - Always catch specific exceptions
3. **No hardcoded paths** - Use config or env vars
4. **No missing encoding** - Always `encoding="utf-8"`
5. **No `.env` commits** - Keep secrets out of VCS
6. **No mixed language** - Korean or English per block, not both
7. **No empty agent responses** - Agent must always call tool or return summary after transfer

---

## Git Commit Guidelines

- **커밋 메시지는 한글로 작성**
- 제목은 50자 이내, 본문은 72자 줄바꿈 권장
- 형식: `type(scope): 제목`
  - type: feat, fix, refactor, docs, test, chore 등
  - scope: 변경된 모듈/파일 (선택)
- 예시:

  ```
  feat(adk): 강제 재전처리 옵션 추가

  - set_pipeline_config에 force_preprocessing 파라미터 추가
  - 기존 파일 있어도 처음부터 재실행 가능

  Closes #35
  ```

---

## PR Guidelines

- **PR 제목/본문은 한글로 작성**
- PR 하나당 하나의 목적 (여러 기능 X)
- 형식: `[type] 한글 설명`
- PR Template에 맞게 작성
- 본문 필수 포함 사항:
  - 변경 사항 요약
  - 테스트 결과
  - 관련 이슈 번호
- 머지 전 최소 1명 Approve 필요
