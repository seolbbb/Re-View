# AGENTS.md - Coding Agent Guidelines

## Project Overview
Lecture video processing pipeline: extracts slides, transcribes audio (STT), uses VLM for text extraction, generates summarized notes via LLM fusion.

**Tech Stack**: Python 3.10+, OpenCV, Pydantic v2, Google GenAI, OpenAI Whisper

---

## Build / Run / Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Main pipeline (end-to-end)
python src/run_video_pipeline.py --video "path/to/video.mp4"

# Individual components
python src/process_content.py                                    # Capture only
python src/audio/speech_client.py --media-path "path/to/video"   # STT only
python src/vlm/vlm_engine.py --image "img.jpg" --video-name "x"  # VLM only

# Fusion steps
python src/fusion/run_sync_engine.py --config src/fusion/config.yaml
python src/fusion/run_summarizer.py --config src/fusion/config.yaml
python src/fusion/run_final_summary.py --config src/fusion/config.yaml

# Testing (no formal framework - use dry-run flags)
python src/fusion/run_summarizer.py --config src/fusion/config.yaml --dry_run
python src/fusion/run_sync_engine.py --config src/fusion/config.yaml --limit 2
```

**No linter configured.** When adding, prefer: `ruff format`, `ruff check`, `mypy --strict`

---

## Code Style Guidelines

### Imports (in order)
```python
from __future__ import annotations          # 1. Future annotations
import json, os                             # 2. Standard library
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2                                  # 3. Third-party
from pydantic import BaseModel, ConfigDict
from .config import ConfigBundle            # 4. Local (relative in package)
from src.common.schemas import AudioSegment # 4. Local (absolute)
```

### Type Hints (mandatory)
```python
def process_video(path: str, threshold: float = 8.0) -> List[Dict[str, Any]]:
    ...
```
- Use `Optional[X]` for nullable (not `X | None`)
- Import from `typing`: `List`, `Dict`, `Tuple`, `Any`, `Optional`

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Functions/methods | snake_case | `extract_keyframes()` |
| Classes | PascalCase | `VideoProcessor` |
| Constants | UPPER_SNAKE | `DEFAULT_REQUEST_PARAMS` |
| Private | `_` prefix | `_load_stt_segments()` |

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

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")
```

---

## Project Structure
```
src/
├── audio/      # STT (ClovaSpeech, Whisper)
├── capture/    # Video frame extraction
├── common/     # Shared Pydantic schemas
├── fusion/     # STT+VLM sync, summarization
├── vlm/        # Vision-Language Model
└── utils/      # Utilities
```

**Key Files:**
- `src/run_video_pipeline.py` - Main pipeline
- `src/common/schemas.py` - Data contracts
- `src/fusion/config.yaml` - Pipeline config

---

## Configuration

Required `.env` variables:
```
CLOVA_SPEECH_URL=...
CLOVA_SPEECH_API_KEY=...
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
```

Output artifacts under `data/outputs/{video_name}/`:
- `stt.json`, `manifest.json`, `vlm.json`, `fusion/`

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

---

## Git Commit Guidelines

- **커밋 메시지는 한글로 작성**
- 제목은 50자 이내, 본문은 72자 줄바꿈 권장
- 형식: `type(scope): 제목`
  - type: feat, fix, refactor, docs, test, chore 등
  - scope: 변경된 모듈/파일 (선택)
- 예시:
  ```
  refactor(summarizer): LLM 요약 품질 개선 - source_type 기반 근거 분리
  
  - 모든 요약 항목에 source_type 필드 추가
  - 배경지식 활용한 풍부한 설명 허용
  - renderer.py에 source_type 표시 반영
  
  Closes #19
  ```
