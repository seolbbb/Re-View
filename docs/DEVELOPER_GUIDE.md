# Screentime MVP 개발자 가이드

이 문서는 팀원들이 각자 맡은 기능을 구현할 때 참고할 수 있는 상세 가이드입니다.

## 목차

1. [전체 아키텍처](#1-전체-아키텍처)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [파이프라인 흐름](#3-파이프라인-흐름)
4. [ADK 멀티에이전트 구조](#4-adk-멀티에이전트-구조)
5. [개발 영역별 가이드](#5-개발-영역별-가이드)
   - [DB 구현](#51-db-구현)
   - [Judge 연결 및 수정](#52-judge-연결-및-수정)
   - [Capture 로직 변경](#53-capture-로직-변경)
6. [테스트 방법](#6-테스트-방법)
7. [자주 묻는 질문](#7-자주-묻는-질문)

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User (mp4 upload)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pre-ADK Pipeline (CLI)                               │
│            python src/pre_adk_pipeline.py --video "xxx.mp4"                 │
│                                                                             │
│   ┌─────────────────┐              ┌─────────────────┐                      │
│   │   STT (Clova)   │   Parallel   │     Capture     │                      │
│   │   -> stt.json   │   Execution  │ -> manifest.json│                      │
│   │                 │              │ -> captures/*.png│                      │
│   └─────────────────┘              └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    data/outputs/{video_name}/                               │
│   Pre-ADK Outputs: stt.json, manifest.json, captures/                       │
│                   (Current: FileSystem / Future: DB)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ADK Pipeline (Interactive)                            │
│                       adk web src/adk_pipeline                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Root Agent (screentime_pipeline)                 │   │
│   │                                                                     │   │
│   │   Tools: list_available_videos, set_pipeline_config, get_status     │   │
│   │   Sub-Agents: preprocessing, summarize, judge                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│          │                         │                         │              │
│          ▼                         ▼                         ▼              │
│   ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│   │Preprocessing│  ──────▶  │  Summarize  │  ──────▶  │    Judge    │       │
│   │   Agent     │           │   Agent     │           │   Agent     │       │
│   │             │           │             │           │             │       │
│   │ VLM + Sync  │           │ Sum + Render│           │ Quality Eval│       │
│   └─────────────┘           └─────────────┘           └─────────────┘       │
│                                    │                         │              │
│                                    │◀── FAIL + can_rerun ────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    data/outputs/{video_name}/fusion/                        │
│   ADK Outputs: segment_summaries.jsonl, final_summary_*.md, judge.json      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 디렉터리 구조

```
Screentime-MVP/
├── data/
│   ├── inputs/                     # 입력 비디오 (mp4)
│   └── outputs/                    # 출력 (DB 대체, 비디오별 폴더)
│       └── {video_name}/
│           ├── stt.json            # STT 결과
│           ├── manifest.json       # 캡처 메타데이터
│           ├── captures/           # 캡처 이미지
│           ├── vlm.json            # VLM 결과
│           ├── config.yaml         # Fusion 설정
│           └── fusion/
│               ├── segments_units.jsonl
│               ├── segment_summaries.jsonl
│               ├── judge.json
│               └── outputs/
│                   └── final_summary_*.md
│
├── src/
│   ├── pre_adk_pipeline.py         # Pre-ADK CLI (STT + Capture)
│   │
│   ├── adk_pipeline/               # ADK 멀티에이전트 파이프라인
│   │   ├── __init__.py
│   │   ├── agent.py                # ⭐ Agent 정의 (Root + Sub-agents)
│   │   ├── store.py                # ⭐ VideoStore (DB 추상화 레이어)
│   │   ├── paths.py                # 경로 상수/유틸
│   │   └── tools/
│   │       ├── root_tools.py       # Root Agent 도구
│   │       ├── preprocessing_tools.py  # Preprocessing Agent 도구
│   │       ├── summarize_tools.py  # Summarize Agent 도구
│   │       ├── judge_tools.py      # ⭐ Judge Agent 도구
│   │       └── internal/           # 내부 구현 모듈
│   │           ├── vlm_openrouter.py
│   │           ├── sync_data.py
│   │           ├── summarize.py
│   │           ├── render_md.py
│   │           ├── final_summary.py
│   │           ├── judge_gemini.py # ⭐ Judge 구현 (현재 stub)
│   │           ├── attempts.py
│   │           ├── fusion_config.py
│   │           └── pre_db.py       # ⭐ Pre-ADK 실행 (Capture 포함)
│   │
│   ├── fusion/                     # Fusion 코어 모듈
│   │   ├── config.py
│   │   ├── sync_engine.py
│   │   ├── summarizer.py
│   │   ├── renderer.py
│   │   └── final_summary_composer.py
│   │
│   ├── audio/                      # STT 관련
│   │   └── stt_router.py
│   │
│   └── capture/                    # ⭐ Capture 관련
│       └── process_content.py
│
└── docs/
    └── DEVELOPER_GUIDE.md          # 이 문서
```

---

## 3. 파이프라인 흐름

### 3.1 Pre-ADK 단계

```bash
python src/pre_adk_pipeline.py --video "my_video.mp4"
```

**흐름:**
1. `pre_adk_pipeline.py` → `tools/internal/pre_db.py`
2. `pre_db.py` → `src/audio/stt_router.py` (STT 실행)
3. `pre_db.py` → `src/capture/process_content.py` (Capture 실행)

**산출물:**
- `data/outputs/{video_name}/stt.json`
- `data/outputs/{video_name}/manifest.json`
- `data/outputs/{video_name}/captures/*.png`


### 3.2 ADK 단계

```bash
adk web src/adk_pipeline
```

**흐름:**
1. Root Agent가 사용자와 대화
2. `set_pipeline_config`로 비디오 설정
3. `preprocessing_agent`로 transfer
   - `load_data` → `run_vlm` → `run_sync`
   - 완료 후 Root로 복귀
4. `summarize_agent`로 transfer
   - `run_summarizer` → `render_md` → `write_final_summary`
   - 완료 후 Root로 복귀
5. `judge_agent`로 transfer
   - `evaluate_summary`
   - PASS → 완료 / FAIL + can_rerun → summarize 재실행

---

## 4. ADK 멀티에이전트 구조

### 4.1 Agent 정의 (`agent.py`)

```python
# Sub-Agents
preprocessing_agent = Agent(
    name="preprocessing_agent",
    model="gemini-2.0-flash",
    instruction="...",
    tools=[load_data, run_vlm, run_sync],  # 이 agent가 사용할 도구
)

# Root Agent
root_agent = Agent(
    name="screentime_pipeline",
    model="gemini-2.0-flash",
    instruction="...",
    tools=[list_available_videos, set_pipeline_config, get_pipeline_status],
    sub_agents=[preprocessing_agent, summarize_agent, judge_agent],  # transfer 가능한 agent들
)

# Sub-agent가 Root로 돌아갈 수 있도록 설정
preprocessing_agent._sub_agents = [root_agent]
```

### 4.2 Agent Transfer 흐름

```
Root Agent
    │
    ├── transfer → preprocessing_agent
    │                   │
    │                   ├── load_data()
    │                   ├── run_vlm()
    │                   ├── run_sync()
    │                   │
    │   ◀── transfer ───┘ (완료 후 Root로 복귀)
    │
    ├── transfer → summarize_agent
    │                   │
    │                   ├── run_summarizer()
    │                   ├── render_md()
    │                   ├── write_final_summary()
    │                   │
    │   ◀── transfer ───┘
    │
    ├── transfer → judge_agent
    │                   │
    │                   ├── evaluate_summary()
    │                   │
    │   ◀── transfer ───┘ (결과: PASS/FAIL)
    │
    └── FAIL + can_rerun? → transfer → summarize_agent (재실행)
```

### 4.3 Tool 구조

**Tool = ADK가 호출하는 함수**

```python
# tools/preprocessing_tools.py

def run_vlm(tool_context: ToolContext) -> Dict[str, Any]:
    """ADK Tool 함수 - ToolContext를 통해 상태(state) 접근"""

    # 1. state에서 설정 값 가져오기
    video_name = tool_context.state.get("video_name")

    # 2. VideoStore로 경로 관리
    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    # 3. 내부 구현 호출
    from .internal.vlm_openrouter import run_vlm_openrouter
    run_vlm_openrouter(...)

    # 4. 결과 반환 (Agent에게 전달됨)
    return {"success": True, "vlm_json": str(store.vlm_json())}
```

### 4.4 State 관리

ADK는 `tool_context.state`를 통해 세션 상태를 관리합니다:

```python
# 상태 읽기
video_name = tool_context.state.get("video_name")

# 상태 쓰기
tool_context.state["current_rerun"] = 1
```

**주요 state 키:**
- `video_name`: 현재 처리 중인 비디오 이름
- `max_reruns`: 최대 재실행 횟수
- `current_rerun`: 현재 재실행 횟수
- `summarize_prompt`: 커스텀 요약 프롬프트

---

## 5. 개발 영역별 가이드

### 5.0 코드 의존성 및 수정 가이드

이 프로젝트는 **ADK(Agent) 계층**과 **Core(Business Logic) 계층**이 분리되어 있습니다. 코드를 수정할 때 어떤 파일을 건드려야 하는지 파악하는 것이 중요합니다.

**의존성 흐름:**

```
[ADK Agent] (src/adk_pipeline/agent.py)
    │
    ▼
[ADK Tools] (src/adk_pipeline/tools/*.py)
    │  - 역할: 상태(State) 관리, 경로(VideoStore) 처리, 에러 핸들링
    │  - 예: preprocessing_tools.py, summarize_tools.py
    │
    ▼
[Internal Bridge] (src/adk_pipeline/tools/internal/*.py)
    │  - 역할: ADK Tool과 Core 모듈 간의 연결 고리 (파라미터 변환 등)
    │  - 예: vlm_openrouter.py, sync_data.py, summarize.py
    │
    ▼
[Core Modules] (src/audio, src/capture, src/fusion, src/vlm)
    │  - 역할: 실제 비즈니스 로직, 알고리즘, 데이터 처리
    │  - 예: vlm_engine.py, sync_engine.py, summarizer.py
```

**수정 가이드:**

| 수정 목표 | 수정할 위치 | 예시 |
|---|---|---|
| **핵심 알고리즘 변경** | **Core Modules** (`src/{module}/`) | 캡처 알고리즘 개선, 요약 프롬프트 변경, Sync 로직 수정 |
| **ADK 연동 방식 변경** | **Internal Bridge** (`tools/internal/`) | Core 모듈 호출 파라미터 변경, 리턴값 포맷 변경 |
| **Agent 동작/상태 변경** | **ADK Tools** (`tools/*.py`) | 재실행 카운트 로직, 에러 메시지 처리, 선행 조건 검사 |
| **새로운 Tool 추가** | **ADK Tools** + **Agent** | 새 기능을 Agent에 노출 |

---

### 5.1 DB 구현

**목표**: 현재 파일시스템 기반 저장소를 DB로 교체

**현재 구조:**
```
data/outputs/{video_name}/
├── stt.json
├── manifest.json
├── captures/
├── vlm.json
└── fusion/
    ├── segment_summaries.jsonl
    └── judge.json
```

**수정해야 할 파일들:**

#### 1) `src/adk_pipeline/store.py` (핵심)

현재 VideoStore는 파일 경로만 제공합니다:

```python
class VideoStore:
    def __init__(self, output_base: Path, video_name: str):
        self._output_base = output_base
        self._video_name = video_name

    def video_root(self) -> Path:
        return self._output_base / self._video_name

    def stt_json(self) -> Path:
        return self.video_root() / "stt.json"

    # ... 다른 경로 메서드들
```

**DB 구현 시 수정 방향:**

```python
class VideoStore:
    def __init__(self, video_name: str, db_client: DatabaseClient):
        self._video_name = video_name
        self._db = db_client

    # 읽기 메서드 추가
    def get_stt_data(self) -> dict:
        return self._db.query("SELECT data FROM stt WHERE video_name = ?", self._video_name)

    # 쓰기 메서드 추가
    def save_stt_data(self, data: dict) -> None:
        self._db.execute("INSERT INTO stt ...", data)

    # 존재 여부 확인
    def has_stt(self) -> bool:
        return self._db.exists("stt", video_name=self._video_name)
```

#### 2) `src/adk_pipeline/tools/root_tools.py`

`list_available_videos` 함수 수정:

```python
# 현재 (파일시스템)
def list_available_videos(tool_context: ToolContext) -> Dict[str, Any]:
    videos = []
    for video_dir in _OUTPUT_BASE.iterdir():
        if video_dir.is_dir():
            # 파일 존재 여부 확인
            ...

# DB 구현 시
def list_available_videos(tool_context: ToolContext) -> Dict[str, Any]:
    db = get_db_client()
    videos = db.query("SELECT DISTINCT video_name FROM videos WHERE pre_adk_complete = true")
    ...
```

#### 3) `src/adk_pipeline/tools/preprocessing_tools.py`

`load_data` 함수 수정:

```python
# 현재
def load_data(tool_context: ToolContext) -> Dict[str, Any]:
    store = VideoStore(...)
    if not store.stt_json().exists():  # 파일 존재 확인
        return {"success": False, "error": "..."}

# DB 구현 시
def load_data(tool_context: ToolContext) -> Dict[str, Any]:
    store = VideoStore(video_name=video_name, db_client=get_db())
    if not store.has_stt():  # DB 존재 확인
        return {"success": False, "error": "..."}
```

#### 4) `src/fusion/io_utils.py`

JSON 읽기/쓰기 유틸리티 → DB 읽기/쓰기로 확장:

```python
# 현재
def read_json(path: Path, desc: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# DB 구현 시 (추가)
def read_from_db(table: str, video_name: str, db: DatabaseClient) -> dict:
    return db.query(f"SELECT data FROM {table} WHERE video_name = ?", video_name)
```

**DB 스키마 예시:**

```sql
-- 비디오 메타데이터
CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    video_name VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    pre_adk_complete BOOLEAN DEFAULT FALSE,
    adk_complete BOOLEAN DEFAULT FALSE
);

-- STT 결과
CREATE TABLE stt_results (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 캡처 메타데이터
CREATE TABLE captures (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    manifest JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 캡처 이미지 (BLOB 또는 S3 URL)
CREATE TABLE capture_images (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    timestamp_ms INTEGER,
    image_url VARCHAR(512),  -- 또는 BYTEA for BLOB
    created_at TIMESTAMP DEFAULT NOW()
);

-- VLM 결과
CREATE TABLE vlm_results (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 요약 결과
CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    attempt INTEGER DEFAULT 1,
    segment_summaries JSONB,
    final_summaries JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Judge 결과
CREATE TABLE judge_results (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id),
    attempt INTEGER,
    passed BOOLEAN,
    score FLOAT,
    reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**테스트:**
```bash
# 1. VideoStore 단위 테스트
pytest tests/test_store.py

# 2. ADK 파이프라인 통합 테스트
adk web src/adk_pipeline
# → "test3_Diffusion 파이프라인 실행해줘"
```

---

### 5.2 Judge 연결 및 수정

**목표**: 현재 stub (항상 PASS)인 Judge를 실제 평가 로직으로 교체

**현재 구조:**

```
tools/judge_tools.py          # ADK Tool (evaluate_summary)
    │
    └── internal/judge_gemini.py  # 실제 구현 (현재 stub)
```

**수정해야 할 파일들:**

#### 1) `src/adk_pipeline/tools/internal/judge_gemini.py` (핵심)

```python
# 현재 (stub - 항상 PASS)
def judge_stub_gemini(*, fusion_dir: Path) -> Dict[str, Any]:
    result = {
        "schema_version": 1,
        "pass": True,  # 항상 True
        "score": 1.0,
        "reason": "stub judge - 항상 통과",
    }
    judge_path = fusion_dir / "judge.json"
    write_json(judge_path, result)
    return result
```

**실제 Judge 구현 방향:**

```python
def judge_with_gemini(*, fusion_dir: Path, config: Optional[dict] = None) -> Dict[str, Any]:
    """Gemini를 사용한 요약 품질 평가.

    평가 기준:
    1. 정보 완전성: 원본 내용이 요약에 충분히 반영되었는가?
    2. 정확성: 요약이 원본과 일치하는가? (hallucination 없는가?)
    3. 일관성: 요약 내부에서 모순이 없는가?
    4. 가독성: 요약이 이해하기 쉬운가?

    Returns:
        {
            "pass": bool,          # 통과 여부
            "score": float,        # 0.0 ~ 1.0
            "reason": str,         # 평가 이유
            "details": {           # 상세 평가
                "completeness": float,
                "accuracy": float,
                "consistency": float,
                "readability": float,
            }
        }
    """
    # 1. 입력 데이터 로드
    segments_units = read_jsonl(fusion_dir / "segments_units.jsonl")
    segment_summaries = read_jsonl(fusion_dir / "segment_summaries.jsonl")

    # 2. Gemini 클라이언트 초기화
    client = init_gemini_client()

    # 3. 평가 프롬프트 구성
    prompt = _build_judge_prompt(segments_units, segment_summaries)

    # 4. Gemini 호출
    response = client.generate_content(prompt)

    # 5. 응답 파싱
    result = _parse_judge_response(response)

    # 6. 통과 기준 적용
    threshold = config.get("pass_threshold", 0.7) if config else 0.7
    result["pass"] = result["score"] >= threshold

    # 7. 결과 저장
    write_json(fusion_dir / "judge.json", result)

    return result


def _build_judge_prompt(segments_units: list, summaries: list) -> str:
    """평가 프롬프트 구성."""
    return f"""
당신은 요약 품질을 평가하는 전문가입니다.

## 원본 데이터 (segments_units)
{json.dumps(segments_units, ensure_ascii=False, indent=2)}

## 생성된 요약 (segment_summaries)
{json.dumps(summaries, ensure_ascii=False, indent=2)}

## 평가 기준
1. 정보 완전성 (0-1): 원본의 핵심 정보가 요약에 포함되어 있는가?
2. 정확성 (0-1): 요약이 원본과 일치하는가? 없는 내용을 만들어내지 않았는가?
3. 일관성 (0-1): 요약 내부에서 모순되는 내용이 없는가?
4. 가독성 (0-1): 요약이 명확하고 이해하기 쉬운가?

## 출력 형식 (JSON)
{{
    "completeness": 0.0-1.0,
    "accuracy": 0.0-1.0,
    "consistency": 0.0-1.0,
    "readability": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reason": "평가 이유 설명"
}}

평가 결과를 JSON으로만 출력하세요.
"""
```

#### 2) `src/adk_pipeline/tools/judge_tools.py`

Tool 인터페이스는 유지하되, 내부 구현만 교체:

```python
def evaluate_summary(tool_context: ToolContext) -> Dict[str, Any]:
    video_name = tool_context.state.get("video_name")
    store = VideoStore(output_base=_OUTPUT_BASE, video_name=video_name)

    try:
        # 변경: stub → 실제 구현
        from .internal.judge_gemini import judge_with_gemini  # 함수명 변경

        # Judge 설정 (optional)
        judge_config = {
            "pass_threshold": 0.7,  # 통과 기준
            "model": "gemini-2.0-flash",
        }

        result = judge_with_gemini(
            fusion_dir=store.fusion_dir(),
            config=judge_config,
        )

        # 재실행 가능 여부
        current_rerun = tool_context.state.get("current_rerun", 1)
        max_reruns = tool_context.state.get("max_reruns", 2)
        can_rerun = not result["pass"] and current_rerun <= max_reruns

        return {
            "success": True,
            "result": "PASS" if result["pass"] else "FAIL",
            "score": result["score"],
            "reason": result["reason"],
            "details": result.get("details", {}),
            "can_rerun": can_rerun,
            "attempt": current_rerun,
        }
    except Exception as e:
        return {"success": False, "error": f"Judge 실행 실패: {e}"}
```

#### 3) 평가 기준 커스터마이징

`src/fusion/config.yaml`에 Judge 설정 추가:

```yaml
judge:
  enabled: true
  pass_threshold: 0.7
  weights:
    completeness: 0.3
    accuracy: 0.3
    consistency: 0.2
    readability: 0.2
  model: "gemini-2.0-flash"
  temperature: 0.1
```

**테스트:**
```bash
# 1. Judge 단위 테스트
python -c "
from src.adk_pipeline.tools.internal.judge_gemini import judge_with_gemini
from pathlib import Path
result = judge_with_gemini(fusion_dir=Path('data/outputs/test3_Diffusion/fusion'))
print(result)
"

# 2. ADK 파이프라인 통합 테스트
adk web src/adk_pipeline
# → 파이프라인 실행 후 Judge 결과 확인
```

---

### 5.3 Capture 로직 변경

**목표**: 캡처 추출 로직 수정 (임계값, 알고리즘, 출력 형식 등)

**현재 구조:**

```
pre_adk_pipeline.py
    │
    └── tools/internal/pre_db.py
            │
            └── src/capture/process_content.py  # 실제 캡처 로직
```

**수정해야 할 파일들:**

#### 1) `src/capture/process_content.py` (핵심)

```python
def process_single_video_capture(
    video_path: str,
    output_base: str,
    scene_threshold: float = 3.0,      # 장면 전환 감지 임계값
    dedupe_threshold: float = 3.0,     # 중복 제거 임계값
    min_interval: float = 0.5,         # 최소 캡처 간격 (초)
) -> List[Dict[str, Any]]:
    """비디오에서 캡처를 추출합니다.

    Args:
        video_path: 입력 비디오 경로
        output_base: 출력 디렉터리
        scene_threshold: 장면 전환 감지 임계값 (낮을수록 민감)
        dedupe_threshold: 중복 제거 임계값 (낮을수록 엄격)
        min_interval: 캡처 간 최소 간격 (초)

    Returns:
        캡처 메타데이터 리스트 (manifest.json에 저장됨)
        [
            {
                "timestamp_ms": 1000,
                "filename": "frame_0001.png",
                "diff_score": 0.85,
            },
            ...
        ]
    """
    # 구현...
```

**수정 예시:**

```python
# 새로운 캡처 알고리즘 추가
def process_single_video_capture_v2(
    video_path: str,
    output_base: str,
    method: str = "scene_detect",  # "scene_detect" | "fixed_interval" | "content_aware"
    **kwargs,
) -> List[Dict[str, Any]]:
    """개선된 캡처 추출.

    Methods:
        - scene_detect: 장면 전환 기반 (기존)
        - fixed_interval: 고정 간격 캡처
        - content_aware: 콘텐츠 변화량 기반
    """
    if method == "scene_detect":
        return _capture_scene_detect(video_path, output_base, **kwargs)
    elif method == "fixed_interval":
        return _capture_fixed_interval(video_path, output_base, **kwargs)
    elif method == "content_aware":
        return _capture_content_aware(video_path, output_base, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _capture_content_aware(
    video_path: str,
    output_base: str,
    sensitivity: float = 0.3,
    max_captures_per_minute: int = 10,
) -> List[Dict[str, Any]]:
    """콘텐츠 변화량 기반 캡처.

    슬라이드 전환, 중요 시각적 변화를 감지합니다.
    """
    # 구현...
```

#### 2) `src/adk_pipeline/tools/internal/pre_db.py`

캡처 함수 호출 부분 수정:

```python
def run_capture(
    *,
    video_path: Path,
    output_base: Path,
    scene_threshold: float,
    dedupe_threshold: float,
    min_interval: float,
    method: str = "scene_detect",  # 새로운 파라미터
) -> List[Dict[str, Any]]:
    """캡처 실행."""
    # 새 함수 사용
    from src.capture.process_content import process_single_video_capture_v2

    metadata = process_single_video_capture_v2(
        str(video_path),
        str(output_base),
        method=method,
        scene_threshold=scene_threshold,
        dedupe_threshold=dedupe_threshold,
        min_interval=min_interval,
    )
    return metadata
```

#### 3) `src/pre_adk_pipeline.py`

CLI 옵션 추가:

```python
parser.add_argument(
    "--capture-method",
    choices=["scene_detect", "fixed_interval", "content_aware"],
    default="scene_detect",
    help="캡처 추출 방법 (기본: scene_detect)",
)

parser.add_argument(
    "--max-captures-per-minute",
    type=int,
    default=10,
    help="분당 최대 캡처 수 (content_aware 방법용)",
)
```

#### 4) manifest.json 스키마 확장

```python
# 현재
{
    "timestamp_ms": 1000,
    "filename": "frame_0001.png",
    "diff_score": 0.85,
}

# 확장
{
    "timestamp_ms": 1000,
    "filename": "frame_0001.png",
    "diff_score": 0.85,
    "capture_method": "content_aware",  # 사용된 방법
    "content_type": "slide_change",     # 캡처 이유
    "ocr_hint": "Title: Introduction",  # OCR 힌트 (optional)
}
```

**테스트:**
```bash
# 1. Capture 단위 테스트
python -c "
from src.capture.process_content import process_single_video_capture
result = process_single_video_capture(
    'data/inputs/test.mp4',
    'data/outputs/test',
    scene_threshold=3.0,
)
print(f'Captured {len(result)} frames')
"

# 2. Pre-ADK 파이프라인 테스트
python src/pre_adk_pipeline.py --video "test.mp4" --capture-threshold 2.0

# 3. 새 캡처 방법 테스트
python src/pre_adk_pipeline.py --video "test.mp4" --capture-method content_aware
```

---

## 6. 테스트 방법

### 6.1 개별 모듈 테스트

```bash
# VideoStore 테스트
python -c "
from src.adk_pipeline.store import VideoStore
from pathlib import Path
store = VideoStore(Path('data/outputs'), 'test3_Diffusion')
print(store.stt_json())
print(store.stt_json().exists())
"

# Tool 테스트 (ToolContext 모킹 필요)
python -c "
from unittest.mock import MagicMock
from src.adk_pipeline.tools.preprocessing_tools import load_data

ctx = MagicMock()
ctx.state = {'video_name': 'test3_Diffusion'}
result = load_data(ctx)
print(result)
"
```

### 6.2 Pre-ADK 파이프라인 테스트

```bash
# 기본 실행
python src/pre_adk_pipeline.py --video "my_video.mp4"

# 옵션 변경
python src/pre_adk_pipeline.py --video "my_video.mp4" \
    --stt-backend whisper \
    --capture-threshold 2.0
```

### 6.3 ADK 파이프라인 테스트

```bash
# ADK Web 실행
adk web src/adk_pipeline

# 브라우저에서 테스트
# 1. "처리 가능한 비디오 목록 보여줘"
# 2. "test3_Diffusion으로 파이프라인 실행해줘"
# 3. 각 단계 진행 확인
```

### 6.4 특정 Agent/Tool 디버깅

```python
# agent.py에 로깅 추가
import logging
logging.basicConfig(level=logging.DEBUG)

# Tool 함수에 디버그 출력 추가
def my_tool(tool_context: ToolContext) -> Dict[str, Any]:
    print(f"[DEBUG] state: {dict(tool_context.state)}")
    # ...
```

---

## 7. 자주 묻는 질문

### Q1: ADK Agent에 새 Tool을 추가하려면?

1. `tools/` 아래에 Tool 함수 정의
2. `agent.py`에서 import 후 Agent의 `tools=[]`에 추가

```python
# tools/my_new_tools.py
def my_new_tool(tool_context: ToolContext) -> Dict[str, Any]:
    return {"result": "success"}

# agent.py
from .tools.my_new_tools import my_new_tool

my_agent = Agent(
    ...,
    tools=[existing_tool, my_new_tool],  # 추가
)
```

### Q2: Sub-Agent를 추가하려면?

1. 새 Agent 정의
2. Root Agent의 `sub_agents=[]`에 추가
3. 새 Agent의 `_sub_agents`에 root_agent 추가 (복귀용)

```python
new_agent = Agent(
    name="new_agent",
    ...,
    tools=[...],
)

root_agent = Agent(
    ...,
    sub_agents=[..., new_agent],  # 추가
)

new_agent._sub_agents = [root_agent]  # 복귀용
```

### Q3: State에 새 값을 추가하려면?

`set_pipeline_config` Tool에서 설정하거나, 다른 Tool에서 직접 설정:

```python
# root_tools.py의 set_pipeline_config 수정
def set_pipeline_config(
    tool_context: ToolContext,
    video_name: str,
    my_new_config: str = None,  # 새 파라미터
) -> Dict[str, Any]:
    tool_context.state["video_name"] = video_name
    tool_context.state["my_new_config"] = my_new_config  # 추가
    ...
```

### Q4: 파일 경로를 DB 쿼리로 바꾸려면?

VideoStore 메서드를 수정하고, 호출하는 곳의 패턴을 변경:

```python
# 현재 패턴
if store.stt_json().exists():
    data = read_json(store.stt_json())

# DB 패턴
if store.has_stt():
    data = store.get_stt_data()
```

### Q5: Judge 통과 기준을 바꾸려면?

`tools/internal/judge_gemini.py`에서:

```python
# 기준 변경
threshold = 0.8  # 기존 0.7에서 상향

# 또는 config에서 읽기
threshold = config.get("pass_threshold", 0.7)
```

---

## 부록: 주요 파일 요약

| 파일 | 역할 | 수정 시점 |
|------|------|----------|
| `agent.py` | Agent 정의 | 새 Agent/Tool 추가 시 |
| `store.py` | 저장소 추상화 | DB 구현 시 |
| `tools/*_tools.py` | ADK Tool 인터페이스 | 새 기능 추가 시 |
| `tools/internal/*.py` | 실제 구현 | 로직 변경 시 |
| `fusion/*.py` | 코어 처리 모듈 | 알고리즘 변경 시 |
| `capture/process_content.py` | 캡처 로직 | 캡처 방식 변경 시 |
| `audio/stt_router.py` | STT 로직 | STT 백엔드 추가 시 |

---

## 연락처

질문이나 이슈가 있으면 팀 채널에 공유해주세요.
