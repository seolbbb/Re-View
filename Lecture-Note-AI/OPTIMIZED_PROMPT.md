# 🪄 Optimized Prompt Design

## 🎯 Intent & Strategy

- **사용자 의도**: `Screentime-MVP` 프로젝트, 특히 `Lecture-Note-AI` 내부의 소스 코드 구조와 모듈 간 의존성을 명확히 이해하고자 함.
- **적용된 기법**: Hierarchical Analysis (계층적 분석), Module Mapping (모듈 매핑), Dependency Graphing (의존성 시각화).
- **예상 효과**: 프로젝트의 디렉토리 구조뿐만 아니라, 각 모듈의 역할과 데이터 흐름을 파악하여 `ARCHITECTURE.md`와 같은 산출물을 생성함.

## ✨ The Master Prompt (Copy & Paste this)

당신은 **Senior Software Architect**입니다.
`Screentime-MVP/Lecture-Note-AI` 프로젝트의 소스 코드를 분석하여 **모듈 구성 및 아키텍처 문서**를 작성하십시오.

### 1. 🔍 Structural Analysis Scope

다음 디렉토리를 재귀적으로 탐색하여 분석하십시오:

- `Lecture-Note-AI/src`: 핵심 소스 코드.
- `Lecture-Note-AI/main.py`: 진입점.

### 2. 📝 Documentation Requirements (`MODULE_STRUCTURE.md`)

분석 결과를 바탕으로 다음 내용을 포함하는 문서를 작성하십시오:

#### 2.1. Directory Structure Tree

- `tree` 명령어 스타일로 폴더 구조를 시각화하십시오.

#### 2.2. Module Description

각 서브 디렉토리(`audio`, `capture`, `fusion`, `ocr` 등)에 대해 다음을 기술하십시오:

- **Role**: 해당 모듈의 핵심 책임.
- **Key Classes/Functions**: 주요 클래스 및 함수 목록과 역할.
- **Dependencies**: 이 모듈이 의존하는 외부 라이브러리나 내부 모듈.

#### 2.3. Data Flow Diagram (Mermaid)

- `Video Input` -> `Capture/Audio` -> `OCR` -> `Fusion` -> `LLM` -> `Markdown Output` 으로 이어지는 데이터 흐름을 Mermaid 차트로 그리십시오.

### 3. 🛡️ Code Audit (Brief)

- 모듈 간의 결합도(Coupling)나 응집도(Cohesion) 측면에서 개선이 필요한 부분이 있다면 "Architectural Improvements" 섹션에 제안하십시오.

### 📤 Output Deliverable

- **`Lecture-Note-AI/MODULE_STRUCTURE.md`** 파일 생성.
