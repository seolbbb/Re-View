# 🪄 Optimized Prompt Design

## 🎯 Intent & Strategy

- **사용자 의도**: `capture_test` 폴더에 임시로 복사된 코드들이 협업 환경(`Screentime-MVP`)에서 문제없이 작동하는지 검증하고, 프로덕션 수준으로 정리(Update)하기를 원함.
- **적용된 기법**: Code Audit (협업 적합성 검사), Path Normalization (경로 정규화), Dependency Check (의존성 확인).
- **예상 효과**: 하드코딩된 경로 제거, 환경 변수 분리, 그리고 `capture_test`의 코드를 `Lecture-Note-AI` 구조로 통합하는 제안.

## ✨ The Master Prompt (Copy & Paste this)

당신은 **Senior Software Architect**입니다.
`Screentime-MVP/capture_test` 디렉토리의 코드를 분석하여 **협업 적합성(Collaboration Readiness)**을 진단하고 수정하십시오.

### 1. 🔍 Analysis Scope

- `capture_test/` 내부의 모든 `.py` 파일.
- `Lecture-Note-AI/src/` (기존 모듈과의 통합 가능성 확인).

### 2. 📝 Diagnosis Points (Zero Tolerance)

1. **Absolute Paths**: `C:\Users\...` 형태의 절대 경로가 단 하나라도 존재하면 즉시 수정 대상으로 지목하십시오.
2. **Missing Dependencies**: `import` 문을 분석하여 `requirements.txt`에 누락된 패키지가 있는지 확인하십시오.
3. **Environment Variables**: API Key나 비밀번호가 코드에 하드코딩되어 있는지 확인하십시오.

### 3. 🛠️ Action Plan

분석 후 다음 작업을 수행하십시오:

- **Refactor**: 하드코딩된 경로를 `os.path`나 `pathlib`을 사용한 상대 경로로 변경.
- **Integrate**: `capture_test`의 기능을 `Lecture-Note-AI` 프로젝트 구조에 맞게 이동하거나 정리 제안.
- **Document**: 변경 사항을 `COLLABORATION_REVIEW.md`에 기록.

### 📤 Output Deliverable

- 수정된 코드 파일들.
- **`COLLABORATION_REVIEW.md`**
