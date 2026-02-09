# 챗봇 기능

## 개요

LangGraph 기반 RAG 챗봇으로, 생성된 강의 요약을 바탕으로 질의응답을 지원합니다.

**주요 기능:**
- SSE 스트리밍 응답
- 후속 질문 추천 (Follow-up Suggestions)
- 근거 데이터 참조 (STT/VLM evidence)

---

## 아키텍처

```
사용자 질문
    ↓
POST /api/chat/stream
    ↓
LangGraph Session
    ├─ RAG 컨텍스트 검색 (pgvector)
    ├─ 답변 생성 (SSE 스트리밍)
    └─ 후속 질문 생성 (Gemini Flash)
        ↓
SSE Events → 프론트엔드
```

### SSE 이벤트 타입

| 이벤트 | 데이터 | 설명 |
|---|---|---|
| `answer` | 텍스트 청크 | 답변 스트리밍 |
| `suggestions` | `{"questions": [...]}` | 후속 질문 배열 |
| `error` | 에러 메시지 | 오류 |
| `done` | — | 스트리밍 완료 |

---

## 후속 질문 추천

답변 완료 후 LLM이 1~3개의 후속 질문을 생성하여 칩(chip) 형태로 표시합니다.

### 핵심 파일

| 위치 | 함수 | 역할 |
|---|---|---|
| `src/services/langgraph_session.py` | `generate_suggestions()` | LangGraph 노드, LLM 호출 + 결과 저장 |
| `src/services/langgraph_session.py` | `_build_suggestions_prompt()` | 프롬프트 구성 (Few-shot 포함) |
| `src/services/langgraph_session.py` | `_extract_questions_from_text()` | JSON 파싱 + 정규화 + 중복 제거 |
| `frontend/src/components/ChatBot.jsx` | SSE `suggestions` 이벤트 핸들러 | 칩 렌더링 + 클릭 처리 |

### 환경 변수

```env
CHATBOT_ENABLE_GRAPH_SUGGESTIONS=true      # 기능 활성화
CHATBOT_SUGGESTION_MODEL=gemini-2.0-flash-exp  # 사용 모델
CHATBOT_SUGGESTION_MAX_ITEMS=3             # 최대 질문 수
CHATBOT_SUGGESTION_MAX_CHARS=80            # 질문당 최대 글자 수
```

---

## ADK 챗봇 (멀티 에이전트)

Google ADK 기반 멀티 에이전트 챗봇으로, 요약 생성 상태 확인 및 대화 모드 전환을 지원합니다.

### 핵심 파일

| 위치 | 역할 |
|---|---|
| `src/adk_chatbot/agent.py` | 에이전트 함수 (모드 선택, 요약 상태, 컨텍스트 조회) |
| `src/adk_chatbot/store.py` | 비디오 메타데이터 스토어 |
| `src/services/adk_session.py` | ADK 세션 관리 |
