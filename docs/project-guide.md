# 프로젝트 가이드

## 핵심 진입점

| 파일 | 용도 |
|---|---|
| `src/run_pipeline_demo_async.py` | **비동기 파이프라인** (Producer → VLM → Fusion 동시 실행) |
| `src/run_pipeline_demo.py` | 순차 파이프라인 (Preprocess → Process) |
| `src/run_preprocess_pipeline.py` | 전처리 전용 (STT + Capture) |
| `src/run_process_pipeline.py` | 처리 전용 (VLM + Fusion + Judge) |
| `src/run_fusion_only.py` | Fusion만 재실행 (프롬프트 수정 후 빠른 반복용) |
| `src/process_api.py` | FastAPI 서버 (REST API + SSE) |

---

## 모듈별 핵심 파일

### STT (`src/audio/`)
- `stt_router.py` — STT 백엔드 선택/라우팅
- `clova_stt.py` — Clova Speech API 호출 및 응답 정규화

### Capture (`src/capture/`)
- `process_content.py` — 캡처 실행 엔트리 (슬라이드 캡처 + manifest 생성)
- `tools/hybrid_extractor.py` — 슬라이드 전환 감지/중복 제거 알고리즘

### VLM (`src/vlm/`)
- `vlm_engine.py` — Qwen VLM 호출 + 결과 수집
- `vlm_fusion.py` — vlm_raw.json → vlm.json 변환

### Fusion (`src/fusion/`)
- `sync_engine.py` — STT/VLM 동기화 + 세그먼트 생성
- `summarizer.py` — LLM 요약 생성

### Judge (`src/judge/`)
- `judge.py` — 요약 품질 평가 (Gemini 기반, 다축 평가)

### Pipeline Async (`src/pipeline/`)
- `orchestrator_async.py` — AsyncPipelineOrchestrator (큐 관리 + 조율)
- `producers_async.py` — AsyncCaptureSttProducer (캡처 + STT 이벤트 생산)
- `vlm_worker_async.py` — 비동기 VLM 워커
- `fusion_worker_async.py` — 비동기 Fusion 워커 (요약 + 판정)
- `contracts.py` — 이벤트/데이터 계약 (EndOfStreamEvent, FusionDoneEvent 등)
- `stages.py` — 파이프라인 단계 함수 (VLM, Fusion, Sync, 렌더링)
- `cancel.py` — 파이프라인 취소/삭제 메커니즘
- `benchmark.py` — 성능 벤치마크

### DB (`src/db/`)
- `pipeline_sync.py` — 파이프라인 완료 후 Supabase 동기화
- `adapters/video_adapter.py` — 비디오 메타데이터 CRUD
- `adapters/capture_adapter.py` — 캡처 이미지 관련 DB 작업
- `adapters/content_adapter.py` — 콘텐츠/세그먼트 DB 작업
- `adapters/job_adapter.py` — 전처리/처리 Job 추적
- `supabase_schema.sql` — DB 스키마 (PostgreSQL + pgvector)

### Services (`src/services/`)
- `pipeline_service.py` — 파이프라인 오케스트레이션 서비스
- `langgraph_session.py` — LangGraph 기반 챗봇 세션 (멀티모달)
- `adk_session.py` — ADK 에이전트 세션 관리
- `chat_llm_config.py` — 챗봇 LLM 설정 로더
- `chat_session_store.py` — 세션 저장/수명 관리
- `summary_backend.py` — 요약 생성 백엔드

### ADK Chatbot (`src/adk_chatbot/`)
- `agent.py` — Google ADK 기반 멀티 에이전트 챗봇
- `store.py` — 비디오 메타데이터 스토어

### LLM (`src/llm/`)
- 글로벌 Rate Limiter + API Key 라우팅

---

## 프론트엔드 (`frontend/src/`)

### Pages
| 파일 | 용도 |
|---|---|
| `HomePage.jsx` | 대시보드 (비디오 목록 + 업로드) |
| `AnalysisPage.jsx` | 분석 결과 (요약 + 챗봇 + 플레이어) |
| `LoadingPage.jsx` | 3단계 로딩 UI (전처리 → 처리 → 완료) |
| `LoginPage.jsx` | 로그인 |
| `SignupPage.jsx` | 회원가입 |

### 주요 Components
| 파일 | 용도 |
|---|---|
| `SummaryPanel.jsx` | 요약 패널 (비동기 진행률 표시) |
| `ChatBot.jsx` | 챗봇 인터페이스 (SSE 스트리밍) |
| `VideoPlayer.jsx` | 비디오 재생기 |
| `UploadArea.jsx` | 드래그&드롭 업로드 |
| `MarkdownRenderer.jsx` | 마크다운 렌더링 (KaTeX 수식 지원) |
| `Header.jsx` / `Sidebar.jsx` | 네비게이션 |

### Hooks
| 파일 | 용도 |
|---|---|
| `useVideoStatusStream.js` | SSE 실시간 상태 스트림 |
| `usePolling.js` | 범용 폴링 |

### API
| 파일 | 용도 |
|---|---|
| `client.js` | HTTP 클라이언트 설정 |
| `videos.js` | 비디오 API 호출 |
| `chat.js` | 챗봇 API 호출 |

---

## 설정 파일

| 경로 | 용도 |
|---|---|
| `config/pipeline/settings.yaml` | 파이프라인 런타임 설정 |
| `config/capture/settings.yaml` | 캡처 엔진 설정 |
| `config/audio/settings.yaml` | STT 설정 |
| `config/vlm/settings.yaml` | VLM 설정 |
| `config/fusion/settings.yaml` | Fusion 설정 |
| `config/judge/settings.yaml` | Judge 설정 |
| `config/*/prompts.yaml` | 각 모듈 프롬프트 |

---

## 출력 구조

기본 출력 폴더: `data/outputs/<video_name>/`

### 배치 모드 OFF
```
data/outputs/<video>/
  ├─ stt.json                          # STT 세그먼트
  ├─ manifest.json                     # 캡처 이미지 목록
  ├─ captures/*.jpg                    # 캡처 이미지
  ├─ vlm.json                          # VLM 분석 결과
  ├─ pipeline_run.json                 # 실행 메타데이터
  ├─ fusion/
  │   ├─ segments_units.jsonl          # STT+VLM 결합 세그먼트
  │   ├─ segment_summaries.jsonl       # 세그먼트 요약
  │   ├─ segment_summaries.md          # 요약 마크다운
  │   ├─ judge.json                    # PASS/FAIL + 평균 점수
  │   └─ judge_segment_reports.jsonl   # 세그먼트별 Judge 상세
  └─ results/
      └─ final_summary_*.md            # 최종 요약
```

### 배치 모드 ON
```
data/outputs/<video>/
  ├─ stt.json / manifest.json / captures/
  ├─ batches/batch_N/
  │   ├─ vlm.json                      # 배치별 VLM
  │   ├─ segments_units.jsonl          # 배치별 동기화
  │   ├─ judge.json                    # 배치별 Judge
  │   └─ judge_segment_reports.jsonl
  ├─ fusion/
  │   ├─ segment_summaries.jsonl       # 전체 누적 요약
  │   └─ segment_summaries.md
  └─ results/
      └─ final_summary_*.md
```

---

## 환경 변수 (.env)

```env
# AI APIs
GOOGLE_API_KEY=your_gemini_key
QWEN_API_KEY_1=your_qwen_key

# STT
CLOVA_SPEECH_URL=your_clova_endpoint
CLOVA_SPEECH_API_KEY=your_clova_key

# Supabase
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Cloudflare R2
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=your_bucket_name
```

프론트엔드: `frontend/.env`
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_API_BASE_URL=http://localhost:8080
```
