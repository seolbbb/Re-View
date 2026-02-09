<div align="center">

# Re:View
**강의 영상을 보지 않아도 이해할 수 있는 AI 강의 노트**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3FCF8E?style=flat-square&logo=supabase&logoColor=white)](https://supabase.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

강의 영상에서 **슬라이드를 자동 추출**하고, **음성을 텍스트로 변환**하여,<br/>
시각 정보와 청각 정보가 완벽히 동기화된 **독립형 강의 노트**를 생성합니다.

[시작하기](#-getting-started) &nbsp;&bull;&nbsp; [아키텍처](#-architecture) &nbsp;&bull;&nbsp; [기여하기](#-contributing) &nbsp;&bull;&nbsp; [팀원 소개](#-team)

![Image](https://github.com/user-attachments/assets/bb4a8ae7-8f00-4b53-a9c9-441211d15f67)
</div>

---

## Overview

Re:View는 단순한 요약 도구가 아닙니다. "저기 보이는 그래프에서..." 같은 모호한 표현 없이, **영상을 보지 않아도 내용을 완벽히 이해할 수 있는 독립형 강의 노트**를 만드는 것이 목표입니다.

AI가 슬라이드의 시각 정보(텍스트, 수식, 도표)와 교수자의 음성 설명을 시간축 기반으로 동기화하여, 근거 있고 체계적인 학습 자료를 자동으로 생성합니다.

### Key Features

- **Smart Slide Capture** &mdash; ORB 특징점 + 지속성 분석 기반으로 슬라이드 전환을 정밀하게 감지하고, Smart ROI와 적응형 리사이징으로 최적의 캡처를 수행합니다.

- **Dual STT Engine** &mdash; Clova Speech(primary)와 OpenAI Whisper(fallback)를 지원하며, 자동 모노 변환으로 최적의 음성 인식 품질을 보장합니다.

- **VLM Slide Analysis** &mdash; Qwen VLM이 슬라이드 속 텍스트, 수식, 도표를 구조화된 형태로 추출합니다. 단순 OCR이 아닌 시각적 맥락까지 파악합니다.

- **Time-Synchronized Fusion** &mdash; STT 텍스트와 VLM 결과를 타임스탬프 기반으로 동기화하여, 교수자의 설명과 슬라이드 내용이 정확히 매칭된 세그먼트를 생성합니다.

- **AI Summarization with Quality Judge** &mdash; Gemini 기반 "독립형 튜터"가 근거 중심의 요약을 생성하고, 다축 자동 평가(정합성/노트 품질/멀티모달 활용도)를 통해 품질을 보장합니다.

- **Batch Processing & Resume** &mdash; 대용량 강의를 배치 단위로 처리하며, 중단된 파이프라인을 마지막 완료 배치부터 자동 재개합니다.

- **RAG Chatbot** &mdash; LangGraph 기반 RAG 챗봇이 생성된 요약을 바탕으로 질의응답을 지원하며, 스트리밍 응답과 후속 질문 추천을 제공합니다.

- **Real-time Status** &mdash; SSE 스트리밍으로 처리 진행 상황을 실시간 모니터링하고, 증분 요약을 점진적으로 전달받습니다.

---

## Architecture

```
                          +-----------------+
                          |    Frontend     |
                          |  React + Vite   |
                          |   (Vercel)      |
                          +--------+--------+
                                   |
                              REST API / SSE
                                   |
                          +--------v--------+
                          |    Backend      |
                          |    FastAPI      |
                          | (Cloud Run)     |
                          +--------+--------+
                                   |
                 +-----------------+-----------------+
                 |                 |                 |
          +------v------+  +------v------+  +-------v------+
          |  Supabase   |  | Cloudflare  |  |   AI APIs    |
          | PostgreSQL  |  |     R2      |  |              |
          |  + pgvector |  |  (Storage)  |  | Gemini / Qwen|
          |  + Auth     |  |             |  | Clova Speech |
          +-------------+  +-------------+  +--------------+
```

### Pipeline Flow

```
Video Upload
    |
    +---> [Preprocess] ----+---> STT (Clova / Whisper)
    |                      +---> Slide Capture (ORB + pHash)
    |
    +---> [Process] -------+---> VLM Analysis (Qwen)
                           +---> Time Sync (STT + VLM Fusion)
                           +---> Summarization (Gemini)
                           +---> Quality Judge (Gemini)
                           |
                           +---> Structured Lecture Notes
```

---

## Tech Stack

### Backend

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Framework | FastAPI + Uvicorn |
| AI/LLM | Google Gemini, Qwen VLM (DashScope), Google ADK |
| STT | Clova Speech API, OpenAI Whisper |
| Computer Vision | OpenCV (ORB, pHash, ROI Detection) |
| Orchestration | LangGraph |
| Database | Supabase (PostgreSQL + pgvector) |
| Storage | Supabase Storage + Cloudflare R2 |
| Container | Docker (multi-stage build) |
| CI/CD | GitHub Actions -> Google Cloud Run |

### Frontend

| Category | Technology |
|----------|-----------|
| Framework | React 19.2 + Vite 7.2 |
| Routing | React Router DOM 7.12 |
| Styling | Tailwind CSS + Custom Design System (Dark/Light) |
| Auth | Supabase Auth (Email + Google OAuth) |
| Math | KaTeX + remark-math + rehype-katex |
| Markdown | react-markdown + rehype-raw |
| PDF Export | html2pdf.js |
| Analytics | Vercel Analytics |
| Deployment | Vercel |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- FFmpeg (audio extraction)

### 1. Clone & Install

```bash
# Clone
git clone https://github.com/seolbbb/Re-View.git
cd Re-View

# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Environment Setup

루트 디렉토리에 `.env` 파일을 생성합니다.

```bash
# AI APIs (Required)
GOOGLE_API_KEY=your_gemini_api_key
QWEN_API_KEY_1=your_qwen_api_key

# STT - Clova Speech
CLOVA_SPEECH_URL=your_clova_endpoint
CLOVA_SPEECH_API_KEY=your_clova_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Cloudflare R2
R2_ENDPOINT_URL=your_r2_endpoint
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=your_bucket_name
```

`frontend/.env` 파일도 생성합니다.

```bash
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=http://localhost:8080
```

### 3. Run

```bash
# Backend (port 8080)
uvicorn src.process_api:app --host 0.0.0.0 --port 8080 --reload

# Frontend (port 5173)
cd frontend && npm run dev
```

### Docker

```bash
docker build -t re-view-backend .
docker run -p 8080:8080 --env-file .env re-view-backend
```

---

## Project Structure

```
Re-View/
├── src/
│   ├── process_api.py          # FastAPI main server
│   ├── audio/                  # STT module (Clova + Whisper)
│   ├── capture/                # Slide capture engine
│   │   └── tools/              # ORB extractor, deduplicator, ROI, pHash
│   ├── vlm/                    # Qwen VLM slide analysis
│   ├── fusion/                 # Time sync + Gemini summarizer + judge
│   ├── judge/                  # Multi-axis quality evaluation
│   ├── pipeline/               # Pipeline orchestration & stages
│   ├── db/                     # Supabase adapter + R2 storage
│   │   └── adapters/           # Video, Capture, Content, Job adapters
│   ├── services/               # Chat, LangGraph session, pipeline service
│   ├── llm/                    # Global limiter + API key routing
│   └── adk_chatbot/            # Google ADK multi-agent chatbot
├── frontend/
│   └── src/
│       ├── api/                # HTTP client, video/chat API
│       ├── components/         # ChatBot, VideoPlayer, SummaryPanel, ...
│       ├── context/            # Auth, Theme, Video contexts
│       ├── hooks/              # usePolling, useVideoStatusStream
│       └── pages/              # Home, Login, Signup, Analysis, Loading
├── config/
│   ├── audio/                  # STT settings
│   ├── capture/                # Capture engine settings
│   ├── fusion/                 # Summarizer prompts & settings
│   ├── vlm/                    # VLM prompts & settings
│   ├── judge/                  # Judge prompts & settings
│   └── pipeline/               # Pipeline runtime settings
├── tests/                      # Pytest test suite
├── docs/                       # Technical documentation
├── .github/                    # CI/CD workflows + issue templates
├── Dockerfile                  # Multi-stage backend build
└── requirements.txt            # Python dependencies
```

---

## CLI Usage

### End-to-End Pipeline

```bash
python src/run_pipeline_demo.py --video "data/inputs/lecture.mp4"
```

### Step-by-Step

```bash
# Step 1: Preprocess (STT + Capture)
python src/run_preprocess_pipeline.py --video "data/inputs/lecture.mp4"

# Step 2: Process (VLM + Sync + Summarize + Judge)
python src/run_process_pipeline.py --video_name "lecture"
```

### Re-run Fusion Only

프롬프트를 수정한 후 Fusion 단계만 재실행합니다.

```bash
python src/run_fusion_only.py --video_name "lecture"
```

---

## Contributing

프로젝트에 기여해주셔서 감사합니다! 아래 가이드를 참고해주세요.

1. 이 레포지토리를 Fork합니다.
2. Feature 브랜치를 생성합니다. (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다. (`git commit -m 'feat: add amazing feature'`)
4. 브랜치에 Push합니다. (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다.

버그 리포트, 기능 제안은 [Issues](https://github.com/seolbbb/Re-View/issues)에 남겨주세요.

---

## Team

<div align="center">

<table>
  <tr>
    <td align="center" width="150">
      <a href="https://github.com/seolbbb">
        <img src="https://avatars.githubusercontent.com/u/85074180?v=4" width="100" height="100" style="border-radius:50%;" alt="seolbbb"/><br/>
        <sub><b>seolbbb</b></sub>
      </a><br/>
      <sub>설성범</sub>
    </td>
    <td align="center" width="150">
      <a href="https://github.com/iamcmj">
        <img src="https://avatars.githubusercontent.com/u/162732589?v=4" width="100" height="100" style="border-radius:50%;" alt="iamcmj"/><br/>
        <sub><b>iamcmj</b></sub>
      </a><br/>
      <sub>조민재</sub>
    </td>
    <td align="center" width="150">
      <a href="https://github.com/dltkdwns0730">
        <img src="https://avatars.githubusercontent.com/u/208967935?v=4" width="100" height="100" style="border-radius:50%;" alt="dltkdwns0730"/><br/>
        <sub><b>dltkdwns0730</b></sub>
      </a><br/>
      <sub>이상준</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <a href="https://github.com/kdmin3318">
        <img src="https://avatars.githubusercontent.com/u/206398757?v=4" width="100" height="100" style="border-radius:50%;" alt="kdmin3318"/><br/>
        <sub><b>kdmin3318</b></sub>
      </a>
    </td>
    <td align="center" width="150">
      <a href="https://github.com/tjrrms">
        <img src="https://avatars.githubusercontent.com/u/229415141?v=4" width="100" height="100" style="border-radius:50%;" alt="tjrrms"/><br/>
        <sub><b>tjrrms</b></sub>
      </a>
    </td>
    <td align="center" width="150">
      <a href="https://github.com/Hu-tech-hub">
        <img src="https://avatars.githubusercontent.com/u/177188868?v=4" width="100" height="100" style="border-radius:50%;" alt="Hu-tech-hub"/><br/>
        <sub><b>Hu-tech-hub</b></sub>
      </a>
    </td>
  </tr>
</table>

</div>

---

## Resources

| Resource | Description |
|----------|-------------|
| [Pipeline Architecture](./docs/pipeline.md) | 파이프라인 아키텍처, API 엔드포인트, DB 구조 |
| [Project Guide](./docs/project-guide.md) | 코드베이스 구조, 모듈별 핵심 파일, 설정 |
| [Chatbot Feature](./docs/chatbot.md) | 챗봇 기능 (RAG, 후속 질문, SSE 스트리밍) |
| [Supabase Schema](./src/db/supabase_schema.sql) | 데이터베이스 스키마 (PostgreSQL + pgvector) |

---

<div align="center">

Made with dedication by the **Re:View** team

</div>
