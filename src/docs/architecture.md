# Screentime MVP Architecture

## 배포 구조

```
User → Frontend (SPA, Vercel) → Backend (단일 FastAPI) → DB (Supabase)
                               → Storage (Supabase, Signed URL 직접 업로드)
```

## 영상 업로드 흐름 (Signed URL)

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant F as Frontend (SPA)
  participant B as Backend (FastAPI)
  participant S as Storage (Supabase)
  participant D as DB (Supabase)

  U->>F: 영상 선택
  F->>B: POST /api/videos/upload/init {filename, content_type}
  B->>D: insert video 레코드
  B->>S: create_signed_upload_url()
  B->>F: {video_id, upload_url, storage_key}
  F->>S: PUT video (signed URL 직접 업로드)
  F->>B: POST /api/videos/upload/complete {video_id, storage_key}
  B->>D: status → PREPROCESSING
  B->>S: download video → temp
  B->>B: preprocess + process (background task)
  B->>D: captures, stt, segments, summaries 저장
  B->>D: status → DONE / FAILED
```

## 영상 스트리밍/썸네일

```mermaid
sequenceDiagram
  participant F as Frontend
  participant B as Backend
  participant S as Storage

  F->>B: GET /api/videos/{id}/stream
  B->>S: create_signed_url(video_storage_key)
  B->>F: 302 Redirect → signed URL
  F->>S: GET video (signed URL)

  F->>B: GET /api/videos/{id}/thumbnail
  B->>S: create_signed_url(capture_storage_path)
  B->>F: 302 Redirect → signed URL
  F->>S: GET thumbnail (signed URL)
```

## 채팅 흐름

```mermaid
sequenceDiagram
  participant U as User
  participant F as Frontend
  participant B as Backend
  participant D as DB

  U->>F: 질문 입력
  F->>B: POST /api/chat {video_id, message, session_id}
  B->>D: summaries 조회
  B->>F: 응답
```

## 주요 컴포넌트

| 컴포넌트 | 기술 | 역할 |
|---------|------|------|
| Frontend | React + Vite (Vercel) | SPA, 영상 업로드/재생/요약 표시 |
| Backend | FastAPI (단일 서버) | API, 파이프라인 실행, Storage/DB 연동 |
| DB | Supabase PostgreSQL | videos, captures, stt_results, segments, summaries |
| Storage | Supabase Storage | videos 버킷 (영상), captures 버킷 (썸네일) |

## 환경변수

### Frontend (Vite)
- `VITE_API_BASE_URL` — 백엔드 API URL (개발: 빈 문자열 → Vite 프록시, 프로덕션: 절대 URL)

### Backend
- `SUPABASE_URL` — Supabase 프로젝트 URL
- `SUPABASE_KEY` — Supabase anon/service key
- `CORS_ORIGINS` — 허용 origin 목록 (쉼표 구분, 기본: `http://localhost:5173,http://localhost:5174`)
