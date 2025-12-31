PRD v1.0: 강의/발표 영상 자동 요약 파이프라인 (Google ADK Orchestration)

작성일: 2025-12-30 (Asia/Seoul)
대상: Boostcamp CV-02 팀 내부 개발/운영 기준 문서
입력: 로컬 영상 파일
출력: 구간별 요약 + 최종 요약 노트(Markdown), 중간 산출물(JSON/JSONL), 실행 로그/메타데이터

1. 문제 정의 및 목표

1.1 문제

- 영상에는 음성(설명) + 시각(슬라이드/데모/화이트보드/코드/그래프) 정보가 함께 존재한다.
- STT만으로는 시각 근거가 누락되고, 프레임 기반 추출만으로는 설명 맥락이 누락된다.
- 긴 영상 처리 시 비용/시간이 커 재시도, 재개(resume), 중간 산출물 저장이 필수다.

  1.2 목표

- 영상에서 “구간(segment)” 단위로 STT(한국어)와 VLM(프레임 요약)을 타임라인으로 동기화해, 근거 기반 구간 요약과 전체 요약 노트를 생성한다.
- 단계별 산출물과 실행 상태를 표준화된 포맷으로 저장해 부분 재실행이 가능하게 한다.
- LLM Judge를 통해 환각/비약을 탐지하고, 해당 구간만 재생성하는 검증 루프를 제공한다.

  1.3 비목표

- 실시간(라이브) 처리
- 완전한 UI(웹 편집기) 제공
- 그래프/도표 수치의 완전 추출(초기에는 요약 중심)

2. 사용자/이해관계자 및 성공 기준

2.1 주요 사용자

- 수강생/학습자: 강의 내용을 빠르게 복습
- 팀/조직: 세미나·회의 녹화 요약, 액션아이템 정리
- 제작자: 노트 자동 생성 후 배포

  2.2 성공 기준(정량/정성)

- 커버리지: 주요 슬라이드 변화 지점이 요약에 반영(“제목/핵심 bullet” 수준)
- 정합성: 구간 요약이 해당 구간 STT/VLM 근거에서 벗어나지 않음
- 재현성/운영성: 실패 시 전체 재실행 없이 실패 단계부터 재개 가능
- 비용 통제: 캡처/LLM 호출량이 설정으로 제어 가능(1분당 10장 기준선)

3. 시스템 개요(End-to-End)

3.1 파이프라인 단계

1. ADK 오케스트레이션
2. 영상 확보/전처리(로컬 파일, 오디오 추출, 필요 시 “자르기”)
3. 변화 감지 + 캡처(Pixel diff, 1분 10장)
4. STT(네이버 STT, 한국어)
5. VLM(캡처 이미지→요약 텍스트, 출력: timestamp/extracted_text)
6. 정렬/병합(싱크: 시간축 세그먼트 생성, image_refs 제외)
7. 중간 포맷(JSONL: 세그먼트 원라인 저장)
8. LLM 요약(구간별 + 전체 Markdown)
9. 검증 루프(Judge → 문제 구간만 재생성)

3.2 아티팩트 저장(권장)

- runs/{run_id}/ 하위에 단계별 디렉터리(멱등성 확보)
- 모든 단계는 “입력/출력 계약”을 가진다.

4. 오케스트레이션 설계(Google ADK)

4.1 ADK 사용 방식(권장)

- 워크플로는 예측 가능한 순차 파이프라인이므로 기본은 Sequential workflow로 구성한다. ADK는 워크플로 에이전트(Sequential/Parallel/Loop)로 파이프라인을 정의할 수 있다. Google GitHub
- 단계별 실패 처리, 재시도, 체크포인트는 “커스텀 에이전트(BaseAgent 상속)” 또는 단계 래퍼에서 구현한다. ADK는 BaseAgent 상속으로 임의 제어흐름/상태/이벤트 처리가 가능하다. Google GitHub
- 참고로 ADK 커뮤니티에서는 툴/LLM 호출에 대한 retry/checkpoint 메커니즘 요구가 별도로 논의될 정도로, 프로젝트 레이어에서 정책을 갖추는 것이 안전하다. GitHub

  4.2 상태 모델(State Machine)

각 Step은 아래 상태를 가진다.

- PENDING → RUNNING → SUCCEEDED
- RUNNING → FAILED → (RETRYING → RUNNING …) → FAILED(terminal)
- SKIPPED(조건부 스킵: 예, 오디오 없음)

  4.3 재시도 정책(초안)

- 대상: 외부 API 호출(STT/VLM/LLM), 일시적 I/O 오류
- 최대 재시도: 기본 3회(설정 가능)
- 백오프: 2^k 초(상한 설정)
- 비재시도 오류: 입력 포맷 오류, 인증키 오류 등 “영구 오류”

  4.4 실행 메타데이터(run.json)

- run_id, started_at, config_hash
- input_video_path, duration, resolution
- steps: {name, status, started_at, ended_at, artifacts, error_summary}
- versions: stt_provider_version, vlm_model_id, llm_model_id, prompt_version

5. 단계별 상세 요구사항

5.1 영상 확보/전처리

입력

- local_video_path

기능

- 파일 검증: 존재/확장자/코덱 정보 추출
- 오디오 추출: STT 입력용 오디오 생성
- 입력 크기 제한 대응(“자르기”): 기준 미정(결정 필요)

출력(아티팩트)

- audio/{run_id}.wav (또는 provider 요구 포맷)
- preprocess/manifest.json (원본 메타 + 추출 결과 + (선택) 분할 목록)

에러/예외

- 오디오 트랙 없음: STT step SKIPPED, 이후 sync에서 transcript_text 비어있을 수 있음

결정 필요: “자르기” 기준

- 후보 1: 영상 길이 기준(예: 30분/60분 단위)
- 후보 2: STT/VLM/LLM 제공자의 요청 크기 제한 기반(용량/길이)
- 후보 3: 캡처 변화 구간 기준 분할(슬라이드 덩어리 단위)

  5.2 변화 감지 + 캡처 (Pixel diff, 1분 10장)

입력

- video (원본 또는 분할 chunk)

핵심 파라미터

- diff_threshold: Pixel diff 임계값(정규화 방식 포함)
- fps_sample: 분석용 샘플링 fps(예: 1~3 fps 권장; 결정 필요)
- cap_per_minute: 기본 10(= 6초 최소 간격에 준함)
- min_interval_sec: 기본 6초(= 60/10)
- hard*max_caps: 전체 캡처 상한(예: duration_min * cap*per_minute * 1.2) (결정 필요)

처리 규칙(권장 초안)

- 프레임을 일정 샘플링(fps_sample)으로 읽고, 이전 선택 프레임 대비 pixel diff가 threshold 이상이면 “변화 후보”로 기록
- 변화 후보 중에서도 min_interval_sec를 만족하는 것만 캡처 저장
- 1분 창(sliding 또는 tumbling window) 내 후보가 10장 초과 시:
  - 우선순위: diff score가 큰 상위 10장만 유지(권장) 또는 시간 균등 샘플(대안)

출력

- captures/\*.png (또는 jpg)
- captures/manifest.json
  - [{timestamp_ms, frame_index, file_name, diff_score}]

비고

- 이후 sync 결과에는 image_refs를 넣지 않지만, 디버깅/근거 추적을 위해 captures/manifest는 유지한다.

  5.3 STT (NAVER STT, 한국어)

입력

- 영상
- audio file (전처리 산출물)

STT 제공자 확정(현 기준)

- NAVER Cloud Platform의 CLOVA Speech Recognition STT API를 1차 타겟으로 한다. 이 계열 문서에서는 언어와 오디오 데이터(여러 포맷)를 입력으로 받아 텍스트로 변환한다고 명시되어 있다. Ncloud Docs+1
- 서비스는 Classic/VPC 환경에서 제공된다고 안내되어 있다.
- 오디오 포맷은 MP3, AAC, AC3, OGG, FLAC, WAV 등을 지원한다고 안내되어 있다.

요구사항

- 언어: 한국어 고정
- 타임스탬프 포함 세그먼트 단위 반환(최소: start_ms, end_ms, text)
  - 제공자 API가 “단어 단위/문장 단위 타임스탬프”를 얼마나 지원하는지에 따라 어댑터에서 표준화

출력(stt.json)

- schema_version: 1
- segments: [{start_ms, end_ms, text}]
- (옵션) confidence, raw_response 저장

추후 전환(로컬 STT)

- provider adapter 인터페이스를 고정하여, “NAVER → Local(예: Whisper)” 교체가 파이프라인 전체에 영향이 없도록 한다.

결정 필요(정확한 표준화 수준)

- STT가 “단어 단위 타임스탬프”를 주는 경우, 그대로 저장할지 / 문장 단위로 다시 묶을지

  5.4 VLM (단독 요약, 출력 최소화)

입력

- captures/manifest.json + 이미지 파일

요구사항

- VLM 호출 단위: 이미지 1장 = 1회(기본)
- 출력 필드 고정: timestamp_ms, extracted_text만 생성
- extracted_text의 성격
  - 화면에 보이는 핵심 텍스트/요점 요약(슬라이드 제목, 주요 bullet)
  - 그래프/도표는 “무엇을 말하는지”를 과장 없이 설명(모르면 모른다고 표기하도록 프롬프트 설계)

출력(vlm.json)

- schema_version: 1
- items: [{timestamp_ms, extracted_text}]

결정 필요(토큰/비용 최적화)

- 1장당 1회 호출이 부담이면:
  _ 후보 1: 1분당 대표 5장만 VLM
  _ 후보 2: OCR/간이 필터로 텍스트가 많은 프레임만 VLM \* 후보 3: 특정 구간(변화량 큰 프레임)만 VLM

  5.5 정렬/병합(싱크) – image_refs 제외

입력

- stt.json (segments)
- vlm.json (items)

출력(sync.json)

- schema_version: 1
- segments: [
  {segment_id, start_ms, end_ms, transcript_text, visual_text}
  ]

중요: image_refs 제외(요구 반영)

- visual_text는 “해당 구간에 매칭된 VLM extracted_text를 병합한 결과 문자열”로만 제공
- 이미지 파일/프레임 참조는 sync 산출물에서 제거한다(다만 captures/manifest는 run 아티팩트로 유지)

결정 필요 1: 구간화(segmentation) 기준

- 후보 A: 캡처 변화 시점 기반(슬라이드 단위)
  - 장점: 시각 정보와 결합이 자연스럽다
  - 단점: 설명이 길게 이어지면 구간이 너무 커질 수 있다
- 후보 B: STT 침묵/문장 경계 기반
  - 장점: 말 흐름 중심 요약
  - 단점: 시각 변화와 어긋날 수 있다
- 후보 C: 고정 윈도우(예: 30초/60초)
  - 장점: 구현 단순, 안정적
  - 단점: 의미 단위가 깨질 수 있다
- 후보 D: 하이브리드(캡처 경계 우선 + STT로 세분화)
  - 장점: 두 신호를 함께 사용
  - 단점: 규칙 복잡

결정 필요 2: 충돌 규칙(다대다 매칭)
VLM item(시각)과 STT segment(음성)의 매칭은 시간축 기준으로 수행하되, 아래 중 선택 필요.

- 규칙 1: 구간 내 VLM을 “모두 병합”(길이 제한 필요)
- 규칙 2: 구간 내 VLM을 “가장 가까운 1개만 채택”
- 규칙 3: 구간 내 VLM을 “상위 N개(예: 2~3개)만 채택”(diff_score 큰 것 우선)
- 규칙 4: “슬라이드가 바뀐 직후 1장”만 채택(대표성 우선)

결정 필요 3: visual_text 병합 방식

- 단순 이어붙이기
- 중복 제거(유사 문장 제거)
- 길이 제한(예: 800~1200자) 초과 시 요약 압축(간단 규칙 or 소형 LLM)

  5.6 중간 포맷(JSONL)

목적

- LLM 입력 토큰 절감
- 재요약/재검증을 위한 표준화

입력

- sync.json segments

출력(segments.jsonl)

- 1 line = 1 segment

JSONL 스키마(초안, v1)

- run_id: string
- segment_id: int
- start_ms: int
- end_ms: int
- transcript_text: string
- visual_text: string

주의

- image_refs 없음(요구 반영)
- 필요 시 traceability는 “segment_id ↔ 캡처 시점”을 별도 매핑 파일로 유지(결정 필요: 필요/불필요)

  5.7 LLM 요약

입력

- segments.jsonl

출력

- outputs/segment_summaries.md
- outputs/final_summary.md

구간 요약 요구사항

- 각 segment에 대해:
  - 핵심 요점 3~5개(가능하면)
  - 용어/개념 정의(등장 시)
  - 과장/추정 금지(근거 부족 시 “확인 불가”)

전체 요약 문서 구조(실험)

- 포맷 A: 시간 순(타임라인 노트)
- 포맷 B: 주제별 재구성(섹션 추정)
- 포맷 C: TL;DR + 시간 순(하이브리드)

사용자 결정: “3개 다 해보고 결정”

- 구현 요구사항:
  _ 동일 run에서 3개 포맷을 모두 생성할 수 있게 옵션화
  _ 비교를 위한 평가 템플릿(아래 7장) 포함

  5.8 검증 루프(Judge)

입력

- 구간 요약(세그먼트 단위)

검증 목표

- 환각/비약/근거 없는 일반화 탐지
- 문제 구간만 재생성(확정: 10A)

동작

- Judge는 각 segment_summary에 대해:
  - “근거로부터 도출 가능한가?”
  - “원문(STT/VLM)에 없는 구체 수치/고유명사/결론이 생성됐는가?”
  - “서로 모순되는 주장 존재 여부”
- 실패 판정 시:
  - 해당 segment만 재요약(입력은 해당 segment의 transcript_text + visual_text만)
  - 재생성 횟수 상한(예: 2회) 설정(결정 필요)

출력

- judge/report.json
  - [{segment_id, verdict(PASS/FAIL), reasons, flagged_sentences, regen_count}]
- 수정된 segment_summaries.md(최종본)

6. 인터페이스/모듈 경계(팀 병렬 개발용)

권장 모듈

- orchestrator (ADK)
- preprocess
- capture
- stt_provider_adapter (NAVER → Local 교체 가능)
- vlm_provider_adapter
- sync_engine
- jsonl_writer
- summarizer
- judge

모듈 간 계약은 “파일 기반”을 1차로 권장

- 이유: 장애/재시도/재개가 단순해지고, 팀이 서로의 런타임 환경 차이를 흡수하기 쉬움

7. 실험 설계(문서 구조 3안 비교)

평가 항목(권장)

- 이해 용이성(빠른 복습): 1~5
- 검색 용이성(특정 주제 찾기): 1~5
- 근거성(타임라인 대비 정합): 1~5
- 생성 비용/시간: 상대 비교

실험 절차(권장)

- 동일 영상 2개(슬라이드형 1, 데모형 1) 선정
- A/B/C 포맷 모두 생성
- 팀 내 3명 이상이 평가표 작성
- 평균 + 코멘트 기반으로 1개 포맷을 기본값으로 확정
