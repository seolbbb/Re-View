# Re:View TODO

> 📌 **최종 업데이트**: 2026-01-11  
> 이 문서는 `docs/PROJECT_DIRECTION.md` 기반으로 현재 진행 중인 작업과 남은 과제를 정리합니다.

---

## 🔥 우선순위 높음 (Performance Optimization)

### VLM 비동기 처리

- [ ] `AsyncOpenAI` 도입으로 VLM 호출 병렬화
- [ ] 비동기 환경에서 최적 Batch Size 재측정
- [ ] Rate Limiting / Throttling 로직 구현

### Summarizer 최적화

- [x] 입력 경량화: `segments_units.jsonl`에서 중복 데이터(`transcript_text` + `transcript_units`) 정리
- [x] Map-Reduce 구조로 세그먼트별 병렬 요약 처리
- [x] 토큰 사용량 모니터링 및 비용 최적화

### Judge 병렬화

- [x] 세그먼트별 병렬 평가 구조로 변경
- [x] 평가 기준 고도화

---

## 🟡 중간 우선순위 (UX & Architecture)

### 스트리밍 파이프라인

- [ ] 세그먼트 2~3개 단위로 파이프라인 분할 실행
- [ ] 앞부분 완료 시 즉시 사용자에게 표시 (체감 대기 시간 단축)

### Fast Summary (초안 생성)

- [ ] STT 결과만으로 빠른 초안 생성
- [ ] VLM/LLM 분석 중 사용자에게 먼저 결과 제공

---

## 🟢 낮은 우선순위 (Infrastructure)

### DB 도입

- [ ] 로컬 파일 시스템(JSON/JSONL) → Database 마이그레이션
- [ ] `VideoStore` 클래스 DB 연동 구현
- [ ] 메타데이터, 캡처 정보, 요약 결과 체계적 관리

### 피벗 전략 검토

- [ ] Latency 최적화 한계 시 Video RAG 서비스로 피벗 가능성 검토

---

## ✅ 완료된 항목

- [x] ADK 기반 멀티에이전트 파이프라인 구축
- [x] Pre-ADK (STT + Capture) 분리
- [x] VLM (OpenRouter Qwen) 연동
- [x] Fusion (STT + VLM 동기화) 구현
- [x] Summarizer (Gemini) 연동
- [x] Judge Agent 기본 구조
- [x] 재실행 로직 (FAIL 시 Summarize 재시도)
