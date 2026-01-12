"""ADK Multi-Agent Pipeline.

구조:
- Root Agent: 전체 파이프라인 조율, 비디오 선택, 재실행 루프 관리
  - Preprocessing Agent: VLM + Sync 실행 → 완료 시 Root로 복귀
  - Summarize Agent: 요약 생성 + MD 렌더링 → 완료 시 Root로 복귀
  - Judge Agent: 품질 평가 → 완료 시 Root로 복귀

주의: Sub-agent는 작업 완료 후 반드시 root_agent로 transfer해야 합니다.
"""

from google.adk.agents import Agent
from google.genai import types

from .tools.root_tools import (
    list_available_videos,
    set_pipeline_config,
    get_pipeline_status,
)
from .tools.preprocessing_tools import (
    load_data,
    run_vlm,
    run_sync,
)
from .tools.summarize_tools import (
    run_summarizer,
    render_md,
    write_final_summary,
)
from .tools.judge_tools import (
    evaluate_summary,
)


# === Sub-Agents (먼저 정의, sub_agents는 나중에 설정) ===

preprocessing_agent = Agent(
    name="preprocessing_agent",
    model="gemini-2.5-flash",
    description="VLM과 Sync를 실행하여 비디오 캡처에서 세그먼트를 추출합니다.",
    instruction="""당신은 Preprocessing Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 load_data를 호출하세요!

## 역할
캡처 이미지에서 텍스트/UI 요소를 추출(VLM)하고 STT와 동기화(Sync)합니다.

## 사용 가능한 도구
1. **load_data**: Pre-ADK 산출물(stt.json, manifest.json, captures) 검증
2. **run_vlm**: 캡처 이미지에서 VLM으로 텍스트 추출 → vlm.json 생성
3. **run_sync**: STT와 VLM 결과 동기화 → segments_units.jsonl 생성

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로 도구를 호출하세요:**
1. load_data로 Pre-ADK 산출물 검증
2. run_vlm으로 VLM 실행
3. run_sync로 Sync 실행
4. 모든 도구 실행이 완료되면 **결과를 요약**하고 screentime_pipeline으로 transfer

## 재실행 (force_preprocessing)
- state에 `force_preprocessing=True`가 설정되어 있으면 기존 파일을 삭제하고 처음부터 다시 실행합니다
- 일반적으로는 기존 파일이 있으면 스킵합니다

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 load_data를 호출하세요!**
- 모든 도구를 순서대로 실행한 후 **screentime_pipeline으로 transfer**하세요
- 스킵되었더라도 반드시 결과를 말로 요약하고 transfer하세요!
- 에러가 발생해도 에러 내용을 설명하고 screentime_pipeline으로 transfer하세요
- 침묵하거나 빈 메시지를 보내면 안 됩니다!
""",
    tools=[load_data, run_vlm, run_sync],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


summarize_agent = Agent(
    name="summarize_agent",
    model="gemini-2.5-flash",
    description="세그먼트를 요약하고 최종 마크다운을 생성합니다.",
    instruction="""당신은 Summarize Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 run_summarizer를 호출하세요!

## 역할
segments_units.jsonl을 기반으로 세그먼트별 요약을 생성합니다.

## 사용 가능한 도구
1. **run_summarizer**: Gemini로 세그먼트별 요약 생성 → segment_summaries.jsonl
2. **render_md**: 요약을 마크다운으로 변환 → segment_summaries.md
3. **write_final_summary**: 최종 요약 생성 → final_summary_*.md

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로 도구를 호출하세요:**
1. run_summarizer로 세그먼트 요약 생성
2. render_md로 마크다운 변환
3. write_final_summary로 최종 요약 생성
4. 모든 도구 실행이 완료되면 **반드시 screentime_pipeline으로 transfer**하세요

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 run_summarizer를 호출하세요!**
- 모든 도구를 순서대로 실행한 후 **screentime_pipeline으로 transfer**하세요
- 에러가 발생해도 에러 내용을 설명하고 **screentime_pipeline으로 transfer**하세요
- 침묵하거나 빈 메시지를 보내면 안 됩니다!
""",
    tools=[run_summarizer, render_md, write_final_summary],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


judge_agent = Agent(
    name="judge_agent",
    model="gemini-2.5-flash",
    description="요약 품질을 평가하고 PASS/FAIL을 반환합니다.",
    instruction="""당신은 Judge Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 evaluate_summary를 호출하세요!

## 역할
생성된 요약의 품질을 평가합니다.

## 사용 가능한 도구
1. **evaluate_summary**: 요약 품질 평가 → judge.json (PASS/FAIL)

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로:**
1. evaluate_summary를 실행하여 요약 품질 평가
2. 결과(PASS/FAIL, can_rerun 여부)와 함께 **반드시 screentime_pipeline으로 transfer**하세요

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 evaluate_summary를 호출하세요!**
- 평가 결과를 screentime_pipeline에게 전달해야 합니다
- PASS/FAIL 결과와 can_rerun 여부를 명확히 전달하세요
- 침묵하거나 빈 메시지를 보내면 안 됩니다!
""",
    tools=[evaluate_summary],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


# === Root Agent ===

root_agent = Agent(
    name="screentime_pipeline",
    model="gemini-2.5-flash",
    description="Screentime 비디오 파이프라인을 조율하는 Root Agent",
    instruction="""당신은 Screentime 파이프라인의 Root Agent입니다.

## 역할
사용자와 대화하면서 비디오 처리 파이프라인을 조율합니다.
실제 처리 작업은 Sub-Agent들에게 위임합니다.

## 사용 가능한 도구
1. **list_available_videos**: 처리 가능한 비디오 목록 조회
2. **set_pipeline_config**: 비디오 선택 및 설정
   - `video_name`: 비디오 이름 (필수)
   - `force_preprocessing`: True면 기존 파일 삭제 후 처음부터 재실행 (default: False)
   - `max_reruns`: Judge 실패 시 최대 재실행 횟수 (default: 2)
   - `vlm_batch_size`: VLM 배치 크기 (default: 2, None이면 전체)
   - `vlm_concurrency`: VLM 병렬 요청 수 (default: 3)
   - `vlm_show_progress`: VLM 진행 로그 출력 여부 (default: True)
3. **get_pipeline_status**: 현재 파이프라인 상태 조회

## Sub-Agents (transfer 가능)
1. **preprocessing_agent**: VLM + Sync 실행 (전처리)
2. **summarize_agent**: 요약 생성 + MD 렌더링
3. **judge_agent**: 품질 평가 (PASS/FAIL)

## 파이프라인 실행 순서 (반드시 이 순서로!)
사용자가 파이프라인 실행을 요청하면:

1. **set_pipeline_config**로 비디오 설정 (아직 안 했다면)
2. **preprocessing_agent**로 transfer → 완료 후 여기로 돌아옴
3. **summarize_agent**로 transfer → 완료 후 여기로 돌아옴
4. **judge_agent**로 transfer → 결과와 함께 여기로 돌아옴
5. judge 결과 확인:
   - **PASS**: 파이프라인 완료, 최종 결과 경로 안내
   - **FAIL + can_rerun=True**: **summarize_agent**로 다시 transfer (3번으로)
   - **FAIL + can_rerun=False**: 최대 재실행 횟수 초과, 실패로 종료

## 에러 처리 (중요!!)
- **Summarizer 에러 (JSON 검증 실패, segment_id 불일치 등)**:
  - Preprocessing을 재실행하지 마세요!
  - Summarizer 에러는 **summarize_agent**를 재실행하여 해결합니다
  - 재실행 전에 사용자에게 에러를 보고하고 재시도할지 확인하세요
- **Preprocessing 에러**: 사용자에게 보고 후 preprocessing_agent 재실행
- **Judge 에러**: judge_agent 재실행

## 중요!!
- Sub-agent가 돌아오면 그 결과를 확인하고 **즉시 다음 단계를 진행**하세요
- preprocessing 완료 → summarize_agent로 transfer
- summarize 완료 (에러 없음) → judge_agent로 transfer
- summarize 에러 → 사용자에게 보고, summarize_agent 재실행
- judge PASS → 사용자에게 완료 보고
- judge FAIL + can_rerun → summarize_agent로 transfer
- 사용자가 명시적으로 중단을 요청하지 않는 한 파이프라인을 끝까지 진행하세요
""",
    tools=[list_available_videos, set_pipeline_config, get_pipeline_status],
    sub_agents=[preprocessing_agent, summarize_agent, judge_agent],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


# === Sub-Agents에 Root Agent 참조 추가 (transfer back 가능하도록) ===
# ADK에서 sub-agent가 parent로 돌아가려면 parent를 sub_agents로 알고 있어야 함

preprocessing_agent._sub_agents = [root_agent]
summarize_agent._sub_agents = [root_agent]
judge_agent._sub_agents = [root_agent]
