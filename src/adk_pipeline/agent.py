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
    render_batch_md,
    write_final_summary,
)
from .tools.judge_tools import (
    evaluate_summary,
)
from .tools.batch_tools import (
    init_batch_mode,
    get_batch_info,
    get_current_batch_time_range,
    mark_batch_complete,
    get_previous_context,
)
from .tools.merge_tools import (
    merge_all_batches,
    generate_final_summary as generate_final_summary_tool,
    merge_and_finalize,
)


# === Sub-Agents (먼저 정의, sub_agents는 나중에 설정) ===

preprocessing_agent = Agent(
    name="preprocessing_agent",
    model="gemini-2.5-flash",
    description="VLM과 Sync를 실행하여 비디오 캡처에서 세그먼트를 추출합니다.",
    instruction="""Preprocessing Agent입니다.

## 도구 순서
1. load_data (첫 배치면 실행, 아니면 자동 스킵)
2. init_batch_mode (첫 배치면 실행)
3. run_vlm
4. run_sync
5. Root로 transfer

## 에러 처리
도구 실패 시 에러 메시지를 포함하여 Root로 transfer하세요.
""",
    tools=[load_data, init_batch_mode, run_vlm, run_sync],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


summarize_agent = Agent(
    name="summarize_agent",
    model="gemini-2.5-flash",
    description="세그먼트를 요약하고 배치별 MD를 생성합니다.",
    instruction="""Summarize Agent입니다.

## 도구 순서
1. run_summarizer (현재 배치 요약 생성, fusion에 누적 저장)
2. render_batch_md (현재 배치 요약을 MD으로 렌더링)
3. Root로 transfer (batch_summaries_md 경로 포함)

## 에러 처리
도구 실패 시 에러 메시지를 포함하여 Root로 transfer하세요.
""",
    tools=[run_summarizer, render_batch_md],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


judge_agent = Agent(
    name="judge_agent",
    model="gemini-2.5-flash",
    description="요약 품질을 평가하고 PASS/FAIL을 반환합니다.",
    instruction="""Judge Agent입니다.

## 도구 순서
1. evaluate_summary (품질 평가, PASS/FAIL 반환)
2. Root로 transfer (PASS/FAIL 결과 포함)

## 에러 처리
도구 실패 시 에러 메시지를 포함하여 Root로 transfer하세요.
""",
    tools=[evaluate_summary],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)



merge_agent = Agent(
    name="merge_agent",
    model="gemini-2.5-flash",
    description="모든 배치 결과 병합 및 최종 요약 생성",
    instruction="""Merge Agent입니다.

## 도구 순서
1. merge_and_finalize (병합 + render_md + final_summary 한번에 실행)
2. Root로 transfer (최종 요약 파일 경로 포함)

## 에러 처리
도구 실패 시 에러 메시지를 포함하여 Root로 transfer하세요.
""",
    tools=[merge_all_batches, render_md, generate_final_summary_tool, merge_and_finalize],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)



root_agent = Agent(
    name="screentime_pipeline",
    model="gemini-2.5-flash",
    description="ReView 비디오 파이프라인을 조율하는 Root Agent",
    instruction="""당신은 ReView 파이프라인의 Root Agent입니다.

## 역할
사용자와 대화하면서 비디오 처리 파이프라인을 조율합니다.

## 사용 가능한 도구

### 기본 도구
1. **list_available_videos**: 처리 가능한 비디오 목록 조회
2. **set_pipeline_config**: 비디오 선택 및 설정
   - video_name: 비디오 이름 (필수)
   - batch_size: 배치당 캡처 개수 (기본: 4장)
   - batch_mode: 배치 모드 여부 (기본: True)
3. **get_pipeline_status**: 현재 파이프라인 상태 조회

### 배치 관리 도구
4. **init_batch_mode**: 배치 모드 초기화
5. **get_batch_info**: 현재 배치 상태 조회 (all_completed 확인용)
6. **get_current_batch_time_range**: 현재 배치의 시간 범위 조회
7. **mark_batch_complete**: 배치 완료 표시 후 다음 배치로 이동
8. **get_previous_context**: 이전 배치 요약 context 조회

## Sub-Agents
1. **preprocessing_agent**: VLM + Sync 실행
2. **summarize_agent**: 요약 생성 + 배치 MD 렌더링
3. **judge_agent**: 품질 평가 (PASS/FAIL)
4. **merge_agent**: 모든 배치 병합 + 최종 요약

## 파이프라인 실행 순서

사용자가 "test3 요약해줘" 같이 요청하면:

**STEP 1**: set_pipeline_config(video_name="test3_xxx") 호출
**STEP 2**: preprocessing_agent로 transfer
**STEP 3**: summarize_agent로 transfer
**STEP 4**: judge_agent로 transfer
**STEP 5**: "배치 X 완료" 사용자에게 알림
**STEP 6**: get_batch_info 호출하여 all_completed 확인

**STEP 7** (조건 분기):
- all_completed=False: mark_batch_complete 후 STEP 2로 돌아감
- all_completed=True: merge_agent로 transfer

**STEP 8**: "최종 요약 완료!" 사용자에게 알림

## 에러 처리
- Sub-agent 에러 시 해당 agent 재실행 (최대 2회)
- 사용자가 중단 요청하지 않는 한 파이프라인 끝까지 진행
""",
    tools=[
        list_available_videos,
        set_pipeline_config,
        get_pipeline_status,
        init_batch_mode,
        get_batch_info,
        get_current_batch_time_range,
        mark_batch_complete,
        get_previous_context,
    ],
    sub_agents=[
        preprocessing_agent,
        summarize_agent,
        judge_agent,
        merge_agent,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
    ),
)


# === Sub-Agents에 Root Agent 참조 추가 (transfer back 가능하도록) ===
# ADK에서 sub-agent가 parent로 돌아가려면 parent를 sub_agents로 알고 있어야 함

preprocessing_agent._sub_agents = [root_agent]
summarize_agent._sub_agents = [root_agent]
judge_agent._sub_agents = [root_agent]
merge_agent._sub_agents = [root_agent]
