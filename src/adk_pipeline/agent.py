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

도구 순서: load_data → init_batch_mode → run_vlm → run_sync → screentime_pipeline으로 transfer

(각 도구가 상황에 맞게 자동으로 스킵됩니다)
""",
    tools=[load_data, init_batch_mode, run_vlm, run_sync],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


summarize_agent = Agent(
    name="summarize_agent",
    model="gemini-2.5-flash",
    description="세그먼트를 요약합니다.",
    instruction="""Summarize Agent입니다.

도구 순서: run_summarizer → screentime_pipeline으로 transfer
""",
    tools=[run_summarizer],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


judge_agent = Agent(
    name="judge_agent",
    model="gemini-2.5-flash",
    description="요약 품질을 평가하고 PASS/FAIL을 반환합니다.",
    instruction="""Judge Agent입니다.

도구 순서: evaluate_summary → screentime_pipeline으로 transfer
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

도구 순서: merge_and_finalize → screentime_pipeline으로 transfer
""",
    tools=[merge_all_batches, render_md, generate_final_summary_tool, merge_and_finalize],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)



root_agent = Agent(
    name="screentime_pipeline",
    model="gemini-2.5-flash",
    description="Screentime 비디오 파이프라인을 조율하는 Root Agent",
    instruction="""당신은 Screentime 파이프라인의 Root Agent입니다.

## 역할
사용자와 대화하면서 비디오 처리 파이프라인을 조율합니다.
실제 처리 작업은 Sub-Agent들에게 위임합니다.

## 🚨 중요: 기본 동작 = 배치 모드
파이프라인은 **배치 모드**가 기본입니다. 10장씩 분할 처리하여 사용자가 앞부분 요약을 먼저 볼 수 있습니다.

## 사용 가능한 도구

### 기본 도구
1. **list_available_videos**: 처리 가능한 비디오 목록 조회
2. **set_pipeline_config**: 비디오 선택 및 설정
   - `video_name`: 비디오 이름 (필수)
   - `batch_size`: 배치당 캡처 개수 (default: 5장)
   - `batch_mode`: True면 배치 모드 (default: True)
   - `force_preprocessing`: True면 기존 파일 삭제 후 재실행 (default: False)
   - `max_reruns`: Judge 실패 시 최대 재실행 횟수 (default: 2)
3. **get_pipeline_status**: 현재 파이프라인 상태 조회

### 배치 관리 도구
4. **init_batch_mode**: 배치 모드 초기화 (manifest에서 캡처 수 확인, 배치 개수 결정)
5. **get_batch_info**: 현재 배치 상태 조회
6. **get_current_batch_time_range**: 현재 배치의 시간/인덱스 범위 조회
7. **mark_batch_complete**: 현재 배치 완료 표시, 다음 배치로 이동
8. **get_previous_context**: 이전 배치의 요약 context 조회

## Sub-Agents (transfer 가능)
1. **preprocessing_agent**: VLM + Sync (배치 모드면 현재 배치만 처리)
2. **summarize_agent**: 요약 생성 (배치 모드면 현재 배치만, fusion에 누적)
3. **judge_agent**: 품질 평가
4. **merge_agent**: 모든 배치 병합 + 최종 요약 (배치 모드에서만)

## 파이프라인 실행 순서 (배치 모드)

사용자가 "test3 해봐" 같이 요청하면:

1. **set_pipeline_config(video_name="test3_Diffusion")**  ← batch_mode=True 자동
2. **preprocessing_agent**로 transfer (load_data + init_batch_mode + VLM + Sync)
3. **summarize_agent**로 transfer (요약 생성)
4. **judge_agent**로 transfer (배치 평가)
5. 🎉 "배치 0 완료!" 결과 표시
6. **mark_batch_complete** → 다음 배치로 이동
7. 2-6 반복 (모든 배치 완료까지)
8. **merge_agent**로 transfer (병합 + 최종 요약)
9. 🎉 최종 결과 보고

## 🚨 중요!!
- Sub-agent가 돌아오면 그 결과를 확인하고 **즉시 다음 단계를 진행**하세요
- 배치 모드에서는 각 배치 완료 후 사용자에게 진행 상황을 알려주세요
- 에러가 발생해도 해당 단계의 agent를 재실행하세요 (preprocessing 에러 → preprocessing 재실행)
- 사용자가 명시적으로 중단을 요청하지 않는 한 파이프라인을 끝까지 진행하세요
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
        temperature=0.1,
    ),
)


# === Sub-Agents에 Root Agent 참조 추가 (transfer back 가능하도록) ===
# ADK에서 sub-agent가 parent로 돌아가려면 parent를 sub_agents로 알고 있어야 함

preprocessing_agent._sub_agents = [root_agent]
summarize_agent._sub_agents = [root_agent]
judge_agent._sub_agents = [root_agent]
merge_agent._sub_agents = [root_agent]
