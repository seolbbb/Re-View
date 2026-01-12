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
    instruction="""당신은 Preprocessing Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 load_data를 호출하세요!

## 역할
캡처 이미지에서 텍스트/UI 요소를 추출(VLM)하고 STT와 동기화(Sync)합니다.
배치 모드일 때는 현재 배치의 캡처만 처리합니다.

## 사용 가능한 도구
1. **load_data**: Pre-ADK 산출물 검증
2. **init_batch_mode**: 배치 모드 초기화 (첫 배치에서만)
3. **run_vlm**: VLM 실행 → vlm.json 생성 (배치 모드면 현재 배치만)
4. **run_sync**: Sync 실행 → segments_units.jsonl (배치 모드면 현재 배치만)

## 워크플로우

**배치 모드 (첫 배치일 때):**
1. load_data → Pre-ADK 검증
2. init_batch_mode → "총 N장을 M개 배치로 처리"
3. run_vlm → 현재 배치 VLM
4. run_sync → 현재 배치 Sync
5. screentime_pipeline으로 transfer

**배치 모드 (이후 배치일 때):**
1. run_vlm → 현재 배치 VLM
2. run_sync → 현재 배치 Sync
3. screentime_pipeline으로 transfer

**일반 모드:**
1. load_data → Pre-ADK 검증
2. run_vlm → 전체 VLM
3. run_sync → 전체 Sync
4. screentime_pipeline으로 transfer

## 🚨 중요!!
- **Transfer 받으면 절대 빈 응답 금지!**
- 에러 시에도 screentime_pipeline으로 transfer하세요
""",
    tools=[load_data, init_batch_mode, run_vlm, run_sync],
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

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 도구를 호출하세요!

## 역할
생성된 요약의 품질을 평가합니다.

## 사용 가능한 도구
1. **evaluate_summary**: 일반 모드에서 전체 요약 품질 평가 → judge.json (PASS/FAIL)
2. **evaluate_batch_summary**: 배치 모드에서 현재 배치 요약 품질 평가

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로:**
1. 배치 모드면 evaluate_batch_summary, 아니면 evaluate_summary 실행
2. 결과(PASS/FAIL, can_rerun 여부)와 함께 **반드시 screentime_pipeline으로 transfer**하세요

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 평가 도구를 호출하세요!**
- 평가 결과를 screentime_pipeline에게 전달해야 합니다
- PASS/FAIL 결과와 can_rerun 여부를 명확히 전달하세요
- 침묵하거나 빈 메시지를 보내면 안 됩니다!
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
    instruction="""당신은 Merge Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 merge_all_batches를 호출하세요!

## 역할
모든 배치의 결과를 병합하고 최종 요약을 생성합니다.

## 사용 가능한 도구
1. **merge_all_batches**: 모든 배치 파일 병합 (vlm.json, segments_units.jsonl, segment_summaries.jsonl)
2. **generate_final_summary_tool**: LLM으로 전체 통합 요약 생성
3. **merge_and_finalize**: 병합 + 최종 요약을 한번에 실행

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로 도구를 호출하세요:**
1. merge_all_batches로 배치 파일 병합
2. generate_final_summary_tool로 최종 요약 생성
   (또는 merge_and_finalize로 한번에 실행)
3. 모든 도구 실행이 완료되면 **결과를 요약**하고 screentime_pipeline으로 transfer

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 merge_all_batches를 호출하세요!**
- 병합된 파일 수, 세그먼트 수 등을 결과에 포함하세요
- 최종 요약 파일 경로를 결과에 포함하세요
""",
    tools=[merge_all_batches, generate_final_summary_tool, merge_and_finalize],
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
   - `batch_size`: 배치당 캡처 개수 (default: 10장)
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
