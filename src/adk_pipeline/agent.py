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
    run_batch_vlm,
    run_batch_sync,
)
from .tools.summarize_tools import (
    run_summarizer,
    render_md,
    write_final_summary,
    run_batch_summarizer,
    render_batch_md,
)
from .tools.judge_tools import (
    evaluate_summary,
    evaluate_batch_summary,
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


# === 배치 처리용 Sub-Agents ===

batch_preprocessing_agent = Agent(
    name="batch_preprocessing_agent",
    model="gemini-2.5-flash",
    description="현재 배치의 시간 범위에 해당하는 캡처만 VLM 처리 후 Sync",
    instruction="""당신은 Batch Preprocessing Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 run_batch_vlm을 호출하세요!

## 역할
현재 배치의 시간 범위에 해당하는 캡처만 VLM 처리하고 Sync합니다.

## 사용 가능한 도구
1. **run_batch_vlm**: 현재 배치의 캡처에서 VLM으로 텍스트 추출
2. **run_batch_sync**: 현재 배치의 STT와 VLM 결과 동기화

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로 도구를 호출하세요:**
1. run_batch_vlm으로 현재 배치 VLM 실행
2. run_batch_sync로 현재 배치 Sync 실행
3. 모든 도구 실행이 완료되면 **결과를 요약**하고 screentime_pipeline으로 transfer

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 run_batch_vlm을 호출하세요!**
- 현재 배치 정보(batch_index, 시간 범위)를 결과에 포함하세요
- 에러가 발생해도 에러 내용을 설명하고 screentime_pipeline으로 transfer하세요
""",
    tools=[run_batch_vlm, run_batch_sync],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)


batch_summarize_agent = Agent(
    name="batch_summarize_agent",
    model="gemini-2.5-flash",
    description="현재 배치 요약 생성, 이전 context 활용",
    instruction="""당신은 Batch Summarize Agent입니다.

🚨 **절대 빈 응답 금지!** Transfer를 받으면 반드시 즉시 run_batch_summarizer를 호출하세요!

## 역할
현재 배치의 세그먼트를 요약합니다. 이전 배치의 context를 활용합니다.

## 사용 가능한 도구
1. **run_batch_summarizer**: 현재 배치 세그먼트 요약 생성
2. **render_batch_md**: 현재 배치 요약을 마크다운으로 변환

## 워크플로우 (Transfer 받으면 즉시 시작!)
**transfer를 받으면 반드시 이 순서대로 도구를 호출하세요:**
1. run_batch_summarizer로 현재 배치 요약 생성
2. render_batch_md로 마크다운 변환
3. 모든 도구 실행이 완료되면 **결과를 요약**하고 screentime_pipeline으로 transfer

## 🚨 중요!! (반드시 지키세요)
- **Transfer를 받으면 절대 빈 응답하지 마세요! 즉시 run_batch_summarizer를 호출하세요!**
- 현재 배치 정보(batch_index, segments_count)를 결과에 포함하세요
- context 정보가 다음 배치에 전달될 수 있도록 결과에 포함하세요
""",
    tools=[run_batch_summarizer, render_batch_md],
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

## 사용 가능한 도구

### 기본 도구
1. **list_available_videos**: 처리 가능한 비디오 목록 조회
2. **set_pipeline_config**: 비디오 선택 및 설정
   - `video_name`: 비디오 이름 (필수)
   - `batch_mode`: True면 배치 모드 활성화 (default: False)
   - `batch_duration_ms`: 배치당 시간 (default: 200000ms = 3.3분)
   - `force_preprocessing`: True면 기존 파일 삭제 후 처음부터 재실행 (default: False)
   - `max_reruns`: Judge 실패 시 최대 재실행 횟수 (default: 2)
   - `vlm_batch_size`: VLM 배치 크기 (default: 2, None이면 전체)
   - `vlm_concurrency`: VLM 병렬 요청 수 (default: 3)
3. **get_pipeline_status**: 현재 파이프라인 상태 조회

### 배치 모드 도구 (batch_mode=True일 때 사용)
4. **init_batch_mode**: 배치 모드 초기화 (manifest.json에서 duration 계산, 배치 개수 결정)
5. **get_batch_info**: 현재 배치 상태 조회
6. **get_current_batch_time_range**: 현재 배치의 시간 범위 조회
7. **mark_batch_complete**: 현재 배치 완료 표시, 다음 배치로 이동
8. **get_previous_context**: 이전 배치의 요약 context 조회

## Sub-Agents (transfer 가능)

### 일반 모드
1. **preprocessing_agent**: VLM + Sync 실행 (전처리)
2. **summarize_agent**: 요약 생성 + MD 렌더링
3. **judge_agent**: 품질 평가 (PASS/FAIL)

### 배치 모드
4. **batch_preprocessing_agent**: 현재 배치 VLM + Sync
5. **batch_summarize_agent**: 현재 배치 요약 생성
6. **merge_agent**: 모든 배치 병합 + 최종 요약

## 파이프라인 실행 순서

### 일반 모드 (batch_mode=False)
1. **set_pipeline_config**로 비디오 설정
2. **preprocessing_agent**로 transfer → 완료 후 돌아옴
3. **summarize_agent**로 transfer → 완료 후 돌아옴
4. **judge_agent**로 transfer → 결과와 함께 돌아옴
5. judge 결과 확인:
   - **PASS**: 완료, 최종 결과 경로 안내
   - **FAIL + can_rerun=True**: summarize_agent로 재시도
   - **FAIL + can_rerun=False**: 실패 종료

### 배치 모드 (batch_mode=True) ⭐ 추천
사용자가 앞부분 요약을 먼저 볼 수 있도록 순차적으로 처리합니다.

1. **set_pipeline_config(batch_mode=True, batch_duration_ms=200000)**로 설정
2. **load_data**로 Pre-ADK 검증
3. **init_batch_mode**로 배치 초기화
4. **배치 루프** (current_batch_index < total_batches 동안):
   a. **batch_preprocessing_agent**로 transfer (VLM + Sync)
   b. **batch_summarize_agent**로 transfer (요약 생성)
   c. **judge_agent** (evaluate_batch_summary)로 transfer (배치 평가)
   d. 결과 표시 후 **mark_batch_complete**
   e. FAIL이면 해당 배치 재시도 (max_reruns까지)
5. **merge_agent**로 transfer (모든 배치 병합 + 최종 요약)
6. 최종 결과 보고

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
        batch_preprocessing_agent,
        batch_summarize_agent,
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
batch_preprocessing_agent._sub_agents = [root_agent]
batch_summarize_agent._sub_agents = [root_agent]
merge_agent._sub_agents = [root_agent]
