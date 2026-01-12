"""Gemini 기반 구간 요약 생성."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ConfigBundle
from .io_utils import ensure_output_root, print_jsonl_head, read_jsonl, update_token_usage


PROMPT_VERSION = "sum_v1.5"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiClientBundle:
    client: Any
    backend: str
    model: str


def _load_genai() -> Any:
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai 패키지가 필요합니다. requirements.txt를 확인하세요."
        ) from exc
    return genai


def _init_gemini_client(config: ConfigBundle) -> GeminiClientBundle:
    genai = _load_genai()
    llm_cfg = config.raw.llm_gemini
    if llm_cfg.backend == "developer_api":
        api_key = None
        for env_name in llm_cfg.developer_api.api_key_env_candidates:
            api_key = os.getenv(env_name)
            if api_key:
                break
        if not api_key:
            raise ValueError(
                "Developer API 키가 없습니다. GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정하세요."
            )
        client = genai.Client(api_key=api_key)
        return GeminiClientBundle(
            client=client, backend="developer_api", model=llm_cfg.model
        )

    if llm_cfg.backend == "vertex_ai":
        project = llm_cfg.vertex_ai.project
        location = llm_cfg.vertex_ai.location
        if not project or not location:
            raise ValueError(
                "Vertex AI는 project/location이 필요합니다. config를 확인하세요."
            )
        if llm_cfg.vertex_ai.auth_mode == "adc":
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv(llm_cfg.vertex_ai.api_key_env)
            if not api_key:
                raise ValueError("Vertex AI express_api_key 모드용 API 키가 없습니다.")
            client = genai.Client(
                vertexai=True, project=project, location=location, api_key=api_key
            )
        return GeminiClientBundle(
            client=client, backend="vertex_ai", model=llm_cfg.model
        )

    raise ValueError(f"지원하지 않는 backend: {llm_cfg.backend}")


def _build_response_schema() -> Dict[str, Any]:
    evidence_schema = {"type": "array", "items": {"type": "string"}}
    source_type_schema = {
        "type": "string",
        "enum": ["direct", "inferred", "background"],
    }
    bullet_schema = {
        "type": "object",
        "required": [
            "bullet_id",
            "claim",
            "source_type",
            "evidence_refs",
            "confidence",
            "notes",
        ],
        "properties": {
            "bullet_id": {"type": "string"},
            "claim": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    definition_schema = {
        "type": "object",
        "required": [
            "term",
            "definition",
            "source_type",
            "evidence_refs",
            "confidence",
            "notes",
        ],
        "properties": {
            "term": {"type": "string"},
            "definition": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    explanation_schema = {
        "type": "object",
        "required": ["point", "source_type", "evidence_refs", "confidence", "notes"],
        "properties": {
            "point": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    question_schema = {
        "type": "object",
        "required": ["question", "source_type", "evidence_refs", "confidence", "notes"],
        "properties": {
            "question": {"type": "string"},
            "source_type": source_type_schema,
            "evidence_refs": evidence_schema,
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "notes": {"type": "string"},
        },
    }
    summary_schema = {
        "type": "object",
        "required": ["bullets", "definitions", "explanations", "open_questions"],
        "properties": {
            "bullets": {"type": "array", "items": bullet_schema},
            "definitions": {"type": "array", "items": definition_schema},
            "explanations": {"type": "array", "items": explanation_schema},
            "open_questions": {"type": "array", "items": question_schema},
        },
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["segment_id", "summary"],
            "properties": {
                "segment_id": {"type": "integer"},
                "summary": summary_schema,
            },
        },
    }


def _build_batch_prompt(
    segments: List[Dict[str, Any]],
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
    previous_context: Optional[str] = None,
) -> str:
    jsonl_text = "\n".join(
        json.dumps(segment, ensure_ascii=False) for segment in segments
    )
    claim_rule = (
        f"- claim은 {claim_max_chars}자 이하의 한 문장"
        if claim_max_chars > 0
        else "- claim은 한 문장으로 작성 (길이 제한 없음)"
    )
    if bullets_min > 0 and bullets_max > 0:
        bullets_rule = f"- bullets는 {bullets_min}~{bullets_max}개 (권장)"
    elif bullets_min > 0:
        bullets_rule = f"- bullets는 최소 {bullets_min}개 (상한 없음)"
    else:
        bullets_rule = "- bullets는 필요한 만큼 작성"

    # 이전 배치 context 섹션
    context_section = ""
    if previous_context:
        context_section = f"""
========================
이전 배치 요약 (맥락 유지용)
========================
{previous_context}

위 내용은 이전 배치에서 다룬 핵심 내용입니다.
- 현재 배치 요약 시 위 내용과 일관성을 유지하세요.
- 동일한 용어/개념은 같은 방식으로 설명하세요.
- 중복 설명은 피하고, 새로운 정보에 집중하세요.
- 이전 내용과 연결되는 부분이 있으면 자연스럽게 연결하세요.
========================

"""

    prompt = f"""{context_section}당신은 "요약가"가 아니라 "초학자 튜터(독립형 강의 노트 작성자)"입니다.
목표는 입력(STT/VLM)을 그대로 줄이는 것이 아니라, 사용자가 영상을/슬라이드를 보지 않아도 오직 당신의 출력(JSON)만 읽고 이해할 수 있도록 '설명'과 '연결'을 만들어 주는 것입니다.

- 모든 출력은 한국어로 작성하되, 아래 용어 표기 규칙을 반드시 지키세요.
- 출력은 반드시 "순수 JSON 배열"만 반환하세요. (설명 문장/코드블록/마크다운 금지)
- 각 세그먼트는 오직 해당 세그먼트 입력만 근거로 하되, 일반적인 배경지식은 허용됩니다(아래 source_type 규칙 준수).
- 사용자에게는 STT/VLM 입력이 보이지 않습니다. 사용자가 볼 수 있는 것은 오직 당신의 출력(JSON)뿐입니다.

========================
출력 JSON 형식 (필수)
========================
배열의 각 원소:
{{
  "segment_id": int,
  "summary": {{
    "bullets": [...],
    "definitions": [...],
    "explanations": [...],
    "open_questions": [...]
  }}
}}

========================
가장 중요한 품질 목표 (우선순위)
========================
1) "독립형 노트": 사용자가 영상을/슬라이드를/그림을 보지 않아도 이해할 수 있어야 합니다.
2) "처음 듣는 학생이 이해"가 최우선입니다.
3) 단순 패러프레이즈(입력 문장 구조를 그대로 바꾼 문장)를 피하세요.
4) 각 세그먼트마다 최소 1개는 "왜(why)" 또는 "어떻게(how)"를 명시적으로 설명하세요.
5) 수식/기호가 나오면 가능한 한 입력에 있는 전체 수식을 먼저 제시하고, 이어서 각 기호/항의 의미를 풀어 쓰세요.
   - "이 수식/이 항/여기" 같은 지시어는 금지합니다. 반드시 수식 이름/항 이름/기호를 명시하세요.
6) 강의 맥락에서 중요한 연결(앞에서 왜 이걸 말하는지, 다음으로 무엇을 위해 쓰는지)을 최소 1개는 써 주세요.
   - 단, "앞에서/다음에/방금" 같은 시간 지시어 대신, 연결되는 개념/목표를 명시적으로 적으세요.

========================
요약 규칙
========================
{bullets_rule}
{claim_rule}

- 수식/기호는 입력에서 확인되면 그대로 제시한다.
- 입력에 전체 수식이 없더라도 핵심 관계를 복원해 표현할 수 있다(수식 대신 문장 설명도 허용).
  이때 source_type은 inferred 또는 background로 표기하고, confidence는 medium/low로 낮추며,
  notes에 "입력에 전체 수식 없음, 요약/재구성" 같은 설명을 1문장 남긴다.
- 입력과 무관한 수식/기호를 사실처럼 단정하지 않는다.

- bullets는 "강의의 뼈대(학생이 외워야 할 핵심)"입니다.
  * 첫 번째 bullet(가장 위)은 '가장 중요한 개념 + 왜 중요한지(한 문장 안에서)'가 되도록 작성하세요.
  * 나머지 bullets는 (정의/전개/비교/결론/주의) 중 누락된 축을 채우세요.
- definitions는 학생이 '검색 없이' 이해할 수 있도록 1~2문장으로 명확히 쓰되,
  필요하면 괄호로 쉬운 말 풀이를 1개 덧붙일 수 있습니다. (예: "… (쉽게 말해, …)")

========================
독립형 노트 원칙 (가장 중요, 다른 규칙보다 우선)
========================
- 사용자는 영상을/슬라이드를/그림을 "보고 있지 않다". 사용자가 볼 수 있는 것은 오직 당신의 출력(JSON)뿐이다.
- 따라서 다음 표현을 전부 금지한다: "슬라이드", "화면", "그림(을) 보면", "보시면", "위/아래", "여기", "방금", "앞에서", "다음으로", "지금 보이는".
- 시각 정보(visual_text/visual_units)는 '사용자가 보는 이미지'가 아니라 '텍스트로 추출된 장면 정보'다.
  시각 정보에 근거한 내용을 쓰려면, 그 시각 내용을 출력 내부에서 먼저 문장으로 재서술한 다음 해설하라.
  (예: "예시로 x_0는 선명한 이미지, x_t는 노이즈가 섞인 이미지, x_T는 거의 순수 노이즈라는 단계적 변화가 제시된다.")

========================
정의(Definitions) 커버리지 규칙 (필수)
========================
- 각 segment의 definitions는 해당 segment에서 등장하는 "핵심 용어/약어/수식 기호"를 빠짐없이 포함해야 합니다.
- "등장"의 범위는 다음 3가지의 합집합입니다.
  (A) transcript_text / transcript_units에 실제로 등장한 전문 용어/약어/영문 표현
  (B) visual_text / visual_units의 Title/Labels/Equation/Description에 실제로 등장한 전문 용어/약어/수식 기호
  (C) 당신이 bullets/explanations/open_questions에서 사용한 전문 용어/약어/수식 기호
- 포함 대상의 예시(유형): ELBO, KL divergence, likelihood, prior matching, denoising matching, objective function,
  Bayes rule, reparameterization trick, q(·), p(·), p_{{\\theta}}(·), E_q[·], x_0/x_1/x_t/x_T, α_t, ε, I 등.
- 제외 가능: 일반어(예: "생각", "정답", "문제", "결과"), 감탄사/군더더기 표현, 지시어.
- 정의 누락 금지: bullets/explanations/open_questions에 등장하는 용어/기호가 definitions에 없으면 출력은 실패로 간주합니다.
- 정의 개수 최소치: definitions의 길이는 "후보 용어 개수" 이상이어야 합니다.
  단, 관련 기호는 묶어서 1개 항목으로 정의해도 됩니다(예: {{x_0, x_1, x_t, x_T}}를 하나의 term으로 묶어 정의).
- 정의 길이 제한: 각 definition은 1~2문장(최대 3문장)으로 간결하게 작성합니다.
- 작성 절차(권장): (1) 먼저 용어/기호 목록을 만들고 (2) definitions를 채운 뒤 (3) bullets/explanations를 쓰되,
  새 용어를 추가로 쓰게 되면 definitions에도 반드시 추가합니다.

========================
시각/지시어 금지 규칙 (필수, 강화)
========================
- "이/해당/위/아래/여기/수식의/그림의/슬라이드의/마지막 항" 같은 지시어만으로 대상을 지칭하지 마세요.
- "이 수식은", "수식의 마지막 항인", "슬라이드의 고양이 그림은" 같은 문장 형태를 금지합니다.
- 반드시 대상 이름을 먼저 명시하세요.
  좋은 예: "ELBO 3항 분해 식", "Denoising matching 항", "KL(q(x_T|x_0) || p(x_T))", "Reparameterization trick",
          "정규분포 $N(x_t; sqrt(α_t) x_{{t-1}}, (1-α_t) I)$"
- 그림/도식/표/그래프가 있으면, 무엇을 보여주는지 "텍스트로" 서술하세요. (사용자가 실제 그림을 보지 못한다는 전제)

========================
수식/도식 선제 제시 규칙 (필수)
========================
- 수식이 있으면 explanations에서 해당 수식을 처음 설명하는 시점에 반드시 전체 수식(입력에 있는 형태 그대로)을 1회 이상 먼저 제시하라.
  - 제시 방식: 문장 안에 수식을 반드시 `$...$` 또는 `$$...$$`로 감싸서 포함한다.
  - LaTeX 백슬래시 명령을 사용할 수 있다. 단, 출력은 JSON이므로 문자열 값 내부의 백슬래시는 반드시 두 번 연속으로 출력해야 한다(예: `\\theta`, `\\alpha_t`, `\\mathcal{{L}}`).
  - 표기 정규화(권장): `p_theta` 같은 평문 표기 대신 LaTeX 표기인 `$p_{{\\theta}}$`를 사용하라.
  - 표기 정규화(권장): `sum_{{t=2}}^T`/`product_{{t=2}}^T` 같은 평문 표기 대신 `\\sum_{{t=2}}^T`/`\\prod_{{t=2}}^T`를 사용하라.
  - 이후에 각 기호/항의 의미를 풀어쓴다.
- 도식/예시 이미지가 있으면 explanations에서 처음 활용하는 시점에 반드시 다음을 먼저 텍스트로 정의하라:
  1) 무엇이 등장하는지(예: x_0, x_t, x_T의 예시)
  2) 무엇이 변하는지(예: 노이즈가 증가하여 선명→흐림→거의 순수 노이즈)
  3) 왜 이 예시를 쓰는지(예: forward noising / reverse denoising 직관 제공)

========================
source_type / evidence_refs (추적 가능성 유지)
========================
각 항목(bullets/definitions/explanations/open_questions)에 source_type을 반드시 포함하세요.

- "direct": 입력 텍스트를 직접 인용/가까운 패러프레이즈. evidence_refs 필수.
- "inferred": 입력 여러 조각을 종합해 논리적으로 재구성/연결. evidence_refs 필수(근거 unit_id).
- "background": 널리 알려진 정의/수학적 성질/기본 성질/표준 해석 등 일반 배경지식.
  evidence_refs는 빈 배열([]) 허용.
  단, notes에 "어떤 배경지식인지(짧게)"를 반드시 적으세요.

중요: evidence_refs 때문에 설명이 위축되면 안 됩니다.
- 강의에서 말한 내용은 direct/inferred로 근거를 달고,
- 이해를 돕기 위한 일반 설명은 background로 분리하세요.

========================
출력 포맷 규칙 (필수)
========================
- bullet_id 형식: "SEGMENT_ID-INDEX" (INDEX는 1부터 시작)

- bullets 항목:
{{
  "bullet_id": "1-1",
  "claim": "학생이 외울 핵심 1~2문장(가능하면 왜/효과 포함). 독립형 노트로서 완결된 문장으로 작성.",
  "source_type": "direct|inferred|background",
  "evidence_refs": ["t1","v2"],
  "confidence": "high|medium|low",
  "notes": "핵심을 이해시키는 보조 힌트 0~1문장 (필요할 때만). 금지어(슬라이드/그림/위/아래/여기/이 수식) 사용 금지."
}}

- definitions 항목:
{{
  "term": "용어(규칙에 따라 영어 우선). term에는 일반 용어뿐 아니라 수식 기호/표현도 허용됨.",
  "definition": "초학자 기준 정의 1~2문장 + (선택) 쉬운 말 풀이 1개",
  "source_type": "direct|inferred|background",
  "evidence_refs": ["v2"],
  "confidence": "high|medium|low",
  "notes": "오해 포인트가 있으면 1문장. background 사용 시: 어떤 일반 지식인지 1문장으로 명시."
}}
- term 예시(기호/표현 포함):
  "$x_0, x_t, x_T$", "$p_{{\\theta}}(x_{{t-1}} | x_t)$", "$q(x_{{t-1}} | x_t, x_0)$", "$\\alpha_t$", "$\\varepsilon ~ \\mathcal{{N}}(0, I)$", "$E_{{q(\\cdot)}}[\\cdot]$"

========================
JSON 문자열 안정성 규칙 (필수)
========================
- 문자열 값(claim/definition/point/question/notes/term) 내부에는 큰따옴표 문자(")를 절대 포함하지 마세요.
  - JSON 문법을 위한 큰따옴표(키/값을 감싸는 따옴표)는 예외입니다.
  - 인용이 필요하면 괄호() 또는 따옴표 대신 ‘ ’ 같은 문자를 사용하세요.
- LaTeX 등으로 백슬래시(역슬래시) 문자를 써야 한다면, JSON 문자열 안에서는 반드시 `\\` 형태로 이중 이스케이프해서 출력하세요.

- explanations 항목(가장 중요: '가르치기'):
{{
  "point": "4~8문장. 반드시 독립형 노트로 완결된 문장만 사용. (1) 직관/큰그림 → (2) 정확한 의미/정의 → (3) 왜 필요한지(동기) → (4) 입력 속 예시/문맥 연결(텍스트로 재서술) → (5) 흔한 오해/주의점. 수식을 다루면 먼저 전체 수식을 제시한 뒤 기호/항을 설명.",
  "source_type": "direct|inferred|background",
  "evidence_refs": ["t3","v2"],
  "confidence": "high|medium|low",
  "notes": "background 사용 시: 어떤 일반 지식인지 1문장으로 명시. 금지어(슬라이드/그림/위/아래/여기/이 수식) 사용 금지."
}}

- open_questions 항목:
{{
  "question": "학생이 자연스럽게 가질 질문 1문장. 독립형 노트 기준으로 맥락이 통하도록 작성.",
  "source_type": "direct|inferred|background",
  "evidence_refs": ["t2"],
  "confidence": "high|medium|low",
  "notes": ""
}}

========================
필수 '설명' 최소치 (강제)
========================
- 각 segment마다 explanations를 최소 3개 작성하세요.
- explanations 3개는 역할이 겹치면 안 됩니다. 다음 3종류를 최소 1개씩 포함:
  A) 수식/기호 해설(있으면 최우선): "수식에서 특정 기호는 ~을 뜻한다" 형태로 풀어쓰기 (전체 수식 선제 제시 포함)
  B) 비교/대조: 두 개념의 차이(언제/왜/어떤 결과가 다른지)
  C) 동기/용도: "왜 이 용어/수식/가정을 도입하는지"를 학생 관점으로 설명

========================
패러프레이즈 방지 규칙 (중요)
========================
- 입력 문장을 그대로 바꿔 말하는 문장만 나열하지 마세요.
- 최소 1개 bullet 또는 explanation에는 반드시 다음 중 하나를 포함:
  * "즉, …"로 요지를 재구성
  * "왜냐하면 …"로 이유 제시
  * "예를 들어 …"로 간단 예시(입력 맥락 기반, 단 사용자가 그림을 본다는 전제 없이 텍스트로 설명)
  * "주의: …"로 흔한 오해 교정

========================
용어 표기 규칙(중요, 다른 규칙보다 우선)
========================
- 출력은 기본적으로 한국어로 작성한다.
- 단, 아래 범주의 전문 용어는 한국어 번역 대신 영어 원어 표기를 우선 사용한다.
  1) 알고리즘/모델/방법론 명칭 (예: Expectation-Maximization (EM), Variational Inference, Mean-field Variational Inference, ELBO)
  2) 확률/통계/최적화 핵심 개념 (예: log marginal likelihood, posterior, prior, likelihood, KL divergence, stationary point, stationary function, functional derivative)
  3) 수식에 직접 등장하는 심볼/표현 (예: log p(x), q(z), p(z|x), L(q))

- 괄호 규칙:
  * 처음 등장할 때만 "영어 (약어)" 또는 "영어 (한국어 번역)" 중 하나로 병기한다.
    - 기본: 영어 (약어) 형태 권장. 예: Expectation-Maximization (EM)
    - 한국어 병기는 필요할 때만. 예: log marginal likelihood (로그 마지널 라이클리후드)
  * 이후에는 영어/약어만 사용한다.

========================
입력(JSONL, 1줄=1세그먼트)
========================
{jsonl_text}

========================
마지막 점검(출력 전에 스스로 확인)
========================
- 각 segment에 explanations 3개 이상인가?
- 최소 1개는 why/how를 명시했는가?
- background 항목은 notes에 '어떤 배경지식인지'가 적혔는가?
- 금지 표현이 없는가? (슬라이드/화면/그림을 보면/보시면/위/아래/여기/방금/앞에서/다음으로/이 수식/마지막 항)
- 수식/관계는 처음 설명하는 시점에 전체 수식을 1회 이상 먼저 제시했는가?
- 시각 정보는 사용자가 그림을 본다는 전제 없이 텍스트로 재서술했는가?
- definitions가 (A) transcript, (B) visual, (C) bullets/explanations/open_questions에서 등장한 핵심 용어/기호를 모두 포함하는가?
- 재구성한 수식/관계는 source_type/notes/confidence로 출처와 확실성을 표시했는가?
- JSON이 깨지지 않았는가? (코드블록/마크다운 금지, 순수 JSON 배열만 출력)
"""

    return prompt


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_text_from_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    return part_text
    raise ValueError("Gemini 응답에서 텍스트를 추출할 수 없습니다.")


def _generate_content(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
) -> str:
    client = client_bundle.client
    config = {
        "temperature": temperature,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
    }
    try:
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
            timeout=timeout_sec,
        )
    except TypeError:
        response = client.models.generate_content(
            model=client_bundle.model,
            contents=prompt,
            config=config,
        )
    return _extract_text_from_response(response)


def _normalize_evidence_refs(evidence_refs: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(evidence_refs, list):
        candidates = [str(item) for item in evidence_refs]
    elif isinstance(evidence_refs, dict):
        candidates.extend(
            [str(item) for item in evidence_refs.get("transcript_unit_ids", [])]
        )
        candidates.extend(
            [str(item) for item in evidence_refs.get("visual_unit_ids", [])]
        )
    elif isinstance(evidence_refs, str):
        candidates = [evidence_refs]
    else:
        return []

    normalized: List[str] = []
    seen = set()
    for item in candidates:
        if not (item.startswith("t") or item.startswith("v")):
            continue
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _normalize_confidence(value: Any) -> str:
    confidence = str(value or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        return "low"
    return confidence


def _normalize_source_type(value: Any) -> str:
    source_type = str(value or "direct").lower()
    if source_type not in {"direct", "inferred", "background"}:
        return "direct"
    return source_type


def _validate_summary_payload(
    payload: Dict[str, Any],
    segment_id: int,
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int,
) -> Dict[str, Any]:
    bullets = payload.get("bullets") or []
    definitions = payload.get("definitions") or []
    explanations = payload.get("explanations") or []
    open_questions = payload.get("open_questions") or []

    if not isinstance(bullets, list):
        raise ValueError("bullets 형식이 올바르지 않습니다.")
    if not isinstance(definitions, list):
        raise ValueError("definitions 형식이 올바르지 않습니다.")
    if not isinstance(explanations, list):
        raise ValueError("explanations 형식이 올바르지 않습니다.")
    if not isinstance(open_questions, list):
        raise ValueError("open_questions 형식이 올바르지 않습니다.")
    if bullets_min > 0 and len(bullets) < bullets_min:
        raise ValueError("bullets 개수가 부족합니다.")

    normalized_bullets: List[Dict[str, Any]] = []
    max_items = bullets_max if bullets_max > 0 else len(bullets)
    for idx, bullet in enumerate(bullets[:max_items], start=1):
        if not isinstance(bullet, dict):
            continue
        claim = str(bullet.get("claim", "")).strip()
        if claim_max_chars > 0 and len(claim) > claim_max_chars:
            raise ValueError("claim 길이가 초과되었습니다.")
        source_type = _normalize_source_type(bullet.get("source_type"))
        evidence_refs = _normalize_evidence_refs(bullet.get("evidence_refs"))
        confidence = _normalize_confidence(bullet.get("confidence"))
        notes = str(bullet.get("notes", "")).strip()
        bullet_id = f"{segment_id}-{idx}"
        normalized_bullets.append(
            {
                "bullet_id": bullet_id,
                "claim": claim,
                "source_type": source_type,
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "notes": notes,
            }
        )

    normalized_definitions: List[Dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not term or not definition:
            continue
        source_type = _normalize_source_type(item.get("source_type"))
        evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
        confidence = _normalize_confidence(item.get("confidence"))
        notes = str(item.get("notes", "")).strip()
        normalized_definitions.append(
            {
                "term": term,
                "definition": definition,
                "source_type": source_type,
                "evidence_refs": evidence_refs,
                "confidence": confidence,
                "notes": notes,
            }
        )

    normalized_explanations: List[Dict[str, Any]] = []
    for item in explanations:
        if isinstance(item, dict):
            point = str(item.get("point", "")).strip()
            if not point:
                continue
            source_type = _normalize_source_type(item.get("source_type"))
            evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
            confidence = _normalize_confidence(item.get("confidence"))
            notes = str(item.get("notes", "")).strip()
            normalized_explanations.append(
                {
                    "point": point,
                    "source_type": source_type,
                    "evidence_refs": evidence_refs,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized_explanations.append(
            {
                "point": text,
                "source_type": "direct",
                "evidence_refs": [],
                "confidence": "low",
                "notes": "확인 불가",
            }
        )

    normalized_questions: List[Dict[str, Any]] = []
    for item in open_questions:
        if isinstance(item, dict):
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            source_type = _normalize_source_type(item.get("source_type"))
            evidence_refs = _normalize_evidence_refs(item.get("evidence_refs"))
            confidence = _normalize_confidence(item.get("confidence"))
            notes = str(item.get("notes", "")).strip()
            normalized_questions.append(
                {
                    "question": question,
                    "source_type": source_type,
                    "evidence_refs": evidence_refs,
                    "confidence": confidence,
                    "notes": notes,
                }
            )
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized_questions.append(
            {
                "question": text,
                "source_type": "direct",
                "evidence_refs": [],
                "confidence": "low",
                "notes": "확인 불가",
            }
        )

    return {
        "bullets": normalized_bullets,
        "definitions": normalized_definitions,
        "explanations": normalized_explanations,
        "open_questions": normalized_questions,
    }


def _parse_json_response(text: str) -> Any:
    cleaned = _strip_code_fences(text)
    return json.loads(cleaned)


def _repair_prompt(
    bad_json: str, bullets_min: int, bullets_max: int, claim_max_chars: int
) -> str:
    claim_rule = (
        f"- claim은 {claim_max_chars}자 이하"
        if claim_max_chars > 0
        else "- claim은 한 문장 (길이 제한 없음)"
    )
    if bullets_min > 0 and bullets_max > 0:
        bullets_rule = f"- bullets는 {bullets_min}~{bullets_max}개"
    elif bullets_min > 0:
        bullets_rule = f"- bullets는 최소 {bullets_min}개 (상한 없음)"
    else:
        bullets_rule = "- bullets는 필요한 만큼"
    return f"""아래는 잘못된 JSON 출력입니다. 반드시 유효한 JSON만 반환하세요.
출력은 JSON 배열이며, 각 원소는 segment_id와 summary를 포함해야 합니다.
summary는 bullets/definitions/explanations/open_questions 구조입니다. 설명 없이 JSON만 출력하세요.

규칙:
{bullets_rule}
{claim_rule}
- evidence_refs는 unit_id(t*, v*) 배열만 사용
- confidence는 low|medium|high

잘못된 출력:
{bad_json}
"""


def _run_with_retries(
    client_bundle: GeminiClientBundle,
    prompt: str,
    response_schema: Dict[str, Any],
    temperature: float,
    response_mime_type: str,
    timeout_sec: int,
    max_retries: int,
    backoff_sec: List[int],
) -> str:
    attempt = 0
    while True:
        try:
            return _generate_content(
                client_bundle,
                prompt,
                response_schema,
                temperature,
                response_mime_type,
                timeout_sec,
            )
        except Exception as exc:
            if attempt >= max_retries:
                logger.error(f"GenerateContent failed after {max_retries} retries: {exc}")
                raise
            sleep_for = (
                backoff_sec[min(attempt, len(backoff_sec) - 1)] if backoff_sec else 1
            )
            logger.warning(
                f"GenerateContent failed (attempt {attempt+1}/{max_retries}). "
                f"Retrying in {sleep_for}s. Error: {exc}"
            )
            time.sleep(max(sleep_for, 0))
            attempt += 1


def run_summarizer(
    config: ConfigBundle, limit: Optional[int] = None, dry_run: bool = False
) -> None:
    paths = config.paths
    ensure_output_root(paths.output_root)
    output_dir = paths.output_root / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = output_dir / "segments_units.jsonl"
    if not input_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {input_jsonl}")

    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(input_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        raise ValueError("요약할 세그먼트가 없습니다.")

    prompt = _build_batch_prompt(segments, claim_max_chars, bullets_min, bullets_max)

    if dry_run:
        print(f"[DRY RUN] segments={len(segments)} (LLM 미호출, 출력 미생성)")
        return

    response_schema = _build_response_schema()
    client_bundle = _init_gemini_client(config)

    # Count input tokens and save to token_usage.json
    try:
        token_result = client_bundle.client.models.count_tokens(
            model=client_bundle.model,
            contents=prompt
        )
        input_tokens = token_result.total_tokens
        update_token_usage(
            output_dir=output_dir,
            component="summarizer",
            input_tokens=input_tokens,
            model=client_bundle.model,
            extra={"segments_count": len(segments)}
        )
        print(f"[TOKEN] summarizer input_tokens={input_tokens}")
    except Exception as exc:
        print(f"[TOKEN] count_tokens failed: {exc}")

    output_jsonl = output_dir / "segment_summaries.jsonl"
    output_handle = None
    try:
        output_handle = output_jsonl.open("w", encoding="utf-8")

        llm_text = _run_with_retries(
            client_bundle,
            prompt,
            response_schema,
            config.raw.summarizer.temperature,
            config.raw.llm_gemini.response_mime_type,
            config.raw.llm_gemini.timeout_sec,
            config.raw.llm_gemini.max_retries,
            config.raw.llm_gemini.backoff_sec,
        )

        last_error: Optional[Exception] = None
        attempts = config.raw.summarizer.json_repair_attempts
        for _ in range(attempts + 1):
            try:
                payload = _parse_json_response(llm_text)
                if not isinstance(payload, list):
                    raise ValueError("응답이 JSON 배열 형식이 아닙니다.")

                summary_map: Dict[int, Dict[str, Any]] = {}
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("응답 배열의 항목 형식이 올바르지 않습니다.")
                    if "segment_id" not in item:
                        raise ValueError("응답에 segment_id가 없습니다.")
                    segment_id = int(item.get("segment_id"))
                    if segment_id in summary_map:
                        raise ValueError(f"중복 segment_id 발견: {segment_id}")
                    summary_map[segment_id] = _validate_summary_payload(
                        item.get("summary", {}),
                        segment_id,
                        claim_max_chars,
                        bullets_min,
                        bullets_max,
                    )

                expected_ids = [int(seg.get("segment_id")) for seg in segments]
                if set(summary_map.keys()) != set(expected_ids):
                    missing = sorted(set(expected_ids) - set(summary_map.keys()))
                    extra = sorted(set(summary_map.keys()) - set(expected_ids))
                    raise ValueError(
                        f"segment_id 불일치 (missing={missing}, extra={extra})"
                    )

                for segment in segments:
                    segment_id = int(segment.get("segment_id"))
                    summary = summary_map[segment_id]
                    record = {
                        "run_id": segment.get("run_id"),
                        "segment_id": segment.get("segment_id"),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "summary": summary,
                        "version": {
                            "schema_version": 1,
                            "prompt_version": PROMPT_VERSION,
                            "llm_model_id": config.raw.llm_gemini.model,
                            "temperature": config.raw.summarizer.temperature,
                            "backend": config.raw.llm_gemini.backend,
                        },
                    }
                    output_handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                    )
                    output_handle.write("\n")

                last_error = None
                break
            except Exception as exc:
                last_error = exc
                repair_prompt = _repair_prompt(
                    llm_text, bullets_min, bullets_max, claim_max_chars
                )
                llm_text = _run_with_retries(
                    client_bundle,
                    repair_prompt,
                    response_schema,
                    config.raw.summarizer.temperature,
                    config.raw.llm_gemini.response_mime_type,
                    config.raw.llm_gemini.timeout_sec,
                    config.raw.llm_gemini.max_retries,
                    config.raw.llm_gemini.backoff_sec,
                )
        if last_error:
            raise RuntimeError(f"LLM JSON/검증 실패: {last_error}")
    finally:
        if output_handle:
            output_handle.close()

    print_jsonl_head(output_jsonl, max_lines=2)


def run_batch_summarizer(
    *,
    segments_units_jsonl: Path,
    output_dir: Path,
    config: ConfigBundle,
    previous_context: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """배치별 세그먼트 요약을 생성합니다.

    Args:
        segments_units_jsonl: 배치의 segments_units.jsonl 경로
        output_dir: 배치별 출력 디렉토리
        config: ConfigBundle 인스턴스
        previous_context: 이전 배치의 요약 context (선택)
        limit: 처리할 세그먼트 수 제한 (선택)

    Returns:
        success: 실행 성공 여부
        segment_summaries_jsonl: 생성된 파일 경로
        segments_count: 처리된 세그먼트 수
        context: 다음 배치에 전달할 context
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not segments_units_jsonl.exists():
        raise FileNotFoundError(f"segments_units.jsonl이 없습니다: {segments_units_jsonl}")

    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars

    segments: List[Dict[str, Any]] = []
    for segment in read_jsonl(segments_units_jsonl):
        if limit is not None and len(segments) >= limit:
            break
        segments.append(segment)

    if not segments:
        # 세그먼트가 없으면 빈 파일 생성
        output_jsonl = output_dir / "segment_summaries.jsonl"
        output_jsonl.write_text("", encoding="utf-8")
        return {
            "success": True,
            "segment_summaries_jsonl": str(output_jsonl),
            "segments_count": 0,
            "context": "",
        }

    prompt = _build_batch_prompt(
        segments, claim_max_chars, bullets_min, bullets_max, previous_context
    )

    response_schema = _build_response_schema()
    client_bundle = _init_gemini_client(config)

    # Token 사용량 기록
    try:
        token_result = client_bundle.client.models.count_tokens(
            model=client_bundle.model,
            contents=prompt
        )
        input_tokens = token_result.total_tokens
        update_token_usage(
            output_dir=output_dir,
            component="batch_summarizer",
            input_tokens=input_tokens,
            model=client_bundle.model,
            extra={"segments_count": len(segments), "has_context": previous_context is not None}
        )
        print(f"[TOKEN] batch_summarizer input_tokens={input_tokens}")
    except Exception as exc:
        print(f"[TOKEN] count_tokens failed: {exc}")

    output_jsonl = output_dir / "segment_summaries.jsonl"
    output_handle = None
    try:
        output_handle = output_jsonl.open("w", encoding="utf-8")

        llm_text = _run_with_retries(
            client_bundle,
            prompt,
            response_schema,
            config.raw.summarizer.temperature,
            config.raw.llm_gemini.response_mime_type,
            config.raw.llm_gemini.timeout_sec,
            config.raw.llm_gemini.max_retries,
            config.raw.llm_gemini.backoff_sec,
        )

        last_error: Optional[Exception] = None
        attempts = config.raw.summarizer.json_repair_attempts
        for _ in range(attempts + 1):
            try:
                payload = _parse_json_response(llm_text)
                if not isinstance(payload, list):
                    raise ValueError("응답이 JSON 배열 형식이 아닙니다.")

                summary_map: Dict[int, Dict[str, Any]] = {}
                for item in payload:
                    if not isinstance(item, dict):
                        raise ValueError("응답 배열의 항목 형식이 올바르지 않습니다.")
                    if "segment_id" not in item:
                        raise ValueError("응답에 segment_id가 없습니다.")
                    sid = int(item["segment_id"])
                    summary_map[sid] = _validate_summary_payload(
                        item, bullets_min, bullets_max, claim_max_chars
                    )

                # 파일에 기록
                for segment in segments:
                    segment_id = int(segment.get("segment_id"))
                    summary = summary_map[segment_id]
                    record = {
                        "run_id": segment.get("run_id"),
                        "segment_id": segment.get("segment_id"),
                        "start_ms": segment.get("start_ms"),
                        "end_ms": segment.get("end_ms"),
                        "summary": summary,
                        "version": {
                            "schema_version": 1,
                            "prompt_version": PROMPT_VERSION,
                            "llm_model_id": config.raw.llm_gemini.model,
                            "temperature": config.raw.summarizer.temperature,
                            "backend": config.raw.llm_gemini.backend,
                        },
                    }
                    output_handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                    )
                    output_handle.write("\n")

                last_error = None
                break
            except Exception as exc:
                last_error = exc
                repair_prompt = _repair_prompt(
                    llm_text, bullets_min, bullets_max, claim_max_chars
                )
                llm_text = _run_with_retries(
                    client_bundle,
                    repair_prompt,
                    response_schema,
                    config.raw.summarizer.temperature,
                    config.raw.llm_gemini.response_mime_type,
                    config.raw.llm_gemini.timeout_sec,
                    config.raw.llm_gemini.max_retries,
                    config.raw.llm_gemini.backoff_sec,
                )
        if last_error:
            raise RuntimeError(f"LLM JSON/검증 실패: {last_error}")
    finally:
        if output_handle:
            output_handle.close()

    # 다음 배치를 위한 context 추출
    next_context = extract_batch_context(output_jsonl)

    print_jsonl_head(output_jsonl, max_lines=2)

    return {
        "success": True,
        "segment_summaries_jsonl": str(output_jsonl),
        "segments_count": len(segments),
        "context": next_context,
    }


def extract_batch_context(summaries_jsonl: Path, max_chars: int = 500) -> str:
    """배치 요약에서 다음 배치를 위한 context를 추출합니다.

    각 segment의 첫 번째 bullet의 claim만 추출하여 간결한 context를 생성합니다.

    Args:
        summaries_jsonl: segment_summaries.jsonl 경로
        max_chars: 최대 문자 수 (기본: 500)

    Returns:
        context 문자열 (max_chars 이하)
    """
    if not summaries_jsonl.exists():
        return ""

    context_parts = []
    for summary_record in read_jsonl(summaries_jsonl):
        segment_id = summary_record.get("segment_id", "?")
        summary = summary_record.get("summary", {})
        bullets = summary.get("bullets", [])
        if bullets:
            # 첫 번째 bullet의 claim만 추출
            first_claim = bullets[0].get("claim", "")
            if first_claim:
                context_parts.append(f"[Seg {segment_id}] {first_claim}")

    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    return context
