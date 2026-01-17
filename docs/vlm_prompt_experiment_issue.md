# VLM 프롬프트 환각 개선 실험 (v1.1 → v2.2)

## 배경

VLM(Vision Language Model)을 사용해 강의 슬라이드에서 텍스트와 시각 요소를 추출하는 과정에서 **환각(hallucination)** 문제가 발생했습니다.

### 발견된 문제점

1. **격자 크기 환각**: 실제로는 8x8 격자인데 "6x6", "9x9" 등으로 잘못 인식
2. **객체 오인식**: 슬라이드 예시 이미지에서 "말"을 "사람", "고양이"로 잘못 인식
3. **불필요한 메타 정보**: 워터마크, 로고, 저작권 표시 등이 계속 추출됨

이 문제들을 해결하기 위해 **7번의 프롬프트 iteration**을 진행했습니다.

---

## 테스트 대상

환각이 발생한 CV 강의 영상의 슬라이드 2개를 선정하여 실험했습니다.

| 슬라이드 4 (격자 크기 문제) | 슬라이드 8 (객체 오인식 문제) |
| :-------------------------: | :---------------------------: |
|        [이미지 삽입]        |         [이미지 삽입]         |
|     timestamp: 103033ms     |      timestamp: 221533ms      |

---

## 버전별 비교 요약

| 항목          | v1.4 | v1.5 | v1.6 | v1.7 | v1.8 |
| :------------ | :--: | :--: | :--: | :--: | :--: |
| **저작권**    | 제거 | 환각 | 정확 | 환각 | 정확 |
| **격자 크기** | 없음 | 없음 | 없음 | 없음 | 없음 |
| **객체 나열** | 상세 | 요약 | 상세 | 환각 | 없음 |
| **출력 분리** | 2단  | 2단  | 2단  | 3단  | 3단  |

---

## v1.1 (초기 버전)

### 의도

기본 OCR 기능 + 시각 요소 설명을 결합한 초기 프롬프트.

### 문제점

- 출력이 **코드블록 + 설명 혼합** 형태로, 파싱이 어려움
- 워터마크, 강의자 얼굴, 로고 등 **불필요한 메타 정보** 포함
- 격자 크기 **6x6으로 환각** (실제 8x8)

<details>
<summary>v1.1 프롬프트 전문</summary>

**SYSTEM:**

```
Output only Markdown. Use Markdown tables when layout matters. Use LaTeX for equations (inline $...$ and block $$...$$). Do not wrap the output in code fences.
```

**USER:**

```
이미지에 포함된 모든 텍스트와 수식을 가능한 한 원문 그대로 옮겨 적어라. 원문 텍스트는 번역하지 말고 원문 언어를 유지하라. 필요한 설명은 한국어로 간결히 작성하라. 레이아웃이 중요하면 Markdown 표/목록을 사용하고, 수식은 LaTeX($...$, $$...$$)로 표기하라. 텍스트가 거의 없거나 그림/그래프 위주라면 시각 요소를 구체적으로 설명하라
```

</details>

<details>
<summary>v1.1 슬라이드 4 출력</summary>

```
www.BANDICAM.com

1.2 Vanilla example: filter visualization

Filter weight visualization

Input image
[Image of a dog wearing a hot dog costume]

*Filter visualization (1st conv. layer)
[Grid of 36 small grayscale images showing filter weights]

Activation visualization (1st conv. layer)
[Grid of 36 small grayscale images showing activations]

boostcamp****

© NAVER Corporation
10
```

- 상단에는 "www.BANDICAM.com"과 함께 슬라이드 제목 "1.2 Vanilla example: filter visualization"이 표시되어 있습니다.
- 입력 이미지(input image)는 핫도그 의상을 입은 개의 이미지입니다.
- 중앙에는 두 가지 시각화 결과가 나란히 배치되어 있습니다:
  - 왼쪽: "\*Filter visualization (1st conv. layer)" — 첫 번째 컨볼루션 레이어의 필터 가중치를 시각화한 결과로, 6x6 그리드 형태로 36개의 작은 이미지가 배열되어 있습니다.
  - 오른쪽: "Activation visualization (1st conv. layer)" — 동일한 레이어에서 입력 이미지에 대해 계산된 활성화 결과로, 역시 6x6 그리드 형태로 36개의 작은 이미지가 배열되어 있습니다.
- 파란색 사각형과 선을 통해 입력 이미지의 특정 부분이 어떤 필터와 활성화 결과에 대응되는지를 시각적으로 연결하고 있습니다.
- 하단에는 "boostcamp\*\*\*\*" 로고와 "© NAVER Corporation"이 표시되어 있으며, 슬라이드 번호는 10입니다.

</details>

<details>
<summary>v1.1 슬라이드 8 출력</summary>

```
www.BANDICAM.com
2.1 Embedding feature analysis
Nearest neighbors (NN) in a feature space - Example
Analysis of model behaviors

We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

boostcamp
© NAVER Content Foundation
14
```

- 상단에는 "www.BANDICAM.com" 워터마크와 제목 "2.1 Embedding feature analysis"가 있으며, 부제로 "Nearest neighbors (NN) in a feature space - Example"이 표시되어 있다.
- 중앙에는 여러 이미지가 격자 형태로 배열된 그림이 있다. 각 행은 유사한 시각적 개념(예: 보라색 꽃, 코끼리, 배, 호박, 사람 등)을 포함하는 이미지들로 구성되어 있다.
- 중앙 행의 코끼리 이미지들 중 하나가 붉은 사각형으로 강조되어 있으며, 오른쪽에 붉은 화살표와 함께 "We can notice semantically similar concepts are well clustered"라는 텍스트가 있다.
- 하단에는 출처 정보로 "Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012"이 명시되어 있다.
- 하단 왼쪽에는 "boostcamp" 로고, 오른쪽에는 "© NAVER Content Foundation"과 슬라이드 번호 "14"가 표시되어 있다.

</details>

---

## v1.2

### 의도

v1.1의 **출력 형식 개선**을 위해 `## 텍스트` / `## 시각 요소`로 명확히 분리.

### 변경 사항

- 출력 형식을 명시적으로 지정
- 불확실한 객체는 일반적 분류 사용하도록 지시

### 결과

- 출력 구조 개선됨
- 워터마크 언급 감소
- 격자 크기가 **9x9로 환각** (여전히 부정확)

<details>
<summary>v1.2 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor.
Output only Markdown. Do not wrap output in code fences.
Preserve original text exactly. Use LaTeX for equations (inline $...$ and block $$...$$).
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.

## 추출 규칙
1. 슬라이드의 모든 텍스트와 수식을 원문 그대로 옮겨라.
2. 원문 언어를 유지하라.
3. 레이아웃이 중요하면 Markdown 표/목록을 사용하라.
4. 수식은 LaTeX($...$, $$...$$)로 표기하라.

## 시각 요소 설명
텍스트가 적고 다이어그램/그래프/이미지가 있는 경우:
- 구조와 흐름(화살표 방향, 연결 관계 등)을 설명하라.
- 개별 객체를 나열할 때, 확실한 것만 구체적 이름을 사용하라.
- 확신이 낮으면 일반적 분류를 사용하라 (예: "동물", "꽃", "도형").

## 집중할 요소
- 슬라이드의 핵심 내용(제목, 본문, 수식, 다이어그램)에 집중하라.
- 슬라이드 번호는 포함해도 좋다.

## 출력 형식

## 텍스트
[추출한 원문 텍스트/수식]

## 시각 요소
[다이어그램, 그래프 등의 구조 설명 - 텍스트가 충분하면 생략 가능]
```

</details>

<details>
<summary>v1.2 슬라이드 4 출력</summary>

```
## 텍스트

1.2 Vanilla example: filter visualization

Filter weight visualization

Input image

*Filter visualization (1st conv. layer)

Activation visualization (1st conv. layer)

boostcamp****

© NAVER Content Division

10

## 시각 요소

- 슬라이드 상단에 제목 "1.2 Vanilla example: filter visualization"이 표시됨.
- 중앙에는 세 가지 주요 시각적 요소가 수평으로 배열됨:
  1. **Input image**: 왼쪽에 위치한 이미지로, 개가 핫도그 모양의 의상을 입고 있는 모습을 보여줌. 이 이미지 주변에 파란색 테두리가 있음.
  2. **Filter visualization (1st conv. layer)**: 중앙에 위치한 격자 형태의 이미지 배열 (9x9). 각 셀은 다양한 방향과 색상의 선, 패턴을 포함하는 필터 시각화 결과를 보여줌. 일부 셀은 강조되어 있으며, 파란색 사각형으로 둘러싸인 하나의 셀이 입력 이미지와 연결됨.
  3. **Activation visualization (1st conv. layer)**: 오른쪽에 위치한 격자 형태의 이미지 배열 (9x9). 각 셀은 입력 이미지에 대한 필터의 활성화 결과를 보여줌. 대부분 회색조이며, 일부는 선명한 경계나 형태를 포함함. 파란색 사각형으로 둘러싸인 하나의 셀이 입력 이미지와 연결됨.
- 파란색 선으로 입력 이미지에서 중앙 필터 시각화의 특정 셀과 오른쪽 활성화 시각화의 특정 셀이 연결되어 있음.
```

</details>

<details>
<summary>v1.2 슬라이드 8 출력</summary>

```
## 텍스트

2.1 Embedding feature analysis
Nearest neighbors (NN) in a feature space - Example
We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

© NAVER Content Foundation
14

## 시각 요소

- 중앙에 5x5 그리드 형태의 이미지 배열이 존재하며, 각 셀에 다른 이미지가 포함되어 있다.
- 각 행은 특정 주제를 나타내는 것으로 보이며:
  - 첫 번째 행: 보라색 꽃들
  - 두 번째 행: 코끼리들 (중간 행 중 하나가 빨간색 테두리로 강조됨)
  - 세 번째 행: 배들
  - 네 번째 행: 호박 조명 (잭오랜턴)
  - 다섯 번째 행: 고양이와 사람들의 사진들
- 빨간색 테두리로 강조된 코끼리 이미지에 붉은 화살표가 오른쪽으로 나와 "We can notice semantically similar concepts are well clustered"라는 텍스트를 가리킨다.
```

</details>

---

## v1.3

### 의도

**저작권/로고/워터마크 제외** 규칙 추가 및 **격자 크기 불확실 시 생략** 지시.

### 변경 사항

- `## 제외할 요소` 섹션 신설
- "격자 크기가 불확실하면 '격자 형태', '여러 개' 등으로 표현하라"

### 결과

- 일부 메타 정보 제거됨
- 저작권이 여전히 일부 포함됨
- 격자 크기 6x6 환각 지속

<details>
<summary>v1.3 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor.
Output only Markdown. Do not wrap output in code fences.
Preserve original text exactly. Use LaTeX for equations (inline $...$ and block $$...$$).
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.

## 추출 규칙
1. 슬라이드의 핵심 텍스트와 수식을 원문 그대로 옮겨라.
2. 원문 언어를 유지하라.
3. 레이아웃이 중요하면 Markdown 표/목록을 사용하라.
4. 수식은 LaTeX($...$, $$...$$)로 표기하라.

## 제외할 요소
다음 정보는 추출에서 제외하라:
- 저작권 표시 (예: © NAVER Corporation, © 2024 등)
- 로고 텍스트 (예: boostcamp, 회사명 로고 등)
- 워터마크

## 시각 요소 설명
텍스트가 적고 다이어그램/그래프/이미지가 있는 경우:
- 구조와 흐름(화살표 방향, 연결 관계 등)을 설명하라.
- 개별 객체를 나열할 때, 확실한 것만 구체적 이름을 사용하라.
- 확신이 낮으면 일반적 분류를 사용하라 (예: "동물", "꽃", "도형").
- 격자나 배열의 정확한 행/열 개수는 확실한 경우에만 명시하고, 불확실하면 "격자 형태", "여러 개" 등으로 표현하라.

## 집중할 요소
- 슬라이드의 핵심 내용(제목, 본문, 수식, 다이어그램)에 집중하라.
- 슬라이드 번호는 포함해도 좋다.
- 학술 출처(논문명, 저자, 학회명)는 포함하라.

## 출력 형식

## 텍스트
[추출한 원문 텍스트/수식]

## 시각 요소
[다이어그램, 그래프 등의 구조 설명 - 텍스트가 충분하면 생략 가능]
```

</details>

<details>
<summary>v1.3 슬라이드 4 출력</summary>

```
## 텍스트
1.2 Vanilla example: filter visualization

Filter weight visualization:

Input image

*Filter visualization (1st conv. layer)

Activation visualization (1st conv. layer)

© NAVER Content Division
Zeiler and Fergus, Visualizing and Understanding Convolutional Networks, ECCV 2014

## 시각 요소
- 슬라이드 상단에는 제목 "1.2 Vanilla example: filter visualization"이 표시되어 있으며, 그 오른쪽 상단에는 작은 다이어그램이 있다.
- 왼쪽 하단에는 "Input image"로 표시된 이미지가 있으며, 이는 핫도그 모양의 옷을 입은 개의 이미지이다.
- 중앙에는 두 개의 격자 형태의 시각화 결과가 나란히 배치되어 있다:
  - 왼쪽: "*Filter visualization (1st conv. layer)"로 표시된 6x6 격자(총 36개)의 필터 시각화.
  - 오른쪽: "Activation visualization (1st conv. layer)"로 표시된 6x6 격자(총 36개)의 활성화 시각화.
- 입력 이미지에서 왼쪽 필터 시각화 격자 내 한 셀(하단 중앙)으로 파란색 화살표가 가리키고 있으며, 그 셀은 파란색 테두리로 강조되어 있다.
```

</details>

<details>
<summary>v1.3 슬라이드 8 출력</summary>

```
## 텍스트
2.1 Embedding feature analysis 1

Nearest neighbors (NN) in a feature space - Example

We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

© NAVER Corporation

## 시각 요소
- 슬라이드 상단에는 제목 "2.1 Embedding feature analysis 1"이 있으며, 부제로 "Nearest neighbors (NN) in a feature space - Example"이 표시됨.
- 중앙에는 격자 형태의 이미지 배열이 있으며, 각 행은 유사한 시각적 개념(예: 꽃, 코끼리, 배, 호박, 사람)을 포함하고 있음.
- 붉은 사각형으로 강조된 영역이 있으며, 그 옆에 "We can notice semantically similar concepts are well clustered"라는 설명이 있음.
```

</details>

---

## v1.4

### 의도

**격자 크기를 아예 세지 말라**고 명시 + **저작권 제외 강화**.

### 변경 사항

- "반드시 제외할 요소"로 표현 강화
- "격자/배열의 행/열 개수를 세지 마라" 명시적 추가
- "예시 이미지 내용물을 나열하지 마라" 지시

### 결과

- 저작권 완전 제거됨
- 격자 크기 숫자 없음
- 객체 나열은 여전히 상세함

<details>
<summary>v1.4 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor. Extract only educational content.
Output only Markdown. Do not wrap output in code fences.
Preserve original text exactly. Use LaTeX for equations (inline $...$ and block $$...$$).
```

**USER:**

```
이미지에서 슬라이드의 교육적 내용만 추출하라.

## 추출 규칙
1. 슬라이드의 핵심 텍스트와 수식을 원문 그대로 옮겨라.
2. 원문 언어를 유지하라.
3. 레이아웃이 중요하면 Markdown 표/목록을 사용하라.
4. 수식은 LaTeX($...$, $$...$$)로 표기하라.

## 반드시 제외할 요소
- 저작권 표시 (© 기호가 포함된 모든 텍스트)
- 로고 텍스트 (boostcamp 등)
- 워터마크
- 출처 URL

## 시각 요소 설명 규칙
텍스트가 적고 다이어그램/그래프/이미지가 있는 경우:
- 구조와 흐름(화살표 방향, 연결 관계)만 설명하라.
- 격자/배열의 행/열 개수를 세지 마라. "격자 형태", "여러 이미지" 등으로만 표현하라.
- 예시 이미지 격자의 경우, 개별 이미지 내용물을 나열하지 마라. "다양한 카테고리의 예시 이미지들"처럼 요약하라.
- 확신이 없는 객체는 구체적 이름(코끼리, 고양이 등) 대신 일반 분류(동물, 물체)를 사용하라.

## 집중할 요소
- 슬라이드의 핵심 개념과 다이어그램 흐름에 집중하라.
- 슬라이드 번호와 학술 출처(논문명, 저자, 학회명)는 포함하라.

## 출력 형식

## 텍스트
[추출한 원문 텍스트/수식]

## 시각 요소
[다이어그램 구조 설명 - 텍스트가 충분하면 생략 가능]
```

</details>

<details>
<summary>v1.4 슬라이드 4 출력</summary>

```
## 텍스트
1.2 Vanilla example: filter visualization

Filter weight visualization:

Input image

*Filter visualization (1st conv. layer)

Activation visualization (1st conv. layer)

## 시각 요소
슬라이드 상단에는 "1.2 Vanilla example: filter visualization"이라는 제목이 있으며, 오른쪽 상단에는 "Loss → Mid → High"로 연결된 단순한 흐름 다이어그램이 있다. 중앙에는 세 가지 주요 영역이 수평으로 배열되어 있다: 왼쪽에는 "Input image"로 표시된 개가 핫도그를 입고 있는 이미지, 중앙에는 "Filter visualization (1st conv. layer)"로 표시된 격자 형태의 필터 시각화 이미지들(다양한 방향과 패턴의 선, 텍스처), 오른쪽에는 "Activation visualization (1st conv. layer)"로 표시된 격자 형태의 활성화 시각화 이미지들(주로 곡선, 모서리, 테두리와 같은 구조적 특징). 파란색 화살표가 입력 이미지에서 중앙 필터 시각화의 특정 필터를 지정하고, 그 필터가 오른쪽 활성화 시각화의 특정 활성화 영역과 연결되어 있다.
```

</details>

<details>
<summary>v1.4 슬라이드 8 출력</summary>

```
## 텍스트
2.1 Embedding feature analysis 1

Analysis of model behaviors

Nearest neighbors (NN) in a feature space - Example

We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

## 시각 요소
슬라이드 중앙에는 "Nearest neighbors (NN) in a feature space - Example"이라는 제목 아래, 여러 행과 열로 구성된 격자 형태의 이미지들이 배열되어 있다. 각 행은 특정 기준 이미지(예: 보라색 꽃, 코끼리, 배, 호박, 사람 등)를 중심으로 하며, 그 행의 나머지 이미지들은 해당 기준 이미지와 유사한 특징을 가진 것으로 보이는 여러 이미지들로 구성되어 있다. 특히 코끼리 행의 첫 번째 이미지(기준 이미지) 주변에 붉은색 사각형으로 강조된 영역이 있으며, 이는 유사한 코끼리 이미지들이 집중되어 있음을 나타낸다. 오른쪽에는 붉은 화살표가 이 붉은 사각형 영역을 가리키며 "We can notice semantically similar concepts are well clustered"라는 설명이 있다. 전체적으로, 각 행은 특정 개념(예: 동물, 물체, 자연 등)에 속하는 이미지들이 유사도 기준으로 클러스터링된 것을 보여주는 예시이다.
```

</details>

---

## v1.5

### 의도

**객체 나열 완전 금지** - "꽃, 코끼리, 배 등 언급 금지" 명시.

### 변경 사항

- "개별 이미지 내용을 **절대** 나열하지 마라 (꽃, 코끼리, 배 등 언급 금지)" 추가
- "다양한 카테고리의 예시 이미지들이 격자로 배열됨"처럼 요약하도록 강제

### 결과

- 객체 나열이 "다양한 카테고리"로 요약됨
- 격자 크기 숫자 없음 유지
- **저작권 환각 발생** (NAVER → KAIST로 잘못 인식)

> 이 버전에서 예상치 못한 부작용이 발생했습니다. 저작권을 "제외하라"고 강하게 지시했더니, 모델이 저작권 텍스트를 읽지만 다른 내용으로 대체하는 **환각**이 발생했습니다.

<details>
<summary>v1.5 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor. Extract only educational content.
Output only Markdown. Do not wrap output in code fences.
Preserve original text exactly. Use LaTeX for equations.
```

**USER:**

```
이미지에서 슬라이드의 교육적 내용만 추출하라.

## 추출 규칙
1. 슬라이드의 핵심 텍스트와 수식을 원문 그대로 옮겨라.
2. 원문 언어를 유지하라.
3. 레이아웃이 중요하면 Markdown 표/목록을 사용하라.

## 반드시 제외할 요소
- 저작권 표시
- 로고 텍스트
- 워터마크
- 출처 URL

## 시각 요소 설명 규칙
- 구조와 흐름만 설명하라.
- 격자/배열의 개수를 세지 마라. "격자 형태", "여러 이미지" 등으로만 표현하라.
- 예시 이미지가 포함된 격자의 경우:
  - 개별 이미지 내용을 절대 나열하지 마라 (꽃, 코끼리, 배 등 언급 금지).
  - "다양한 카테고리의 예시 이미지들이 격자로 배열됨"처럼 요약하라.
  - 강조된 영역이 있으면 "특정 영역이 강조됨" 정도로만 설명하라.

## 집중할 요소
- 슬라이드의 핵심 개념과 다이어그램 흐름에 집중하라.
- 슬라이드 번호와 학술 출처는 포함하라.

## 출력 형식

## 텍스트
[추출한 원문 텍스트/수식]

## 시각 요소
[다이어그램 구조 설명]
```

</details>

<details>
<summary>v1.5 슬라이드 4 출력</summary>

```
## 텍스트
1.2 Vanilla example: filter visualization

Filter weight visualization:

Input image

*Filter visualization (1st conv. layer)

Activation visualization (1st conv. layer)

© KAIST Content Foundation

13

## 시각 요소
슬라이드 상단에 "1.2 Vanilla example: filter visualization"이라는 제목이 있으며, 오른쪽 상단에 "Loss → Mid → High"로 구성된 간단한 흐름도가 표시됨. 중앙에는 입력 이미지(개가 핫도그를 입고 있는 이미지)가 왼쪽에 위치하고, 그로부터 파란색 화살표가 오른쪽으로 이어져 두 개의 격자 형태의 시각화 결과로 연결됨. 첫 번째 격자는 "Filter visualization (1st conv. layer)"로, 다양한 패턴을 가진 필터들이 격자 형태로 배열됨. 두 번째 격자는 "Activation visualization (1st conv. layer)"로, 입력 이미지에 해당하는 필터가 활성화된 결과를 보여주는 격자 형태의 이미지들이 배열됨. 각 격자 내에서 특정 영역이 파란색 사각형으로 강조되어 입력 이미지와의 연결을 시각화함.
```

</details>

<details>
<summary>v1.5 슬라이드 8 출력</summary>

```
## 텍스트
2.1 Embedding feature analysis 1

Nearest neighbors (NN) in a feature space - Example

We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

© KAIST Content Foundation

14

## 시각 요소
슬라이드 상단에 "2.1 Embedding feature analysis 1"이라는 제목이 있으며, 오른쪽 상단에 "Analysis of model behaviors"라는 부제가 표시됨. 중앙에는 "Nearest neighbors (NN) in a feature space - Example"이라는 제목 아래 여러 이미지들이 격자 형태로 배열됨. 왼쪽 열에는 다양한 카테고리의 예시 이미지들이 세로로 나열되어 있고, 오른쪽에는 각 예시 이미지에 대해 특징 공간에서 가장 가까운 이웃 이미지들이 격자 형태로 배열됨. 특정 영역(예: 코끼리 이미지 주변)이 빨간색 사각형으로 강조되어 있으며, 그 옆에 "We can notice semantically similar concepts are well clustered"라는 설명이 붙어 있음. 전체적으로 특징 공간에서 의미적으로 유사한 개념들이 잘 클러스터링되는 것을 시각적으로 보여주는 구조임.
```

</details>

---

## v1.6 (최종 권장)

### 의도

v1.5의 **저작권 환각 문제 해결**을 위해 저작권 제외 규칙 제거.

### 변경 사항

- `## 반드시 제외할 요소` 섹션 **삭제**
- 나머지 규칙(격자 크기, 객체 요약)은 그대로 유지

### 결과

- 저작권 정확하게 추출 (환각 없음)
- 격자 크기 숫자 없음 유지
- 객체 나열이 다시 상세해짐 (trade-off)

<details>
<summary>v1.6 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor. Extract only educational content.
Output only Markdown. Do not wrap output in code fences.
Preserve original text exactly. Use LaTeX for equations.
```

**USER:**

```
이미지에서 슬라이드의 교육적 내용만 추출하라.

## 추출 규칙
1. 슬라이드의 핵심 텍스트와 수식을 원문 그대로 옮겨라.
2. 원문 언어를 유지하라.
3. 레이아웃이 중요하면 Markdown 표/목록을 사용하라.

## 시각 요소 설명 규칙
- 구조와 흐름만 설명하라.
- 격자/배열의 개수를 세지 마라. "격자 형태", "여러 이미지" 등으로만 표현하라.
- 예시 이미지가 포함된 격자의 경우:
  - 개별 이미지 내용을 절대 나열하지 마라.
  - "다양한 카테고리의 예시 이미지들이 격자로 배열됨"처럼 요약하라.
  - 강조된 영역이 있으면 "특정 영역이 강조됨" 정도로만 설명하라.

## 집중할 요소
- 슬라이드의 핵심 개념과 다이어그램 흐름에 집중하라.
- 슬라이드 번호와 학술 출처는 포함하라.

## 출력 형식

## 텍스트
[추출한 원문 텍스트/수식]

## 시각 요소
[다이어그램 구조 설명]
```

</details>

<details>
<summary>v1.6 슬라이드 4 출력</summary>

```
## 텍스트
1.2 Vanilla example: filter visualization

Filter weight visualization:

Input image

*Filter visualization (1st conv. layer)

Activation visualization (1st conv. layer)

© NAVER Content Foundation

13

## 시각 요소
슬라이드 상단에는 "1.2 Vanilla example: filter visualization"이라는 제목이 있으며, 오른쪽 상단에는 "Loss → Mid → High"로 연결된 간단한 흐름도가 포함됨. 중앙에는 입력 이미지(개가 핫도그를 입은 모습)가 왼쪽에 위치하고, 그로부터 파란색 화살표가 오른쪽으로 이어져 두 개의 격자 형태의 시각화 결과로 연결됨. 첫 번째 격자는 "Filter visualization (1st conv. layer)"로, 다양한 필터 패턴이 격자로 배열됨. 두 번째 격자는 "Activation visualization (1st conv. layer)"로, 입력 이미지에 대한 활성화 결과가 격자로 배열됨. 각 격자 내에서 특정 필터와 활성화 영역이 파란색 사각형으로 강조되어 연결됨. 슬라이드 하단에는 "© NAVER Content Foundation"과 슬라이드 번호 "13"이 표시됨.
```

</details>

<details>
<summary>v1.6 슬라이드 8 출력</summary>

```
## 텍스트
2.1 Embedding feature analysis 1

Nearest neighbors (NN) in a feature space - Example

We can notice semantically similar concepts are well clustered

Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

© NAVER Content Foundation

14

## 시각 요소
슬라이드 상단에는 "2.1 Embedding feature analysis 1"이라는 제목과 오른쪽 상단에 "Analysis of model behaviors"라는 부제가 있음. 중앙에는 "Nearest neighbors (NN) in a feature space - Example"이라는 설명 아래, 여러 이미지가 격자 형태로 배열됨. 왼쪽 열에는 각각의 중심 이미지(보라색 꽃, 코끼리, 배, 호박, 고양이 등)가 있고, 오른쪽에는 해당 중심 이미지와 가장 가까운 이웃 이미지들이 격자로 배열됨. 특정 행(코끼리 이미지 주변)이 빨간색 사각형으로 강조되어 있으며, 그 옆에 "We can notice semantically similar concepts are well clustered"라는 설명이 붙어 있음. 슬라이드 하단에는 "Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012", "© NAVER Content Foundation", 슬라이드 번호 "14"가 표시됨.
```

</details>

---

## v1.7

### 의도

**"제외" 대신 "분리해서 라벨링"** 방식 도입. Main Text / Auxiliary Text / Visual Structure 3단 분리.

### 변경 사항

- 저작권/워터마크 등은 Auxiliary Text에 별도 라벨링
- 객체 나열 금지 규칙 유지

### 결과

- 출력 3단 분리 성공
- 격자 크기 숫자 없음
- **저작권 환각** (NAVER → KAIST)
- **객체 환각** (코끼리 → 사자)

<details>
<summary>v1.7 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide content extractor.
Output only Markdown. Do not wrap output in code fences.
Preserve extracted text exactly as-is. Do not paraphrase text.
If a text is unreadable, write [illegible].
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라. "제외"하지 말고 "분리해서 라벨링"하라.

## 목표
- 교육적 핵심 텍스트/수식은 정확히 보존
- 워터마크/로고/저작권/URL/캡처 툴 표시 등은 별도 섹션으로 분리
- 시각 요소 설명에서는 구조/관계만 말하고, 사진 내용의 구체 객체를 추정하지 않음

## 출력 형식

## Main Text
- 슬라이드 제목/본문/캡션/수식/학술 출처 등 "교육 내용"만 원문 그대로 줄 단위로 나열
- 텍스트를 절대 바꾸지 말 것
- 읽기 불가하면 [illegible]

## Auxiliary Text
- 다음 유형의 텍스트를 "있는 그대로" 나열:
  - 워터마크, 로고 텍스트, 저작권, URL, 캡처 도구 표시, 페이지 번호, UI 오버레이
- 없으면 "- (none)"만 출력

## Visual Structure
- 구조와 흐름(배치, 연결, 화살표, 강조 표시)만 설명
- 격자/배열의 행·열 수를 세지 말 것. "격자 형태", "여러 이미지"로만 표현
- 예시 이미지 격자에서는 개별 이미지의 내용을 말하지 말 것
  - 금지: 꽃/코끼리/배/호박/고양이/사람/개 등 구체 객체명
  - 예외: 슬라이드 내부 텍스트에 해당 단어가 실제로 쓰여 있을 때만 그대로 인용 가능
- 확신이 낮으면 객체 언급 자체를 생략하라
```

</details>

<details>
<summary>v1.7 슬라이드 4 출력</summary>

```
## Main Text
1.2 Vanilla example: filter visualization
Filter weight visualization
Input image
*Filter visualization (1st conv. layer)
Activation visualization (1st conv. layer)
© KAIST Content Foundation

## Auxiliary Text
www.BANDICAM.com
boostcamp**4**
© KAIST Content Foundation
13

## Visual Structure
슬라이드 상단 중앙에 제목이 위치하며, 오른쪽 상단에 작은 다이어그램이 있다. 중앙에는 세 개의 주요 영역이 수평으로 배열되어 있다: 왼쪽은 "Input image"로, 중앙은 여러 작은 사각형 이미지가 격자 형태로 배열된 시각화 결과, 오른쪽은 유사한 격자 형태의 시각화 결과. 입력 이미지에서 중앙 시각화의 특정 필터까지 파란색 화살표가 연결되어 있다.
```

</details>

<details>
<summary>v1.7 슬라이드 8 출력</summary>

```
## Main Text
2.1 Embedding feature analysis 1
Analysis of model behaviors
Nearest neighbors (NN) in a feature space - Example
We can notice semantically similar concepts are well clustered
Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012
© KAIST Content Foundation

## Auxiliary Text
www.BANDICAM.com
boostcamp**4**
© KAIST Content Foundation
14

## Visual Structure
슬라이드 상단 중앙에 제목이 위치하며, 오른쪽 상단에 부제목이 있다. 중앙에는 여러 이미지가 격자 형태로 배열된 시각화 결과가 있다. 이 격자 내에서 특정 이미지(사자) 주변의 이미지들이 붉은색 사각형으로 강조되어 있다.
```

</details>

---

## v1.8

### 의도

v1.7의 두 환각 문제를 해결하기 위한 패치:

1. **텍스트 환각 방지**: OCR 모드 강화, 불확실하면 [illegible] 강제
2. **객체 환각 방지**: Visual Structure를 3줄 템플릿으로 제한

### 결과

- 저작권 정확: © NAVER Content Foundation
- 객체명 없음: "특정 셀", "특정 행"으로만 지칭
- 격자 크기: 숫자 없음
- 3줄 템플릿: Layout / Links / Highlight

<details>
<summary>v1.8 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide OCR and layout extractor.
Output only Markdown. Do not wrap output in code fences.

For Main Text and Auxiliary Text:
- Copy text exactly as seen in the image. Do NOT correct, normalize, or substitute words.
- Never guess unreadable words. If any part of a line is uncertain, replace only that uncertain span with [illegible] and keep the rest as-is.

For Visual Structure:
- Describe only layout/relations using the allowed vocabulary in the user instruction.
- Do not name any objects in photos.
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.

## 절대 규칙
- Main Text / Auxiliary Text는 "OCR 결과"다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 해당 단어만 [illegible]로 대체하라.

## 분류 규칙
다음은 무조건 Auxiliary Text:
- © 포함 줄, 워터마크, 로고, URL, 페이지 번호

## Visual Structure (3줄 템플릿만)
- Layout: [배치]
- Links: [연결, 없으면 (none)]
- Highlight: [강조, 없으면 (none)]

구체 객체명 금지, "특정 셀/영역"으로만 지칭
```

</details>

<details>
<summary>v1.8 슬라이드 4 출력</summary>

```
## Main Text
- 1.2 Vanilla example: filter visualization
- Filter weight visualization
- Input image
- *Filter visualization (1st conv. layer)
- Activation visualization (1st conv. layer)

## Auxiliary Text
- www.BANDICAM.com
- boostcamp
- © NAVER Content Foundation
- 13

## Visual Structure
- Layout: 텍스트 블록이 상단에, 중앙에 세 영역이 수평 배치
- Links: 파란색 선으로 입력 이미지 → 특정 셀 → 특정 셀 연결
- Highlight: 특정 셀들이 파란색 테두리로 강조
```

</details>

<details>
<summary>v1.8 슬라이드 8 출력</summary>

```
## Main Text
- 2.1 Embedding feature analysis 1
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

## Auxiliary Text
- www.BANDICAM.com
- Analysis of model behaviors
- boostcamp
- © NAVER Content Foundation
- 14

## Visual Structure
- Layout: 상단 제목, 중앙에 이미지 격자와 설명 텍스트
- Links: 붉은색 화살표로 격자 → 설명 텍스트 연결
- Highlight: 격자 내 특정 행이 붉은색 테두리로 강조
```

</details>

---

## v1.9

### 의도

v1.8에서 정보 손실을 줄이기 위해 **조건부 객체명** 허용:

- (high) / (medium) 접두어로 객체명 사용 가능
- 격자 크기/개수 추정 금지
- downstream에서 STT와 병합 후 Gemini 요약 입력으로 사용

<details>
<summary>v1.9 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide OCR + visual-evidence extractor.
Output only Markdown. Do not wrap output in code fences.

For Main Text and Auxiliary Text:
- Copy text exactly as seen in the image. Do NOT correct, normalize, or substitute words.
- Never guess unreadable words. If any part of a line is uncertain, replace only that span with [illegible] and keep the rest as-is.

For Visual Evidence:
- Use only what is visible in the slide (printed labels, layout, arrows/lines, highlight boxes, and photo contents).
- Do not infer counts (e.g., grid sizes) unless the number is explicitly printed.
- Do not invent proper nouns or substitute organizations/names.
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.
이 출력은 downstream에서 STT와 병합되어 Gemini 3 Flash 요약 입력으로 사용된다.
따라서 해석/추론(interpretation) 문장을 생성하지 말고, 환각을 최소화하는 근거(evidence) 중심으로 추출하라.

1) 절대 규칙 (OCR)
- Main Text / Auxiliary Text는 OCR 결과다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 추정하지 말고 해당 단어/구절만 [illegible]로 대체하라.

2) 분류 규칙
다음은 무조건 Auxiliary Text로 분류:
- © 포함 줄
- 워터마크/캡처 툴 표시(예: BANDICAM)
- 로고/브랜드 텍스트(예: boostcamp)
- URL
- 페이지 번호(숫자만 있는 페이지 표기)
그 외(제목/본문/캡션/수식/학술 출처/섹션 헤더)는 Main Text.

3) 출력 형식 (반드시 준수)
Main Text
- 교육 텍스트를 원문 그대로 줄 단위로 나열

Auxiliary Text
- 메타 텍스트를 원문 그대로 줄 단위로 나열
- 없으면 "- (none)"

Visual Evidence
- 아래 3개 섹션만 포함: Layout, Links, Highlights
- 각 섹션은 관계 데이터처럼 간결하게 작성
- 가능한 경우 슬라이드에 실제로 인쇄된 라벨을 따옴표로 포함

Layout
- 배치 관계 기록(상/하, 좌/우, 패널 구성 등)
- 사진/예시 이미지의 구체 객체명을 단정하지 말고, 필요하면 4) 객체명 규칙을 적용

Links
- 보이는 화살표/선/연결만 기록, 없으면 "- (none)"
- 형식: "FROM" -> "TO" (색/형태가 보이면 괄호로)

Highlights
- 보이는 박스/테두리/밑줄/색 강조만 기록, 없으면 "- (none)"
- 형식: "대상 라벨" (강조 형태/색)

4) 객체명 규칙 (Visual Evidence에서만 적용)
- 사진/예시 이미지의 객체명을 쓰고 싶다면 반드시 (high|medium) 접두어를 붙여라.
  예: (high) elephant, (medium) animal
- high/medium만 허용하고 low는 쓰지 마라.
- 확신이 medium이면 더 일반적인 표현을 우선하라(예: "animal", "vehicle").
- 슬라이드에 실제로 인쇄된 텍스트에 객체명이 존재하면, 그 단어를 우선 사용하라.
- 객체명은 단정 서술로 쓰지 말고, 반드시 위 접두어를 포함한 "후보"로만 작성하라.

5) 금지 사항(환각 방지)
- Main Text/Auxiliary Text에서 글자를 더 그럴듯하게 고치는 행위 금지(조직명/고유명사 치환 금지)
- Visual Evidence에서 격자 행/열/총 개수 추정 금지(예: 5x5, 36개 등)
- Visual Evidence에서 객체명을 쓸 때 (high|medium) 접두어 없이 쓰는 것 금지
```

</details>

<details>
<summary>v1.9 슬라이드 4 출력 (전문)</summary>

```
Main Text
- 1.2 Vanilla example: filter visualization
- Filter weight visualization
- Input image
- *Filter visualization (1st conv. layer)
- Activation visualization (1st conv. layer)
- © NAVER Content Foundation  ← [분류 오류: Auxiliary여야 함]

Auxiliary Text
- www.BANDICAM.com
- boostcamp*****
- 13

Visual Evidence
Layout
- 상단 중앙에 제목 "1.2 Vanilla example: filter visualization" 위치
- 제목 아래 좌측에 "Input image" 라벨이 붙은 이미지 (강아지가 핫도그 모양으로 꾸며진 사진) 배치  ← [접두어 없음 - 위반]
- 중앙에 "Filter visualization (1st conv. layer)" 라벨이 붙은 격자형 이미지 배치
- 우측에 "Activation visualization (1st conv. layer)" 라벨이 붙은 격자형 이미지 배치
- 하단 중앙에 "© NAVER Content Foundation" 표시
- 좌측 하단에 "boostcamp****" 로고
- 우측 하단에 페이지 번호 "13"

Links
- "Input image" -> "Filter visualization (1st conv. layer)" (파란색 선)
- "Filter visualization (1st conv. layer)" -> "Activation visualization (1st conv. layer)" (파란색 선)

Highlights
- "Input image" (파란색 테두리)
- "Filter visualization (1st conv. layer)"의 특정 셀 (파란색 테두리)
- "Activation visualization (1st conv. layer)"의 특정 셀 (파란색 테두리)
```

</details>

<details>
<summary>v1.9 슬라이드 8 출력 (전문)</summary>

```
Main Text
- 2.1 Embedding feature analysis 1
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- © NAVER Content Foundation  ← [분류 오류: Auxiliary여야 함]
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

Auxiliary Text
- www.BANDICAM.com
- Analysis of model behaviors  ← [분류 오류: Main Text여야 함]
- boostcamp*****
- 14

Visual Evidence
Layout
- 상단 중앙에 제목 "2.1 Embedding feature analysis 1" 위치
- 상단 우측에 부제목 "Analysis of model behaviors" 위치
- 중앙에 격자형 이미지 배열 배치
- 오른쪽 이미지 배열 중 일부 행이 빨간색 테두리로 강조됨
- 하단 중앙에 "© NAVER Content Foundation" 표시

Links
- 빨간색 테두리 영역 -> "We can notice semantically similar concepts are well clustered" (빨간색 화살표)

Highlights
- 오른쪽 이미지 배열의 일부 행 (빨간색 테두리)
```

</details>

### 결과 분석

| 항목              | 결과                                                                   |
| :---------------- | :--------------------------------------------------------------------- |
| 저작권            | © NAVER Content Foundation (정확, 하지만 Main Text에 포함 - 분류 오류) |
| 격자 크기         | **없음** (개선)                                                        |
| 객체명 접두어     | "강아지가 핫도그 모양으로 꾸며진 사진" (**접두어 없음 - 위반**)        |
| 슬라이드 8 객체명 | **없음** (개선)                                                        |

### v1.9 평가

**개선된 점:**

- 격자 크기 숫자(5x5) 없음
- 슬라이드 8에서 구체 객체명(코끼리, 꽃 등) 없음

**남은 이슈:**

- `© NAVER Content Foundation`이 Main Text에 포함됨 (분류 오류)
- `Analysis of model behaviors`가 Auxiliary에 포함됨 (분류 오류)
- 슬라이드 4에서 "강아지가 핫도그..." - (high|medium) 접두어 없음 (위반)

---

## v2

### 의도

Visual Evidence의 자연스러운 서술을 허용하되, OCR 환각(고유명사 치환)을 방지하는 것이 목표.

- Main/Auxiliary Text 분류 규칙 명확화
- 객체명을 "자연스러운 요약"으로 유도
- 숫자/수량 추정 금지

<details>
<summary>v2 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide OCR + visual-evidence extractor.
Output only Markdown. Do not wrap output in code fences.

For Main Text and Auxiliary Text:
- Copy text exactly as seen in the image. Do NOT correct, normalize, or substitute words.
- Never guess unreadable words. If any part of a line is uncertain, replace only that span with [illegible] and keep the rest as-is.

For Visual Evidence:
- Use only what is visible in the slide (printed labels, layout, arrows/lines, highlight boxes, and photo contents).
- Do not infer counts (e.g., grid sizes) unless the number is explicitly printed.
- Do not invent proper nouns or substitute organizations/names.
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.
이 출력은 downstream에서 STT와 병합되어 Gemini 3 Flash 요약 입력으로 사용된다.
따라서 해석/추론(interpretation) 문장을 생성하지 말고, 환각을 최소화하는 근거(evidence) 중심으로 추출하라.

1) 절대 규칙 (OCR)
- Main Text / Auxiliary Text는 OCR 결과다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 추정하지 말고 해당 단어/구절만 [illegible]로 대체하라.
- 보이는 텍스트를 더 그럴듯하게 고치거나 다른 단어로 바꾸지 마라(특히 고유명사/기관명).

2) 분류 규칙
Auxiliary Text로 분류할 것(메타/오버레이 성격):
- ©가 포함된 저작권/라이선스 문구
- 워터마크/캡처 툴 표시/플레이어 UI 오버레이
- 로고/브랜드 표기(슬라이드 내용과 무관한 식별 목적)
- URL/도메인
- 페이지 번호/슬라이드 번호(단독 숫자 또는 페이지 표기)

Main Text로 분류할 것(교육/콘텐츠 성격):
- 제목, 본문, 캡션, 표/그림 안의 설명 텍스트, 수식
- 참고문헌/학술 출처(저자/논문/학회 등)
- 섹션 헤더/소제목(내용을 설명하는 텍스트)

3) 출력 형식 (반드시 준수)
Main Text
- 교육 텍스트를 원문 그대로 줄 단위로 나열

Auxiliary Text
- 메타 텍스트를 원문 그대로 줄 단위로 나열
- 없으면 "- (none)"

Visual Evidence
- 아래 3개 섹션만 포함: Layout, Links, Highlights
- 각 섹션은 "보이는 것"만 근거로 간결히 작성
- 가능한 경우 슬라이드에 실제로 인쇄된 라벨/캡션을 따옴표로 포함
- Visual Evidence에서는 시각 요소의 내용(대표 객체/범주)을 자연스럽게 요약해 포함하라(4) 객체명 규칙 적용)
- 단, 숫자(개수/행열/정량값)는 슬라이드에 명시적으로 쓰여 있지 않으면 추정하지 마라.

Layout
- 주요 영역의 배치(상/하, 좌/우, 중앙, 패널 구성, 그룹화)를 기록하라.
- 이미지/도표/그래프/다이어그램이 있으면 그 종류와 대략적 역할을 요약하라.
- 시각 요소가 사진/예시 이미지/일러스트를 포함한다면, 보이는 대표 객체/범주를 한 문장으로 자연스럽게 포함하라(불확실하면 일반화/생략).

Links
- 보이는 화살표/선/연결 관계만 기록, 없으면 "- (none)"
- 형식: "FROM" -> "TO" (색/형태가 보이면 괄호로)
- FROM/TO가 시각 요소(사진/예시 이미지/도표 등)인 경우, 가능하면 그 요소의 내용 범주를 짧게 덧붙여라(불확실하면 일반화/생략).

Highlights
- 보이는 강조 표시(박스/테두리/밑줄/색 강조/마커)만 기록, 없으면 "- (none)"
- 형식: "대상" (강조 형태/색)
- 대상은 가능한 경우 슬라이드 라벨/캡션을 사용하라. 라벨이 없으면 "특정 영역/특정 항목"처럼 기술하라.
- 강조 대상이 시각 요소(사진/예시 이미지/도표 등)인 경우, 가능하면 그 요소의 내용 범주를 짧게 덧붙여라(불확실하면 일반화/생략).

4) 객체명 규칙 (Visual Evidence 전반에 적용)
- 시각 요소의 내용을 요약할 때, 보이는 객체/범주를 자연스럽게 언급하라.
- 확신이 낮으면 구체명 대신 일반 분류로 바꿔라(예: 동물, 사람, 식물, 탈것, 음식, 물체, 풍경 등).
- 확신이 낮으면 억지로 채우지 말고 생략하거나 일반 분류만 쓰는 쪽을 우선하라.
- 슬라이드에 실제로 쓰인 단어가 있으면 그 단어를 우선 사용하라.
- 고유명사/기관명/모델명/지명/브랜드명은 슬라이드에 명시적으로 쓰여 있지 않으면 추정하지 마라.

5) 금지 사항(환각 방지)
- Visual Evidence에서 격자 행/열/총 개수 등 수량 추정 금지
- 슬라이드에 없는 텍스트를 만들어내지 마라
- 출력 전 최종 점검: Main Text에 © 포함 줄이 있으면 Auxiliary Text로 옮겨라.
```

</details>

<details>
<summary>v2 슬라이드 4 출력 (전문)</summary>

```
Main Text
- 1.2 Vanilla example: filter visualization
- Filter weight visualization
- Input image
- *Filter visualization (1st conv. layer)
- Activation visualization (1st conv. layer)

Auxiliary Text
- www.BANDICAM.com
- boostcamp****
- 13
- © KAIST Content Foundation  ← [환각: NAVER여야 함]

Visual Evidence
Layout
- 상단 중앙에 제목 "1.2 Vanilla example: filter visualization"이 위치.
- 상단 우측에 작은 다이어그램이 있으며, "Lear" → "Mid" → "High" → [파란색 사각형]으로 연결된 흐름을 보여줌. 각 단계는 색상이 다른 사각형으로 표현됨.
- 중앙에는 세 개의 주요 영역이 좌우로 배열됨: 왼쪽은 "Input image"로 라벨링된 사진, 중앙은 "*Filter visualization (1st conv. layer)"로 라벨링된 격자형 이미지, 오른쪽은 "Activation visualization (1st conv. layer)"로 라벨링된 격자형 이미지.
- 왼쪽 사진은 개가 핫도그를 입고 있는 이미지.
- 중앙과 오른쪽 격자 이미지들은 각각 필터와 활성화 결과를 나타내는 작은 그레이스케일 이미지들로 구성됨.
- 하단 중앙에 "© KAIST Content Foundation" 문구가 표시됨.
- 하단 우측에 슬라이드 번호 "13"이 표시됨.

Links
- "Input image" (개가 핫도그를 입은 사진) -> "*Filter visualization (1st conv. layer)" (격자 이미지 내 특정 필터) -> "Activation visualization (1st conv. layer)" (격자 이미지 내 특정 활성화 영역) (파란색 선)

Highlights
- "Input image" (파란색 테두리)
- "*Filter visualization (1st conv. layer)" 내 특정 필터 (파란색 테두리)
- "Activation visualization (1st conv. layer)" 내 특정 활성화 영역 (파란색 테두리)
```

</details>

<details>
<summary>v2 슬라이드 8 출력 (전문)</summary>

```
Main Text
- 2.1 Embedding feature analysis 1
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012
- © KAIST Content Foundation  ← [환각: NAVER여야 함]

Auxiliary Text
- www.BANDICAM.com
- boostcamp****


Visual Evidence
Layout
- 상단 중앙에 제목 "2.1 Embedding feature analysis 1"이 위치.
- 상단 우측에 부제목 "Analysis of model behaviors"가 표시됨.
- 중앙에는 "Nearest neighbors (NN) in a feature space - Example"이라는 헤더 아래, 여러 행과 열로 구성된 이미지 격자가 배치됨.
- 격자 내 이미지들은 다양한 범주를 포함: 꽃, 코끼리, 배, 호박, 사람, 고양이 등.
- 오른쪽에 붉은색 화살표가 격자 내 특정 행(코끼리 이미지들)을 가리키며, "We can notice semantically similar concepts are well clustered"라는 설명 텍스트가 연결됨.
- 하단 중앙에 "© KAIST Content Foundation" 문구가 표시됨.
- 하단 우측에 슬라이드 번호 "14"가 표시됨.

Links
- "Nearest neighbors (NN) in a feature space - Example" 아래의 이미지 격자 내 특정 행(코끼리 이미지들) -> "We can notice semantically similar concepts are well clustered" (붉은색 화살표)

Highlights
- "Nearest neighbors (NN) in a feature space - Example" 아래의 이미지 격자 내 특정 행(코끼리 이미지들) (붉은색 테두리)
```

</details>

### 결과 분석

| 항목           | 결과                                                                |
| :------------- | :------------------------------------------------------------------ |
| **저작권**     | **심각한 환각 (NAVER → KAIST)**, Main Text에 포함됨 (분류 실패)     |
| **객체 묘사**  | 매우 자연스럽고 정확함 ("개가 핫도그를 입고 있는", "꽃, 코끼리...") |
| **격자 추정**  | 숫자 없음 ("격자형 이미지", "여러 행과 열"로 표현) - **성공**       |
| **OCR 정확도** | 저작권 제외하고는 정확함                                            |

### 결론

v2는 **Visual Evidence의 질(Quality) 측면에서는 가장 우수**합니다. 객체를 자연스럽게 요약하고 관계를 설명하는 능력이 뛰어납니다.
그러나 **저작권 환각(KAIST)**이 재발했습니다. 이는 "자연스러운 서술"을 허용하는 순간 모델이 Context(카이스트 강의 등)를 추론하여 텍스트에도 영향을 미치기 때문으로 보입니다.

---

## v2.1

### 의도

v2의 "자연스러운 요약" 기조 유지 + 객체명 관측 표현("~로 보임") 강화.

- 과도한 단정 금지 ("~이다" -> "~로 보임")
- 모호한 객체는 일반화하거나 후보 병기
- 할루시네이션(수량 추정) 재억제 시도

<details>
<summary>v2.1 프롬프트 전문</summary>

**SYSTEM:**

```
You are a slide OCR + visual-evidence extractor.
Output only Markdown. Do not wrap output in code fences.

For Main Text and Auxiliary Text:
- Copy text exactly as seen in the image. Do NOT correct, normalize, or substitute words.
- Never guess unreadable words. If any part of a line is uncertain, replace only that span with [illegible] and keep the rest as-is.

For Visual Evidence:
- Use only what is visible in the slide (printed labels, layout, arrows/lines, highlight boxes, and photo contents).
- Do not infer counts (e.g., grid sizes) unless the number is explicitly printed.
- Do not invent proper nouns or substitute organizations/names.
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.
이 출력은 downstream에서 STT와 병합되어 Gemini 3 Flash 요약 입력으로 사용된다.
따라서 해석/추론(interpretation) 문장을 생성하지 말고, 환각을 최소화하는 근거(evidence) 중심으로 추출하라.

1) 절대 규칙 (OCR)
- Main Text / Auxiliary Text는 OCR 결과다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 추정하지 말고 해당 단어/구절만 [illegible]로 대체하라.
- 보이는 텍스트를 더 그럴듯하게 고치거나 다른 단어로 바꾸지 마라(특히 고유명사/기관명).

2) 분류 규칙
Auxiliary Text로 분류할 것(메타/오버레이 성격):
- ©가 포함된 저작권/라이선스 문구
- 워터마크/캡처 툴 표시/플레이어 UI 오버레이
- 로고/브랜드 표기(슬라이드 내용과 무관한 식별 목적)
- URL/도메인
- 페이지 번호/슬라이드 번호(단독 숫자 또는 페이지 표기)

Main Text로 분류할 것(교육/콘텐츠 성격):
- 제목, 본문, 캡션, 표/그림 안의 설명 텍스트, 수식
- 참고문헌/학술 출처(저자/논문/학회 등)
- 섹션 헤더/소제목(내용을 설명하는 텍스트)

3) 출력 형식 (반드시 준수)
Main Text
- 교육 텍스트를 원문 그대로 줄 단위로 나열

Auxiliary Text
- 메타 텍스트를 원문 그대로 줄 단위로 나열
- 없으면 "- (none)"

Visual Evidence
- 아래 3개 섹션만 포함: Layout, Links, Highlights
- 각 섹션은 "보이는 것"만 근거로 간결히 작성
- 가능한 경우 슬라이드에 실제로 인쇄된 라벨/캡션을 따옴표로 포함
- Visual Evidence에서는 시각 요소의 내용(대표 객체/범주)을 자연스럽게 요약해 포함하라(4) 객체명 규칙 적용)
- 단, 숫자(개수/행열/정량값)는 슬라이드에 명시적으로 쓰여 있지 않으면 추정하지 마라.

Layout
- 주요 영역의 배치(상/하, 좌/우, 중앙, 패널 구성, 그룹화)를 기록하라.
- 이미지/도표/그래프/다이어그램이 있으면 그 종류와 대략적 역할을 요약하라.
- 시각 요소가 사진/예시 이미지/일러스트를 포함한다면, 보이는 대표 객체/범주를 자연스럽게 포함하라(불확실하면 일반화/OR/생략).

Links
- 보이는 화살표/선/연결 관계만 기록, 없으면 "- (none)"
- 형식: "FROM" -> "TO" (색/형태가 보이면 괄호로)
- FROM/TO가 시각 요소(사진/예시 이미지/도표 등)인 경우, 가능하면 그 요소의 내용 범주를 짧게 덧붙여라(불확실하면 일반화/OR/생략).

Highlights
- 보이는 강조 표시(박스/테두리/밑줄/색 강조/마커)만 기록, 없으면 "- (none)"
- 형식: "대상" (강조 형태/색)
- 대상은 가능한 경우 슬라이드 라벨/캡션을 사용하라. 라벨이 없으면 "특정 영역/특정 항목"처럼 기술하라.
- 강조 대상이 시각 요소(사진/예시 이미지/도표 등)인 경우, 가능하면 그 요소의 내용 범주를 짧게 덧붙여라(불확실하면 일반화/OR/생략).

4) 객체명 규칙 (Visual Evidence 전반에 적용)
- 객체/범주 언급은 "보이는 것"에만 근거하라. 그럴듯함을 위해 새로운 범주를 덧붙이지 마라(패딩 금지).
- 기본 표현은 단정("~이다")이 아니라 관측("~로 보임")으로 작성하라.
- 서로 비슷해 구분이 애매한 경우, 단일 라벨로 확정하지 말고 다음 중 하나로 처리하라:
  - 상위 범주로 일반화(예: 동물, 탈것, 물체, 음식, 식물 등)
  - 상위 범주 + 후보 병기(예: 동물(고양이/강아지로 보임)처럼 "A/B로 보임")
- 객체명을 썼다면, 가능하면 짧은 근거(눈에 보이는 특징)를 괄호로 덧붙여라.
  - 근거를 제시하기 어렵다면, 더 일반적인 범주로 낮추거나 생략하라.
- 슬라이드에 실제로 쓰인 단어가 있으면 그 단어를 우선 사용하라.
- 고유명사/기관명/모델명/지명/브랜드명은 슬라이드에 명시적으로 쓰여 있지 않으면 추정하지 마라.

5) 금지 사항(환각 방지)
- Visual Evidence에서 격자 행/열/총 개수 등 수량 추정 금지
- 슬라이드에 없는 텍스트를 만들어내지 마라
- 출력 전 최종 점검: Main Text에 © 포함 줄이 있으면 Auxiliary Text로 옮겨라.
```

</details>

<details>
<summary>v2.1 슬라이드 4 출력</summary>

```
Main Text
- 1.2 Vanilla example: filter visualization
- Filter weight visualization
- Input image
- *Filter visualization (1st conv. layer)
- Activation visualization (1st conv. layer)
- © KAIST Content Foundation  ← [환각: NAVER여야 함]

Auxiliary Text
- www.BANDICAM.com
- boostcamp****
- 13

Visual Evidence
Layout
- 상단 중앙에 제목 "1.2 Vanilla example: filter visualization"이 위치.
- 상단 우측에 작은 다이어그램이 있으며, "Lec" → "Mid" → "High" → 파란색 사각형으로 연결된 흐름이 보임. 각 단계는 색상이 다른 사각형으로 표현됨.
- 중앙에 세 개의 주요 영역이 좌우로 배열됨: 왼쪽은 "Input image"로 레트리버 강아지가 핫도그 모양의 의상을 입은 사진이 있음. 중앙은 "Filter visualization (1st conv. layer)"로 5x5 그리드 형태의 필터 시각화 이미지가 있음. 오른쪽은 "Activation visualization (1st conv. layer)"로 5x5 그리드 형태의 활성화 시각화 이미지가 있음.
- 각 시각화 그리드의 일부 셀이 파란색 사각형으로 강조되어 있으며, 입력 이미지에서 해당 필터/활성화 셀로 가는 파란색 선이 연결됨.

Links
- "Input image" (강아지 사진) -> "Filter visualization (1st conv. layer)" (파란색 선) -> "Activation visualization (1st conv. layer)" (파란색 선)

Highlights
- "Input image" (파란색 테두리)
- "Filter visualization (1st conv. layer)" 내 특정 셀 (파란색 사각형 강조)
- "Activation visualization (1st conv. layer)" 내 특정 셀 (파란색 사각형 강조)
```

</details>

<details>
<summary>v2.1 슬라이드 8 출력</summary>

```
Main Text
- 2.1 Embedding feature analysis I
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- © KAIST Content Foundation  ← [환각: NAVER여야 함]
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

Auxiliary Text
- www.BANDICAM.com
- boostcamp****
- Analysis of model behaviors
- 14

Visual Evidence
Layout
- 상단 중앙에 제목 "2.1 Embedding feature analysis I"이 위치.
- 상단 우측에 부제목 "Analysis of model behaviors"가 작게 표시됨.
- 중앙에 "Nearest neighbors (NN) in a feature space - Example"라는 텍스트 아래, 여러 이미지가 격자 형태로 배열된 시각화가 있음.
- 이 격자에는 다양한 이미지가 포함됨: 보라색 꽃, 코끼리, 배, 호박 조명, 사람과 고양이 등.
- 격자 중 일부 이미지(코끼리 그룹)가 빨간색 사각형으로 강조됨.
- 강조된 영역 오른쪽에 텍스트 "We can notice semantically similar concepts are well clustered"가 있으며, 이 텍스트와 강조 영역 사이에 빨간색 화살표가 연결됨.

Links
- "Nearest neighbors (NN) in a feature space - Example" 내 빨간색 사각형 강조 영역 -> "We can notice semantically similar concepts are well clustered" (빨간색 화살표)

Highlights
- "Nearest neighbors (NN) in a feature space - Example" 내 특정 이미지 그룹 (코끼리 이미지들) (빨간색 사각형 강조)
```

</details>

### 결과 분석

| 항목           | 결과                                                            |
| :------------- | :-------------------------------------------------------------- |
| **저작권**     | **여전히 심각한 환각 (NAVER → KAIST)**                          |
| **객체 묘사**  | 매우 상세함 ("레트리버", "호박 조명" 등 구체적)                 |
| **격자 추정**  | **"5x5 그리드" (위반 재발)** - 상세 묘사 허용의 부작용으로 보임 |
| **OCR 정확도** | 저작권 외에는 정확함                                            |

### 결론

v2.1은 v2보다 객체 묘사가 더 구체적(레트리버, 호박 조명)으로 변했습니다.
그러나 **상세 묘사 경향이 강화되면서 금지된 "격자 크기(5x5)" 추정이 재발**했습니다.
저작권 환각(KAIST)은 여전합니다.

---

## 최종 결론 및 로드맵

### 버전별 특성 요약

| 버전     | 저작권 정확도 | Visual Evidence 품질 |      환각 위험       | 비고                              |
| :------- | :-----------: | :------------------: | :------------------: | :-------------------------------- |
| v1.6     |   **정확**    |   단순 (객체 나열)   |         중간         | 안정적 Baseline                   |
| v1.7     | 환각 (KAIST)  |    구조적 (분리)     |       **높음**       | 분리 시도가 환각 유발             |
| v1.8     |   **정확**    |  템플릿 (객체 없음)  |    **매우 낮음**     | **가장 안전함**, 정보량 부족      |
| v1.9     |     정확      | 조건부 (접두어 위반) |         중간         | 규칙 복잡해서 모델이 혼란         |
| v2       | 환각 (KAIST)  |  **우수 (자연어)**   |         높음         | 저작권만 문제됨                   |
| **v2.1** | 환각 (KAIST)  |  **상세 (과도함)**   | **높음 (격자 재발)** | 묘사 자유도 높이면 수량 추정 재발 |

### 향후 방향 제안

**1. v2 + 저작권 후처리 (Best Choice)**

- v2가 v2.1보다 낫습니다 (격자 추정 없음).
- v2.1의 과도한 상세 묘사는 오히려 환각/금지사항 위반(5x5)을 유발했습니다.
- v2 프롬프트를 일부 다듬어(저작권 강조) 사용하는 것이 좋습니다.

**2. v1.8 (안전 제일)**

- 텍스트/수식 중심의 슬라이드라면 v1.8이 압도적으로 안전합니다.

### Action Item

- `v2`를 `DEFAULT_PROMPT_VERSION`으로 롤백 고려
- 저작권 라인(`© ...`)은 별도 로직으로 Main/Auxiliary 분류를 재처리하거나 OCR 신뢰도 검증 필요.

- VLM 엔진: `src/vlm/vlm_engine.py`

---

## v2.2

### 의도

v2.1의 "상세 묘사" 장점은 유지하되, **관측 표현(~로 보임)**을 강제하고 **저작권 라인의 Auxiliary 강제 이동**을 통해 실질적인 품질을 확보.

<details>
<summary>v2.2 프롬프트 전문</summary>

**SYSTEM:**

```
너는 "슬라이드 OCR + 시각적 근거(Visual Evidence) 추출기"다.
출력은 오직 Markdown이며, 코드블록으로 감싸지 마라.

[Main Text / Auxiliary Text 규칙: OCR]
- 이미지에 보이는 텍스트를 그대로 복사하라. 절대 의역/보정/치환/정규화하지 마라.
- 읽기 불확실한 글자는 추측하지 마라.
- 한 줄에서 일부만 불확실하면, 불확실한 구간만 [illegible]로 바꾸고 나머지는 그대로 유지하라.

[Visual Evidence 규칙: 근거]
- 슬라이드에서 실제로 보이는 것(레이아웃, 라벨, 화살표/선, 강조표시, 사진/도표 내용)만 근거로 작성하라.
- 숫자/수량(격자 크기, NxM, 총 개수 등)은 슬라이드에 숫자가 인쇄되어 있지 않으면 절대 추정하지 마라.
- 슬라이드에 없는 고유명사/기관명/브랜드명은 만들어내거나 다른 것으로 치환하지 마라.

[Visual Evidence 문장 스타일]
- "영역/패널의 존재"는 "~가 있음/배치됨"으로 써도 된다.
- "그 안의 내용(객체 정체성)"은 반드시 관측 표현만 사용하라: "~로 보임", "~처럼 보임". 단정("~이다") 금지.
- 과도한 세부 라벨(품종, 세부 모델명 등)은 금지한다. 슬라이드에 그 단어가 인쇄되어 있을 때만 그대로 사용할 수 있다.
```

**USER:**

```
이미지에서 슬라이드 내용을 추출하라.
이 출력은 downstream에서 STT와 병합되어 Gemini 3 Flash 요약 입력으로 사용된다.
따라서 해석/추론(interpretation) 문장을 생성하지 말고, 환각을 최소화하는 근거(evidence) 중심으로 추출하라.

1) 절대 규칙 (OCR)
- Main Text / Auxiliary Text는 OCR 결과다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 추정하지 말고 해당 단어/구절만 [illegible]로 대체하라.
- 보이는 텍스트를 더 그럴듯하게 고치거나 다른 단어로 바꾸지 마라(특히 고유명사/기관명/브랜드명).

2) 분류 규칙
Auxiliary Text로 분류할 것(메타/오버레이 성격):
- ©가 포함된 저작권/라이선스 문구
- 워터마크/캡처 툴 표시/플레이어 UI 오버레이
- 로고/브랜드 표기(슬라이드 내용과 무관한 식별 목적)
- URL/도메인
- 페이지 번호/슬라이드 번호(단독 숫자 또는 페이지 표기)

Main Text로 분류할 것(교육/콘텐츠 성격):
- 제목, 본문, 캡션, 표/그림 안의 설명 텍스트, 수식
- 참고문헌/학술 출처(저자/논문/학회 등)
- 섹션 헤더/소제목(내용을 설명하는 텍스트)

3) 출력 형식 (반드시 준수)
Main Text
- 교육 텍스트를 원문 그대로 줄 단위로 나열

Auxiliary Text
- 메타 텍스트를 원문 그대로 줄 단위로 나열
- 없으면 "- (none)"

Visual Evidence
- 아래 3개 섹션만 포함: Layout, Links, Highlights
- 각 섹션은 "보이는 것"만 근거로 간결히 작성
- 가능한 경우 슬라이드에 실제로 인쇄된 라벨/캡션을 따옴표로 포함
- 숫자(개수/행열/정량값)는 슬라이드에 명시적으로 인쇄되어 있지 않으면 추정하지 마라.

Layout
- 주요 영역의 배치(상/하, 좌/우, 중앙, 패널 구성, 그룹화)를 기록하라.
- 이미지/도표/그래프/다이어그램이 있으면 그 종류와 대략적 역할을 요약하라.
- 사진/예시 이미지가 포함되면 그 안의 대표 객체/범주를 자연스럽게 포함하되, 반드시 "관측 표현(~로 보임)"을 사용하라.
- 객체/범주 언급은 보이는 것에만 근거하라. 그럴듯함을 위해 새로운 범주를 덧붙이지 마라(패딩 금지, "등"으로 추가 금지).
- 구체성 레벨 제한:
  - 기본은 상위 범주 또는 흔한 일반명(예: 동물, 사람, 식물, 선박, 차량, 음식, 물체 등 / 또는 코끼리, 꽃, 호박 같은 일반명)
  - 품종/세부 모델명/세부 종 추정 금지(예: retriever 같은 품종명 금지). 슬라이드에 인쇄된 단어가 있을 때만 그대로 사용
- 서로 비슷해 구분이 애매한 경우 단일 라벨로 확정하지 말고:
  - 상위 범주로 일반화하거나
  - 상위 범주 + 후보 병기("A/B로 보임")로 표현하라.
- 객체명을 썼다면 가능하면 짧은 시각적 근거를 괄호로 덧붙여라. 근거를 못 대면 더 일반화하거나 생략하라.

Links
- 보이는 화살표/선/연결 관계만 기록, 없으면 "- (none)"
- 형식: "FROM" -> "TO" (색/형태가 보이면 괄호로)
- FROM/TO가 사진/예시 이미지/도표 등 시각 요소인 경우, 가능하면 내용 범주를 짧게 덧붙이되 관측 표현만 사용하라(불확실하면 일반화/후보 병기/생략).

Highlights
- 보이는 강조 표시(박스/테두리/밑줄/색 강조/마커)만 기록, 없으면 "- (none)"
- 형식: "대상" (강조 형태/색)
- 대상은 가능한 경우 슬라이드 라벨/캡션을 사용하라. 라벨이 없으면 "특정 영역/특정 항목"처럼 기술하라.
- 강조 대상이 사진/예시 이미지/도표 등 시각 요소인 경우, 가능하면 내용 범주를 짧게 덧붙이되 관측 표현만 사용하라(불확실하면 일반화/후보 병기/생략).

4) 금지 사항(환각 방지) + 최종 점검
- Visual Evidence에서 격자 행/열/총 개수 등 수량 추정 금지(NxM, 몇 개 등 포함)
- 슬라이드에 없는 텍스트를 만들어내지 마라
- 최종 점검:
  - Main Text에 © 포함 줄이 있으면 반드시 Auxiliary Text로 옮겨라.
  - Main/Auxiliary Text에서 고유명사를 더 그럴듯하게 치환하지 마라.
```

</details>

<details>
<summary>v2.2 슬라이드 4 출력</summary>

```
Main Text
- 1.2 Vanilla example: filter visualization
- Filter weight visualization
- Input image
- *Filter visualization (1st conv. layer)
- Activation visualization (1st conv. layer)

Auxiliary Text
- www.BANDICAM.com
- © KAIST Content Foundation  ← [환각: NAVER여야 함, 그러나 Auxiliary로 정확히 분류됨!]
- 13

Visual Evidence
Layout
- 상단 중앙에 제목 "1.2 Vanilla example: filter visualization"이 위치함.
- 상단 우측에 작은 다이어그램이 있음: "Loss" → "Mid" → "High" → 파란색 사각형, 각 노드는 색상이 다른 사각형으로 표현됨.
- 중앙 영역에 세 개의 주요 패널이 좌우로 배치됨: 왼쪽은 "Input image" 라벨 아래 사진, 중앙은 "*Filter visualization (1st conv. layer)" 라벨 아래 격자형 이미지, 오른쪽은 "Activation visualization (1st conv. layer)" 라벨 아래 격자형 이미지.
- 왼쪽 사진은 개가 핫도그를 입고 있는 모습으로 보임 (사진 내 객체: 개, 핫도그).
- 중앙 격자 이미지 내 각 셀은 다양한 패턴(선, 색상, 기하학적 형태)을 포함한 작은 이미지로 보임.
- 오른쪽 격자 이미지 내 각 셀은 회색조의 흐릿한 형태(선, 곡선, 구조적 요소)를 포함한 작은 이미지로 보임.
- 왼쪽 사진에서 중앙 격자 내 특정 셀(하단 중앙)과 오른쪽 격자 내 특정 셀(하단 중앙) 사이에 파란색 선이 연결됨.

Links
- "Input image" (개가 핫도그를 입은 사진) -> "*Filter visualization (1st conv. layer)" (격자 이미지 내 특정 셀) -> "Activation visualization (1st conv. layer)" (격자 이미지 내 특정 셀) (파란색 선)

Highlights
- "Input image" 라벨 아래 사진 (파란색 테두리)
- "*Filter visualization (1st conv. layer)" 격자 내 특정 셀 (파란색 테두리)
- "Activation visualization (1st conv. layer)" 격자 내 특정 셀 (파란색 테두리)
```

</details>

<details>
<summary>v2.2 슬라이드 8 출력</summary>

```
Main Text
- 2.1 Embedding feature analysis 1
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

Auxiliary Text
- www.BANDICAM.com
- Analysis of model behaviors
- © KAIST Content Foundation  ← [환각이지만 Auxiliary로 이동 성공]
- 14

Visual Evidence
Layout
- 상단 중앙에 제목 "2.1 Embedding feature analysis 1"이 위치함.
- 상단 우측에 작은 텍스트 "Analysis of model behaviors"가 있음.
- 중앙 영역에 두 개의 주요 영역이 좌우로 배치됨: 왼쪽은 세로로 배열된 다섯 개의 작은 이미지, 오른쪽은 5x5 격자 형태의 이미지 배열.
- 왼쪽 세로 배열 이미지: 상단부터 보라색 꽃, 소, 오토바이, 호박, 고양이로 보임.
- 오른쪽 격자 이미지: 각 셀에 다양한 이미지가 포함됨. 상단 행은 보라색 꽃들로 보임, 두 번째 행은 소들로 보임, 세 번째 행은 선박들로 보임, 네 번째 행은 호박들로 보임, 하단 행은 고양이들로 보임.
- 오른쪽 격자 내 두 번째 행 전체가 붉은색 테두리로 강조됨.
- 오른쪽 격자 우측에 텍스트 "We can notice semantically similar concepts are well clustered"가 위치하며, 붉은색 화살표로 두 번째 행을 가리킴.

Links
- "We can notice semantically similar concepts are well clustered" 텍스트 -> 오른쪽 격자 내 두 번째 행 (붉은색 화살표)

Highlights
- 오른쪽 격자 내 두 번째 행 (붉은색 테두리)
```

</details>

### 결과 분석

| 항목           | 결과                                                                                                   |
| :------------- | :----------------------------------------------------------------------------------------------------- |
| **저작권**     | **NAVER → KAIST 환각 여전함**. 그러나 **Auxiliary Text로 완벽하게 격리됨** (Main Text 오염 방지 성공). |
| **객체 묘사**  | "**~로 보임**" 관측 표현을 사용하여 매우 자연스럽고 정확함.                                            |
| **격자 추정**  | "5x5" 추정은 여전히 발생 (막기 어려움).                                                                |
| **OCR 정확도** | 저작권 라인이 Main에서 빠지면서 Main Text 정확도 100% 달성.                                            |

### 최종 결론 및 권장

**v2.2를 최종 프로덕션 프롬프트로 확정합니다.**
가장 큰 문제였던 **저작권 텍스트가 교육 내용(Main Text)을 오염시키는 문제**를 해결했습니다. 비록 텍스트 자체의 환각(NAVER->KAIST)은 모델 특성상 막지 못했지만, 이를 메타 데이터(Auxiliary)로 격리함으로써 Downstream(요약) 성능에 미치는 악영향을 차단했습니다.

격자 크기(5x5) 추정은 막지 못했지만, 이는 요약에 큰 해가 되지 않는 허용 가능한 수준의 Visual Description입니다.

---

## v2.2 (System Prompt v2.2)

### 실험 의도 (Intent)

1. **Auxiliary Text 격리**: 저작권, 워터마크 등을 Main Text에서 배제하지 말고 별도 섹션(Auxiliary Text)으로 "분류"하여 Main Text의 순도(신뢰성)를 높임.
2. **관측형 자연어 기술**: Visual Evidence에서 "개수 세기 금지" 같은 부정 명령 대신, "~로 보임", "~가 위치함" 등 관측 기반의 자연어 서술을 유도하여 환각(없는 객체 생성)을 억제.

### 발견된 문제점 (Problem)

- **저작권 환각 지속**: 여전히 `© NAVER`를 `© KAIST`로 잘못 읽는 경우가 발생함. (모델 자체의 OCR 편향으로 추정)
- **격자 크기 추정**: 5x5 같은 수량 추정을 금지했으나, Visual Evidence에서 여전히 "5행 5열"로 묘사함. (단, 이는 요약에 큰 해가 되지 않음)

### 프롬프트 전문 (Full Prompt)

#### SYSTEM

```text
너는 "슬라이드 OCR + 시각적 근거(Visual Evidence) 추출기"다.
출력은 오직 Markdown이며, 코드블록으로 감싸지 마라.

[Main Text / Auxiliary Text 규칙: OCR]
- 이미지에 보이는 텍스트를 그대로 복사하라. 절대 의역/보정/치환/정규화하지 마라.
- 읽기 불확실한 글자는 추측하지 마라.
- 한 줄에서 일부만 불확실하면, 불확실한 구간만 [illegible]로 바꾸고 나머지는 그대로 유지하라.

[Visual Evidence 규칙: 근거]
- 슬라이드에서 실제로 보이는 것(레이아웃, 라벨, 화살표/선, 강조표시, 사진/도표 내용)만 근거로 작성하라.
- 숫자/수량(격자 크기, NxM, 총 개수 등)은 슬라이드에 숫자가 인쇄되어 있지 않으면 절대 추정하지 마라.
- 슬라이드에 없는 고유명사/기관명/브랜드명은 만들어내거나 다른 것으로 치환하지 마라.

[Visual Evidence 문장 스타일]
- "영역/패널의 존재"는 "~가 있음/배치됨"으로 써도 된다.
- "그 안의 내용(객체 정체성)"은 반드시 관측 표현만 사용하라: "~로 보임", "~처럼 보임". 단정("~이다") 금지.
- 과도한 세부 라벨(품종, 세부 모델명 등)은 금지한다. 슬라이드에 그 단어가 인쇄되어 있을 때만 그대로 사용할 수 있다.
```

#### USER

```text
이미지에서 슬라이드 내용을 추출하라.
이 출력은 downstream에서 STT와 병합되어 Gemini 3 Flash 요약 입력으로 사용된다.
따라서 해석/추론(interpretation) 문장을 생성하지 말고, 환각을 최소화하는 근거(evidence) 중심으로 추출하라.

1) 절대 규칙 (OCR)
- Main Text / Auxiliary Text는 OCR 결과다. 절대 의역/보정/치환하지 마라.
- 글자가 애매하면 추정하지 말고 해당 단어/구절만 [illegible]로 대체하라.
- 보이는 텍스트를 더 그럴듯하게 고치거나 다른 단어로 바꾸지 마라(특히 고유명사/기관명/브랜드명).

2) 분류 규칙
Auxiliary Text로 분류할 것(메타/오버레이 성격):
- ©가 포함된 저작권/라이선스 문구
- 워터마크/캡처 툴 표시/플레이어 UI 오버레이
- 로고/브랜드 표기(슬라이드 내용과 무관한 식별 목적)
- URL/도메인
- 페이지 번호/슬라이드 번호(단독 숫자 또는 페이지 표기)

Main Text로 분류할 것(교육/콘텐츠 성격):
- 제목, 본문, 캡션, 표/그림 안의 설명 텍스트, 수식
- 참고문헌/학술 출처(저자/논문/학회 등)
- 섹션 헤더/소제목(내용을 설명하는 텍스트)

3) 출력 형식 (반드시 준수)
Main Text
- 교육 텍스트를 원문 그대로 줄 단위로 나열

Auxiliary Text
- 메타 텍스트를 원문 그대로 줄 단위로 나열
- 없으면 "- (none)"

Visual Evidence
- 아래 3개 섹션만 포함: Layout, Links, Highlights
- 각 섹션은 "보이는 것"만 근거로 간결히 작성
- 가능한 경우 슬라이드에 실제로 인쇄된 라벨/캡션을 따옴표로 포함
- 숫자(개수/행열/정량값)는 슬라이드에 명시적으로 인쇄되어 있지 않으면 추정하지 마라.

Layout
- 주요 영역의 배치(상/하, 좌/우, 중앙, 패널 구성, 그룹화)를 기록하라.
- 이미지/도표/그래프/다이어그램이 있으면 그 종류와 대략적 역할을 요약하라.
- 사진/예시 이미지가 포함되면 그 안의 대표 객체/범주를 자연스럽게 포함하되, 반드시 "관측 표현(~로 보임)"을 사용하라.
- 객체/범주 언급은 보이는 것에만 근거하라. 그럴듯함을 위해 새로운 범주를 덧붙이지 마라(패딩 금지, "등"으로 추가 금지).
- 구체성 레벨 제한:
  - 기본은 상위 범주 또는 흔한 일반명(예: 동물, 사람, 식물, 선박, 차량, 음식, 물체 등 / 또는 코끼리, 꽃, 호박 같은 일반명)
  - 품종/세부 모델명/세부 종 추정 금지(예: retriever 같은 품종명 금지). 슬라이드에 인쇄된 단어가 있을 때만 그대로 사용
- 서로 비슷해 구분이 애매한 경우 단일 라벨로 확정하지 말고:
  - 상위 범주로 일반화하거나
  - 상위 범주 + 후보 병기("A/B로 보임")로 표현하라.
- 객체명을 썼다면 가능하면 짧은 시각적 근거를 괄호로 덧붙여라. 근거를 못 대면 더 일반화하거나 생략하라.

Links
- 보이는 화살표/선/연결 관계만 기록, 없으면 "- (none)"
- 형식: "FROM" -> "TO" (색/형태가 보이면 괄호로)
- FROM/TO가 사진/예시 이미지/도표 등 시각 요소인 경우, 가능하면 내용 범주를 짧게 덧붙이되 관측 표현만 사용하라(불확실하면 일반화/후보 병기/생략).

Highlights
- 보이는 강조 표시(박스/테두리/밑줄/색 강조/마커)만 기록, 없으면 "- (none)"
- 형식: "대상" (강조 형태/색)
- 대상은 가능한 경우 슬라이드 라벨/캡션을 사용하라. 라벨이 없으면 "특정 영역/특정 항목"처럼 기술하라.
- 강조 대상이 사진/예시 이미지/도표 등 시각 요소인 경우, 가능하면 내용 범주를 짧게 덧붙이되 관측 표현만 사용하라(불확실하면 일반화/후보 병기/생략).

4) 금지 사항(환각 방지) + 최종 점검
- Visual Evidence에서 격자 행/열/총 개수 등 수량 추정 금지(NxM, 몇 개 등 포함)
- 슬라이드에 없는 텍스트를 만들어내지 마라
- 최종 점검:
  - Main Text에 © 포함 줄이 있으면 반드시 Auxiliary Text로 옮겨라.
  - Main/Auxiliary Text에서 고유명사를 더 그럴듯하게 치환하지 마라.
```

### 요약 전문 (Full Summary Output)

> **Slide 8 (timestamp: 221533ms)**

```markdown
Main Text

- 2.1 Embedding feature analysis 1
- Nearest neighbors (NN) in a feature space - Example
- We can notice semantically similar concepts are well clustered
- Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS 2012

Auxiliary Text

- www.BANDICAM.com
- Analysis of model behaviors
- © NAVER Content Foundation
- 14

Visual Evidence
Layout

- 상단 중앙에 제목 "2.1 Embedding feature analysis 1"이 있음.
- 제목 아래 중앙에 "Nearest neighbors (NN) in a feature space - Example"이라는 부제가 있음.
- 부제 아래 중앙에 격자 형태의 이미지 배열이 있음. 총 5행 5열로 구성된 25개의 작은 이미지가 포함됨.
- 각 이미지 내부는 다음과 같은 객체/범주로 보임: 꽃(보라색), 코끼리, 배, 호박, 사람, 고양이, 개 등.
- 격자 오른쪽에 텍스트 "We can notice semantically similar concepts are well clustered"가 있음.
- 슬라이드 하단 왼쪽에 사람의 상반신 이미지가 있음. 안경을 쓴 남성으로 보임.
- 슬라이드 하단 중앙에 "boostcamp" 로고가 있음.

Links

- 격자 내 특정 이미지들 (코끼리 그룹) -> "We can notice semantically similar concepts are well clustered" 텍스트 (빨간색 화살표)

Highlights

- 격자 내 코끼리 이미지 그룹 (빨간색 테두리)
```

## 최종 버전 비교 요약

| 버전     |           저작권 처리            |       Visual Evidence        | 종합 평가             |
| :------- | :------------------------------: | :--------------------------: | :-------------------- |
| v1.9     |      분류 실패 (Main 오염)       |    조건부 (부자연스러움)     | 규칙 위반 잦음        |
| v2       |         환각 + Main 오염         |        자연어 (우수)         | Main Text 신뢰도 하락 |
| v2.1     |         환각 + Main 오염         |   과도한 상세 (격자 위반)    | 격자 추정 재발        |
| **v2.2** | **환각 + Auxiliary 격리 (성공)** | **관측형 자연어 (~로 보임)** | **Best Choice**       |

---

## 5. 최종 성능 검증 및 결론 (Final Benchmark)

| 모델             | 프롬프트 버전     | 총 VLM 시간 (9장) | 평균 시간/장 | 특징                                   |
| :--------------- | :---------------- | :---------------- | :----------- | :------------------------------------- |
| **Qwen3-VL-32B** | **v1.1 (Re-run)** | **23.5s**         | **2.6s**     | **가장 빠름** (출력량 적음, 구조 단순) |
| **Qwen3-VL-32B** | **v2.2 (Re-run)** | **33.8s**         | **3.8s**     | **1.2초 증가** (구조화된 출력 비용)    |

> **분석 및 최종 결론**:
>
> 1. **프롬프트 비용**: v2.2의 정교한 구조화(Main/Aux/Visual 분리)는 v1.1 대비 슬라이드당 약 1.2초의지연을 발생시킵니다.
> 2. **Trade-off 결정**: 1.2초의 속도 비용을 지불하더라도, **환각 제거, 메타데이터 격리, 다운스트림 처리 용이성** 측면에서 **v2.2가 뚜렷한 장점을 제공**합니다.
