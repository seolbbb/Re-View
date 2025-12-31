# Final Summary B (주제별 재구성)
## functional
- (00:37–01:48) Evidence Lower Bound (ELBO) 또는 log marginal likelihood (로그 마지널 라이클리후드)의 하한을 최대화하는 것이 목표이다.
- (00:37–01:48) Functional derivative (변분 미분)는 함수를 입력으로 받아 실수를 출력하는 functional의 변화를 다룬다.
- (03:04–04:17) MFVI를 유도하는 세 번째 방법으로 functional derivative를 상세히 살펴본다.
- (03:04–04:17) Functional (펑셔널)은 함수 f(x)를 입력으로 받아 실수를 내뱉는 '함수에 대한 함수'이다.
- (03:04–04:17) Functional derivative는 VI뿐만 아니라 다양한 분야에서 광범위하게 사용되는 도구이다.
- (03:04–04:17) 결국 우리가 해결해야 할 문제는 functional의 최대치를 구하는 상황으로 귀결된다.
- (04:17–05:37) Functional derivative는 입력 함수가 변할 때 functional의 출력값이 어떻게 변하는지 그 변화량을 추정한다.
- (04:17–05:37) Functional derivative는 variational derivative (변분 미분)라고도 불린다.
- (00:37–01:48) Functional derivative (변분 미분): 입력이 함수인 functional에서, 해당 함수가 변할 때 functional이 어떻게 변하는지 측정하는 것이다.
- (03:04–04:17) Mean-field assumption (민 필드 가정): 복잡한 joint 분포를 개별적인 q들의 곱으로 분리하여 각각을 최대화할 수 있다고 가정하는 사항이다.
- (04:17–05:37) Functional (펑셔널): 함수를 입력으로 받아 실수를 출력하는 함수로, ELBO가 이에 해당한다.
- (00:00–00:37) EM 알고리즘과 VI 사이의 공통점과 차이점 및 두 개념 사이의 깊은 관련성을 분석한다.
- (00:37–01:48) MFVI를 유도하기 위해 log p(x, z)의 기댓값 등을 활용하는 트릭을 사용한다.
- (00:37–01:48) 이상적인 q는 posterior (사후 확률)이지만, 이를 정확히 구할 수 없는 경우가 많다.
- (01:48–03:04) 변수가 많아질수록 q의 구조가 매우 복잡해질 수 있다.
- (01:48–03:04) 서로 다른 분포(Gaussian, Dirichlet, Multinomial)를 따르는 변수들의 joint (조인트) 분포를 정의하기 어렵다.
- (01:48–03:04) ELBO 수식 내의 기댓값 E_{i != j}는 j를 제외한 나머지 변수들에 대한 평균을 의미한다.
- (03:04–04:17) 일반적인 calculus (미적분)는 실수를 입력받아 실수를 출력하는 함수를 다룬다.
- 근거 segment 범위: 1–5
## 입력으로
- (00:37–01:48) 전통적인 미적분은 실수를 입력으로 받지만, 변분법은 함수 자체를 입력으로 취급한다.
- (03:04–04:17) Variational calculus (변분법)는 일반적인 미적분과 달리 함수를 입력으로 받는다.
- (04:17–05:37) f(x)를 입력으로 받는 f는 '함수에 대한 함수'인 functional로 정의된다.
- (04:17–05:37) ELBO는 확률 분포 함수 q_j를 입력으로 받아 값을 결정하므로 functional의 대표적인 예시이다.
- 근거 segment 범위: 2–5
## 함수
- (04:17–05:37) 변분법에서는 미분 방정식을 풀어 functional을 정지시키는 stationary function (정지 함수) f(x)를 찾는다.
- (05:37–06:05) 함수 f(x)에서 최대가 되는 x를 찾기 위해 미분값 f'(x)가 0이 되는 지점을 활용한다.
- (04:17–05:37) stationary function (정지 함수): Functional의 변화율이 0이 되게 하는 함수로, 변분법의 미분 방정식을 통해 구한다.
- 근거 segment 범위: 5–6
## q를
- (00:37–01:48) ELBO를 최대화하기 위한 최적의 q를 찾는 과정이 MFVI의 핵심이다.
- (00:37–01:48) ELBO를 최대화함으로써 q를 posterior에 가깝게 만들 수 있다.
- (01:48–03:04) Posterior를 정확히 구할 수 없으므로 ELBO를 최대화하는 q를 찾는 것이 목적이다.
- (01:48–03:04) 복잡한 문제를 해결하기 위해 q를 q1, q2, q3와 같이 각각 쪼개서 분석한다.
- (03:04–04:17) 각각의 q를 독립적으로 최대화하려는 시도가 Mean-field assumption (민 필드 어썸션)의 핵심이다.
- (00:37–01:48) Evidence Lower Bound (ELBO): log marginal likelihood의 로어 바운드로, 이를 최대화하여 q를 posterior에 근사시킨다.
- (01:48–03:04) Latent Dirichlet Allocation (LDA): 여러 개의 latent variable을 포함하고 있어 q를 직접 정의하거나 유도하기 어려운 모델의 예시이다.
- 근거 segment 범위: 2–4
## variational
- (00:00–00:37) Variational Inference (변분 추론)의 다양한 유도 방법과 심화 내용을 학습한다.
- (00:00–00:37) Mean-field Variational Inference (MFVI)를 다르게 유도하는 방법을 살펴본다.
- (00:00–00:37) Variational Inference (VI)가 실제 환경에서 어떻게 application (어플리케이션) 되는지 확인한다.
- (00:00–00:37) Mean-field Variational Inference (MFVI): 변분 추론의 한 종류로, 이번 강의에서 개념 학습을 완료하고 새로운 유도 방법을 배울 대상이다.
- (03:04–04:17) Variational calculus (변분법): 입력이 함수인 functional을 다루는 미적분학의 한 분야이다.
- 근거 segment 범위: 1–4
## latent
- (01:48–03:04) q_j는 전체 latent variable (레이턴트 베리어블) 중 일부를 나타내는 간소화된 버전이다.
- (01:48–03:04) Latent Dirichlet Allocation (LDA)와 같이 latent variable이 여러 개인 모델은 직접적인 유도가 어렵다.
- (01:48–03:04) latent variable (잠재 변수): 데이터 이외의 관측되지 않는 변수들로, VI를 통해 이들의 분포를 추정하고자 한다.
- 근거 segment 범위: 3–3
## stationary
- (00:00–00:37) Expectation-Maximization (EM) 알고리즘의 정의와 특징을 다룬다.
- (04:17–05:37) 일반 미적분에서는 df/dx = 0을 풀어 stationary point (정지점)를 찾는다.
- (05:37–06:05) f'(x) = 0을 만족하는 지점을 stationary point라고 정의한다.
- (05:37–06:05) Stationary point를 찾은 후에는 해당 점이 극대(max)인지 극소(min)인지 판별하는 추가 과정이 필요할 수 있다.
- (00:00–00:37) Expectation-Maximization (EM): 익스펙테이션 n 맥시마이제이션으로 불리는 알고리즘으로, VI와 밀접한 관련이 있다.
- (05:37–06:05) stationary point (정지점): 함수의 도함수 f'(x) 또는 dy/dx가 0이 되는 지점을 의미한다.
- 근거 segment 범위: 1–6
## 미분
- (04:17–05:37) Functional의 최대치를 구하기 위해 이러한 미분 개념들을 준비 과정으로 학습한다.
- 근거 segment 범위: 5–5