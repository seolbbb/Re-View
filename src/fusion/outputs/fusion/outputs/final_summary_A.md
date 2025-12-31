# Final Summary A (시간 순 타임라인)
#### Segment 1 (00:00–00:37)
- Variational Inference (변분 추론)의 다양한 유도 방법과 심화 내용을 학습한다.
- Mean-field Variational Inference (MFVI)를 다르게 유도하는 방법을 살펴본다.
- Variational Inference (VI)가 실제 환경에서 어떻게 application (어플리케이션) 되는지 확인한다.
#### Segment 2 (00:37–01:48)
- MFVI를 유도하기 위해 log p(x, z)의 기댓값 등을 활용하는 트릭을 사용한다.
- Evidence Lower Bound (ELBO) 또는 log marginal likelihood (로그 마지널 라이클리후드)의 하한을 최대화하는 것이 목표이다.
- ELBO를 최대화하기 위한 최적의 q를 찾는 과정이 MFVI의 핵심이다.
#### Segment 3 (01:48–03:04)
- Posterior를 정확히 구할 수 없으므로 ELBO를 최대화하는 q를 찾는 것이 목적이다.
- 변수가 많아질수록 q의 구조가 매우 복잡해질 수 있다.
- q_j는 전체 latent variable (레이턴트 베리어블) 중 일부를 나타내는 간소화된 버전이다.
#### Segment 4 (03:04–04:17)
- 각각의 q를 독립적으로 최대화하려는 시도가 Mean-field assumption (민 필드 어썸션)의 핵심이다.
- MFVI를 유도하는 세 번째 방법으로 functional derivative를 상세히 살펴본다.
- Variational calculus (변분법)는 일반적인 미적분과 달리 함수를 입력으로 받는다.
#### Segment 5 (04:17–05:37)
- f(x)를 입력으로 받는 f는 '함수에 대한 함수'인 functional로 정의된다.
- ELBO는 확률 분포 함수 q_j를 입력으로 받아 값을 결정하므로 functional의 대표적인 예시이다.
- Functional derivative는 입력 함수가 변할 때 functional의 출력값이 어떻게 변하는지 그 변화량을 추정한다.
#### Segment 6 (05:37–06:05)
- 함수 f(x)에서 최대가 되는 x를 찾기 위해 미분값 f'(x)가 0이 되는 지점을 활용한다.
- f'(x) = 0을 만족하는 지점을 stationary point라고 정의한다.
- Stationary point를 찾은 후에는 해당 점이 극대(max)인지 극소(min)인지 판별하는 추가 과정이 필요할 수 있다.