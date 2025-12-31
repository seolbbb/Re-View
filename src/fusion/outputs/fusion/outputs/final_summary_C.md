# Final Summary C (TL;DR + 시간 순)
## TL;DR
- Variational Inference (변분 추론)의 다양한 유도 방법과 심화 내용을 학습한다.
- MFVI를 유도하기 위해 log p(x, z)의 기댓값 등을 활용하는 트릭을 사용한다.
- Posterior를 정확히 구할 수 없으므로 ELBO를 최대화하는 q를 찾는 것이 목적이다.
- 각각의 q를 독립적으로 최대화하려는 시도가 Mean-field assumption (민 필드 어썸션)의 핵심이다.
- f(x)를 입력으로 받는 f는 '함수에 대한 함수'인 functional로 정의된다.
- 함수 f(x)에서 최대가 되는 x를 찾기 위해 미분값 f'(x)가 0이 되는 지점을 활용한다.
## 시간 순 요약
#### Segment 1 (00:00–00:37)
- Variational Inference (변분 추론)의 다양한 유도 방법과 심화 내용을 학습한다.
#### Segment 2 (00:37–01:48)
- MFVI를 유도하기 위해 log p(x, z)의 기댓값 등을 활용하는 트릭을 사용한다.
#### Segment 3 (01:48–03:04)
- Posterior를 정확히 구할 수 없으므로 ELBO를 최대화하는 q를 찾는 것이 목적이다.
#### Segment 4 (03:04–04:17)
- 각각의 q를 독립적으로 최대화하려는 시도가 Mean-field assumption (민 필드 어썸션)의 핵심이다.
#### Segment 5 (04:17–05:37)
- f(x)를 입력으로 받는 f는 '함수에 대한 함수'인 functional로 정의된다.
#### Segment 6 (05:37–06:05)
- 함수 f(x)에서 최대가 되는 x를 찾기 위해 미분값 f'(x)가 0이 되는 지점을 활용한다.