# Segment Summaries (Natural Language)

## Segment 1 (00:00–00:37)
이번 시간에는 Variational Inference (변분 추론)의 심화 과정으로 Mean-field Variational Inference (MFVI)를 유도하는 또 다른 방법을 살펴봅니다. 특히 이 방법론이 실제 어떻게 응용되는지 확인하고, 널리 알려진 Expectation-Maximization (EM) 알고리즘과 어떤 관계가 있는지 분석합니다. 두 개념 사이의 공통점과 차이점을 이해하는 것이 이번 학습의 주요 목표입니다.

## Segment 2 (00:37–01:48)
MFVI를 유도하기 위해 log marginal likelihood (로그 마지널 라이클리후드)의 하한인 ELBO를 최대화하는 과정을 검토합니다. 이론적으로 ELBO를 최대화하는 최적의 q(z)는 posterior (사후 확률)이지만, 이를 직접 구하기 어렵기 때문에 functional derivative (변분 미분)라는 도구를 사용합니다. 이는 일반적인 미적분과 달리 함수를 입력으로 받는 functional (범함수)의 최대치를 찾는 과정입니다.

## Segment 3 (01:48–03:04)
posterior를 정확히 계산할 수 없는 상황에서 q(z)는 매우 복잡한 형태를 띨 수 있습니다. 특히 Latent Dirichlet Allocation (LDA)처럼 다양한 종류의 latent variable (잠재 변수)이 포함된 경우, 이들의 조인트 분포를 하나의 형식으로 정의하기가 불가능에 가깝습니다. 따라서 복잡한 전체 분포를 다루는 대신 각 변수를 q_1, q_2와 같이 개별적으로 쪼개서 분석하는 전략을 취하게 됩니다.

## Segment 4 (03:04–04:17)
여러 변수를 독립적으로 분리하여 최적화하는 방식을 Mean-field assumption (민 필드 가정)이라고 부르며, 이를 바탕으로 functional derivative를 활용한 유도법을 배웁니다. 일반적인 미적분이 실수를 입력받아 실수를 내뱉는 것과 달리, variational calculus (변분법)는 함수 자체를 입력으로 받아 실수를 출력하는 functional을 다룹니다. 즉, 입력값이 함수라는 점이 기존의 전통적인 미적분 체계와 가장 크게 다른 점입니다.

## Segment 5 (04:17–05:37)
함수를 입력으로 받아 실수를 결과로 내놓는 구조를 functional이라고 정의하며, q_j라는 함수를 입력받는 ELBO가 그 대표적인 사례입니다. functional derivative는 입력 함수가 미세하게 변할 때 functional의 전체 값이 어떻게 변화하는지를 추정하는 도구입니다. 이를 활용해 최적의 함수를 찾기 위해서는 먼저 functional의 최대치를 구하기 위한 수학적 준비 단계를 거쳐야 합니다.

## Segment 6 (05:37–06:05)
일반적인 함수에서 최댓값을 찾으려면 함수의 미분값이 0이 되는 지점인 stationary point (정지점)를 찾는 것이 가장 기본입니다. 이와 마찬가지로 functional의 세계에서도 최적의 상태를 찾기 위해 미분 방정식을 풀어 stationary function (정지 함수)을 구하는 접근 방식을 취합니다. 이러한 기초적인 최적화 원리를 바탕으로 복잡한 변분 문제를 해결하는 준비 작업을 진행하게 됩니다.
