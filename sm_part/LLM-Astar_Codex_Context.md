> **현재 평가에서 사용 금지 — 이전 결과 폐기**
>
> 2026-10-28 정정: 이 문서의 이전 SM 실험 수치와 선정 결론은 현재 평가 근거로 폐기되었습니다. 현재 유효한 S1~S6 결과는 GM 기준·Python 3.11 재실험입니다. 과거 기록은 추적을 위해서만 보존합니다.
> [현재 공식 결과](gm_retest_20261028/results/s1_s6_gm_20261028/report.md)

---

# SWE3011 LLM-A* 과제 / 논문 리뷰 정리
## Codex 코드리뷰용 컨텍스트

이 문서는 SWE3011 Introduction to Artificial Intelligence (Fall 2026) Homework Project의 과제 안내서와
논문 **“LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning”**
(arXiv:2407.02511v2) 내용을 바탕으로, 이후 Codex에게 코드리뷰를 맡기기 위한 공통 컨텍스트를 정리한 것이다.

---

# 1. 과제 전체 설명

## 1.1 과제 목표

이 과제의 핵심은 다음 4단계다.

1. 논문의 **LLM-A\*** 알고리즘을 이해한다.
2. 논문의 주요 실험 결과인 **Table 1을 재현**한다.
3. 원본 LLM-A\*의 **한계를 분석**한다.
4. 해당 한계를 개선한 **우리만의 알고리즘을 설계·구현하고 실험으로 검증**한다.

즉, 단순히 논문 코드를 실행하는 과제가 아니라 다음 비교가 가능해야 한다.

- A*
- Original LLM-A*
- Proposed Method (우리 개선 방법)

---

## 1.2 팀 구성

- 개인 또는 팀 프로젝트 가능
- 팀은 최대 3명
- 팀 프로젝트라도 각 학생이 자신의 기여도를 개별적으로 정확히 보고해야 함

---

# 2. 평가 환경과 중요한 조건

## 2.1 LLM Backbone

개발/재현 과정에서는 접근 가능한 다른 LLM을 사용할 수 있다.

예:
- GPT 계열
- Claude
- Gemini
- Llama 등

하지만 **최종 코드 평가는 주로 GPT-4o mini를 backbone model로 사용**한다.

따라서 코드가 특정 모델 출력 형식이나 특정 모델 특성에 강하게 의존하면 위험하다.

중요한 해석:

> GPT-4o mini 자체를 학습하거나 수정하는 과제가 아니라,
> GPT-4o mini를 LLM-A*의 waypoint 생성용 LLM으로 넣었을 때
> 우리가 만든 알고리즘이 정상적으로 작동해야 한다는 의미다.

---

## 2.2 Hidden Environment / Hidden Map

조교는 공개된 환경뿐 아니라 **미공개 환경과 데이터셋**에서 코드를 평가한다.

따라서 다음은 피해야 한다.

- 공개 map의 좌표를 하드코딩
- 특정 파일명/맵 구조에만 맞춘 분기
- 공개 테스트셋에만 맞춘 heuristic 튜닝
- 수동 수정이 필요한 코드

제출 코드는 baseline repository가 요구하는 입력/출력 형식을 따라야 하며,
새로운 환경에서도 수정 없이 실행되어야 한다.

---

## 2.3 성능만 보는 과제가 아님

평가 항목에는 다음이 포함된다.

- 알고리즘 성능
- Table 1 재현
- 제안 방법의 명확성
- 기존 LLM-A*와의 차이
- 실험 설계의 공정성
- 결과 해석
- 실패 사례 분석
- limitation
- reproducibility
- 보고서의 설명력

논문 결과와 재현 결과가 달라도,
그 차이를 제대로 분석하면 그 자체로 감점 사유가 되지는 않는다.

---

# 3. 제출해야 할 내용

## 3.1 Project Report

파일명:
`TeamName_Report`

포함되어야 할 핵심 내용:

- Experimental Setup
- Methodology
- Table 1 Reproduction
- 원 논문 결과와 재현 결과 비교
- 원본 LLM-A*의 한계
- Proposed Algorithm
- Experimental Results
- A* / LLM-A* / Proposed Method 비교
- 결과 분석
- Failure Cases
- Ablation 또는 단계별 비교(적절한 경우)
- Limitations
- Reproducibility 정보
  - LLM 설정
  - Prompt
  - 환경 설정
  - 실행 방법
- References

보고서는 ACL template을 따라야 한다.

---

## 3.2 Code

파일명:
`TeamName_CodeFile.zip`

필수 내용:

- 전체 코드
- `requirements.txt`
- `README`

README에는 최소한 다음이 포함되어야 한다.

- Table 1 실험 재현 방법
- Proposed Algorithm 실행 방법

추가 주의:

- baseline repository와 호환되는 입출력 형식을 유지해야 함
- hidden map에서 수정 없이 실행되어야 함
- API key / access token / password 등 credential을 절대 포함하면 안 됨

---

## 3.3 Team Contribution Report

개인 제출이다.

- 최소 반 페이지
- 팀원끼리 내용을 공유하거나 맞춰 쓰면 안 됨
- 구체적인 기여 내용 기록

예:
- 구현한 module
- 작성한 report section
- 수행한 experiment
- 처리한 issue / pull request
- 회의 참여
- 작업 날짜
- 결과물

팀원들의 보고서와 실제 프로젝트 artifact를 비교하여 free-riding 여부를 확인할 수 있다.

---

## 3.4 Peer Review

다른 팀 프로젝트에 대해 작성한다.

필수:

- Strengths 3개
- Weaknesses 3개
- Suggestions 3개

총 9개 항목이며,
각 항목은 최소 2~3문장으로 근거와 함께 설명해야 한다.

---

## 3.5 Response to Peer Review

받은 리뷰 각각에 대해:

- Agree / Disagree
- 그 이유

를 작성해야 한다.

---

# 4. 일정

- Project / Code / Report 제출:
  - 11월 22일 23:59
- Peer Review:
  - 11월 23일 ~ 12월 6일 23:59
- Response to Peer Review:
  - 12월 7일 ~ 12월 16일 23:59

---

# 5. 추천 작업 순서

1. 논문 이해
2. 공식 GitHub 실행
3. A* / LLM-A* 기본 실행 확인
4. Table 1 재현
5. 재현 결과와 논문 결과 비교
6. Original LLM-A* 한계 분석
7. 개선 알고리즘 선정
8. 개선 알고리즘 구현
9. A* / LLM-A* / Proposed Method 비교
10. Ablation / Failure Case 분석
11. Report 작성
12. README 및 제출 코드 정리
13. 새 환경에서 재실행 테스트

---

# 6. 논문 핵심 요약

## 6.1 문제의식

논문은 다음 두 가지 문제를 결합해서 해결하려고 한다.

### A*

장점:
- 정확한 search
- valid path
- admissible heuristic 조건에서 optimal path 보장 가능

단점:
- 환경이 커질수록 많은 node를 탐색
- operation 증가
- storage 증가

### LLM

장점:
- 전체 환경을 보고 큰 방향을 잡는 global reasoning 능력

단점:
- 세밀한 spatial reasoning이 약함
- 긴 경로 계획에서 hallucination / collision 가능
- LLM-only path generation은 valid path 비율이 매우 낮음

논문의 아이디어:

> LLM은 큰 방향만 제안하고,
> 실제 경로 탐색은 A*에게 맡긴다.

---

# 7. A*에서 알아야 할 핵심 개념

## 7.1 기본 수식

A*는 다음 값을 사용한다.

`f(s) = g(s) + h(s)`

- `g(s)`:
  - 시작점에서 현재 state까지 실제로 사용한 cost
- `h(s)`:
  - 현재 state에서 goal까지 남은 cost의 추정값
- `f(s)`:
  - 해당 state를 탐색할 우선순위

A*는 OPEN list에서 `f`가 가장 작은 state를 선택한다.

---

## 7.2 OPEN / CLOSED

- OPEN:
  - 앞으로 탐색할 후보 state
- CLOSED:
  - 이미 탐색한 state

A*는 OPEN에서 가장 낮은 f-cost를 가진 state를 하나 선택하고,
주변 neighbor를 확장한 뒤 해당 state를 CLOSED로 이동시킨다.

---

## 7.3 Admissible Heuristic

heuristic `h(n)`이 실제 남은 최단 cost를 **절대 초과해서 예측하지 않는 것**이다.

즉:

`h(n) <= true remaining optimal cost`

중요한 점:

A*는 첫 번째로 좋아 보이는 경로를 선택하고 끝내는 알고리즘이 아니다.

예를 들어 어떤 경로를 먼저 탐색하더라도,
더 좋은 경로가 아직 존재할 가능성이 있으면 OPEN에 남아 계속 비교한다.

admissible heuristic이면 f-value가 해당 후보 경로가 가질 수 있는 최종 cost의
낙관적인 lower bound 역할을 한다.

따라서 진짜 더 좋은 경로를 지나치게 비싸다고 평가해서 버리는 일이 없고,
goal이 최저 f-value로 선택될 때 더 싼 미탐색 경로가 남아 있을 수 없기 때문에
optimal path를 보장할 수 있다.

즉:

- 잘못된 경로를 먼저 탐색할 수는 있음
- 하지만 그것을 바로 최종 답으로 확정하지는 않음
- 모든 경로를 다 보는 것도 아님
- 이미 찾은 최적 후보보다 더 좋아질 가능성이 없는 후보는 탐색할 필요가 없음

---

# 8. LLM-A* 알고리즘

## 8.1 핵심 아이디어

LLM이 전체 최종 경로를 직접 실행하는 것이 아니다.

LLM은 start와 goal 사이에 존재할 수 있는
**중간 목표점(waypoint / target state)**들을 생성한다.

이를 논문에서는 `TARGET list T`라고 부른다.

예:

`T = [Start, t1, t2, t3, Goal]`

LLM:
- 큰 방향 제안

A*:
- 실제 collision-free path 계산

---

## 8.2 핵심 수식

기존 A*:

`f(s) = g(s) + h(s)`

LLM-A*:

`f(s) = g(s) + h(s) + cost(t, s)`

- `t`:
  - 현재 LLM waypoint
- `cost(t, s)`:
  - 현재 state와 waypoint 사이의 거리/비용

이 추가 cost는 **실제 이동 cost g를 증가시키는 것이 아니라**
search priority인 `f`에 penalty를 추가하는 역할이다.

예:

- Node A → waypoint 거리 2
- Node B → waypoint 거리 10

그러면:

- A: 기존 f + 2
- B: 기존 f + 10

따라서 A가 먼저 탐색될 가능성이 높아진다.

즉:

- waypoint와 가까움 → penalty 작음 → 우선순위 높음
- waypoint와 멂 → penalty 큼 → 우선순위 낮음

결과적으로 search를 LLM waypoint 방향으로 끌어당긴다.

---

## 8.3 Waypoint 전환

현재 waypoint `t1`에 도달하면:

`t = t2`

로 전환한다.

그리고 기존 OPEN list에 저장된 state들의 f-cost를
새 waypoint 기준으로 다시 계산한다.

따라서 LLM-A* heuristic은 search 중간에 계속 바뀌는
dynamic heuristic처럼 동작한다.

---

# 9. LLM waypoint 후처리

논문에서 기본적으로 수행하는 검증은 다음과 같다.

1. Target list에 Start가 없으면 추가
2. Goal이 없으면 추가
3. waypoint가 obstacle 내부에 있으면 제거

하지만:

- obstacle 밖이더라도 비효율적인 waypoint
- 심하게 우회하는 waypoint
- 잘못된 global direction

등의 품질 자체는 정교하게 평가하지 않는다.

이는 개선 가능한 지점이다.

---

# 10. LLM에게 주는 입력

논문은 map image 자체를 LLM에 넣는 것이 아니라
환경을 텍스트 형태로 표현한다.

예:

```text
Start Point: [5, 5]
Goal Point: [20, 20]

Horizontal Barriers:
[[10, 0, 25], [15, 30, 50]]

Vertical Barriers:
[[25, 10, 22]]
```

출력:

```text
Generated Path:
[[5, 5], [26, 9], [25, 23], [20, 20]]
```

이 Generated Path는 최종 실행 경로라기보다는
LLM-A*에서 사용할 waypoint sequence로 사용된다.

---

# 11. Prompt 방식 3가지

## 11.1 Few-shot

정답 예시를 몇 개 보여주고,
그 패턴을 참고해 새로운 문제를 풀도록 한다.

논문:
- 5-shot 사용

---

## 11.2 Chain of Thought (CoT)

예시 답뿐 아니라
중간 reasoning 과정도 함께 제공한다.

예:
- 이 장애물을 피하려면 오른쪽으로 가야 한다
- 그 다음 goal 방향으로 이동한다

논문:
- 3-shot CoT

---

## 11.3 Recursive Path Evaluation (RePE)

전체 path를 한 번에 만들기보다 반복적으로:

1. 환경 분석
2. 다음 waypoint 선택
3. waypoint 평가
4. 문제 있으면 수정
5. 다음 waypoint 선택

하는 방식이다.

---

# 12. LLM-only 결과 해석

GPT-3.5를 LLM-only path planner로 사용한 경우:

- Few-shot: Valid Path 12.80%
- CoT: Valid Path 15.20%
- RePE: Valid Path 7.80%

즉 LLM 혼자 end-to-end path planning을 하면
대부분의 경로가 collision-free하고 complete한 valid path가 되지 못했다.

이 결과는 논문의 핵심 주장과 연결된다.

> LLM은 path planner 자체보다는
> A*의 search direction을 보조하는 역할에 더 적합하다.

---

# 13. 실험 데이터셋

논문 기본 실험:

- 100 maps
- 각 map당 10개의 start-goal pair
- 총 1000 samples
- 기본 map size: 50 × 30

각 환경은 대략 다음 정보로 구성된다.

- x range
- y range
- horizontal barriers
- vertical barriers
- start-goal pairs

논문은 continuous environment라고 표현하지만,
실제 LLM과 A*의 search는 discrete points/actions로 동작한다.

---

# 14. 평가 지표

## 14.1 Operation Ratio

탐색 과정에서 사용한 operation의 양을
A* 대비 비율로 나타낸다.

낮을수록 좋다.

예:

`50%`

→ A*의 약 절반 수준의 operation을 사용했다는 의미.

---

## 14.2 Storage Ratio

탐색 과정에서 유지/저장한 state의 양,
즉 메모리 사용량과 관련된 지표다.

낮을수록 좋다.

---

## 14.3 Relative Path Length

A*의 optimal path length를 기준으로
생성된 경로 길이를 비교한다.

A* = 100%

예:
`102.5%`

→ A* 최단 경로보다 약 2.5% 길다.

---

## 14.4 Valid Path Ratio

전체 시도 중:

- obstacle과 충돌하지 않고
- start에서 goal까지 실제로 도달 가능한

path 비율.

높을수록 좋다.

---

# 15. Table 1 핵심 결과

대표적인 결과:

| Method | Operation Ratio | Storage Ratio | Relative Path Length | Valid Path |
|---|---:|---:|---:|---:|
| A* | 100 | 100 | 100 | 100 |
| Dynamic WA* (w=2) | 60.91 | 78.53 | 100.24 | 100 |
| GPT-3.5 LLM-A* Few-shot | 57.39 | 74.96 | 102.44 | 100 |
| LLAMA3 LLM-A* Few-shot | 44.59 | 64.02 | 102.47 | 100 |

핵심 해석:

LLAMA3 Few-shot LLM-A* 기준:

- A* operation의 44.59% 사용
- A* storage의 64.02% 사용
- path는 A*보다 약 2.47% 길음
- valid path ratio는 100%

즉:

> 경로가 약간 길어지는 것을 허용하는 대신
> 탐색량과 저장량을 크게 줄인다.

---

# 16. Ratio 표현 주의

논문 metric 정의상:

`Operation Ratio = 44.59%`

는:

> A*가 사용하는 operation의 44.59%만 사용

한다는 뜻이다.

따라서 실제 감소량으로 표현하면:

`100 - 44.59 = 55.41% 감소`

가 자연스럽다.

논문 본문 일부의 “44.59% 감소”라는 표현과
Table 1 metric 정의 사이에는 해석상의 혼동이 있으므로,
코드 및 보고서에서는 metric 정의를 기준으로 확인할 것.

---

# 17. Geometric Mean

Table 1의 여러 ratio를 aggregation할 때
단순 arithmetic mean이 아니라 **geometric mean**을 사용한다.

N개 sample의 ratio `R_i`가 있을 때:

`G = (Π R_i)^(1/N)`

따라서 Table 1 재현 코드 리뷰 시 반드시 확인해야 한다.

- sample별 ratio를 구하는지
- 그 ratio에 geometric mean을 적용하는지
- 단순 평균을 사용하고 있지는 않은지

---

# 18. Prompt 방식 결과

LLM-A*에서는 대체로 Few-shot이 operation/storage 측면에서 가장 좋은 성능을 보였다.

LLAMA3 기준:

- Few-shot Operation: 44.59
- CoT Operation: 47.60
- RePE Operation: 64.08

즉 더 긴 reasoning을 시킨다고 항상 좋아지는 것은 아니다.

논문은 긴 sequential spatial reasoning에서
hallucination / misunderstanding 문제가 생길 수 있다고 해석한다.

---

# 19. Dynamic Weighted A*

비교 baseline으로 Dynamic Weighted A*가 등장한다.

설정:

- initial weight = 2
- decay = 0.99

역할:

LLM을 사용하지 않고도 heuristic의 영향력을 키워
탐색량을 줄이는 기존 방식과 비교하기 위함이다.

따라서 논문의 실질적인 비교 질문 중 하나는:

> 단순히 heuristic weight를 조절하는 것보다
> LLM waypoint를 활용하는 것이 더 좋은가?

라고 볼 수 있다.

---

# 20. LLM-A*의 핵심 Trade-off

Original A*:

- 높은 optimality guarantee
- search/resource cost 증가 가능

LLM-A*:

- operation 감소
- storage 감소
- path validity 유지
- 하지만 optimality guarantee 상실

논문에서는 약 90%의 LLM-A* path가 optimal이었다고 기술하지만,
항상 optimal path를 보장하지는 못한다.

이유:

`h_A*(n) + c_LLM(n)`

형태의 추가 heuristic term 때문에
실제 remaining optimal cost보다 heuristic이 커질 수 있고,
admissibility가 깨질 수 있기 때문이다.

---

# 21. 논문의 핵심 장점

1. LLM에게 전체 최종 경로를 맡기지 않음
2. LLM은 global guidance만 담당
3. A*는 precise local search 담당
4. 기존 A* 구조를 크게 바꾸지 않음
5. 큰 환경에서 operation / storage 감소 가능
6. LLM-only보다 훨씬 높은 valid path ratio

핵심 구조:

`LLM global reasoning + A* deterministic local search`

---

# 22. 논문의 중요한 한계 / 개선 후보

코드리뷰와 이후 proposed method 설계에서 특히 확인할 항목.

## 22.1 Optimality Guarantee 상실

추가 heuristic term 때문에 admissibility가 깨질 수 있다.

질문:
- 이를 완화할 수 있는가?
- waypoint 영향력을 조절할 수 있는가?

---

## 22.2 LLM waypoint를 거의 동일하게 신뢰

현재 기본식:

`f = g + h + cost(t, n)`

LLM waypoint의 품질/신뢰도에 따라 영향력을 다르게 주지 않는다.

개선 후보:

`f = g + h + α_t * cost(t, n)`

예:

- 좋은 waypoint → alpha 크게
- 의심스러운 waypoint → alpha 작게
- 매우 나쁜 waypoint → 제거

---

## 22.3 Waypoint validation이 단순함

현재는 주로:

- start/goal 보정
- obstacle 내부 waypoint 제거

정도다.

개선 가능:

- detour score
- visibility
- obstacle proximity
- progress toward goal
- local feasibility
- waypoint confidence

등을 활용할 수 있다.

---

## 22.4 OPEN list f-cost 재계산 비용

waypoint가 바뀔 때마다
OPEN에 존재하는 state들의 f-cost를 다시 계산한다.

따라서:

- target 수 증가
- OPEN 크기 증가

시 추가 비용이 커질 수 있다.

검토 후보:

- lazy update
- incremental priority update
- stale key 처리 방식

---

## 22.5 LLM inference cost / latency

논문의 주요 metric은:

- operation
- storage
- path length
- valid path

중심이다.

하지만 실제 시스템 관점에서는:

- LLM API latency
- token usage
- cost
- number of LLM calls

도 중요하다.

과제 안내서도 cost-efficiency analysis를 추가 평가 요소로 인정한다.

---

## 22.6 Model / Prompt Generalization

논문 실험 모델은 주로:

- GPT-3.5-TURBO
- LLAMA3-8B-16bit

이다.

과제 최종 평가는 주로 GPT-4o mini이므로,
현재 구현이 특정 모델 출력에 과하게 의존하지 않는지 확인해야 한다.

---

# 23. 현재 가장 유력한 개선 방향 후보

아직 확정된 방법은 아니지만,
지금까지 논문을 읽으며 가장 자연스럽게 나온 후보는:

## Adaptive Waypoint Weighting

기존:

`f(n) = g(n) + h(n) + d(n, t)`

개선:

`f(n) = g(n) + h(n) + α_t * d(n, t)`

여기서 `α_t`를 waypoint 품질이나 상황에 따라 조정한다.

목표:

- 좋은 LLM guidance는 적극 활용
- 잘못된 waypoint의 악영향 감소
- GPT-4o mini 등 model variation에 대한 robustness 향상
- path length 손실 감소 가능성

단, 이것은 현재까지의 **개선 아이디어 후보**이지
과제에서 반드시 이 방식을 써야 하는 것은 아니다.

---

# 24. Codex 코드리뷰 시 특히 확인해야 할 것

이 문서 이후 Codex에게 코드를 리뷰시킬 때는 아래를 중심으로 확인하면 된다.

## A. Repository / 실행 구조

- baseline repository 원본 구조 확인
- entry point 확인
- 환경 생성 코드 위치
- A* 구현 위치
- LLM-A* 구현 위치
- prompt 코드 위치
- experiment/evaluation 코드 위치
- 결과 집계 코드 위치

## B. Algorithm 1과 코드 일치 여부

특히:

- TARGET list 생성 방식
- Start / Goal 보정
- obstacle 내부 target 제거
- 현재 target 전환 조건
- OPEN / CLOSED 처리
- `g` update
- `f = g + h + cost(t, s)` 구현
- target 변경 후 OPEN f-cost 재계산 여부

## C. Table 1 재현 가능성

확인:

- map 수
- start-goal pair 수
- 50×30 설정
- A*
- Dynamic WA*
- LLM-only
- LLM-A*
- Few-shot
- CoT
- RePE
- GPT/Llama model 설정
- random seed
- operation 정의
- storage 정의
- path length 계산
- valid path 판정
- geometric mean 계산

## D. Operation / Storage 정의 확인

특히 코드가 논문에서 의미하는 operation/storage를
정확히 무엇으로 count하는지 확인해야 한다.

단순 실행 시간이나 Python 메모리 사용량이 아니라
논문 repository가 정의한 search metric과 일치해야 한다.

## E. Hidden map 대응

- 특정 map index 하드코딩 여부
- 공개 좌표 하드코딩 여부
- 새로운 map size 대응
- 입력 형식에 대한 일반성
- 수동 설정 없이 실행 가능한지

## F. GPT-4o mini 대응

- model name/config 교체 가능 여부
- prompt/output parser가 특정 모델 출력에 종속되는지
- malformed output 예외 처리
- retry 정책
- invalid waypoint 처리

## G. Reproducibility

- requirements.txt 완전성
- 실행 명령
- seed
- 환경 설정
- model setting
- prompt
- 결과 저장 방법
- README 내용

## H. 제출 안전성

- API key 하드코딩 여부
- `.env` 포함 여부
- token/password 노출 여부

---

# 25. 가장 짧은 전체 요약

과제:

`논문 이해 → Table 1 재현 → 한계 분석 → 개선 알고리즘 → 비교 실험 → 보고서`

논문:

`A*는 정확하지만 많이 탐색`
+
`LLM은 큰 방향은 잘 보지만 정확한 path는 못 만듦`
→
`LLM이 waypoint 생성`
→
`A*가 waypoint를 heuristic에 추가`
→
`operation/storage 감소`
→
`대신 optimality guarantee 일부 상실`

핵심 수식:

`A*: f = g + h`

`LLM-A*: f = g + h + cost(current_waypoint, node)`

코드리뷰 핵심:

1. 논문 Algorithm 1과 실제 구현이 일치하는가
2. Table 1을 실제로 재현할 수 있는가
3. operation/storage/path length/valid ratio 정의가 정확한가
4. geometric mean이 맞게 계산되는가
5. GPT-4o mini와 hidden map에서도 일반적으로 동작하는가
6. 개선하기 좋은 구조와 병목은 어디인가
