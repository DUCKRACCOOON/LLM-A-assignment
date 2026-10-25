# K4 — LLM 안내를 사용하는 FOCAL 탐색

규민 담당 K4, `gm`, 2026-10-25. LLM이 탐색 순서를 제안하고 알고리즘은 경로 비용 제한을 지키도록 역할을 나눈다. 기존 K3의 거리 하한을 그대로 재사용하며, 큐 관리와 재개방은 K4 전용으로 구현했다.

- [과정·결과·선정 판단](REPORT.md)
- [실행 전 고정 규약](PROTOCOL.md)
- [탐색·LLM 안내 점수](search.py)
- [독립 참조 구현과 검증](verify.py)
- [캐시 확인·비교 실험·통계 집계](run.py)
- [실행 환경·코드·입력 해시](results/run_20261025/manifest.json)

## g+2h, K3와 다른 점

기존 g+2h는 `지금까지 이동한 거리＋2×목표까지 직선거리`를 최소화한다. K3는 그 목표 거리 계산에 벽 우회를 반영한다. 두 방법은 값이 가장 작은 후보 하나를 바로 선택한다.

K4는 먼저 **비용상 허용할 후보 집합**을 만들고 그 안에서 LLM 안내로 순서를 정한다.

```text
h0(n) = K3의 octile＋벽 하한
f(n) = g(n) + h0(n)
fmin = OPEN에 있는 모든 후보의 f 중 최솟값
FOCAL = f(n) <= 1.05*fmin 인 후보들

FOCAL 안에서 LLM 안내 점수가 가장 작은 후보를 선택한다.
```

주 설정 1.05는 **최단거리의 105%까지 허용**한다는 뜻이다. 예를 들어 최단거리가 100이라면 105 이내의 경로를 찾는다. 기존 g+2h의 숫자 2는 h를 키우는 가중치이고, FOCAL의 1.05는 후보의 비용 자격을 제한하는 계수다. 서로 같은 파라미터처럼 비교하지 않는다.

## LLM의 좌표를 어떻게 사용하나

기존 실제 GPT 응답에서 마지막 2차원 좌표 목록을 파싱한다. 시작과 목표를 양 끝에 보완하고 점들을 잇는 꺾은 선을 만든다. 후보 n에서 이 선에 가장 가까운 투영점을 p라고 하면 다음 점수를 쓴다.

```text
guide(n) = n에서 p까지 거리 + p부터 안내 선을 따라 목표까지 남은 길이
```

같은 거리의 투영점이 여러 개면 목표 쪽으로 더 진행한 지점을 사용한다. 이 점수는 **FOCAL 안의 순위에만** 쓰며 하한으로 쓰지 않는다. 현재 후보가 안내 선에 가깝고 안내 경로의 뒤쪽으로 진행했으면 낮은 점수를 받는다.

LLM 선분이 장애물을 통과하거나 좌표가 지도 밖에 있어도 실제 이동 간선으로 사용하지 않는다. 이런 잘못된 안내가 탐색 순서를 악화시킬 수는 있지만, 모든 실제 이동과 최종 경로는 원래 충돌 규칙으로 검사한다. 경로 전체가 유효해야 안내를 채택하는 V-Gate나 경로 수리 알고리즘을 추가한 것은 아니다.

좌표 목록이 없거나 schema가 잘못되거나 계산이 비유한 값이면 h0를 보조 순위로 사용하고 `guide_error`에 기록한다. 지도별 정답을 보고 안내를 선택·교체하지 않는다. 모델은 재호출하지 않는다.

## 최단거리를 미리 모르는데 어떻게 제한하나

올바른 하한 h0와 더 나은 경로의 재개방을 유지하면 OPEN에는 최단경로 비용 이하의 f를 가진 경계 후보가 남아 있다. 따라서 `fmin <= 실제 최단거리`다. 목표를 FOCAL에서 선택할 때 목표의 h0는 0이므로 다음 관계가 성립한다.

```text
반환 비용 <= 목표의 g <= 1.05*fmin <= 1.05*실제 최단거리
```

LLM 점수가 틀려도 이 자격 검사를 우회할 수 없다. 이 논리는 admissible 하한, OPEN의 정확한 최소값, 더 작은 g를 발견한 노드의 재개방, FOCAL 내 목표 선택을 전제로 한다. 코드에서 각 조건을 유지하고 독립 Dijkstra로 검증했다. 수치 비교에는 작은 부동소수점 허용 오차를 둔다.

## 구현 구조와 비교 지표

OPEN은 f 순서, 대기 heap은 아직 자격이 없는 후보의 f 순서, FOCAL은 안내 점수 순서로 관리한다. g가 갱신된 이전 버전과 이미 선택한 후보의 오래된 heap 항목은 무효화한다. fmin이 낮아져 허용 범위가 줄어들면 자격을 다시 구성한다.

더 짧은 경로를 찾으면 CLOSED의 위치도 다시 열기 때문에 기존 K3의 재개방 금지 엔진과는 차이가 있다. 이를 구분하기 위해 같은 새 엔진으로 단일 큐 g+2h0를 실행하는 `clean_wastar_wall` 대조군을 함께 둔다.

| 지표 | 의미 |
| --- | --- |
| operation | OPEN＋대기＋FOCAL의 모든 heap pop 합계. 정리·승격·stale pop 포함 |
| expansions | 유효 후보를 선택한 횟수. 재확장과 목표 선택 포함 |
| unique_pops | 선택된 서로 다른 위치 수 |
| reopenings, reexpansions | CLOSED를 다시 연 사건 수, 같은 위치의 추가 선택 횟수 |
| storage | g 사전 항목 수, 무한 비용 항목 포함. 전체 메모리 아님 |
| peak_heap_entries | 세 heap에 실제 들어 있는 항목 수의 최대 합, stale 포함 |
| heap_pushes, rebuild_scans | heap 삽입 및 자격 재구성의 추가 작업 |
| guide_segment_checks | 안내 점수 계산을 위해 검사한 선분 수 |
| total_seconds | Geometry 생성부터 반환 경로 검증까지의 계획기 시간 |

K3/기존 코드의 operation은 단일 큐 pop 수다. K4의 FOCAL 선택만 세어 이를 전체 연산량이라고 제시하면 비교가 유리하게 왜곡될 수 있으므로 전체 heap pop을 주 비교에 사용한다. 어느 쪽의 pop 수도 전체 CPU 연산 수와 같지는 않아 시간을 함께 읽어야 한다. 저장량 역시 여러 heap과 안내 선을 포함한 실제 RAM 바이트가 아니다.

## 데이터·LLM 비용의 범위

공개 1,000문제는 K3와 같다. 추가 30문제는 기존 `results/generalization`의 seed=20261013 지도 10개다. **이 30문제는 K3의 새 지도 120문제와 다른 집합이다.** 이번 30문제에서도 K3 두 변형과 기존 g+2h를 같은 입력으로 다시 측정한다.

GPT-4o mini `gpt-4o-mini-2024-07-18`의 Few-shot 응답 1,030개를 재사용했다. 요청 내용과 프롬프트·입력 일치를 확인하고, 선택한 응답과 원 파일 경로·요청/응답 해시를 보존한다. 생성 결과를 보고 잘 나온 것만 고르지 않는다. 원 프롬프트는 경계를 명시하지 않으므로 바깥 좌표 안내가 있을 수 있다.

이번 실행의 신규 API 호출·비용은 0이다. 하지만 실제 새 문제에서 LLM 안내를 얻으려면 생성 지연·비용이 든다. 주 시간표는 응답이 이미 있는 **계획기 시간**이다. `historical_api_seconds`와 토큰 수는 별도로 저장하며, 이를 현재 계획기 시간에 더한 `estimated_fresh_seconds`는 과거와 현재의 합성 추정치일 뿐 신규 실측이 아니다.

## 다시 실행하기

저장소 루트의 gm에서 실행한다. API 키·GPU는 필요 없다. Python과 NumPy가 필요하고 그림 생성에는 Matplotlib이 필요하다. 버전은 manifest에 기록했다.

```powershell
python -m experiments.K4.verify
python -m experiments.K4.run --output experiments/K4/results/recheck_01
python -m experiments.K4.report --input experiments/K4/results/recheck_01
```

실험은 이미 있는 출력 폴더를 덮어쓰지 않는다. 새로운 폴더를 지정해야 한다. 그림과 요약 표는 저장된 결과에서 다시 만들 수 있다.

직접 호출할 때는 기존 query 사전과 LLM 응답 문자열을 전달한다.

```python
from experiments.K4.search import run_once

k4 = run_once(query, 'focal_llm_w105', text=cached_response_text)
control = run_once(query, 'focal_goal_w105')
k3 = run_once(query, 'g2h_wall_octile')
baseline = run_once(query, 'g2h_euclidean')
print(k4['length'], k4['bound_limit'], k4['operation'], k4['total_seconds'])
```

`run_once`는 단일 프로세스 실험용이며 trial 사이에 원본 Geometry 이웃 캐시를 비운다. 웹 서비스의 병렬 실행에 그대로 붙이도록 만든 API는 아니다. 선택 trace와 전체 OPEN 스캔 검사를 켜는 `trace=True`, `audit=True`는 검증용이며 본 시간 측정에서는 끈다.

## 출처와 구현 범위

FOCAL 후보 제한은 [Cohen et al., Anytime Focal Search with Applications, IJCAI 2018](https://ijcai.org/proceedings/2018/199) 및 [Fickert et al., New Results in Bounded-Suboptimal Search, AAAI 2022](https://www.cs.unh.edu/~ruml/papers/rrd-aaai-2022.pdf)의 일반 Focal Search 설명에 근거한다. 이번 구현은 그 틀에 우리 좌표 투영 안내를 적용한 것이다. 논문들의 AFS/EES 전체나 LLM의 통계적 신뢰도 모델을 재현한 것은 아니다.

격자·충돌은 [MIT 라이선스 원본의 기존 재현](../../UPSTREAM.md), 하한은 [K3](../K3/README.md), 독립 그래프 검증은 K3의 정수 선분 교차·Dijkstra를 재사용했다. K4 신규 코드·실험·문서는 규민 담당 작업에서 Codex 지원으로 작성했다. 원 논문 재현, K3, K4 결과를 별도 기록으로 보존한다.
