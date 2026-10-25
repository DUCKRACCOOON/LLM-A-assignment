# K6 5% 제한의 LLM 제거·고정 가중치 대조

기존 [K6 실험](../K6/REPORT.md)에서 남았던 질문인 **"w=1.05에서 LLM 안내와 적응 α가 실제로 도움이 되는가?"**를 확인하는 후속 실험이다. gm에서 진행하며 기존 K6의 탐색 코드는 수정하지 않고 재사용한다.

[과정·결과·판단](REPORT.md), [실행 전 규약](PROTOCOL.md), [전체 수치표](results/run_20261025/analysis/TABLES.md)를 함께 본다.

## 비교하는 것

| 조건 | 바꾸는 부분 |
| --- | --- |
| K6 적응 α, w=1.05 | 이전 K6 조건 그대로. 초기 0.5, 8회 관측으로 가중치 조절 |
| K6 LLM 제거, w=1.05 | 응답 파싱·안내를 제거하고 같은 엔진에서 g+1.05h0로 탐색 |
| K6 고정 α=0.5, w=1.05 | 같은 경유점·필터·엔진, 중간 경유점의 영향력만 0.5로 고정 |
| K6 고정 α=1, w=1.05 | 같은 조건에서 영향력 1로 고정 |
| 기존 g+2h | 기존 직선거리 기준 |
| K3 octile＋벽 하한 | K6가 사용하는 하한의 기존 엔진 비교 |
| K3 직선거리＋벽 하한 | 기존 성능 1순위 후보 비교 |

K6 네 조건은 모두 octile＋벽 하한 h0와 같은 재개방·정렬·종료 규칙을 쓴다. 중간 경유점이 없거나 마지막 목표 단계에서는 고정 방식도 α=0이다. 고정 α도 target이 바뀔 때 재정렬할 수 있다. K3는 w=2이고 기존 CLOSED 규칙을 유지하므로 K3와의 차이를 LLM의 효과로 해석하지 않는다.

LLM 유무는 같은 엔진의 적응/고정 조건과 no-guide를 비교해서 판단한다. 적응 조절은 적응과 고정 조건의 비교로 판단한다. 여기서 'LLM 기여'는 안내 시스템을 추가할 실용적 가치이며 모델의 의미 이해만을 다른 요인과 완전히 분리한 인과 추정은 아니다.

## 파일과 실행

| 파일 | 역할 |
| --- | --- |
| [search.py](search.py) | 설정만 달리하는 실행 함수. 기존 K6 graph_search·Controller·필터 사용 |
| [verify.py](verify.py) | 480조건의 전체 스캔 참조·독립 기하·기존 적응 조건 일치·no-guide 별도 엔진 일치 |
| [run.py](run.py) | 7조건×5회 측정, 원시 결과·과거 동등성·통계 |
| [report.py](report.py) | 비교표·그림·유효 안내 부분집합·사례 |
| [audit.py](audit.py) | 저장 해시·반복·경로 대조·중앙값·모든 상대 비율 재집계 |

저장소 최상위에서 실행한다. 기존 numpy·matplotlib 환경을 사용하며 API 키는 필요 없다.

```powershell
python -m experiments.K6_w105.verify
python -m experiments.K6_w105.run --output experiments/K6_w105/results/my_new_run
python -m experiments.K6_w105.report --input experiments/K6_w105/results/my_new_run
python -m experiments.K6_w105.audit --input experiments/K6_w105/results/my_new_run
```

기존 출력 폴더는 덮어쓰지 않는다. 같은 시간에 다른 성능 실험을 실행하지 않는다. K6 네 조건과 기준 3조건은 모두 결정적이며 5회 반복은 시간 중앙값을 위한 것이다.

```python
from experiments.K6_w105.search import run_once
guided = run_once(query, 'adaptive_wall_w105', saved_response_text)
unguided = run_once(query, 'no_guide_wall_w105')
fixed = run_once(query, 'fixed_half_wall_w105', saved_response_text)
```

평가 데이터는 기존 공개 1,000문제·별도 30문제, 응답은 동일 GPT-4o mini standard 1,030개다. 새로운 미지 평가가 아니며 K3의 별도 120문제와 다르다. 파일럿 30문제는 공개 데이터에 포함된다. 이번 검증을 위해 새 모델 요청을 보내지 않는다.

## 기록과 비용

`evaluations.jsonl.gz`에는 모든 반복의 경로·실제 시간·큐/제어 지표가 있다. `per_case.jsonl.gz`는 시간 중앙값과 모든 진단 지표, CSV는 간단한 비교 지표, `summary.json`은 문제별 상대 비율 기하평균·지도 bootstrap이다. 예전 K6의 시간과 새 시간은 실행 시점·반복 수가 다르므로 섞지 않는다.

pop에는 heapify를 위한 OPEN 재계산을 세지 않는다. rekey_scans와 실제 시간을 함께 확인한다. 시간에는 파싱·필터·준비·탐색·제어·재정렬·경로 검증이 포함되며 파일 읽기·독립 정답 산출·모델 생성은 제외한다. storage는 전체 RAM이 아닌 g 사전 크기다.

초기 실행은 첫 baseline 시행에서 내부 시간 필드 `seconds`를 비시간 동등성 항목으로 비교해 중단됐다. [실패 기록](development/failed_20261025_timing_parity/manifest.json)을 보존했다. 비교에서 `seconds`와 `total_seconds`를 제외한 뒤 새 폴더에서 전 과정을 다시 실행했다. 탐색 코드·가중치·데이터는 이 수정으로 바뀌지 않았다. 이 1회는 최종 성능 집계에 포함하지 않는다.

기존 K6의 근거 문헌·수식 설명은 [K6 README](../K6/README.md)에 있다. Codex가 구현·검증·실험·집계·문서 초안을 지원했고, 최종 제출 전 사용자의 이해 및 문장 검토는 별도다. 이 모듈은 비교 실험이며 과제 제출용 Proposed를 자동으로 바꾸지 않는다.
