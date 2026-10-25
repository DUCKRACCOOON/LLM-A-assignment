# S3 계획서 · 탐색 정체 시에만 GPT 호출

## 목적과 정체 조건

처음부터 GPT를 호출하는 기본 LLM-A* `B`와 달리, `g+2h`로 시작해 고정된 정체 조건을 만족할 때만 한 번 호출한다. [기존 고정 프로토콜](../../results/improvements/protocol.json)과 [delayed_search 구현](../../reproduction/improve_core.py)을 정리한 문서이며 새로운 사전등록이 아니다.

시작-목표의 Euclidean 거리를 `d0`라 할 때 다음을 고정했다.

1. 검사 간격은 중복 포함 heap-pop `max(10, ceil(d0))`회다.
2. 지금까지 pop된 노드 중 목표까지의 최소 거리를 `best`로 기록한다.
3. 직전 검사 이후 `best`의 개선량이 `0.05 × d0`보다 작고 탐색 큐가 남아 있으면 GPT를 한 번 호출한다.
4. 프롬프트는 원래 시작·목표·지도와 기존 프롬프트 그대로다. 현재 노드나 탐색 상태를 새 입력으로 추가하지 않는다.
5. 기존 `g`, `parent`, `closed`, OPEN을 유지하고 경유점을 삽입한 뒤 OPEN 우선순위만 다시 계산한다. 처음부터 재탐색하지 않는다.

검사 임계값을 본실험 결과에 맞춰 조절하지 않는다. 목표에서 멀어지는 우회가 필요할 수 있으므로 이 거리 기준을 일반적인 정체의 완전한 판정으로 표현하지 않는다.

## 대조군과 자료

`B`는 즉시 GPT 안내, `N`은 끝까지 GPT 없는 g+2h, `A`는 A*다. E1 정제나 E2 수정은 결합하지 않는다. 원본 100지도·1,000문제에 standard(Few-shot)·cot·repe를 적용하고, pilot 10문제와 고정 시드 240702511의 10문제×1~10배 standard 확대를 사용한다. GPT 모델은 `gpt-3.5-turbo-0125`, temperature 0, max_tokens 1000이다.

최초 응답 수집 단계에서 실제 E3 호출 시점과 실행 trace를 기록하고 B·E1·E2와 응답을 공유한다. 최종 재생에서는 경로·길이·연산·저장·유효성·경유점·trigger·호출 수가 획득 trace와 같은지 검사한다. 호출하지 않은 문제도 분모에 포함한다.

## 계측 및 실행

호출 전후를 합친 heap-pop, `len(g)`, 경로 길이·유효성, 호출 여부, 호출 시점, API 지연과 OPEN 재계산을 포함한 실행 기록을 보존한다. 원래 heap-pop에는 OPEN 재계산과 API 비용이 모두 표현되지 않으므로 별도로 해석한다. 논리 호출 감소와 실제 서비스 지연·청구 비용 감소를 구별한다.

Linux/Python 3.12.3 [환경](../../results/improvements/environment.json)에서 실행한 자료다. `sm_part`에서 원자료 복원 후 집계하는 명령은 다음과 같다.

```text
python reproduction/improve_run.py report
```

새 API 실행 CLI는 `python reproduction/improve_run.py all --workers 8`이다. 이번 정리에서 실행하지 않았으며 재실험은 기존 결과와 분리한 복사본에서 수행한다. 정체 조건·상태 재사용·총 pop 집계는 확인됐고, 독립 미관측 지도와 온라인 지연시간 비교는 현재 자료 밖의 보완 항목이다.
