# J5 — NBS와 양방향 탐색

개인 브랜치 `jy`, 2026-10-27. 공유 실행기에서 공통 비교군과 함께 평가한다.

- [실행 규약](PROTOCOL.md), [결과·한계](REPORT.md)
- [실제 구현](../JY/search.py), [공유 실행기](../JY/run.py), [검증](../JY/test_search.py)
- [원시 결과](../JY/results/run_20261027/), [대응 수치표](../JY/results/run_20261027/analysis/TABLES.md)

실험 조건: `j5_nbs_octile / j5_nbs_euclidean / j5_bidijkstra`.

NBS는 양쪽 frontier의 f값과 g 합으로 쌍 하한 max(fF,fB,gF+gB)을 정하고 최소 하한 쌍을 확장한다. waiting(f)/ready(g) 네 큐를 사용하며 첫 만남으로 종료하지 않는다. 양방향 Dijkstra도 독립 대조한다.

일관적 하한인 현재 격자에서 방향별 정점 확장과 간선 relax는 O(V+E), 큐 작업은 O((V+E) log V), 저장량은 양방향 O(V+E). 별도 audit의 Cartesian 전수 비교는 O(|OPEN_F|·|OPEN_B|)이며 온라인 실험에서는 비활성화한다. 쌍 하한 정리는 실측 총 pop/시간의 우위를 보장하지 않는다.

공개 1,000문제, 기존 별도 30문제, 새 합성 120문제를 동일 이동·충돌 규칙으로 비교한다.
저장 응답이 없는 새 합성 문제에서는 LLM 조건을 실행하지 않는다. API 호출 0회.
실행 명령은 [공유 README](../JY/README.md)를 따른다. 같은 결과를 방법별로 중복 복제하지 않는다.

사용자가 직접 설명할 핵심은 변경한 우선순위/종료 조건, 포함한 준비 비용, 비교군과의 차이,
경로 품질과 속도의 관계다. 출처와 Codex 지원은 공유 규약 및 결과 보고서에 기록한다.
