# J5 전체 데이터셋 1회 평가

J4 다음 J5(NBS·양방향 탐색)를 공개100지도×10쌍=1,000문제에서 평가했다.
9조건×1회=9,000회이며 정확 조건7,000회 모두 독립 Dijkstra 최단거리와 일치했다. 새 API 호출은0회다.

- [사전 규약](PROTOCOL.md), [전체 결과](results/run_20261111/REPORT.md)
- [독립 검산](results/run_20261111/audit.json), [전체 집계](results/run_20261111/summary.json)
- [원시 경로·시간](results/run_20261111/evaluations.jsonl.gz), [문제별 CSV](results/run_20261111/per_case.csv)
- [소형 쌍 선택 검증](results/run_20261111/preflight.json), [35개 무작위 지도 검사](results/run_20261111/unit_tests.json)
- [입력·소스·환경](results/run_20261111/protocol.json), [LLMA 대조 응답](results/run_20261111/responses.jsonl.gz)

주 NBS octile의 로컬 평균은4.546ms로 같은 실행의 단일 octile A* 3.297ms보다 높았다.
양방향 Dijkstra보다는 빨랐으나 강한 단일 정확 A*에 대한 이득은 확인하지 못했다.
모든 heap pop에는 큐 승격·stale 제거도 포함했고 양쪽 방향의 비용을 합산했다.

## 실행

Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
GPU·학습 가중치·API 키는 필요 없다. 저장소 루트에서 실행한다.

```powershell
python -m experiments.J5_full.run --output experiments/J5_full/results/new_run
python -m experiments.J5_full.report --input experiments/J5_full/results/new_run
python -m experiments.J5_full.audit --input experiments/J5_full/results/new_run
```

기존 output은 덮어쓰지 않는다. 원 LLMA 대조만 저장 standard1,000응답을 재생하고,
J5/Euclidean NBS/양방향 Dijkstra는온라인LLM을 쓰지 않는다. 실행 중 네트워크/API 연결을 차단한다.

원 검색 코드는 [기존 JY 구현](../JY/search.py)이며 [adapter](adapter.py)는 기존 NBS와
같은 clean astar의 h=0 단방향 Dijkstra 대조를 연결한다. 작은 검증에서만 Cartesian pair reference를 켜고
데이터셋 시간 측정에서는 audit=False를 사용한다. 보고서·검사 명령은 저장 결과만 읽고 탐색을 재실행하지 않는다.

준연 담당 J5 평가에 Codex가 실행 통합·독립 검증·분석 작성을 지원했다. 이전 코드와 결과는 보존했다.
