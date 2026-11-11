# J3 전체 공개 데이터셋 1회 평가

J1/J2 다음 순서인 J3(유효 초기 경로 상한 가지치기)를 공개100지도×10쌍=1,000문제에서 평가했다.
9조건×1회=9,000회이며 기존 결과를 합산하지 않았다. 새 API 호출은0회다.

- [사전 고정 규약](PROTOCOL.md), [전체 결과](results/run_20261111/REPORT.md)
- [독립 검증](results/run_20261111/audit.json), [전체 집계](results/run_20261111/summary.json)
- [원시 경로·시간](results/run_20261111/evaluations.jsonl.gz), [문제별 CSV](results/run_20261111/per_case.csv)
- [동일 저장 응답](results/run_20261111/responses.jsonl.gz), [입력·소스·환경](results/run_20261111/protocol.json)

주 J3의 모든 경로가 최단이었으며 정확 탐색 조건7,000회 모두 독립 Dijkstra와 일치했다.
원 LLMA 대비 로컬 평균은2.681→4.369ms, 과거 API 지연을 합친 재구성 총평균은5.43864→5.44033초다.
동일2단계 무가지치기 대비로는 줄었지만, 단일 정확 탐색보다 빠르다는 근거는 얻지 못했다.

## 실행

Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
GPU·학습 가중치·API 키는 필요 없다. 저장소 루트에서 다음처럼 실행한다.

```powershell
python -m experiments.J3_full.run --output experiments/J3_full/results/new_run
python -m experiments.J3_full.report --input experiments/J3_full/results/new_run
python -m experiments.J3_full.audit --input experiments/J3_full/results/new_run
```

기존 output은 덮어쓰지 않는다. 저장 standard1,000응답의 요청·입력·SHA를 검증해 고정하고
실행 중 네트워크/API 연결을 차단한다. 보고서·검사 명령은 저장 결과만 읽으며 탐색을 재실행하지 않는다.

J3/J3_no_prune/J3_noLLM/J3_noLLM_no_prune과 Astar/LLMA/G2H/Octile/WallAstar를 함께 평가한다.
LLM이 직접 출력한 polyline을 바로 상한으로 쓰는 것이 아니라 원 LLM-A*의 합법 격자 경로를 seed로 얻는다.
실제 알고리즘은 [기존 JY 구현](../JY/search.py)이며 [adapter](adapter.py)는 방법명·측정/비용 필드만 연결한다.

준연 담당의 전체 재평가이며 Codex가 실행 통합·검증·문서 작성을 지원했다. 기존 J1/J2/JY 결과는 보존했다.
