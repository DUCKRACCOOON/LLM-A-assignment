# J6 전체 데이터셋 1회 평가

마지막 J6(OPEN 우선순위 갱신 최적화)를 공개100지도×10쌍=1,000문제에서 평가했다.
7조건×1회=7,000회이며 신규 API 호출은0회다. 기존 코드·결과를 보존했다.

- [사전 규약](PROTOCOL.md), [전체 결과](results/run_20261111/REPORT.md)
- [독립 검산](results/run_20261111/audit.json), [전체·부분집합 집계](results/run_20261111/summary.json)
- [유효 선택열 포함 원시 시행](results/run_20261111/evaluations.jsonl.gz), [문제별 CSV](results/run_20261111/per_case.csv)
- [강제 target 목록 검사](results/run_20261111/unit_tests.json), [입력·소스·환경](results/run_20261111/protocol.json)

full/lazy의 경로·유효 선택열·확장은1,000쌍 모두 같았다. score 평가는543,034→422,010회로22.29% 감소했고
lazy refresh pop은80,601회 추가됐다. 로컬 평균은2.661→2.637ms로 약0.88% 감소했다.
시간1회·동일 trace append 계측을 포함한 결과이므로 반복 안정성이나 미계측 production의 큰 개선으로 주장하지 않는다.
J6 계열은 최단성 보장이 없다. API 포함 총시간은 과거 지연과 현재 로컬의 재구성 추정이다.

## 실행

Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
GPU·학습 가중치·API 키는 필요 없다. 저장소 루트에서 실행한다.

```powershell
python -m experiments.J6_full.run --output experiments/J6_full/results/new_run
python -m experiments.J6_full.report --input experiments/J6_full/results/new_run
python -m experiments.J6_full.audit --input experiments/J6_full/results/new_run
```

기존 output은 덮어쓰지 않는다. LLMA/full/lazy는같은 저장 standard 응답을 각 논리1회 사용하고,
실행 중 네트워크/API 연결을 차단한다. 원시 응답과 trace를 보존하며 보고서·검사는 검색을 다시 실행하지 않는다.

검색 코드는 [기존 JY 구현](../JY/search.py)이며 [adapter](adapter.py)는 full/lazy 모두 trace=True로 실행한다.
해시 계산·선택열 대조는 타이머 밖이다. 원 LLMA와 clean full/lazy의 엔진 차이는 갱신 최적화 효과와 구분한다.
부분집합 bootstrap은 지도별 사례 수가 달라도 사례 가중을 유지하도록 지도별 log합과 사례 수를 함께 재표집한다.

준연 담당 J6 평가에 Codex가 실행 통합·검증·분석 작성을 지원했다. [담당6개 전체 평가 목록](../FULL_PUBLIC_20261111.md)을 참고한다.
