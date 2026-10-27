# J3~J6 — 준연 방법군 비교 실험

개인 브랜치 `jy`, 2026-10-27. 동적 경유점 생성은 보류하고, 기존 비교 결과가 확인되지 않은
J3 초기 해 상한 가지치기, J4 ALT, J5 NBS·양방향 탐색, J6 OPEN 갱신을 구현·비교한다.
기존 `gm` 실험을 보존한 별도 Git worktree에서 시작했다.

- [실행 전 고정 규약](PROTOCOL.md)
- [구현](search.py), [실행기](run.py), [정확성 검사](test_search.py)
- [집계·사례 그림](report.py), [저장 기록 독립 검사](audit.py)
- [실제 결과와 해석](REPORT.md)
- [전체 비교표](results/run_20261027/analysis/TABLES.md)
- [환경·입력·소스 해시](results/run_20261027/manifest.json)
- [검증 결과](results/run_20261027/final_audit.json)

## 다시 실행하기

저장소 루트에서 실행한다. Python 3.10+, 탐색은 표준 라이브러리, 집계는 NumPy,
그림은 Matplotlib을 사용한다. `requirements.txt`를 따른다. GPU·API 키가 필요 없다.
기존 main 응답을 재사용하며 새 모델/API 호출 코드는 포함하지 않는다.

```powershell
python -m experiments.JY.test_search
python -m experiments.JY.test_additional
python -m experiments.JY.run --output experiments/JY/results/recheck_01
python -m experiments.JY.report --input experiments/JY/results/recheck_01
python -m experiments.JY.write_reports --input experiments/JY/results/recheck_01
python -m experiments.JY.audit --input experiments/JY/results/recheck_01
python -m experiments.JY.finalize --input experiments/JY/results/recheck_01
```

같은 output 경로가 있으면 runner는 중단한다. `--smoke`는 작은 실행 경로 확인용이며
전체 성능 근거가 아니다. `results/smoke_01`은 구현 초기 실행 경로 검증 기록이다.
`public/transfer/synthetic`별 `evaluations.jsonl.gz`에 매 반복의 경로·카운터·시간이 있다.
`preprocessing.jsonl.gz`는 ALT 지도별 실제 준비 비용, `reuse.json`은 준비 1회를 포함한
동일 지도 첫 Q개 질의 비용이다. `per_case.csv`는 3회 시간의 중앙값으로 집계했다.

기존 A*/LLM-A*/g+2h와 새 clean engine의 큐 처리가 다르므로 동일 구현이라고 부르지 않는다.
J3~J5는 최단 경로를 목표로 하며 가중 탐색과 효율을 비교할 때 길이 차이를 함께 본다.
J6의 full/lazy 두 조건은 동일 1e-9 정수 우선순위와 정적 LLM 경유점을 사용한다.
동적 재호출, 새 waypoint 생성, K6 적응 α 실험은 이 범위에 없다.

구현·분석은 사용자 지시에 따라 Codex가 지원했다. 방법 ID는 팀 분담을 따른다.
팀원 개인의 실제 기여, 제출용 `Proposed` 선정, 보고서 통합은 이 실험 완료만으로 확정하지 않는다.
