# J4 전체 데이터셋 1회 평가

J3 다음 J4(ALT 랜드마크 휴리스틱)를 공개100지도×10쌍=1,000문제에서 평가했다.
8조건×1회=8,000회이며 실제 전처리3,000회를 포함한다. 신규 API 호출은0회다.

- [사전 고정 규약](PROTOCOL.md), [전체 결과](results/run_20261111/REPORT.md)
- [독립 검증](results/run_20261111/audit.json), [전체 집계·재사용 시나리오](results/run_20261111/summary.json)
- [원시 경로·시간](results/run_20261111/evaluations.jsonl.gz), [문제별 CSV](results/run_20261111/per_case.csv)
- [입력·소스·환경](results/run_20261111/protocol.json), [LLMA 대조 응답](results/run_20261111/responses.jsonl.gz)

정확 조건6,000회 모두 독립 최단거리와 일치했다. ALT4의 전처리 포함 평균은31.877ms로
단일 clean octile A*의3.627ms보다 높았다. 같은 지도10질의·전처리1회 합산 추정도
비용합 비율1.0364로 더 높았다. 준비 후 질의만의 절감과 전체 비용을 구분한다.

## 실행

Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
GPU·학습 가중치·API 키는 필요 없다. 저장소 루트에서 실행한다.

```powershell
python -m experiments.J4_full.run --output experiments/J4_full/results/new_run
python -m experiments.J4_full.report --input experiments/J4_full/results/new_run
python -m experiments.J4_full.audit --input experiments/J4_full/results/new_run
```

기존 output은 덮어쓰지 않는다. 원 LLMA 대조의 standard1,000응답만 저장 재생하며
J4/ALT8/GraphOctile는온라인LLM을 쓰지 않는다. 실행 중 네트워크/API 연결을 차단한다.

전처리는 각 질의에서 새로 실행했다. 재사용 결과는 첫 전처리1회+첫Q질의 준비 후 비용의
합산 추정이며, 같은 index를 상주시킨 연속 실행을 추가한 결과는 아니다. Q는1/5/10으로 고정했다.

실제 검색·전처리 코드는 [기존 JY 구현](../JY/search.py)이며 [adapter](adapter.py)는 준비·검색 비용을 구분한다.
준연 담당 J4 평가에 Codex가 실행 통합·독립 검증·분석 작성을 지원했다. 이전 결과는 보존했다.
