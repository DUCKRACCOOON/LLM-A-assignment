# J2 전체 공개 데이터셋 1회 평가

J1 다음 순서인 J2(Fast Marching 거리장 안내)를 공개 100지도×10쌍=1,000문제에서 평가한다.
각 문제·조건을1회만 실행하며 J1이나 COMMON18의 과거 측정값과 합산하지 않는다.

2026-11-11 완료: 1,000문제×7조건=7,000회, 전체 경로 검사 통과, 비용 상한 위반0건,
신규 API 호출0회. J2의 재구성 총평균은14.642초로 원 LLMA의5.439초보다 증가했다.
균일 FMM 대조군의 최단 일치율98.6%에 비해 J2는90.1%였다. 자세한 비용·품질 판단은 결과 보고서를 따른다.

- [사전 고정 규약](PROTOCOL.md), [전체 결과](results/run_20261111/REPORT.md)
- [독립 검증](results/run_20261111/audit.json), [전체 집계](results/run_20261111/summary.json)
- [원시 경로·시간](results/run_20261111/evaluations.jsonl.gz), [문제별 CSV](results/run_20261111/per_case.csv)
- [저장 응답3,000건](results/run_20261111/responses.jsonl.gz), [입력·소스·환경](results/run_20261111/protocol.json)

## 실행

Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
GPU·학습 모델 가중치·API 키는 필요 없다. 저장소 루트에서 다음처럼 실행한다.

```powershell
python -m experiments.J2_full.run --output experiments/J2_full/results/new_run
python -m experiments.J2_full.report --input experiments/J2_full/results/new_run
python -m experiments.J2_full.audit --input experiments/J2_full/results/new_run
```

기존 output은 덮어쓰지 않는다. 저장 응답 재생 중 네트워크/API 연결을 차단한다.
J2/J2_exact는 standard/CoT/RePE 세 응답, 원 LLMA는 standard만 사용한다.
논리 호출·토큰과 과거 지연은 사용하는 응답에만 부과하며 신규 API 호출은0회다.

7조건은 Astar/LLMA/G2H/FOCAL0/J2_uniform/J2/J2_exact다. J2/J2_uniform은 같은 w1.05의 직접 대조이고
J2_exact(w1)는 별도 품질 조건이다. 정확한 8방향 비용은 FMM 보조 거리장이 아니라 anchor 탐색으로 보장한다.

최초 J2 구현·실험은 승민이며 [출처](../COMMON18/SM_PROVENANCE.json)를 유지한다.
이번 전체 평가·검토는 준연 담당이며 Codex가 실행 통합·검증·분석 작성을 지원했다. 기존 결과는 보존한다.
