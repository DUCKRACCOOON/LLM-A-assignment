# J1부터 순서대로 전체 데이터셋 평가

사용자 담당 J1~J6 중 첫 번째 J1(CHOMP-inspired 경로 수리)을 공개 전체 1,000문제에서 평가한다.
각 조건은 문제당1회만 실행한다. 기존 COMMON18/SM 결과와 합산하지 않는다.

2026-11-11 완료: 1,000문제×7조건=7,000회, 전체 경로 검사 통과, 비용 상한 위반0건,
새 API 호출0회. J1은 원 LLM-A*보다 경로 품질은 개선했지만 로컬 평균은48.84배,
재구성 총평균은2.45% 증가했다. 수리 성공은124/1,000건이며 상세 판단은 결과 보고서를 따른다.

- [사전 고정 규약](PROTOCOL.md)
- [J1 전체 결과](results/run_20261111/REPORT.md)
- [독립 검사](results/run_20261111/audit.json), [전체 집계](results/run_20261111/summary.json)
- [원시 경로·시간](results/run_20261111/evaluations.jsonl.gz), [동일 저장 응답](results/run_20261111/responses.jsonl.gz)
- [입력·소스·환경 기록](results/run_20261111/protocol.json), [문제별 CSV](results/run_20261111/per_case.csv)

## 실행

저장소 루트에서 Python 3.12와 [COMMON18 선택 의존성](../COMMON18/requirements-common.txt)을 사용한다.
로컬 의존성 디렉터리 발견·native1스레드 설정도 COMMON18과 같다. GPU·모델 가중치·API 키는 필요 없다.

```powershell
python -m experiments.J1_full.run --output experiments/J1_full/results/new_run
python -m experiments.J1_full.report --input experiments/J1_full/results/new_run
python -m experiments.J1_full.audit --input experiments/J1_full/results/new_run
```

기존 output은 덮어쓰지 않는다. API 코드를 새로 호출하지 않고 results/main의 검증된 응답을 복사해 재생한다.
보고서·검사 명령은 저장 원자료만 읽으며 검색을 다시 실행하지 않는다.

7조건은 Astar/LLMA/G2H/FOCAL0/J1_straight/J1/J1_exact다. J1의 같은 구조 대조는
J1_straight이며, exact는 별도 품질 조건이다. 원 CHOMP 논문 전체의 재현이나 동적 경유점 생성은 아니다.

최초 CHOMP-inspired 구현·실험은 승민이며 코드 출처와 COMMON18 수정은
[원본 출처](../COMMON18/SM_PROVENANCE.json), [수정 기록](../COMMON18/IMPLEMENTATION_NOTES.md)에 있다.
이번 전체 평가·검토는 준연 담당이며 Codex가 구현 통합·실행·검증·분석 작성을 지원했다.
J2~J6의 새 평가를 이 패키지에서 실행하지 않는다.
