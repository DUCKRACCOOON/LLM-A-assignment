# S1~S6 공통 재실험

실험 계획과 원자료는 [결과 폴더](../../results/s1_s6_gm_20261028/plan.md)에 있다.

- 실행 환경: 프로젝트 루트의 `.venv-gm311/Scripts/python.exe`, Python 3.11.18.
- 원본 GM 코드·입력은 `vendor/gm`의 고정 snapshot이다. 원본 비교군의 구현은 수정하지 않는다.
- S4~S6의 기존 SM 코드는 루트 `reproduction/`에서 가져오고 실행 직전 해시를 고정한다.
- 온라인 수집과 학습이 끝난 뒤에만 본 시간 벤치마크를 수행한다.
- `run_pipeline.py`는 진행 중인 전체 수집의 완료를 기다린 뒤 평가·검증·Markdown 집계를 순차 실행한다.
- `collect.py collect`와 `programs.py generate`는 응답 캐시가 없을 때 실제 API를 호출한다. 나머지 단계는 새 API를 호출하지 않는다.
- 인증 정보는 기존 로컬 파일에서 메모리로만 읽고 공식 OpenAI API에만 전달한다. 결과에 키를 저장하지 않는다.

실험 절차:

```powershell
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/common.py
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/preflight.py
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/collect.py probe
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/collect.py collect --stage pilot --workers 6
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/train.py
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/evaluate.py oracle
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/programs.py generate
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/programs.py select
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/collect.py collect --stage all --workers 24
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/evaluate.py all
.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/report.py all
```

기존 출력이 있으면 요청·코드·설정 해시를 확인해 재개한다. 코드나 조건이 달라지면 기존 실행의 결과를 섞지 말고 새 실행 폴더를 사용한다. 파일럿은 기능 확인용 1회이며 본 결과의 5회 시간 반복에 합산하지 않는다.
