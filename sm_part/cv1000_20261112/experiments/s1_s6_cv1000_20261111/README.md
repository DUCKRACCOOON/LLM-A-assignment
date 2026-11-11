# S1~S6 지도 단위 5-fold 재실험

계획: `results/s1_s6_cv1000_20261111/plan.md`.
출력: `results/s1_s6_cv1000_20261111/runs/run_20261111_01/`.
기존 2026-10-28 코드와 결과는 수정하지 않는다.

Python 3.11 환경에서 프로젝트 루트를 작업 디렉터리로 사용한다.

```powershell
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/prepare.py
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/preflight.py
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/train_all.py
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/collect_all.py --programs-only
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/offline_prepare.py
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/evaluate.py oracle
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/collect_all.py
.venv-gm311/Scripts/python.exe experiments/s1_s6_cv1000_20261111/finish.py
```

`collect_all.py`는 실제 유료 API를 호출한다. 동일 요청 해시의 성공 응답은 재사용한다.
원문 무효·길이 제한 종료 응답도 보존하며 자동 전송 재시도는 없다.
공개 1,000문제 최초 응답과 무효 제안별 control/feedback 응답을 공유한다.
S6는 fold별 12개 새 후보를 생성한다. 키 내용은 결과와 로그에 기록하지 않는다.

각 하위 실행은 `CV_FOLD=0..4`를 읽는다. 네트워크는 평가 중 코드에서 차단한다.
S4 학습 통계와 checkpoint는 fold 내부 train/validation에서만 계산한다.
모든 fold 모델·계수·프롬프트·응답·실행 코드 해시를 고정한 다음 외부 평가를 시작한다.

22조건은 원 LLM-A*, A*, g+2h, S1~S6 및 직접 대조군이다.
모든 조건은 같은 Python/장비에서 문제마다 5회 순서 무작위화 반복한다.
S4_mean은 세 시드 평균 집계이며 23번째 실행 조건이 아니다.

최종 `report.md`는 독립 정수 기하/Dijkstra 경로 검증, 입력·응답·분할·해시 대조,
110,000회 결과 행 확인 및 별도 RSS 표본 1,320회가 완료된 후 작성한다.
총응답시간은 실제 수집 API 지연과 오프라인 계획기 시간을 합한 재구성 추정이다.
이번 공개 데이터는 과거 연구 노출이 있어 완전히 미관측인 hidden 평가로 부르지 않는다.
