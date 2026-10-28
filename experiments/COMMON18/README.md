# 18개 방법군 동일 조건 비교

2026-10-28, jy. 서로 다른 브랜치의 옛 비율을 한 표에 섞는 대신, 같은 입력·호스트·온라인
모델·수집 응답·측정 정책으로 모두 다시 실행했다. 동적 경유점 생성은 보류한다.

- [실행 전 규약](PROTOCOL.md), [본평가 전 구현 검토·수정](IMPLEMENTATION_NOTES.md)
- [결과 보고서](results/run_20261028/REPORT.md), [전체 수치](results/run_20261028/summary.json)
- [독립 검증](results/run_20261028/audit.json), [실제 요청·응답](results/run_20261028/generations.jsonl)
- [수집·분할·preflight](prepare.py), [공통 어댑터](backends.py), [타이밍](benchmark.py)
- [격리 메모리 측정](memory.py), [집계](report.py), [독립 audit](audit.py), [회귀 검사](verify.py)
- [SM 구현 출처·hash](SM_PROVENANCE.json), [가져온 원본 사본](vendor_sm/original_candidate_core.py.txt)

18개 primary, 같은 구조의 안내 제거 대조와 정확 변형을 합쳐 최종 45조건이다.
K5 계열 3조건의 확률 시드3개와 시간3회로 평가한다. 고유 문제는 200개이며 시드/반복을
독립 문제 수에 더하지 않는다. 상세한 완료 여부는 benchmark_complete.json/audit.json을 따른다.

200문제 × 시간3회 및 확률 조건3시드, 총 **30,600회**를 완료했고 독립 경로 검사와
24,600회 비용 상한 검사, 495개 집계값 재계산을 통과했다. 신규 API 요청은 1,567회다.
수치상 최속 LLM 조건은 K1_exact(0.831초, 최단 보장)이지만 규칙 대조와 200문제 모두
같은 순위·경로·확장 수를 보였다. 원 LLM-A*(6.551초) 대비 지연 절감이 LLM 판단의
우수성을 입증하지는 않는다. 매 문제 standard 응답을 요구하면 개선은 사실상 없다.

## 환경과 재현

Python 3.12, NumPy/SciPy/Shapely/PyTorch CPU/psutil/Matplotlib. 실제 버전은 실행 산출물에
기록한다. 저장 모델은 기존 SM seed11이며 원 train/validation/calibration/test 분할이
그대로 일치하는지 확인했다. CPU/native 스레드는1이다. GPU 서버와 시간 수치를 섞지 않는다.
설치된 선택 의존성을 local_private/*_deps에서 발견하면 해당 디렉터리를 사용한다.
이 디렉터리와 자격증명은 Git에서 제외된다.

```powershell
python -m experiments.COMMON18.verify
python -m experiments.COMMON18.prepare --output experiments/COMMON18/results/new_run
# 다음 명령은 신규 유료 API 요청을 만든다. 규약의 최대1600회 예산을 확인한다.
python -m experiments.COMMON18.prepare --output experiments/COMMON18/results/new_run --collect
# 이하 실행은 위에서 수집한 같은 응답을 재생하며 API를 호출하지 않는다.
python -m experiments.COMMON18.benchmark --input experiments/COMMON18/results/new_run
python -m experiments.COMMON18.memory --input experiments/COMMON18/results/new_run
python -m experiments.COMMON18.report --input experiments/COMMON18/results/new_run
python -m experiments.COMMON18.audit --input experiments/COMMON18/results/new_run
```

기존 raw 응답이 있는 같은 output에서는 수집 시 SHA가 일치하는 요청만 재사용한다.
API 오류는 기록 후 중단하며 자동으로 응답을 재생성하지 않는다. 완료된 timed output은
덮어쓰지 않는다. 중간에 중단된 timing은 별도 output에서 같은 응답을 보존해 재실행한다.

## 해석할 때 지킬 점

총시간은 실제 기록한 같은 수집의 API 대기와 현재 로컬 계산의 합인 **재구성 지연**이다.
모든 방법을 실제 서비스에서 동시에 3회씩 호출한 시간은 아니다. 기본 응답을 공유하는
방법은 똑같은 지연/토큰을 부과하고, 추가 응답은 그대로 더한다. 조건부 호출은 호출한
문제에만 부과한다. 여러 프롬프트는 순차 운영을 가정한다. 학습·보정·model load/warm-up은 별도다.

품질 제한이 없는 방법의 속도와 최단성 보장 방법을 같은 품질에서의 승패로 표현하지 않는다.
최단성은 원리와 구현 조건으로 설명하고 전체 경로/Dijkstra로 구현을 검사한다.
입력 공개 자료는 연구 전체의 hidden holdout이 아니며, 이 연구에서 독립 일반화 성공을 주장하지 않는다.

전체 RSS는 별도 같은10문제의 새 프로세스 표본 peak다. 하네스·라이브러리·모델 상주분을
포함하고 짧은 순간 peak를 놓칠 수 있다. 저장 상태/heap pop은 실제 bytes/FLOPs가 아니다.

사용자 요청에 따라 Codex가 통합 구현·실행·검증·분석을 지원했다. 기존 gm/sm 코드의
출처를 유지하며 팀원별 사람 기여를 임의로 추정하지 않는다. main이나 제출용 Proposed를
자동 변경하지 않는다. 이전 jy 실험·원 gm 폴더의 동적 코드/결과도 보존한다.
