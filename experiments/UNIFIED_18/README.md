# 18개 방법 단일 블록 비교 (2026-11-12, gm)

SM·JY·GM 세 명이 테스트한 18개 방법(S1~S6, J1~J6, K1~K6)과 A*·원 LLM-A*를 **한 PC, 한 측정 블록**에서 같은 규칙으로 측정하고, 원 LLM-A* 대비 개선·악화를 7개 평가 항목으로 정리했다. 정규화·가중치·종합 점수는 아직 적용하지 않았다(팀 결정 대기).

- 결과 표: [report18.md](results/run_20261112/report18.md)
- 시각화: [tradeoff18.html](results/run_20261112/tradeoff18.html) (브라우저로 열기)
- 문제별 원자료: [per_case.csv](results/run_20261112/per_case.csv), 요약: [summary18.json](results/run_20261112/summary18.json), 검산: [verification18.json](results/run_20261112/verification18.json)

## 규칙

SM 통합 규칙(`origin/sm` `0c0a6082`)을 [UNIFIED_K](../UNIFIED_K/README.md)와 같은 방식으로 따른다.
- 공개 1,000문제(100지도 × 10쌍, 논문과 같은 데이터·샘플 수), SM 지도 단위 5-fold. 학습·보정이 필요한 방법(S4·S6·K2)은 자기 지도를 쓰지 않은 fold 산출물로만 평가(OOF).
- 문제마다 51조건 × 5회, 반복마다 조건 순서를 고정 난수로 섞음. 시간은 5회 중앙값. 결정적 필드는 5회 일치를 강제. 측정 중 네트워크 차단, 1스레드, 측정 전 코드·입력 해시 고정([frozen.json](results/run_20261112/frozen.json)).
- LLM: gpt-4o-mini-2024-07-18(과제 지정 주 모델). 측정은 저장된 응답 재생이며 신규 API 호출은 없다.
- 18개 방법의 분모는 같은 standard 응답을 공유하는 원 LLM-A*(`B_llm`, SM 통합 프롬프트). Table 1 형식 점수표에는 논문 원 프롬프트 응답의 LLM-A*(`B_llm_paper`)도 같은 블록에서 함께 측정했다.
- 통계: 문제별 비율의 기하평균, fold 내 지도 단위 bootstrap 2,000회 95% 구간, 기하 표준편차, 원값 평균 ± 표준편차, 5회 시간 변동계수(과제 공지의 신뢰구간·표준편차 요구).

## 과제 안내서와 맞춘 점

| 안내서 요구 | 이번 처리 |
| --- | --- |
| Table 1은 논문과 같은 실험 설정·샘플 수 | 1,000문제, A* 분모, 문제별 비율 기하평균, operation = 주 탐색 OPEN pop, storage = len(g) ([TABLE1_SPEC.md](TABLE1_SPEC.md)) |
| A*·LLM-A*와 공정 비교 | 18개 방법·A*·LLM-A*를 한 블록에서 교차 순서로 측정 |
| 신뢰구간·표준편차 보고 | 모든 비율에 95% 구간·기하SD, 원값에 ±SD |
| GPT-4o mini 주 모델 | 모든 LLM 응답이 gpt-4o-mini-2024-07-18 |
| 공개 지도 전용 조정 금지 | 학습·선택은 fold 내부 지도만 사용(OOF) |
| 실행 시간 제한 가능성 | 계획기 시간과 API 포함 총응답시간을 함께 보고 |

## 조건 이식과 동등성

- S1~S6: SM 코드·fold 산출물 그대로([S_PORT.md](S_PORT.md)). SM 기록과 결정 필드 일치(S4만 4,000회 중 14회 확장 수 1~3 차이, 경로·길이 동일; CPU 부동소수 차이).
- J1~J6: JY 코드 그대로([J_PORT.md](J_PORT.md)). JY 원 응답 입력 시 JY 기록과 1,000/1,000 일치.
- K1~K6, 공통 기준: UNIFIED_K 그대로. 이번 블록의 결정 필드가 UNIFIED_K 실행과 일치([verification18.json](results/run_20261112/verification18.json)).

## 실행

```powershell
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\evaluate18.py pilot
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\evaluate18.py freeze
# fold 0..4 각각 (이번에는 5개 프로세스 병렬)
$env:CV_FOLD=0; .venv-gm311\Scripts\python.exe experiments\UNIFIED_18\evaluate18.py evaluate
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\audit18.py
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\report18.py
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\report18_md.py
.venv-gm311\Scripts\python.exe experiments\UNIFIED_18\viz18.py
```

`test_s.py`·`test_j.py`는 이식 동등성 검사이며 `origin/sm`·`origin/jy` 커밋이 로컬에 있어야 한다.

## 주의

- 5개 fold를 병렬 프로세스로 실행했다. 같은 문제 안에서 모든 조건이 섞여 실행되므로 비율은 공정하지만, 절대 ms는 단독 실행보다 클 수 있다.
- heap pop은 엔진마다 의미가 달라(배열 스캔 FOCAL, 보조 탐색 등) 엔진 간에는 이웃 확장 수를 우선 본다. 논문 정의 operation/storage는 엔진이 분리할 수 있는 조건만 표시했다.
- 총응답시간은 저장된 실제 API 대기 + 계획기 시간의 재구성 추정이다. 프로세스 RSS는 이 블록에서 측정하지 않았다.
- 공개 자료는 과거 연구에서 이미 노출됐다. TA의 hidden 평가 결과를 대신하지 않는다.
- 이식·측정·문서 작성에 Claude Code(에이전트)의 지원을 받았다. 각 방법의 원 구현은 담당자(SM·JY·GM)의 것이다.
