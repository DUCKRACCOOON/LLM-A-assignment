# K1~K6 통합 평가 규칙 재평가 (2026-11-12, gm)

SM 브랜치의 통합 평가 규칙(`origin/sm` `0c0a6082`, `sm_part/cv1000_20261112`의 [실험 계획](inputs/plan.md))을 그대로 따라 규민 담당 K1~K6를 다시 평가했다.
결과는 [report.md](results/run_20261112/report.md), 독립 검산은 [verification.json](results/run_20261112/verification.json)에 있다.

## 따른 규칙

- 공개 1,000문제(100지도 × 10쌍), SM의 지도 단위 5-fold 분할([fold_manifest](inputs/fold_manifest.json))을 그대로 사용. 각 문제는 자기 지도를 학습·보정에 쓰지 않은 fold에서 한 번만 평가(OOF).
- 주 분모는 원 LLM-A*(`B_llm`). A*, g+2h, A*_octile, 무안내 FOCAL 1.05를 공통 대조군으로 같이 측정.
- 공개 지도와 겹치지 않는 예시 지도 5개로 만든 SM 프롬프트. standard 응답은 SM이 수집한 새 응답 1,000건을 그대로 공유([inputs/raw/public](inputs/raw/public)).
- 문제마다 22조건 × 5회, 반복마다 조건 순서를 고정 난수로 섞음. 문제별 시간은 5회 중앙값. 결정적 지표는 반복 간 일치 강제.
- 측정 중 네트워크 차단, CPU 1스레드, 측정 전 코드·프롬프트·응답·보정값 해시 고정([frozen.json](results/run_20261112/frozen.json)).
- 측정 래퍼는 SM `algorithms.run`과 같은 정책(heapq pop 계수, 최종 격자 검증까지 시간 포함). 기준 7조건의 결정적 지표가 SM 기록과 일치하는지 audit에서 대조.

## 조건 (22)

| 구분 | 조건 | 비용 상한 | 내용 |
| --- | --- | ---: | --- |
| 공통 | A_GM, A_OCT | 1 | 정확 A* (GM 엔진 / SM octile 엔진) |
| 공통 | N_g2h, B_llm | 없음 | g+2h, 원 LLM-A*(주 분모) |
| 공통 | FOCAL_105 | 1.05 | 무안내 FOCAL (K2의 직접 대조) |
| K1 | K1_llm, K1_rule, K1_random | 1.05 | 통로 후보 순위(LLM / 규칙 / 무작위), SM topology + FOCAL |
| K2 | K2_conformal, K2_nocal | 1.05 | standard·CoT·RePE 3응답 거리장, fold 내부 800문제 conformal 보정(α=0.1) / 보정 없음 |
| K3 | K3_wall_euclidean, K3_wall_octile | 없음 | 벽 하한 g+2h (최종 선정안은 euclidean) |
| K4 | K4_llm, K4_noLLM | 1.05 | gm K4 FOCAL 주 설정 / 같은 엔진 LLM 제거 |
| K5 | K5_dts_s0~s2, K5_rr, K5_noLLM | 2 | gm K5 MHA*+DTS 3시드 / 고정 순환 / LLM 제거 |
| K6 | K6_main | 2 | gm K6 적응 α 주 설정 |
| K6 | K6_w105, K6_w105_noguide | 1.05 | 5% 제한 적응 α / 같은 엔진 무안내 |

K1·K2의 새 응답(K1 순위 823건, K2 CoT·RePE 각 1,000건, 실패 0, 약 0.92달러)은 [collect_k.py](collect_k.py)로 수집했다. 후보가 2개 미만인 177문제는 K1 순위 요청을 보내지 않았다.

## 규칙과 다른 점

- Python 3.11.9(공식 서명 설치본)를 사용했다. SM은 3.11.18이다. 이 PC의 Windows 앱 제어 정책이 서명 없는 3.11.18 DLL을 차단했다. NumPy·SciPy·Shapely·PyTorch 버전은 SM과 같다.
- CoT·RePE 예시는 원 템플릿과 같은 위치(1·2·5번째)의 SM 새 예시 지도를 쓰고, 예시의 추론 문장은 기하에서 규칙으로 생성했다(원 예시는 손으로 쓴 문장). [provenance](results/run_20261112/prompts/provenance.json).
- K2 보정은 학습 단계가 없으므로 fold 내부 train+validation 80지도(800문제)를 보정 자료로 썼다. 평가 지도는 쓰지 않았다.

## 실행

```powershell
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\evaluate.py pilot
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\evaluate.py freeze
# CV_FOLD=0..4 각각
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\evaluate.py evaluate
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\memory.py
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\report.py
.venv-gm311\Scripts\python.exe experiments\UNIFIED_K\audit.py
```

`collect_k.py`는 유료 API를 호출한다. 저장 응답으로 측정·검산만 재현할 때는 필요 없다.
SM에서 가져온 코드·입력의 해시는 [SM_PROVENANCE.json](SM_PROVENANCE.json)에 있다. 원 LLM-A* 코드는 MIT 라이선스([vendor](vendor/llm-astar/LICENSE)).
구현·실행·문서 정리에 Claude Code(에이전트 병렬 실행)의 지원을 받았다.
