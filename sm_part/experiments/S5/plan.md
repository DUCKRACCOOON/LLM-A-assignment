# S5 계획서 · 정확한 직사각형 공간 압축

## 원 계획과 구현 범위

기존 [사전 계획 R1](../../results/next_experiments_plan/plan.md)과 [고정 프로토콜](../../results/next_search_v1/protocol_frozen.json)을 S5에 대응시켰다. 새 실험 계획을 사후에 사전등록한 문서가 아니다. 가설은 빈 영역의 내부 정점을 생략하되 합법적인 8방향 격자 이동의 최적 비용을 보존하면 탐색·전체 시간 비용을 줄일 수 있다는 것이다.

1. 원래 선분 장애물 규칙 아래 내부 이동이 유효한 직사각형을 인증한다.
2. `full`은 경계 정점을 유지하고 영역 내부 통과를 정확한 octile 비용의 macro edge로 연결한다.
3. `sparse`는 같은 변 경계 이동, 마주 보는 변의 fan, 이웃 변 대각 연결 등으로 필요한 경로를 표현한다.
4. 내부 시작·목표를 임시로 삽입하고 경로를 같은 길이의 원래 단위 격자 이동으로 복원한다.
5. 인증을 적용하지 않는 길이 0 벽 등에는 원래 격자 탐색으로 fallback한다.

압축 경로가 같은 비용으로 복원되고 원래 경로의 영역 통과를 더 비싸지 않게 대체할 수 있어야 한다. 최적성 보존 논증과 독립 Dijkstra 대조를 함께 요구한다. 구체 구현은 [next_geometry.py](../../reproduction/next_geometry.py)와 [next_search_core.py](../../reproduction/next_search_core.py)다. JPS·Subgoal Graphs의 모든 경로 보존·성능을 대신 검증한 것으로 해석하지 않는다.

## 비교·데이터·채택 기준

개발에서 full/sparse를 비교해 한 변형을 고정한다. `R1_ADAPTIVE`는 원 계획의 조건부 확장으로 실제 실행 범위와 구분한다. 대조군은 `A_OCT`, 기존 g+2h의 `N_LEGACY`, 새 탐색기 기반 `N_CLEAN`, WA 1.01/1.02/2.00, FOCAL 1.01/1.02다. S6 결합의 추가 가치는 `R1_R2/R1`로 별도 판단한다.

| 자료 | 고정 규모·seed |
| --- | --- |
| 개발 | 4유형×15지도×10질의=600문제, 26101410 |
| 검증 | 4유형×15지도×10질의=600문제, 26101411 |
| 최종 | 4유형×50지도×10질의=2,000문제, 26101412 |
| 확대 | 최종 40지도×2질의×배율 1/2/4/8=320조건 |
| 재사용 | 최종 20지도, 목표 고정/변경, Q=1/5/10 |

네 유형은 원본 유사 선분, 조밀한 선분, 복도·막다른 길, 여러 우회다. 지도 분할은 기하 중복 없이 나누고 최종 자료는 소스·선택 설정 고정 뒤 생성한다. 주 결과에 원 LLM-A* 좌표 응답 직접 비교는 포함하지 않는다.

최단 길이 일치 100%·무효 0건이 필수다. 단일 질의 목표는 A_OCT 대비 전체 시간 ≤0.90, 확장 수 ≤0.80이며 시간 구간 상한 1 미만을 요구한다. 실패하면 확대·반복 질의의 조건부 가능성을 진단하며 일반 채택 성공으로 바꾸지 않는다.

## 비용·검증·실행

영역 분할·쿼리 삽입·macro edge 검사·fan 스캔·전체 탐색·복원·fallback을 모두 포함한다. 확장 수 한 번의 비용은 원 격자와 다를 수 있어 heap pop·relax·이웃 수·RSS·전체 시간을 함께 본다. 동일 조건을 5회 측정해 중앙값을 쓰고 지도 단위 층화 bootstrap 10,000회로 구간을 계산한다. 일반 95%는 탐색적, 지정 비교 98.75%는 최대 4개 비교를 고려한다.

독립 선분 그래프/SciPy Dijkstra, 작은 격자 2,000질의, 도달 불가능 100질의 및 경계 사례를 검사한다. 실행 환경은 [environment.json](../../results/next_search_v1/environment.json)을 따른다. 탐색은 단일 CPU 흐름이며 GPU는 공통 묶음의 Llama 생성에 사용됐다.

원자료 복원 후 `sm_part`에서 집계할 수 있다.

```text
python reproduction/next_run.py report
python reproduction/analyze_three_experiments.py
python reproduction/write_three_experiment_report.py
```

원 서버 전체 실행 명령은 `/home/work/fanimal_hidream/llama-reproduction/.venv-llama/bin/python reproduction/next_run.py all`이었다. 고정 서버 모델 경로가 있어 임의 PC에서 그대로 실행 가능한 이식형 명령으로 보장하지 않는다. 이번 정리에서는 새 실행을 하지 않았다.
