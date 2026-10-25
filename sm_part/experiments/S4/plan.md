# S4 계획서 · 학습한 경로 포함 가능성으로 FOCAL 안내

## 범위와 가설

학습 모델의 경로 포함 확률이 동일한 FOCAL 탐색기의 노드 확장과 준비 포함 시간을 줄이는지 평가한다. [고정 프로토콜](../../results/candidates_v1/protocol.json)과 실제 구현을 정리한 문서다. 모델 구조·학습 자료·선택 과정이 공식 TransPath와 같지 않으므로 **TransPath-inspired 소형 모델 실험**으로 한정한다.

## 구현 및 학습

[candidate_core.py](../../reproduction/candidate_core.py)의 `network_inputs`, `make_network`, `focal`과 [candidate_run.py](../../reproduction/candidate_run.py)가 기준이다.

| 항목 | 실제 고정 설정 |
| --- | --- |
| 입력 7채널 | 장애물, 시작점, 목표점, 정규화 x·y, 시작점·목표점까지 octile 거리 |
| 모델 | 합성곱 인코더 24/48/64채널, Transformer 2층·4 heads·FFN 128, 합성곱 디코더 |
| 파라미터 | 160,297개 |
| 정답 영역 | 정확 거리 `d(s,x)+d(x,g) ≤ L*+0.5`를 만족하는 격자 칸 |
| 최적화 | AdamW, lr 0.001, weight_decay 0.0001, batch 32, 60 epochs |
| 손실 | 자유 칸 mask와 양성 가중치를 사용한 BCEWithLogitsLoss |
| seed | 11·22·33 |
| 선택 | 검증 BCE가 가장 낮은 checkpoint; test로 선택하지 않음 |

같은 원본 지도의 10개 질의를 함께 배정한다. seed 261014로 **학습 50지도·500문제 / 검증 10지도·100문제 / 보정 20지도·200문제 / 평가 20지도·200문제**로 분할한다. 보정 집합은 공통 실행의 Conformal용이며 S4의 학습에 쓰지 않는다. 프롬프트 예시 지도 0·2·3·4·5는 학습에만 둔다. 지도 분할 목록은 프로토콜에 모두 있다. 원본 자료는 앞선 연구에서 관찰한 것이어서 연구 전체의 완전한 미관측 평가라고 부르지 않는다.

## 평가·대조군·비용

주 대조군은 같은 그래프·같은 `w=1.1`의 `FOCAL_EUCLIDEAN`이다. admissible anchor는 octile이고 예측은 FOCAL의 보조 순위로 사용한다. 학습하지 않은 `UNTRAINED_11`, A*, Weighted A*(1.1)를 함께 비교한다. S4는 온라인 GPT 응답을 필요로 하지 않는다.

평가용 200문제 외에 새 교대 벽 미로 20지도×5질의=100문제(seed 262026), 평가 지도 중 10개의 첫 문제를 2·4배로 확대한 20조건을 평가한다. 지도 유효성·길이 상한 `L≤1.1L*`, 실제 확장 수, 발견 노드 수·peak OPEN, 준비·GPU 추론·탐색·그래프 생성 비용을 기록한다.

대응 비율은 지도 내 로그 평균 후 지도 단위 bootstrap 2,000회(seed 2026)의 탐색적 95% 구간을 사용한다. 독립된 신경망 seed를 지도 표본 수로 추가하지 않는다. 시간은 공유 서버에서 문제별 한 번 측정했고 모델은 상주·warm-up 완료 상태다. 전송·CUDA 동기화는 포함하며 학습·모델 로딩·warm-up·정답 생성은 별도 비용이다.

## 재현 경로

Python 3.12.3, NumPy 1.26.4, PyTorch 2.7.0a0+7c8ec84dab.nv25.03, CUDA 12.8, RTX A5000의 [환경 기록](../../results/candidates_v1/environment.json)을 기준으로 한다. `sm_part`에서 자료를 복원한 뒤 기존 결과 검증 CLI는 다음과 같다.

```text
python reproduction/candidate_run.py verify
```

전체 학습·평가 CLI는 `python reproduction/candidate_run.py all --gpu 0`이다. 이는 공통 후보 묶음 전체를 실행하고 검증된 S1~S3 최초 응답 캐시도 필요로 한다. 기존 결과를 유지한 별도 복사본과 동일한 CUDA 환경에서 실행한다. 이번 문서 정리에서는 실행하지 않았다. 현재 분담의 실제 모델·분할 공개는 충족하며 공식 모델 재현이나 반복 시간 벤치마크는 별도 작업이다.
