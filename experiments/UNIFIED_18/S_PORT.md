# SM S1~S6 이식 (UNIFIED_18, 2026-11-12)

SM의 22조건 통합 실행(`origin/sm` `0c0a6082`, `sm_part/cv1000_20261112`)에서 공통 기준 7조건을 뺀 S 조건 15개를 UNIFIED_K harness 규약으로 호출할 수 있게 했다. 재학습·재생성은 없다.

## 복사한 산출물 (`inputs_s/`, 2,054파일)

SM run 폴더(`results/s1_s6_cv1000_20261111/runs/run_20261111_01`)와 같은 상대 경로. 원 경로·sha256·커밋은 [PROVENANCE.json](inputs_s/PROVENANCE.json). 2,032개는 SM `frozen.json`의 해시와 대조해 일치했다.

- S4: `folds/fold{0..4}/models/seed{11,22,33}.pt` + `.json`, `training_data.json`, `model_audit.json`, `model_loading.json`, `warmups.json`, `training_status.json`
- S6: `folds/fold*/program_selection.json`(선택 계수), `programs/proposal*.json`, `raw/s6_programs/*`(오프라인 LLM 제안 원응답)
- S2: `raw/public/{id}-feedback.json`, `{id}-control.json` 각 936건(첫 제안이 무효인 문제만 존재)
- `plan.md`, `report.md`
- 제외: `training_data.npz`, `programs/selection*.json`(선택 로그), `evaluations/`(test_s.py가 git에서 직접 읽음)

## 조건·상한·주/대조

| 조건 | family | 상한 | 역할 | 짝 |
| --- | --- | ---: | --- | --- |
| S1 | S1 | 없음 | 주 | V_GATE_NORMALIZED(직접 대조), V_GATE_GM, B_llm |
| V_GATE_NORMALIZED | S1 | 없음 | 대조 | 같은 정규화·gate, 중복 제거·단축 없음 |
| V_GATE_GM | S1 | 없음 | 대조 | GM proposed_search gate만 |
| S2_feedback | S2 | 없음 | 주 (보고서의 "S2") | S2_control, B_llm |
| S2_control | S2 | 없음 | 대조 | 같은 추가 호출, 진단 없는 일반 재검토 |
| S3 | S3 | 없음 | 주 | N_g2h(무안내), B_llm(즉시 안내) |
| S4_seed11/22/33 | S4 | 1.05 | 주 (S4 = 3시드 문제별 평균) | FOCAL_105, S4_untrained |
| S4_untrained | S4 | 1.05 | 대조 | 같은 망, `torch.manual_seed(20261111+fold)` 초기화 |
| S5 | S5 | 1 | 주 | A_OCT(같은 백엔드) |
| S6_program | S6 | 1 | 주 (새 LLM 계수) | S6_rule, S6_random, A_OCT |
| S6_rule | S6 | 1 | 대조 | 시작-목표 선분과 교차하는 벽 |
| S6_random | S6 | 1 | 대조 | 같은 수의 무작위 벽, 시드 `20261111:<fold>:<id>` |
| S6_historical | S6 | 1 | 참고(순위 제외) | 과거 계수 `[0.5,0.3,0.1,0.1]` |

상한은 SM `evaluate.bounds()`와 같다. 기준 7조건(A_GM, N_g2h, B_llm, A_OCT, FOCAL_105, K3_wall_*)은 harness BASE에 이미 있다.

## 구현 ([methods_s.py](methods_s.py))

- 각 fn은 SM `sm_src/algorithms.py::run`의 해당 분기를 줄 단위로 옮기고 SM 함수(`algorithms.s1`, `algorithms.delayed`, `proposed_search`, `Rectangles`+`next_search_core.search`, `next_refinement.solve`, `cv_candidate_core.focal`)를 그대로 호출한다. `algorithms.py`는 `sm_algorithms`라는 이름으로 파일 경로에서 불러 이름 충돌을 피했다. SM `common.diagnosis`는 SM 실행 경로를 읽어 import할 수 없으므로 원문 그대로 복사했다.
- 시간 정책은 SM과 같다. 모델 로딩·선택 계수 읽기·S2 응답 파일 읽기·격자 크기별 warm-up 추론은 시간 밖. `replies(q)`(evaluate18 `context()`가 측정 전에 호출)가 `prepare(fold)`·`warmup(q, fold)`까지 수행한다. 파싱·진단·수정 응답 선택·그래프·추론·탐색·재탐색·벽 검증은 fn 안(시간 안).
- 평가기가 `prepare`를 건너뛰면 fn이 로딩하되 `extra.artefacts_loaded_in_timed_call=True`로 표시한다. 1000문제 시험에서는 0건.
- heapq: `reproduction/core.py`, `next_search_core.py`, `cv_candidate_core.py`, `algorithms.delayed` 모두 `heapq.heappop` 모듈 속성으로 호출한다(확인).

### S2 두 번째 응답 읽기

`inputs_s/raw/public/{case_id}-feedback.json` → `ctx['replies']['s2_feedback']`, `{case_id}-control.json` → `ctx['replies']['s2_control']`. 텍스트 = `response.choices[0].message.content or ''`. `request_sha256`·`response_sha256`를 SM `digest`로 검증한다. **finish_reason으로 거르지 않는다**(feedback 398건·control 287건이 `length`인데 SM도 그대로 사용). 파일은 첫 제안이 SM `diagnosis`로 무효인 936문제에만 있고, 유효한 64문제는 두 번째 응답 없이 첫 응답을 쓴다(api_calls=1). `methods_s.replies(q)`가 `{'s2_feedback': (text, finish_reason, response_sha256), 's2_control': (...)}`를 돌려준다. evaluate18 pilot의 `s2_second` 키는 쓰이지 않는다(pilot 응답은 유효하므로 문제 없음).

### 논문 정의 지표 (`extra`)

`paper_operation`(주 탐색 OPEN 선택 수, goal pop·stale/중복 pop 포함), `paper_storage`(g를 받은 상태 수), `aux_operation`(보조 탐색 pop, S 조건 모두 0). 결정적 비교 필드에는 영향 없음(시험에서 확인).

| 엔진 | paper_operation | paper_storage |
| --- | --- | --- |
| GM 엔진(V_GATE_*, S1, S2_*) | gm.search `operation=len(closed)` | `len(g)`(inf 포함) |
| S3 (`delayed`) | `expansions + 목표 pop 여부`(= len(closed); 엔진이 반환하지 않아 항등식으로 계산) | `len(g)` |
| S5 | `counters.heap_pops`(stale 포함) | `discovered = len(g)`(유한값만 저장하는 엔진) |
| S6_* | 모든 회차 `heap_pops` 합(회차는 보조가 아닌 재탐색) | 회차별 `len(g)` 최대(= SM storage). 회차 간 합집합은 엔진 수정 없이 얻을 수 없어 생략, 회차 합은 `paper_storage_sum_rounds` |
| S4_* (FOCAL) | 배열 선택 수 + anchor heap stale pop 수 (`paper_operation_selections`, `paper_operation_stale_pops`로 분리) | 유한 g 칸 수(`discovered`). dense 배열은 모든 칸을 inf로 할당하며 그 수는 `storage` |

1000문제 전부에서 harness heap pop 수 = 주 OPEN pop(GM·S3·S5·S6는 paper_operation, S4는 stale pop)으로, 보조 탐색 pop이 없음을 확인했다. FOCAL의 paper_operation은 확장 노드의 anchor 항목이 나중에 stale로 한 번 더 빠지므로 heap A*의 pop 수보다 크다(S4 평균 약 500 대 harness operation 약 245).

## 등가성 시험 ([test_s.py](test_s.py), [결과](results/s_port/equivalence.json))

1000문제 × 15조건 1회, OOF fold, CPU torch 1스레드, 네트워크 차단. SM 기록(반복 0)과 `valid, error, length, operation, successor_expansions, storage, api_calls, path_sha256`을 비교.

| 조건 | 일치 | Oracle 유효 | 상한 이내 | 평균 ms (이 PC) | SM 평균 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| V_GATE_GM | 1000 | 1000 | – | 4.5 | 2.07 |
| V_GATE_NORMALIZED | 1000 | 1000 | – | 4.2 | 2.05 |
| S1 | 1000 | 1000 | – | 4.3 | 2.01 |
| S2_feedback | 1000 | 1000 | – | 5.6 | 2.70 |
| S2_control | 1000 | 1000 | – | 5.5 | 2.76 |
| S3 | 1000 | 1000 | – | 5.2 | 2.44 |
| S4_seed11 | 999 | 1000 | 1000 | 147 | 78.7 |
| S4_seed22 | 1000 | 1000 | 1000 | 146 | 78.6 |
| S4_seed33 | 999 | 1000 | 1000 | 146 | 78.7 |
| S4_untrained | 988 | 1000 | 1000 | 146 | 79.1 |
| S5 | 1000 | 1000 | 1000 | 12.3 | 6.63 |
| S6_program | 1000 | 1000 | 1000 | 4.7 | 2.38 |
| S6_rule | 1000 | 1000 | 1000 | 4.6 | 2.37 |
| S6_random | 1000 | 1000 | 1000 | 9.7 | 4.96 |
| S6_historical | 1000 | 1000 | 1000 | 4.6 | 2.37 |

- length는 ulp 차이도 없이 정확히 일치했다(harness가 경로에서 다시 합산해도 같음).
- S4 불일치 14건(seed11 1, seed33 1, untrained 12)은 모두 **경로·길이·유효성은 같고** `operation` 또는 `successor_expansions`만 ±1~3 다르다. FOCAL의 2차 키(망 출력)에서 동률 근처 값의 마지막 비트가 CPU(SM AMD, 이 PC Intel i5-12400)의 float32 커널 차이로 달라져 선택 순서가 바뀐 것으로 본다. 진단 결과는 아래.
- 시간은 참고용 1회 측정이다. 이 PC에서 다른 에이전트 프로세스가 동시에 돌았고 SM과 CPU도 달라 약 2배다. 본 측정은 evaluate18의 5회 중앙값을 쓴다.

S4 불일치 진단(별도 재실행, 4,000회 중 같은 14건이 같은 값으로 재현 → 이 PC 안에서는 결정적):

- 차이 크기: `operation` |Δ| ≤ 2(14건 중 11건), `successor_expansions` |Δ| ≤ 3(4건). 상대 차이 최대 1.4%(m038_p01 seed33 pop 71 대 72), 그 밖에는 0.6% 이하. 경로 sha·길이·유효성·상한은 14건 모두 SM과 같다.
- 같은 망을 float64로 추론한 로짓과 float32 로짓의 최대 차이는 1e-7~8e-6이다. float64 로짓으로 FOCAL을 돌리면 경로는 14건 모두 SM과 같고 heap pop은 6건이 SM 값과 같아졌다. 즉 결과가 망 출력의 마지막 비트에 민감한 동률 처리 문제이며 알고리즘 차이가 아니다.
- 평가에는 SM과 같은 float32 추론을 그대로 쓴다. 이 14건은 이 PC에서 SM 기록과 ±1~3 pop/확장 차이가 나는 것으로 기록한다.

## 차이·주의

- SM S6의 `successor_expansions`는 회차가 2개 이상이면 앞 회차를 두 번 센다(`next_refinement.solve`가 마지막 회차 결과의 `counters`를 전체 합으로 덮어쓰고, SM `run`은 그 위에 회차별 값을 다시 더함). 결정적 비교 필드라 SM 값 그대로 두고, 바로잡은 값을 `extra.successor_expansions_corrected`에 넣었다. 해당 문제 수: S6_program 222, S6_rule 208, S6_historical 208, S6_random 785. 확장 수 비교에 S6를 쓸 때는 보정값을 써야 한다. heap pop(`operation`)과 `paper_operation`은 이 영향이 없다.
- Python 3.11.9(SM 3.11.18), CPU가 다르다(UNIFIED_K README와 같은 사유).
- 이름은 SM 기록과 맞추려고 `S2_feedback`(=S2), `S6_program`(=S6 새 LLM 계수)을 그대로 썼다.
- S6_random의 난수 시드는 `ctx['fold']`를 쓴다(SM은 전역 FOLD). 각 문제는 자기 OOF fold에서만 평가하므로 같다.
