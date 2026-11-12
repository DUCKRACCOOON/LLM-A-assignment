# J1~J6(준연) 통합 하네스 이식 (2026-11-12, gm)

준연의 J1~J6를 UNIFIED_K `harness.run` 계약 그대로(`methods_j.py`의 `METHODS`) 옮겼다. 원 코드는 `origin/jy` `d72632c3`에서 바이트 단위로 복사했고 알고리즘 코드는 한 줄도 고치지 않았다. 필요한 보정은 모두 어댑터(`methods_j.py`)에 두었다.

## 복사한 파일 (`jy_src/`, 해시는 [PROVENANCE.json](jy_src/PROVENANCE.json))

| 위치 | 원 경로(origin/jy) | 용도 |
| --- | --- | --- |
| `vendor_sm/core.py`, `vendor_sm/candidate_core.py`, `vendor_sm/__init__.py` | `experiments/COMMON18/vendor_sm/` | J1 CHOMP 수리·J2 FMM·FOCAL 엔진 (SM 코드의 COMMON18 사본, 준연의 incumbent 반환 수정 포함) |
| `JY/search.py`, `JY/__init__.py` | `experiments/JY/` | J3~J6 엔진 (`run_once`, `LandmarkIndex`, NBS, guided) |
| `reference/*.txt`, `reference/COMMON18_SM_PROVENANCE.json` | `COMMON18/backends.py`, `J3~J6_full/adapter.py` | 준연 원 실행기(참고용, import 안 함) |

`jy_src/__init__.py`만 새로 만든 파일이다. JY 코드가 부르는 `reproduction/core.py`, `experiments/K3/heuristic.py`, `experiments/K3/verification.py`는 gm과 origin/jy가 같은 파일이라 복사하지 않았다.
모든 엔진이 `import heapq` 후 `heapq.heappop(...)`을 쓴다(`from heapq import` 없음). 따라서 패치 없이 `HeapCounts`가 모든 pop을 센다.

## 조건 (13)

| 조건 | 계열 | 역할 | 비용 상한 | API | 내용 · 근거 |
| --- | --- | --- | ---: | ---: | --- |
| J1_main | J1 | 주 | 1.05 | 1 | LLM 경로 CHOMP식 수리 → incumbent + 거리 보조순위 FOCAL. J1 PROTOCOL 11행 |
| J1_straight | J1 | 대조 | 1.05 | 0 | 같은 수리·FOCAL, 직선 초기화. 12~13행(직접 대조는 J1/J1_straight) |
| J2_main | J2 | 주 | 1.05 | 3 | standard·CoT·RePE 거리장 → 속도장 FMM 보조순위 FOCAL. J2 PROTOCOL 10~14행 |
| J2_uniform | J2 | 대조 | 1.05 | 0 | 균일 속도 FMM. 14~15행(직접 대조는 J2/J2_uniform) |
| J3_main | J3 | 주 | 1 | 1 | LLM-A* seed(실패 시 g+2h) 상한 U + octile A* 정련, g+h≥U 가지치기. J3 PROTOCOL 8~11행 |
| J3_no_prune | J3 | 대조 | 1 | 1 | 같은 seed·정련, 가지치기 제거. 12행 |
| J3_noLLM | J3 | 대조 | 1 | 0 | g+2h seed + 같은 가지치기 정련. 13행 |
| J4_main | J4 | 주 | 1 | 0 | ALT 랜드마크 4개, 질의마다 그래프·랜드마크 Dijkstra 새로 생성. J4 PROTOCOL 5~7, 11~12행 |
| J4_graph_octile | J4 | 대조 | 1 | 0 | 같은 그래프·엔진, 랜드마크 0개(octile만). 9행 |
| J5_main | J5 | 주 | 1 | 0 | octile NBS. J5 PROTOCOL 6행 |
| J5_bidijkstra | J5 | 대조 | 1 | 0 | 같은 양방향 함수, 휴리스틱 없음. 6, 9~10행 |
| J6_main | J6 | 주 | 없음 | 1 | guided LLM-A* clean 엔진, target 변경 시 lazy 재계산. J6 PROTOCOL 7~11행(최단성 보장 없음) |
| J6_full | J6 | 대조 | 없음 | 1 | 같은 엔진, target 변경마다 OPEN 전체 재계산+heapify. 7행 |

공통 기준 조건(A_GM, N_g2h, B_llm, A_OCT, FOCAL_105)과 겹치는 Astar/LLMA/G2H/Octile/FOCAL0는 넣지 않았다. 준연 Octile·WallAstar(JY clean A* 엔진)와 J1_exact/J2_exact(w=1), J3_noLLM_no_prune, J4_alt8, J5_euclidean, 단방향 Dijkstra도 주 조건의 직접 대조가 아니어서 넣지 않았다.
J4·J5는 LLM을 쓰지 않는 방법이다(준연 설계 그대로).

## 입력

- standard 응답은 `ctx['text']`(SM 공유 `first`). J2는 `ctx['replies']['k2_cot']`, `['k2_repe']`(UNIFIED_K 수집본)를 함께 쓴다.
- finish_reason은 `extra`에만 기록하고 응답을 버리지 않는다. 준연 J2 PROTOCOL 9행("토큰 한도 종료·무효 안내를 제외하거나 재생성하지 않는다")과 K2 어댑터가 같은 방식이다. 파싱 실패는 원 코드대로 직선 제안이 된다.
- 응답 파싱, Grid 생성, CHOMP 수리, 거리장, FMM, J3 seed 탐색, J4 그래프·랜드마크 전처리(질의마다 새로), NBS 큐 관리, 경로 검증은 모두 `fn` 안이라 시간에 포함된다.

## 지표 정의

| 계열 | storage (준연 정의 유지) | successor_expansions | extra.paper_operation / paper_storage | extra.aux_operation |
| --- | --- | --- | --- | --- |
| J1, J2 | 유한 g 칸 수(`discovered`) | 확장 − (목표 선택 시 1) | OPEN 선택 수(목표 포함) + anchor heap stale pop / 유한 g 칸 수 | J1 0, J2 FMM heap pop |
| J3 | seed g(무한 포함) + 정련 g 합 | seed pop − valid + 정련 확장 | 정련 A* pop / 정련 len(g) | seed 탐색 pop(실패 seed 포함), `aux_storage`=seed len(g) |
| J4 | 질의 len(g) | 질의 확장 | 질의 A* pop / 질의 len(g) | 랜드마크 Dijkstra pop, `aux_storage`=거리표 항목 |
| J5 | 정·역방향 g 합 | 양방향 확장 | 양방향 루프의 모든 pop(확장·승격·stale) / 정·역 g 합 | 0 |
| J6 | len(g) | 확장(목표 제외) | 모든 pop(stale·lazy refresh 포함) / len(g) | 0 |

하네스 `operation`(모든 heapq pop)은 `paper_operation`과 `aux_operation`의 합과 같다. J1·J2만 예외로, FOCAL은 OPEN을 집합에서 배열로 골라 선택 자체가 heap pop이 아니다. 그래서 하네스 값은 anchor stale pop + aux와 같다. 이 관계는 test에서 모든 실행에 대해 확인했다.
J1·J2의 g는 크기 grid.n인 밀집 배열이고 모든 칸이 inf로 채워진다. 그래서 `paper_storage`는 유한 g 칸 수로 두고, 배열 크기는 `extra.allocated_g_entries`에 따로 적었다.

## 어댑터 보정 (원 코드 무수정)

1. `vendor_sm/core.py`는 `Path(__file__).parents[3]`로 `source_code/`를 찾는다. `jy_src/vendor_sm` 아래에서는 한 단계 깊어 경로가 틀어진다. 그래서 같은 파일을 실행하되 `__file__`만 원 위치(`experiments/COMMON18/vendor_sm/core.py`)로 둔다.
2. J6는 `trace=False`로 실행한다. 준연은 full/lazy 선택열 대조를 위해 `trace=True`로 쟀다. trace는 선택 노드를 목록에 덧붙일 뿐 탐색 결정은 같다. 선택열 해시는 test에서 하네스 밖 `trace=True` 재실행으로 대조했다.
3. J5는 준연 본 측정과 같이 `audit=False`(쌍 전수 검산 비용 제외).
4. J3: `jy_src.JY.search`의 모듈 이름 `search`를 읽기 전용 래퍼로 감싸 seed 탐색별 pop·저장량을 기록한다(정련/seed 분리용).
5. Python 3.12 `sum()`: 준연은 Python 3.12.14로 실행했다(모든 J*_full protocol.json). 3.12의 `sum()`은 float를 보정(Neumaier) 합산하고, 우리 3.11.9는 단순 합산한다. float 합이 판단에 쓰이는 곳이 두 군데 있다. (a) `vendor_sm/core.path_length`: CHOMP가 더 짧은 격자 경로만 채택하고 FOCAL incumbent 상한에도 쓰인다. (b) J3 `astar`의 상한 U이다(f≥U 가지치기). 길이가 같은 후보 사이 판정이 마지막 비트에서 뒤집힌다. 보정 없이 돌리면 J1은 1000건 중 997~999건, J3는 처음 20문제에서 storage가 13/20(J3_main), 14/20(J3_noLLM)만 일치했다. 그래서 CPython 3.12 `builtin_sum_impl`을 옮긴 `_sum_py312`를 모듈 전역 `sum`으로 넣었다. `vendor_sm.core`에는 항상 넣고(이 모듈은 J1·J2만 사용), `JY.search`에는 J3 호출 동안만 넣는다. J4~J6에서는 float 합이 보고용 길이에만 쓰인다. J5의 매 반복 int 합은 builtin으로 두어 시간 왜곡을 피했다.

## 동등성 검사 ([test_j.py](test_j.py), 결과 [results/j_port/test_j.json](results/j_port/test_j.json))

입력은 준연이 고정한 응답(`J1/J2/J3/J6_full/results/run_20261111/responses.jsonl.gz`, results/main의 standard·CoT·RePE)이다. 이를 git 객체에서 그대로 읽어 공개 1,000문제 전체에서 13조건을 하네스로 실행했다. 결과는 준연의 문제별 기록(`J*_full/results/run_20261111/evaluations.jsonl.gz`)과 대조했다. 문제 정의(시작·목표·벽)도 UNIFIED_K `cases()`와 같음을 확인했다.

| 조건 | valid·path·length·확장·storage | heap pop(operation) | 선택열 해시 |
| --- | --- | --- | --- |
| J1_main, J1_straight, J2_main, J2_uniform | 1000/1000 (각 항목) | 준연 기록 없음(None) | – |
| J3_main, J3_no_prune, J3_noLLM | 1000/1000 | 1000/1000 | – |
| J4_main, J4_graph_octile | 1000/1000 | 1000/1000 (전처리 pop 포함) | – |
| J5_main, J5_bidijkstra | 1000/1000 | 1000/1000 | – |
| J6_main, J6_full | 1000/1000 | 1000/1000 | 1000/1000 |

- 13,000개 경로 모두 독립 정수 Oracle 검사를 통과했다. `inputs/oracle.json` 최단거리 대비 상한 검사도 11,000건(J6 제외) 위반 0이었다. 준연이 기록한 최단거리와도 일치한다.
- 13,000건 모두 `paper_operation`+`aux_operation`(J1·J2는 anchor stale pop+aux)=하네스 heap pop이다. `paper_operation`=0인 33건은 J1이 검증된 incumbent를 첫 판정에서 바로 반환한 경우다.
- 통합 입력 smoke: 공개 문제 20번째마다 50문제(SM `first` + UNIFIED_K k2_cot/k2_repe), 13조건 × 2회. 경로·상한·Oracle 검사를 모두 통과했고 두 번의 결정적 결과가 같았다. finish_reason은 k2_cot length 3건, k2_repe length 10건이었고 그대로 파싱했다. SM 합성 pilot 30문제(가짜 waypoint) 390회도 통과했다.

문제당 실행 시간(ms, 하네스 total_seconds, 1회. 동등성 검사 1,000문제 중앙값 / 평균):
J1_main 86/177, J1_straight 81/174, J2_main 20/117, J2_uniform 18/115, J3_main 3.8/5.7, J3_no_prune 4.0/5.8, J3_noLLM 2.9/4.9, J4_main 35/37, J4_graph_octile 23/26, J5_main 4.4/6.1, J5_bidijkstra 14/15, J6_main 1.8/3.5, J6_full 1.7/3.4.
J1·J2는 큰 지도에서 밀집 Grid 생성·CHOMP·FMM 비용이 커서 평균이 중앙값의 2~6배다.

## 주의

- J2는 응답 3개를 요구한다. `k2_cot`/`k2_repe`가 없으면 원 코드대로 직선 제안으로 대체되는데, API 호출은 그대로 3으로 센다(`extra.missing_replies`에 기록).
- J4 storage는 준연 정의(질의 g만)를 따른다. 랜드마크 거리표(`aux_storage`)와 그래프(`graph_nodes`, `graph_edges`)는 extra에 따로 있다.
- 시간은 3.11 + `_sum_py312`(Python 구현) 기준이다. 이 함수는 J1의 CHOMP에서 몇십 번, J3에서 2번 불릴 뿐이라 영향이 작다.
