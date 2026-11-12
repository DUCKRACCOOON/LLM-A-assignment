# Table 1 재현 사양과 통합 하네스 대응표

작성 2026-11-12 (gm, 읽기 전용 조사. API 호출 없음). 근거 표기: 논문은 `materials/인지개과제_논문.pdf`의 PDF 쪽(p.)과 절, 코드는 `파일:줄`.
`source_code/`는 upstream `SilinMeng0510/llm-astar@9414ed8`(UPSTREAM.md:5-7)이다. 데이터 `source_code/dataset/environment_50_30.json`과 `experiments/UNIFIED_K/vendor/llm-astar/dataset/environment_50_30.json`은 sha256이 같다(`e46c7e79…`).

## 1. 지표 정의 (논문 + 공식 코드)

### 논문
- 기준은 **A\***이다. Table 1에서 A*=100이다(p.7 4.4절 "A* algorithm serves as the baseline, with an index value of 100").
- 집계: "geometric mean of performance ratios between LLM-A* and A* for operation, storage, path length"(p.6 4.3절). App. D.2(p.15): R_i는 i번째 test case의 (제안/기준) 비율이고 G=(∏R_i)^{1/N}이다. → **문제별 비율 → 기하평균 → ×100(%)**.
- Operation / Storage Ratio: LLM-A*/A*의 기하평균이며 낮을수록 좋다. "50% … uses 50% of the resources"(p.6). operation은 시간 복잡도, storage는 공간 복잡도에 대응한다(p.6 Experiment Subject).
- Relative Path Length: LLM-A*, A*, LLM-only 경로를 *최적 경로*와 비교한 비율의 기하평균이다(p.6). 공식 A*(유클리드, consistent)는 8방향 격자의 최적해이므로 분모는 A* 길이가 된다.
- Valid Path Ratio: 성공(충돌 없음, 완주 가능) 시도의 비율로, 기하평균이 아닌 단순 비율이다(p.6).
- LLM-only는 격자 탐색을 하지 않으므로 operation과 storage를 보고하지 않는다(p.7 Table 1 캡션).
- p.8 4.5절의 Figure 1 예시("140 operations … 859 operations")에서도 operation은 탐색한 상태(explored states) 수의 의미로 쓰였다.

### 공식 코드: 한 문제의 원값
| 항목 | A* (`source_code/llmastar/pather/a_star/a_star.py`) | LLM-A* (`…/llm_a_star/llm_a_star.py`) |
| --- | --- | --- |
| operation | `count` = while 반복 수 = heap pop 수 (46-49, 70) | `len(self.CLOSED)` (121-123, 149) |
| storage | `len(self.g)` (70) | `len(self.g)` (150) |
| length | 복원 경로 유클리드 길이 합 (70) | 같음 (151) |

- **operation = OPEN에서 pop한 횟수**이다. goal pop도 포함한다(pop하고 CLOSED에 넣은 뒤 break, A* 49-53). CLOSED는 list라서 이미 닫힌 상태가 중복(stale)으로 다시 pop되어도 다시 append되고 이웃 루프도 다시 돈다. 따라서 중복 pop도 포함한다. 이웃 확장 수나 고유 상태 수가 아니다.
- **storage = g 딕셔너리 항목 수**이다. 닫히지 않은 모든 이웃에 `g[s_n]=inf`를 먼저 넣으므로(A* 60-61, LLM-A* 138-139) 충돌이나 경계 이웃(비용 inf)도 포함된다. 즉 "한 번이라도 생성(관측)된 고유 상태 수"이다. OPEN+CLOSED의 크기나 peak OPEN이 아니다.
- 이동은 8방향이다(`env/search/env.py:7-8`). 비용은 hypot, 충돌이면 inf이다(a_star.py:110-113).
- LLM-A* 우선순위는 g + ‖s−goal‖ + ‖s−target‖이다(llm_a_star.py:203). 이웃이 target과 같으면 target을 갱신하고 OPEN 전체를 재계산한다(128-131, 162-166). 이 재삽입은 pop이 아니므로 operation에 들어가지 않는다.
- 비율 계산과 기하평균 코드는 upstream에 **없다**(grep 결과 없음). 집계는 논문 App. D를 따른다.
- 우리 재현의 집계는 `reproduction/summarize.py:11-30`이다. **유효 경로만** 대상으로 문제별 method/A* 비율을 구하고 기하평균×100을 취한다. 지표별 분모(n)를 기록한다. valid%는 전체 1000 기준이다. 재현 엔진 `reproduction/core.py:181-184, 222-223`은 공식과 같은 정의(operation=`len(closed)`(중복 포함), storage=`len(g)`(inf 포함))를 쓴다. 공식 코드와의 동등성은 `reproduction/AUDIT.md:29`(30문제×3조건 일치)에서 확인했다.

## 2. 통합 하네스 필드와의 대응 (`experiments/UNIFIED_K/harness.py`)

`run()`의 필드 출처:
- `operation = hc.counts['heap_pops']` (105). 측정 구간 안에서 **모든** `heapq.heappop` 호출 수이다(39-40, 76).
- `successor_expansions`: GM 엔진에서는 `CountGeometry.neighbors` 호출 수이다(33-34, 84). 그 밖의 엔진에서는 각 방법이 자체 정의한 값이다(88, 93, 96).
- `storage`: GM 엔진에서는 `r['storage']`=`len(g)`(inf 포함)이다(86). A_OCT는 유한 g만 센다(89). FOCAL은 격자 전체 dense 배열 크기이다(93). EXTRA는 방법별 값이다(96).
- `length`: 유효할 때만 경로 길이를 넣는다(104). `valid`는 GM `validate_path(grid=True)`와 정수 좌표 검사를 통과해야 한다(98-101).

실측 대조(`results/run_20261112/per_case.csv`, 1000문제):
| 비교 | 결과 |
| --- | --- |
| A_GM (op, storage) vs 재현 `results/main` astar | 1000/1000 일치 (평균 371.995 / 389.11) |
| B_llm `operation − successor_expansions` | 1000건 모두 1 (goal pop) |
| B_llm vs A_GM (Table 1 방식) | op 44.52%, storage 63.59%, len 102.99%, valid 100% |

**판정**
- GM 엔진(`gm.search`: A_GM, N_g2h, B_llm, K3_*, K4/K5/K6 계열의 `len(g)` 엔진)에서는 하네스 `operation`이 공식 operation과, `storage`가 공식 storage와 **정확히 같다**. K3 LowerBound는 heapq를 쓰지 않는다(`experiments/K3/heuristic.py:26-70`).
- **불일치**: `operation`은 "모든 heap pop"이라서 엔진마다 의미가 달라진다.
  - 보조 heap이 있으면 부풀려진다. K1은 `sm_src/next_topology.py:30`의 BFS/Dijkstra pop 때문에 op/A* 228%이고, 같은 방법의 expansions/A*는 60.9%이다.
  - OPEN을 heapq로 관리하지 않으면 빠진다. FOCAL_105와 K2_conformal은 pop=0인 문제가 100건이고 K2_nocal은 66건이다.
  - storage 정의도 섞여 있다. FOCAL·K2는 dense 배열 전체를 세서 storage/A* 515.78%이고, A_OCT·K1은 유한 g만 센다.
- **필요한 추가 필드**(18개 방법 공통): 아래 두 필드를 새로 기록해야 한다. 기존 `operation`과 `storage`는 "통합 프로토콜 지표"로 따로 남긴다.
  - `paper_operation`: 주 탐색 OPEN에서 상태를 선택한 횟수. goal 선택과 stale/중복 선택을 포함하고, 전처리·보조 탐색·순위 계산의 pop은 제외한다. 다회 탐색(재탐색, 다중 큐)이면 모든 회차의 합이다.
  - `paper_storage`: 주 탐색에서 g 값을 받은 고유 상태 수. inf 자리표시를 포함한다(공식 `len(g)`). dense 배열 엔진이면 "g<∞로 갱신되었거나 inf로 관측된 셀 수"를 따로 세야 한다.
  - 대안: 중복 pop을 다시 확장하는 엔진이면 `successor_expansions + int(goal 도달)`가 `paper_operation`과 같다(B_llm에서 확인). lazy-deletion으로 stale pop을 건너뛰는 엔진에서는 이 식이 과소 계측이 되므로 쓸 수 없다.
- Table 1 형식의 분모는 **A_GM**이어야 한다. 통합 보고서의 주 분모는 B_llm이다(`UNIFIED_K/README.md:9`). 길이 분모는 A_GM 길이이며, 이는 `per_case.csv`의 `optimal`과 같다.

## 3. 실험 설정

### 논문과 공식 코드
| 항목 | 논문 | 공식 코드 |
| --- | --- | --- |
| 데이터 | 50×30 지도 100개 × start/goal 10쌍 = **1000 샘플**(p.5 4.1절). Table 1은 50×30 원본 크기(p.6-7) | `dataset/environment_50_30.json`: 100맵 × 10쌍, range [0,51]×[0,31]. `dataset.py:12-14` |
| 프롬프트 | standard 5-shot, CoT 3-shot, RePE 3-shot(p.6 4.2절, App. B Table 2-4 p.13-14) | `pather/llm_a_star/prompt.py:56-96`(standard), `159-189`(CoT), `237-302`(RePE) |
| 모델 | GPT-3.5-TURBO, LLAMA3-8B-16bit(p.6) | `model/chatgpt.py:26-31`: **gpt-3.5-turbo-instruct Completion**, temperature 0, `stop=["\n"]`(25). `max_tokens=1000`(llm_a_star.py:70). Llama: `Meta-Llama-3-8B-Instruct`, bf16, greedy, max_new_tokens 8000(`model/llama3.py:6-25`) |
| 파싱 | – | `utils/utils.py:19-37`: 마지막 `[[…]]`를 literal_eval한다. 목록이 없으면 `[[0,0],[0,0]]`을 반환하고, 필터(llm_a_star.py:99-104)에서 빠져 start/goal만 남는다 |

- **few-shot 예시는 평가 지도에서 가져왔다.** standard 예시 1~5는 공개 데이터의 map 0, 2, 3, 4, 5의 pair 0과 start/goal/장애물이 완전히 같다(직접 대조). CoT와 RePE는 1, 2, 5번째 예시만 쓰므로 m000_p00, m002_p00, m005_p00과 같다(`results/main/prompt_overlap.json:2-19`). 실제로 gpt-3.5 standard의 m000_p00 응답은 예시 경로 `[[5,5],[26,9],[25,23],[20,20]]` 그대로였다.
- **SM 통합 프롬프트**(`UNIFIED_K/inputs/prompts/standard.txt:1-40`, `inputs/plan.md:33-39`): 지시문, 출력 형식, 예시 개수(5)는 같고, 예시만 공개 100지도와 기하가 겹치지 않는 합성 지도 5개(seed 20261112, `examples.json`)로 바꿨다. 모델은 gpt-4o-mini-2024-07-18, temperature 0, max_tokens 1000, 줄바꿈 stop 없음(plan.md:45, `inputs/protocol.json`)이다. 프롬프트 sha `b429d6…`, 원본 sha `da7220…`. 공유 standard 응답 1000건은 `inputs/raw/public/<case>-first.json`에 있다. B_llm의 operation이 원 프롬프트 4o-mini standard와 같은 문제는 181/1000뿐이다.

### `results/main/`의 원 프롬프트 응답
- `generations.jsonl` 6000행 = **1000문제 × {standard, cot, repe} × 2모델**(gpt-4o-mini-2024-07-18 3000, gpt-3.5-turbo-0125 3000). 응답 `model` 필드도 같은 ID이다.
- 전부 status ok, 재시도 0. 요청 user 메시지는 6000/6000이 공식 `gpt_prompt[style].format(...)`과 **바이트 단위로 같다**(직접 대조).
- 요청 파라미터는 Chat Completions, user 메시지 1개, `temperature=0, max_tokens=1000, store=false`이다(`reproduction/experiment.py:94-95`). 줄바꿈 stop은 없다(`reproduction/PROTOCOL.md:9`).
- finish_reason=length: 4o-mini standard 117, cot 35, repe 70; gpt-3.5 repe 7(costs.json 합계 229).
- 필드: `case_id, map_index, pair_index, model, prompt_style, request, request_sha256, manifest, utc, status, response, headers, attempts, seconds`.
- 파싱:
  ```python
  import json
  for line in open('results/main/generations.jsonl', encoding='utf-8'):
      g = json.loads(line)
      text = g['response']['choices'][0]['message'].get('content') or ''
      # (g['case_id'], g['model'], g['prompt_style']) -> text
      # 평가: reproduction.core.search(query, Geometry(query), 'llm_astar', text=text)
  ```
  평가 경로는 `reproduction/experiment.py:164-173`이다(한 응답을 LLM-only와 LLM-A*가 공유한다).
- `evaluations.jsonl`은 15000행이다: astar, dwa, goal_only 각 1000행과 llm_only·llm_astar × 2모델 × 3프롬프트. 행에 operation, storage, length, unique_pops, duplicate_pops, infinite_g 등이 있다.
- **Table 1 재현표는 이미 있다**: `results/main/table1.md`, `table1.csv`, `summary.json`(`reproduction/summarize.py`가 생성). 이번에 evaluations.jsonl에서 독립 재계산한 값과 모든 숫자가 같았다.

## 4. Table 1 수치 대조 (%; 우리 = `results/main`, A* 대비, 유효 경로 기하평균)

| 방법 | 프롬프트 | 논문 GPT-3.5 (op/st/len/valid) | 논문 LLAMA3 | 우리 gpt-3.5-turbo-0125 | 우리 gpt-4o-mini |
| --- | --- | --- | --- | --- | --- |
| A* | – | 100/100/100/100 | | 100/100/100/100 | |
| Dynamic WA* | – | 60.91/78.53/100.24/100 | | 59.27/78.52/100.24/100 (구현 가정) | |
| LLM-only | Few-shot | –/–/119.38/12.80 | –/–/111.05/12.60 | –/–/109.36 (n=64)/6.4 | –/–/116.64 (56)/5.6 |
| LLM-only | CoT | –/–/151.73/15.20 | –/–/114.89/12.00 | –/–/126.54 (79)/7.9 | –/–/122.00 (65)/6.5 |
| LLM-only | RePE | –/–/183.87/7.80 | –/–/138.32/16.40 | –/–/157.36 (121)/12.1 | –/–/117.95 (73)/7.3 |
| LLM-A* | Few-shot | 57.39/74.96/102.44/100 | 44.59/64.02/102.47/100 | 40.41/61.34/102.26/100 | 44.85/64.43/102.65/100 |
| LLM-A* | CoT | 69.50/83.65/102.54/100 | 47.60/66.27/102.46/100 | 55.96/74.34/102.80/100 | 46.78/65.63/102.58/100 |
| LLM-A* | RePE | 85.47/96.53/102.41/100 | 64.08/80.19/102.54/100 | 72.49/86.55/102.39/100 | 50.22/68.32/102.35/100 |

- 논문 수치 출처: p.7 Table 1, `report/paper_table1.json`.
- 참고 1: 예시와 겹치는 문제를 뺀 민감도 분석(`prompt_overlap.json:21-142`)에서 4o-mini standard는 44.81/64.42/102.64(n=995)로, 차이는 0.04%p 이하이다.
- 참고 2: g+2h(goal_only, LLM 없음)는 34.32/56.23/102.02로 모든 LLM-A* 행보다 탐색량이 적다.
- 참고 3: SM 통합 하네스의 B_llm(합성 예시, 4o-mini, standard)은 A_GM 대비 44.52/63.59/102.99이다.

## 5. "논문과 일치하는 Table 1"에 영향을 주는 차이

1. **표본 수**: 논문은 전체 1000문제를 썼다. 통합 프로토콜도 1000문제(OOF 5-fold)지만, fold는 학습이 있는 방법에만 의미가 있다. Table 1에는 fold 없이 1000문제 전체를 써야 한다. 예시와 겹치는 5문제도 논문처럼 포함하고, 제외한 결과는 민감도 분석으로 따로 적는다.
2. **프롬프트**: 논문 Table 1은 원 예시(평가 지도에서 가져온 예시)를 썼다. SM 통합 B_llm은 합성 예시를 쓴다. Table 1 재현에는 `results/main`의 원 프롬프트 응답을 써야 한다. 통합 비교표의 B_llm과는 별개 행으로 표시한다.
3. **모델과 API**: 공식 코드는 gpt-3.5-turbo-instruct Completion에 `stop=["\n"]`을 쓴다. 우리는 Chat Completions(gpt-3.5-turbo-0125 / gpt-4o-mini)를 쓰고 stop을 쓰지 않는다. 공식 CoT·RePE 템플릿은 `Vertical Barriers: …` 줄에서 끝나므로(prompt.py:188-189, 301-302) 줄바꿈 stop을 쓰면 첫 줄("Thought: …")에서 응답이 끊긴다. 그러면 경로 목록이 없어 start/goal fallback이 되므로, 공식 코드 그대로의 CoT/RePE는 사실상 g+2h였을 가능성이 있다. 이는 코드에서 추정한 것이고 논문 수치와 맞지 않는다. Llama3 행은 4o-mini로 대체했다(PROTOCOL.md:8).
4. **재현 시점의 무작위성**: temperature 0이고 1회 생성이다. 반복 샘플링이 없다는 점은 논문과 같다. Figure 3의 "10 trials"는 확장성 실험에만 해당한다(p.8).
5. **유효성과 실패 처리**: 논문 LLM-A*는 100%이고 우리도 100%이다(4o-mini 117건이 length로 잘렸지만 fallback으로 완주). LLM-only 길이를 유효 경로만으로 집계하는지는 논문에 적혀 있지 않다. 우리는 유효 경로만 쓰고 n을 함께 적는다. LLM-only 경로는 polyline이라 격자 최적보다 짧을 수 있다(AUDIT.md:37).
6. **시간 지표**: Table 1에는 시간이 없다. 하네스의 `total_seconds`, 5회 중앙값, end-to-end 시간은 Table 1 밖의 추가 지표이다. Table 1 형식의 열은 operation, storage, length, valid 넷이다.
7. **유효성 기준**: 하네스는 8방향 격자 단계와 정수 좌표까지 검사한다(harness.py:99-100). 공식 코드는 탐색 경로를 별도로 검증하지 않는다. LLM-A* 계열은 어느 쪽이든 100%이다.
8. **기준선**: Table 1 비율의 분모는 A*(A_GM)이다. 통합 보고서는 B_llm을 분모로 쓰고 결과를 `100×(1−비율)` 개선율로 표기한다(plan.md:117). 두 방식을 섞으면 안 된다.
9. **DWA**: upstream에 구현이 없다. 우리 행(59.27/78.52)은 감쇠 규칙을 가정한 구현이다(PROTOCOL.md:23).
10. **원 코드의 부수 동작**: upstream은 입력 dict의 range 상한을 직접 줄인다(a_star.py:30-31). 동일 start/goal이면 예외가 난다. 재현 엔진은 이를 구조화된 결과로 처리하며, 결과 수치에는 영향이 없다(PROTOCOL.md:21).
