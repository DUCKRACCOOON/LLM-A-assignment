# J6 전체 데이터셋 1회 평가

2026-11-11, jy. 마지막 담당 방법 J6(OPEN 갱신 최적화)를 공개100지도×10쌍=1,000문제에서 평가한다.

- 기존 experiments/JY/search.py의 guided 엔진을 그대로 사용한다. 원 LLMA와 달리 stale-g를 건너뛰는 clean 엔진이다.
  저장된 정적 waypoint 목록에서 target가 바뀔 때 큐를 갱신하며, 새 waypoint 생성·재호출·지도 장애물 변경은 없다.
- J6_full은target 변경마다 active OPEN 전체의 score를 다시 계산해 heapify한다.
  주 J6는삼각부등식의 metric 하한과 global offset/epoch를 사용해 오래된 top만 정확히 다시 계산한다.
  g 감소는 버전으로 무효화한다. 오래된 키를 그대로 정확한 키로 사용하는 단순 lazy 방식이 아니다.
- 두 방식의 g+h_goal+d_target, 1e-9 정수 우선순위, target 처리·동률·closed 규칙을 동일하게 유지한다.
  J6 계열은 최단성을 보장하지 않는다. 원 LLMA 대비 전체 차이를 갱신 최적화 하나의 효과로 돌리지 않는다.
- 대조군 Astar/LLMA/G2H/Octile/WallAstar를 포함해7조건, 문제·조건별1회=7,000회다.
  모든 대조를 이번 실행에서 새로 측정하며 과거 J1~J5/JY/COMMON18 결과와 합산하지 않는다.
- CPU/native1스레드, seed20261111로 방법 순서를 섞는다. 파싱·지도 준비·우선순위 평가·rebuild/refresh·검색·복원·필수 검증을 로컬에 포함한다.
  파일 읽기·독립 Oracle/Dijkstra·trace 해시/대조·집계는 타이머 밖이다.
- full/lazy 둘 다 trace=True로 실제 유효 상태 선택열을 같은 방식으로 수집한다. append 계측 비용은 두 방법의 타이머 안에 포함한다.
  각 문제·조건은1회만 실행하며 추가 dataset 재실행으로 trace를 만들지 않는다. 미계측 production 성능이라고 표현하지 않는다.
  trace는 stale/refresh pop을 제외한 유효 상태 선택열이며 목표 선택을 포함한다. 모든 heap pop 열과 같다는 뜻이 아니다.
- 모든1,000쌍에서 유효 선택열·경로·길이·이웃 확장·상태 저장·target 변경·relax 시도 일치를 검사한다.
  operation=expansions+goal선택+stale_pops+lazy_refreshes를 검산하고, target 변경0인 문제도 별도 집계한다.
- rekey_evaluations는초기 삽입·재삽입 score 평가까지 포함한 총 우선순위 계산 수다.
  heap_pushes는실제 heappush 호출 수이며 full의 list 재구성/heapify 항목을 모두 삽입으로 세는 값이 아니다.
  rebuild 비용을 시간과 score 평가로 함께 보고, pop 수를 총 CPU 연산 수로 표현하지 않는다.
- 작은 fixture·무작위 강제 target 목록은기존 JY의 full/lazy 선택열 동등성 검사로 별도 검증한다.
  dataset에서는 독립 정수 Oracle로100지도 간선과 모든 최종 경로를 검사하고 정확 대조군3,000회는최단거리와 대조한다.
  정답은 안내·검색·튜닝에 전달하지 않는다.
- LLMA/J6_full/J6는같은 GPT-4o mini standard 응답을 각각 논리1회 사용한다. 다른 조건은0회다.
  results/main의1,000응답을 요청·입력·SHA와 확인해 고정하고 잘린 응답도 보존한다.
  신규 API 호출0회, 실행 중 네트워크/API 연결 차단. 과거 API 지연+현재 로컬의 총시간은 재구성 추정이다.
- 공개 전체는 기존 연구 노출 자료다. COMMON18 test200과 target변경0/양수 부분집합을 별도 제시하되 고유 문제 수에 더하지 않는다.
  시간1회이므로 반복 안정성을 주장하지 않는다. 지도 bootstrap2000회는 구성의 탐색적 구간이며 API/시간 반복 변동을 포함하지 않는다.
- storage/heap peak는논리 항목 수이고 실제 메모리 bytes가 아니다. 별도 RSS 측정은 없다.
  기존 코드·결과를 보존하며 준연 담당 J6 평가에 Codex가 실행 통합·검증·분석 작성을 지원한다.
