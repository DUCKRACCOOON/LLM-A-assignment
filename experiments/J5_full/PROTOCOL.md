# J5 전체 데이터셋 1회 평가

2026-11-11, jy. J4 다음 J5(NBS·양방향 탐색)를 공개100지도×10쌍=1,000문제에서 평가한다.

- 기존 experiments/JY/search.py의 NBS와 양방향 Dijkstra를 그대로 사용한다. 실제 양방향 대칭 격자 간선을 사용한다.
- 주 J5는octile NBS, 추가 J5_euclidean은Euclidean NBS, J5_bidijkstra는양방향 Dijkstra다.
  NBS는 waiting(f)/ready(g)의4큐, 버전별 stale 제거, 재개방, 쌍 하한과 유효 연결 경로 상한의 종료 규칙을 유지한다.
  최초 만남만으로 종료하는 방식이 아니다. 코드·파라미터를 결과에 따라 바꾸지 않는다.
- 대조 Astar/LLMA/G2H/Octile/WallAstar/단방향 Dijkstra를 포함해9조건, 문제·조건별1회=9,000회다.
  단방향 Dijkstra는기존 clean astar 함수에 h=0을 주는 같은 엔진의 대조다. LLMA/G2H 외7조건은 정확 탐색이다.
- 전처리 없는 단일 질의 비교다. CPU/native1스레드, seed20261111로 방법 순서만 섞는다.
  모든 대조군을 같은 실행에서 새로 측정하며 이전 JY/J1~J4 결과와 합산하지 않는다.
- 파싱·지도 준비·모든 큐 관리/쌍 선택/승격·stale 제거·검색·복원·필수 검증을 전체 로컬 시간에 포함한다.
  파일 읽기·독립 정수 Oracle/Dijkstra·집계는 타이머 밖이다. 방법·질의 사이 이웃 캐시는 공유하지 않는다.
- 실제 모든 heap pop을operation으로 기록한다. NBS에서는 승격 pop+실제 확장 pop+stale pop을 모두 포함한다.
  확장·쌍 수·승격·push·양방향 g 저장량도 구분한다. pop이나논리상태를CPU 연산 수/메모리 bytes로 표현하지 않는다.
- 정확성 검증에서는 작은 고정 fixture와 기존35개 무작위 소형 지도의 Cartesian frontier 쌍 reference를 사용한다.
  주 데이터셋의 시간 측정에서는audit=False로 쌍 전수 검산의 추가 비용을 부과하지 않는다. 주 탐색은 각1회뿐이다.
- 주 데이터셋은 독립 정수 기하 Oracle와 전체100지도 간선을 대조하고, 모든 최종 경로·최단거리·정확 조건을 검사한다.
  정답을 탐색기나 휴리스틱에 전달하지 않는다. NBS expansions=2*pairs,
  operation=expansions+promotions+stale_pops 및 forward/backward 저장량 합을 저장 원자료에서 검산한다.
- J5와 관련 변형은 온라인LLM0회다. 원 LLMA 대조만 results/main의동일 standard 응답1,000건을 검증 후 재생한다.
  신규 API 호출0회, 실행 중 네트워크/API 연결 차단. 과거 API 지연+현재 로컬의 총시간은 재구성 추정이며 현재 서비스 실측이 아니다.
- NBS의확장 관련 이론을 실제 총heap pop/CPU시간의 일정 배수 보장으로 확대하지 않는다.
- 공개 전체는 기존 연구 노출 자료다. COMMON18 test200도 부분집합으로만 집계한다.
  시간1회이므로 반복 안정성은 평가하지 않는다. 지도 bootstrap2000회는 구성의 탐색적 구간이며 시간/API 반복 변동을 포함하지 않는다.
- 논리 storage는양방향g 항목 합이며 고유 정점수나peak bytes가 아니다. heap peak는원 코드의 반복 경계 관측 카운터로만 해석한다.
  별도 RSS 측정은 없다. 기존 코드·결과는 보존하며 J6는이번에 새로 실행하지 않는다.
- 준연 담당 J5 평가에 Codex가 실행 통합·검증·분석 작성을 지원한다.
