# J3~J6 실행 전 고정 규약

2026-10-27 (Asia/Seoul), 개인 브랜치 `jy`. 동적 경유점 생성은 보류한다.
기준 `origin/gm=c56eb83`, 원본/기존 실험 파일은 변경하지 않는다.
다른 브랜치의 `README.md / PROTOCOL.md / REPORT.md / results/` 양식을 따른다.
이 문서는 실행 전 설계이며 실제 완료 여부는 manifest와 결과 보고서가 결정한다.

## 연구 질문과 고정 조건

| ID | 주 조건 | 절제/비교 조건 | 질문 |
|---|---|---|---|
| J3 | g+2h 초기 경로 + octile A* 상한 가지치기 | 같은 2단계에서 가지치기 제거; 저장 LLM-A* 초기 경로로 교체 | 초기 해 생성과 증명 비용을 합쳐도 효율적인가? |
| J4 | graph-farthest 4-landmark ALT + octile A* | 8 landmarks; 단일 질의/같은 지도 전체 질의 재사용 | 강한 하한의 준비 비용을 회수할 수 있는가? |
| J5 | NBS + octile 양방향 하한 | NBS Euclidean; bidirectional Dijkstra; 단방향 A* | 같은 격자에서 양방향 쌍 선택의 확장/시간 이득이 있는가? |
| J6 | metric bound를 이용한 lazy OPEN 갱신 | 동일 clean engine의 full heapify 갱신 | 같은 상태 확장 순서를 보존하면서 전체 비용을 줄이는가? |

공통 대조군: 원본 동등 A*, 원 LLM-A*, g+2h, K3 직선거리+벽,
clean octile A*, clean octile+벽 A*, clean octile g+2h. 전체 18조건.
새 API/모델 호출은 0회. 기존 gpt-4o-mini-2024-07-18 standard 응답을 사용한다.
공개 응답은 gm K4의 검증된 요청/내용 해시와 공개 원입력을 대조한다.
별도 30문제는 기존 generalization 입력/원문 응답을 사용한다.
기존 응답 파일은 복사·해시 기록하며 새 프롬프트/응답을 생성하지 않는다.

원래 8방향, 비용 1/√2, 경계 내부, 닫힌 선분 접촉 금지, 길이 0 벽 무시를 유지한다.
J3~J5는 최단 경로를 목표로 한다. J6는 정적 경유점 안내의 큐 구현 비교이며
최단 보장을 주장하지 않는다. J3의 initial bound는 실제 합법 격자 경로에서만 얻는다.
J3는 ARA* 전체 구현이 아니다. J5는 Chen et al. 2017 NBS Algorithms 1~3의
waiting(f)/ready(g) 구조, 쌍 선택, 재개방과 하한 종료를 구현한다.
논문의 2VC 정리를 실측 총 pop/시간의 2배 보장으로 해석하지 않는다.

J6는 기존 원본 구현과 달리 stale-g를 건너뛰는 clean engine이다.
원본 LLM-A*는 별도 비교군이며 full/lazy 간 차이만 갱신 최적화의 효과다.
양쪽에 동일한 1e-9 단위 정수 우선순위와 (priority,node) 동률을 적용한다.
target 변경 시 삼각부등식으로 안전한 전역 하한 이동을 적용하고 stale top만 재계산한다.
오래된 키를 그대로 exact key로 취급하는 단순 lazy heap은 사용하지 않는다.
전체 공개/별도 문제에서 두 방법의 pop 상태열·경로·고유 확장·target 변경의 일치를 검사한다.

## 데이터·검증·측정

- 공개 100지도×10쌍=1,000문제. 파일럿 map 0,10,...,90 pair 0~2의 30문제는 본실험 부분집합이다.
- transfer는 기존 seed=20261013의 별도 30문제. 양쪽 모두 이전 연구에서 노출된 자료이며 독립 hidden test라고 하지 않는다.
- 새 합성 30지도×4쌍=120문제: K3 생성 규칙, seed=20261027. 생성/제외 이력을 보존한다.
  응답이 없으므로 LLM 관련 조건은 실행하지 않고 분모 차이를 표시한다. 성능을 보고 지도/질의를 고르지 않는다.
- 정수 orientation 기반 독립 Oracle로 각 합법적 격자 간선을 대조하고 역방향 Dijkstra 정답을 구한다.
  oracle/정답은 탐색기에 전달하지 않는다. ALT의 landmark Dijkstra는 온라인 전처리 비용에 포함한다.
- 빈 지도/단일 벽/역방향 벽/교차/끝점/퇴화/동일 S,G/도달 불가 검증.
  소형 무작위 지도에서는 NBS 선택을 OPEN Cartesian product reference와 대조한다.
- 공개 저장 baseline과 path/length/pop/storage 동등성; 새 방법 경로 전체 검증과 정확 조건의 최단거리 일치 검사.
- 각 조건 시간 3회, 고정 seed=20261027로 방법 실행 순서를 섞어 한 프로세스 순차 실행.
  원시 반복별 경로·시간·카운터 보존, 문제별 시간 중앙값을 집계. 실패 행도 먼저 저장하고 검사 실패 시 중단한다.
- 매 query에서 Geometry/cache를 새로 만든다. J3는 동일 query 내 seed/refinement 이웃 캐시를 재사용한다.
- ALT는 지도·landmark 수·반복마다 전처리를 실제 측정하고, query마다 준비된 그래프를 사용한다.
  `query_seconds`는 준비 후 실행, `total_seconds`는 실제 측정 prep+query를 합친 재구성 단일질의 비용이다.
  준비 자체는 query마다 중복 실행하지 않는다. 반복 질의 비용은 지도 내 prep 1회 + 전체 query 합으로 별도 보고한다.
- 시간에는 지도 생성·온라인 준비·탐색·복원·자체 반환 경로 검증이 포함된다.
  파일 I/O, 독립 oracle, 통계와 그림 생성은 제외. 과거 API 지연은 별도 열이며 현재 실측 서비스 시간으로 합산하지 않는다.
- operation은 모든 실제 heap pop. NBS promotion 및 stale pop, J6 refresh pop도 포함한다.
  ALT는 query pop과 preprocessing pop, cold 합계를 분리한다. successor expansions도 별도 보고한다.
  storage는 검색 g 항목 수; J3 두 단계 합, J5 양방향 합. 실제 peak bytes가 아니다.
  ALT 거리표·graph 저장량, queue peak도 별도 열에 둔다.
- 문제별 대응 비율 기하평균과 지도 bootstrap 2,000회 95% 구간, seed=20261027.
  정확성을 위한 0회 확장은 비율 계산에만 max(count,1)을 적용하며 실제 합계는 0 유지한다.
  평균·합계·개선/동일/악화 건수, 최단 일치/최대 초과율, 주요 실패·퇴행 사례도 보고한다.
  다중 비교 보정 없는 탐색적 구간이며 새로운 연구 전체의 확증 결과로 표현하지 않는다.

## 산출물·완료 조건

실행 전 규약·소스·입력 SHA-256, 환경, 기준 commit, 실행 명령을 manifest에 기록한다.
기존 output 덮어쓰기 금지. run_20261027 아래 public/transfer/synthetic의 raw jsonl.gz,
per_case.csv, optima/checks/summary, 분석 그림·한국어 REPORT, artifact hashes와 독립 audit를 남긴다.
새 실험에서 효과가 없거나 느려진 방법도 모두 남긴다. 팀 분담의 담당명은 방법 ID를 따르되
실제 구현·실험은 사용자 지시 아래 Codex 지원으로 수행했다고 기록하며 사람 기여를 추정하지 않는다.

참고: [ALT](https://www.microsoft.com/en-us/research/publication/computing-the-shortest-path-a-search-meets-graph-theory/),
[NBS](https://arxiv.org/abs/1703.03868), [공통 기준](../../docs/공통_실험_기준.md).
