# Waypoint 동작에 관한 대화 정리

설명 대상은 [공식 LLMAStar 구현](../source_code/llmastar/pather/llm_a_star/llm_a_star.py)이다.

## LLM은 언제 waypoint를 주는가

탐색을 시작하기 전에 목록 전체를 한 번 생성한다. 구조화된 입력에서는 waypoint 생성 1회가 필요하고, 자연어 입력은 JSON 해석용 호출이 추가된다. 탐색 중 다음 waypoint를 고르기 위해 LLM을 다시 호출하지 않는다.

`searching()`은 `_initialize_parameters()`로 맵을 설정하고 `_initialize_llm_paths()`로 응답을 파싱·필터링한다. 이후 OPEN 탐색을 시작한다.

## Waypoint들은 여러 선택지인가

목록은 `[Start,W1,W2,Goal]`처럼 순서가 있다. 서로 경쟁하는 여러 대안 경로를 표현하는 것은 아니다. 다만 각 점은 강제로 통과해야 하는 제약이 아니라 탐색 방향을 유도하는 기준이다.

## 각 waypoint를 goal로 바꿔 A*를 반복하는가

최종 goal은 계속 동일하다. 현재 target이 W1일 때는 `f=g+d(n,Goal)+d(n,W1)`, W2로 전환한 뒤에는 `f=g+d(n,Goal)+d(n,W2)`를 사용한다. OPEN, CLOSED, g를 유지하며 탐색을 계속한다.

## Waypoint가 후보에 있으면 무조건 선택하는가

노드 선택은 전체 f값이 가장 작은 후보를 기준으로 한다. 현재 선택한 노드의 이웃 좌표를 살펴보다가 target과 일치하면 다음 target으로 바꾼다. target 자신이 큐에서 선택될 때까지 기다리거나 최종 경로에 반드시 넣는 것은 아니다.

## 최종 경로가 waypoint를 안 거칠 수 있는가

그렇다. waypoint까지 거리 항은 우선순위만 바꾼다. g와 goal 거리도 함께 비교하므로 매 단계 waypoint에 더 가까워지는 것도 보장되지 않는다. 최종 goal을 찾으면 남은 waypoint가 있어도 종료한다.

## Waypoint가 전혀 남지 않으면 일반 A*인가

현재 원본은 시작·목표만 남으면 target을 goal로 잡아 `f=g+2h`가 된다. 이는 일반 A*의 `g+h`와 다르다. 파싱 실패나 fallback 분석에서 반드시 구분해야 한다.
