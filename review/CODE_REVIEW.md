# LLM-A* 코드 리뷰

검토 대상은 [원본 출처](../UPSTREAM.md)의 커밋이다. 이 문서는 대화에서 설명한 코드 구조와 검증한 동작을 정리한다. 원본 코드를 수정한 결과가 아니다.

## 읽는 순서와 실행 구조

1. [AStar.searching](../source_code/llmastar/pather/a_star/a_star.py): 입력 초기화, OPEN 탐색, 경로 복원, 지표.
2. [LLMAStar.searching](../source_code/llmastar/pather/llm_a_star/llm_a_star.py): 기본 탐색에 LLM target 추가.
3. 같은 파일의 `heuristic`, `_initialize_llm_paths`, `_update_target`, `_update_queue`.
4. [prompt.py](../source_code/llmastar/pather/llm_a_star/prompt.py), [chatgpt.py](../source_code/llmastar/model/chatgpt.py), [llama3.py](../source_code/llmastar/model/llama3.py).
5. [환경](../source_code/llmastar/env/search/env.py), [좌표 파싱](../source_code/llmastar/utils/utils.py), [충돌 검사](../source_code/llmastar/utils/filter.py).

이 저장소는 클래스의 `searching(query)`를 호출하는 라이브러리 구조다. Table 1 전체를 실행하는 main이나 집계 도구는 없다. 반환값은 실제 경로가 아닌 지표 딕셔너리이며 경로는 내부에서 그림에 사용한다.

## 입력과 탐색 상태

query에는 `start`, `goal`, `range_x`, `range_y`, `horizontal_barriers`, `vertical_barriers`를 준다. README의 `size`는 두 탐색 구현에서 직접 사용하지 않는다. 8방향 이동에서 직선 비용은 1, 대각선은 sqrt(2)다.

- OPEN: `(f,좌표)`를 저장하는 최소 힙.
- CLOSED: 큐에서 꺼낸 노드 기록. 원본은 집합이 아닌 리스트.
- g: 시작점에서 해당 노드까지 현재 발견한 최소 이동 비용.
- PARENT: 경로 복원용 직전 노드.

이웃으로 가는 새 비용이 기존 g보다 작으면 g와 PARENT를 갱신하고 OPEN에 추가한다. 충돌하는 이동은 비용을 무한대로 설정한다. LLM-A*는 goal 거리와 현재 target 거리를 더한 heuristic을 쓰며, target 변경 때 OPEN의 f를 다시 계산한다. 정확한 waypoint 해석은 [문답](WAYPOINT_QA.md)을 참고한다.

## LLM과 프롬프트

딕셔너리 query는 그대로 사용하고 자연어 query는 모델로 JSON을 얻는다. 생성한 waypoint는 `list_parse`로 추출하고 환경 필터를 거쳐 시작·목표를 보충한다. 현재 target은 목록의 두 번째 점이다.

`standard`, `cot`, `repe`는 각각 5-shot, 3-shot, 3-shot 예시다. RePE는 한 번의 생성이며 별도 환경 피드백 루프가 없다. 반환값의 `llm_output`은 원문 응답이 아닌 후처리 target 목록이다.

## 지표의 실제 의미

- operation: 중복을 포함한 큐 추출 횟수.
- storage: g 딕셔너리의 항목 수. 무한 비용 항목도 포함.
- length: 복원한 경로의 이동 거리 합.

LLM 시간·토큰·OPEN 재계산 작업량이 operation에 직접 포함되는 것은 아니다. 고유 방문 수와 실제 메모리는 추가 지표로 측정해야 한다.

## 오프라인으로 재현한 문제

검증은 [check_baseline.py](check_baseline.py), 실제 출력은 [baseline_evidence.json](baseline_evidence.json)에 있다. 원본 알고리즘을 사용하되 LLM 응답은 고정하고 시각화는 비활성화했다. 실제 API 호출은 하지 않았다.

1. 입력의 range 리스트를 참조한 뒤 상한을 1 줄인다. 동일 query를 연속 사용하면 `[0,7] → [0,6] → [0,5]`가 되어 비교하는 맵이 달라진다. 알고리즘별 깊은 복사 또는 내부 입력 불변성이 필요하다.
2. 도달 불가능한 goal은 A*와 LLM-A* 모두 경로 복원에서 `KeyError`를 낸다. 명시적인 실패 상태가 없다.
3. 잘못된 좌표 목록은 parser가 오류 문자열을 반환하고 후속 처리에서 `IndexError`를 낸다.
4. 좌표가 없는 응답은 시작·목표만 남기며 `g+2h`로 동작한다. 일반 A* fallback이라고 부르면 안 된다.
5. 시작점과 목표점이 같고 target 목록이 한 점이면 `IndexError`가 발생한다.
6. 장애물 끝점이 obs에는 없지만 실제 선분 충돌 검사에는 걸려, 이동 불가능한 waypoint가 필터를 통과한다.
7. 이동 가능한 경계 근처 점은 필터가 제거하고, 정수 이동으로 도달할 수 없는 `[3.5,3.5]`는 통과한다.
8. 가짜 API 전송 객체로 확인한 GPT 요청은 `gpt-3.5-turbo-instruct`와 `stop=["\n"]`를 사용한다. 여러 줄의 CoT/RePE 생성에 영향을 줄 수 있다. 실제 응답 잘림 비율은 아직 측정하지 않았다.

## 정적 검토에서 확인한 추가 사항

- 오래된 중복 OPEN 항목을 꺼냈을 때 건너뛰지 않는다. CLOSED 리스트의 멤버십 검사 비용도 커진다.
- OPEN 재계산은 K개 항목에 반복 heappush하므로 일반적으로 O(K log K) 비용이다.
- target 전환이 충돌 검사보다 먼저다. 이 동작은 논문 의사코드의 순서도 함께 고려해야 한다.
- waypoint 생성 프롬프트에 실제 맵 경계를 전달하지 않는다.
- 매 탐색에서 그림 저장과 `plt.show()`를 호출해 대량 실험에 불편하다. 그림 함수는 전달된 visited 리스트도 변경하므로 실행 후 CLOSED를 볼 때 주의한다.
- Llama 장치가 `cuda:3`으로 고정되어 있다.
- A* 내부 복원 경로는 goal→start, LLM-A*는 start→goal이다. 길이는 같지만 경로 인터페이스는 다르다.
- 원본 requirements는 구형 OpenAI SDK를 고정하고 일부 다른 의존성은 고정하지 않아 실행 환경을 별도로 정리해야 한다.
- 현재 저장소에는 Dynamic WA*와 LLM-only 전체 평가, 기하평균 집계 및 전체 실험 실행 도구가 없다.

## 단일 사례 동작 확인

공개 map 0, pair 0에서 A*는 operation 468, storage 450, length 42.72792206135786이었다. 고유하게 꺼낸 상태는 323개, 중복 추출은 145회, g의 무한 비용 항목은 78개였다.

LLM-A*에 `[[5,5],[26,9],[25,23],[20,20]]`를 수동 응답으로 넣으면 operation 281, storage 298, 같은 경로 길이를 얻었다. 두 경로 모두 원본 충돌 검사상 유효했다. 이는 한 사례의 동작 검증이며 모델 성능·전체 성공률·Table 1 재현 결과가 아니다.

## 재현과 개선에 반영할 원칙

입력 독립성, 로그, 예외 기록, headless 실행, 모델 연결을 먼저 정리한다. 탐색 순서·waypoint 필터·지표 정의의 변경은 원본과 분리한다. 공통 수정과 새 알고리즘의 성능 기여를 섞지 않는다. 같은 응답 캐시로 원본과 제안 방법을 비교하고 원문 응답·파싱 상태·후처리 target·실제 경로를 따로 저장한다.

검증 스크립트의 통과는 위 동작이 재현되었다는 뜻이며, 원본의 모든 기능이나 정확성을 보증하지 않는다. 다음 작업은 [프로젝트 계획](../LLM-Astar_Project_Plan.md)을 따른다.
