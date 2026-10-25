# 재현 실행 규약

2026-10-13. 원 논문: Meng et al., *LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning*, arXiv:2407.02511v2. 공식 코드의 MIT 라이선스와 출처는 `../UPSTREAM.md`, `../source_code/LICENSE`를 따른다.

## 고정 조건

- 공개 데이터 100개 맵 × 10쌍 = 1,000개. 모든 방법에 같은 문제를 사용한다. 원 JSON을 수정하지 않는다.
- 실제 호출을 확인한 주 모델 `gpt-4o-mini-2024-07-18`, 비교 모델 `gpt-3.5-turbo-0125`. 과제 평가 모델과 논문 GPT-3.5 계열의 연결을 우선한다. 논문의 Llama3 행은 GPT-4o mini로 대체했음을 명시하며 원 두 모델의 완전한 재현으로 주장하지 않는다.
- 공식 `gpt_prompt`의 standard/Few-shot, CoT, RePE 문자열을 그대로 읽는다. Chat Completions의 user 메시지 한 개로 전달한다. temperature=0, max_tokens=1000, store=false. 원 코드의 줄바꿈 stop은 다중 행 CoT/RePE를 잘라내므로 사용하지 않는다. 원 Completion API 및 instruct 모델과 다른 조건이다.
- 한 응답을 LLM-only/LLM-A* 두 평가에 공유한다. 기본 생성 6,000회. 각 응답 원문, 요청 전체, 모델 ID, 토큰, finish_reason, 요청 ID, 시각, 지연과 재시도를 저장한다. 인증 헤더는 저장하지 않는다.
- 파일럿은 map index 0,10,...,90의 pair 0,1,2: 30문제/180응답. 결과가 좋은 문제를 사후 선택하지 않는다. 같은 생성 설정을 유지하면 전체 실행에 포함한다.
- 원본의 1,000토큰 한도를 유지한다. 길이 제한으로 잘린 응답과 파싱 실패도 결과다. 잘 나온 경로가 나올 때까지 다시 묻지 않는다. 전송·429·5xx 오류만 최대 4회 시도하고 인증·권한·모델 없음·잔액 오류에서는 작업을 중단한다.
- 필요한 재현 비용은 사용자가 승인했다. 임의로 추가 모델이나 반복 실험을 늘리지 않는다. 환경변수 또는 Git에서 제외한 `.env`로 인증한다.

## 탐색 동작과 알려진 차이

`core.py`는 원본 A*/LLM-A* 동작을 별도 구현했다. 동률은 `(f, (x,y))` 순서, 이동은 원본 8방향 순서, 중복 pop과 무한 g 엔트리는 보존한다. CLOSED의 membership만 set으로 가속하고 원본과 같은 반복 확장을 유지한다. target은 이웃을 관측할 때 충돌 검사보다 먼저 바뀌며 OPEN의 모든 엔트리를 다시 넣는다. 원본의 경계 근처 waypoint 필터와 장애물 끝점 필터의 비대칭도 보존한다.

충돌은 축과 평행한 닫힌 선분 교차로 계산하고 맵별 이웃 비용을 캐시한다. 원본 Shapely에서 교차하지 않는 길이 0의 장애물 선분은 같은 방식으로 무시한다. 경로·operation·storage·length의 원본 동등성은 `python -m reproduction.verify`로 검사한다. 속도 최적화된 구현의 실행 시간으로 원본 코드가 더 빠르다고 주장하지 않는다.

동일 시작·목표 및 도달 불가에서는 원본 예외 대신 구조화한 결과를 반환한다. 평가되지 않는 표현식 등 실제 원본 파싱 실패는 실패로 남긴다. 숫자가 3개 이상인 항목은 원본처럼 앞 두 개를 waypoint로 해석한다. 2차원 schema 위반은 별도로 기록하며 LLM-only와 개선안에서는 2차원 좌표 목록을 요구한다. 좌표 목록이 없는 출력은 원본과 같은 start/goal fallback을 사용한다. 입력 dict를 원본처럼 변경하지 않는다. 이 차이는 보고서에서 명시한다.

Dynamic Weighted A*는 원 저장소에 구현이 없다. 여기서는 초기 w=2, 목표가 아닌 노드를 pop한 뒤 w=max(1,0.99w), 남은 OPEN의 모든 우선순위 재계산으로 고정한다. 논문에 불명확한 감쇠 사건·하한·재정렬 규칙에 대한 **우리 구현 가정**이다.

## 지표

- operation: 중복을 포함한 pop 수. storage: 무한 비용을 포함한 g 엔트리 수. 실제 CPU 연산 수/메모리 바이트와 같지 않다.
- 각 문제의 방법/A* 비율을 구한 뒤 기하평균 ×100. 낮을수록 적게 사용한다.
- valid: 시작·목표 일치, 경계 내부, 전체 경로 충돌 없음. 탐색 경로는 8방향 이웃 이동도 확인한다.
- LLM-only는 응답의 원 좌표를 잇는 polyline이다. 좌표를 추가하거나 탐색으로 수리하지 않는다. 원 Table 1처럼 operation/storage는 정의하지 않는다.
- LLM-only의 실패 경로 길이 집계 규칙은 논문에 명확하지 않다. 주 표는 유효 경로에 한해 길이를 집계하고 분모를 표시한다. 실패를 제거한 성공률은 사용하지 않는다. 공통 성공 부분집합 비교와 무효 polyline의 길이는 보조 분석으로 구분한다.
- 누락 API 응답을 성공이나 완료로 처리하지 않는다. 14행 ×1,000개 상태와 원문 6,000개가 있어야 완료다.

## 추가 진단과 개선

g+2h 대조군은 LLM 없이 target=goal인 원 탐색 엔진이다. 원 Table 1의 DWA와 별도 행이다. 같은 응답을 재사용한 개선 비교와 맵 단위 bootstrap은 새 생성 실험으로 세지 않는다. 실패와 악화도 보존하고 원인 가설은 실제 관측과 구분한다. 개선안은 파일럿 및 원본 결함에서 정하고 전체 결과를 보고 맵별로 유리하게 바꾸지 않는다.

## 실행

```bash
python -m pip install -r requirements.txt
python -m reproduction.verify
python -m reproduction.experiment generate --limit 30 --output results/main --workers 12
python -m reproduction.experiment evaluate --output results/main
python -m reproduction.summarize --output results/main
python -m reproduction.experiment generate --output results/main --workers 24
python -m reproduction.experiment evaluate --output results/main
python -m reproduction.summarize --output results/main
```

같은 폴더에서 generate/evaluate를 반복하면 이미 기록된 요청/평가를 건너뛴다. 평가 알고리즘을 변경해 다시 분석할 때는 기존 결과를 보존하고 **새 결과 폴더**에서 원문 캐시를 사용한다. 코드 해시·데이터 해시·프롬프트 해시·Git 커밋과 환경은 실행 manifest에 저장한다. 저장 응답을 재생하는 평가에는 API 키가 필요 없다.

verify는 원본의 eager import에 필요한 openai/torch/transformers 이름을 오프라인 빈 모듈로 대체하고 고정 응답을 공급한다. 실제 모델 호출을 검사하는 테스트가 아니며 원본 탐색·충돌·경로 복원 코드는 그대로 실행한다. 대형 모델 다운로드 없이 requirements.txt로 검증할 수 있다.

## 파일럿 후 고정한 개선안: V-Gate LLM-A*

파일럿에서 생성 경로 충돌과 malformed 좌표가 관측되었다. 전체 polyline이 유효하고 모든 좌표가 정수인 경우에만 원 LLM-A* 안내를 사용한다. 그렇지 않으면 같은 탐색 엔진의 target=goal(`g+2h`)로 전환한다. 경로를 여러 번 생성하거나 좌표를 사후 최적화하지 않는다. 가정은 "충돌 없는 정수 polyline이 더 신뢰할 만한 안내"이며 성능은 실제 비교로 판단한다. 단순한 검증·fallback 제안으로, 새로운 탐색 이론이나 최적성 보장을 주장하지 않는다.

ablation은 (1) 2차원 좌표 schema만 검증하고 공간 검증을 제거, (2) 검증은 유지하고 fallback을 일반 A*로 변경, (3) LLM을 전혀 쓰지 않는 g+2h이다. 두 모델·세 프롬프트에서 같은 캐시로 비교한다. 제안안이 g+2h보다 못하면 이를 그대로 보고하고 추가 LLM 비용이 정당화되는 범위를 제한한다. 공개 맵에 맞춘 개별 규칙이나 성능을 본 후의 임계값 조정은 하지 않는다.
