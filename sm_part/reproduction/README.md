# LLM-A* GPT API reproduction

논문: LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning, arXiv:2407.02511v2.

이 스냅샷은 완료된 GPT 기반 8개 조건과 GPT로 변경한 확장성 실험을 포함합니다. [GPT 보고서](../results/report/report.html), [Table 1 CSV](../results/report/table1.csv), [완전성 검증](../results/verification.json)을 함께 제공합니다. 진행 중인 Llama3 실행 코드·서버 설정·결과는 이번 커밋 범위에서 제외했습니다. 아래 명령은 저장소 루트에서 실행합니다.

## 최초 설정

저장소의 `source_code/`는 기존 원본 자료입니다. 이번 재현 실행기는 원본 Git 커밋까지 기록하기 위해 `vendor/llm-astar/`의 별도 고정 체크아웃을 사용합니다. 이 디렉터리는 Git 추적 대상에서 제외됩니다.

```powershell
git clone https://github.com/SilinMeng0510/llm-astar.git vendor/llm-astar
git -C vendor/llm-astar checkout 9414ed894d774f71c36e8c0afd0e20e6033b95b1
py -3.12 -m venv .venv
```

저장된 결과의 테스트·검증·보고서 생성에는 API 키가 필요하지 않습니다. 새 응답 수집에만 별도 키 파일이 필요합니다.

## 환경

- Windows / Python 3.12, 프로젝트 가상환경 `.venv`.
- 원본: https://github.com/SilinMeng0510/llm-astar
- 고정 커밋: `9414ed894d774f71c36e8c0afd0e20e6033b95b1`
- 설치 버전은 `requirements-lock.txt`에 기록됩니다.
- 키는 사용자가 제공한 `C:\api\gpt.txt`에서 메모리로만 읽습니다. 키를 프로젝트에 복사하지 마세요.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduction/test_reproduction.py
.\.venv\Scripts\python.exe reproduction/run.py audit
```

## 실행

`collect`는 과금되는 실제 API 호출입니다. 저장된 응답이 있으면 요청 해시를 확인하여 재사용합니다. 테스트/평가/보고서 작성은 API를 호출하지 않습니다.

```powershell
.\.venv\Scripts\python.exe reproduction/api_client.py --key-file C:\api\gpt.txt
.\.venv\Scripts\python.exe reproduction/run.py collect --stage pilot --workers 3
.\.venv\Scripts\python.exe reproduction/run.py evaluate --stage pilot
.\.venv\Scripts\python.exe reproduction/run.py collect --stage main --workers 8
.\.venv\Scripts\python.exe reproduction/run.py evaluate --stage main
.\.venv\Scripts\python.exe reproduction/run.py collect --stage scaling --workers 3
.\.venv\Scripts\python.exe reproduction/run.py evaluate --stage scaling
.\.venv\Scripts\python.exe reproduction/report.py
.\.venv\Scripts\python.exe reproduction/verify_artifacts.py
```

## 프로토콜과 해석

1. 본 평가: 저자 데이터 100개 지도 × 10개 쌍 = 1,000문제. 개발용 10개 합성 문제는 분리합니다.
2. GPT 모델: `gpt-3.5-turbo-0125`, temperature=0, max_tokens=1000, 단일 user 메시지에 저자 프롬프트 원문을 전달합니다. 3종 × 1,000문제 = 3,000개 응답을 GPT-only/LLM-A*에 공유합니다.
3. 논문은 Turbo를 명시하지만 공개 코드의 경유점 생성은 Turbo-Instruct입니다. 이 차이와 Chat Completions 전환을 보고서에 명시합니다. 과거의 정확한 모델 스냅샷 재현이라고 주장하지 않습니다.
4. 공개 코드의 줄바꿈 stop은 개발 문제의 CoT/RePE 경로 출력을 차단하여 제외했습니다. 3개 비교 응답은 `results/pilot/newline_audit`에 있습니다.
5. 논문의 operation/storage는 heap pop 횟수/len(g)입니다. 중복 pop도 유지합니다. 경유점 사이를 독립 A*로 풀지 않습니다. 목표 전환 시 OPEN을 재계산합니다.
6. 원본 입력의 in-place 변경을 방지합니다. 원본 `range=[0,51]`을 경계 좌표 0~50으로 해석합니다. 출력이 없으면 빈 경유점으로 기록하고 시작·목표를 추가합니다. 이때 hybrid는 g+2h이며 일반 A* fallback이 아닙니다.
7. Dynamic Weighted A*는 논문에 주어진 초기 2, 감쇠 .99를 사용하되 누락된 세부 규칙을 재구성했습니다: `max(1,2*.99**pop_count)`, 매 확장 후 OPEN 재계산. 이 비교군의 정확한 논문 구현 일치는 보장되지 않습니다.
8. 실패 응답은 재생성하지 않습니다. HTTP 429(쿼터 부족 제외)/5xx만 제한 재시도합니다. 모델 오류/쿼터 부족은 수집을 중단합니다. 각 요청 직후 결과를 원자적으로 저장합니다.
9. 길이는 유효 경로에 대한 기하평균을 주 통계로 제시합니다. 전체 파싱 가능 경로의 길이·분모도 CSV에 저장합니다. 실패율은 전체 분모이며 숨기지 않습니다.
10. 원본 데이터의 예시 중복 5개는 원본 유지 결과에 포함하고, 합집합 5개를 공통 제외한 995개 결과도 제공합니다.
11. 확장성은 GPT 변경 실험입니다. 고정 시드 240702511로 10개 문제를 뽑아 좌표·장애물을 1~10배 확대합니다. 각 문제의 규모 1 대비 증가율을 산술평균합니다. 원문의 Llama Figure 3과 동일 실험이라고 주장하지 않습니다.
12. 검색 실행 시간은 충돌 캐시에 영향을 받습니다. operation/storage를 실제 지연·RAM으로 해석하지 마세요. temperature=0도 서버의 완전한 결정성을 보장하지 않으므로 저장 응답을 기준으로 재평가합니다.

## 파일

- `reproduction/core.py`: 데이터, 원문 프롬프트, 안전한 좌표 파싱, 경로 검증, 탐색
- `reproduction/api_client.py`: OpenAI 공식 호스트에만 키를 전송하는 API 클라이언트
- `reproduction/run.py`: 데이터 감사, 중단 후 재개 가능한 수집·평가
- `reproduction/test_reproduction.py`: 저자 코드와의 차등 검증, Dijkstra 검증, 기하 검증
- `reproduction/report.py`: CSV/JSON, PNG/PDF 그래프, 한국어 HTML 보고서
- `reproduction/verify_artifacts.py`: 최종 응답 수·평가 행·모델·해시·조건별 완전성 확인
- `results/{pilot,main,scaling}/raw`: 실제 API 요청 및 응답(인증 정보 제외)
- `results/{pilot,main,scaling}/evaluation`: 문제별 최종 경로 및 지표
- `results/report`: 보고서, Table 1 결과, 중복 제외 결과, 그래프

저자 알고리즘을 이식한 부분은 원본 MIT 라이선스를 따릅니다. 원본 고지와 라이선스는 `vendor/llm-astar/LICENSE`에 보존되어 있습니다.
