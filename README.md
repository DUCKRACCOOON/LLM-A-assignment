# LLM-A* assignment

`jy` 브랜치의 J3~J6 구현·비교는 [실험 안내](experiments/JY/README.md)와 [결과 보고서](experiments/JY/REPORT.md)를 참조한다. 아래 `gm` 재현·선정 기록은 기반 브랜치의 자료로 보존했다. 동적 경유점 생성은 이번 실험에서 보류했다.

SWE3011, Fall 2026. 팀원 3명이 각각 전체 재현 후 제출 기반을 선택한다. `gm`은 사용자의 독립 재현이다.

**2026-10-25 최종 개선안 선정: K3 직선거리＋벽 하한 (`g2h_wall_euclidean`, `g+2h_K3`).** [선정 설정·근거·한계·후속 작업](최신/최종_개선안_선정.md)을 기준으로 진행한다. K3–K6 및 K6 5% 제한 대조 결과에 따라 사용자가 확정했다. 이번에는 선정 내용만 정리했으며 제출용 `Proposed` 연결과 보고서·PDF·ZIP 반영은 후속 작업이다.

**2026-10-13: 6,000개 실제 응답 수집과 Table 1의 14개 조건 ×1,000문제 평가를 완료했다.** V-Gate 개선안·ablation 18,000건, 새 맵 30문제, 원본 동등성·전체 경로·독립 Dijkstra 검증도 완료했다. API 전송 오류는 0건이다. 보고서는 제출 전 팀 정보와 설명을 검토할 초안이다.

## 결과

- [규민 개선방법 실험 장부](experiments/README.md): **2026-10-25 K3 직선거리＋벽 최종 선정.** [원 K3 실험 기록](experiments/K3/REPORT.md)은 당시 주 설정 octile＋벽(pop 48.74% 감소)과 직선거리＋벽(pop 51.81% 감소)을 구분해 보존했다. 합성 지도에서는 시간 증가도 확인했다.
- [K4 FOCAL 구현·비교 결과](experiments/K4/REPORT.md): **2026-10-25 완료.** 공개 1,000문제와 기존 별도 지도 30문제에서 5% 비용 제한을 지켰지만 K3보다 연산·시간이 증가했다. LLM 제거 대조군과 12조건의 원시 결과를 보존했다.
- [K5 MHA*＋DTS 구현·비교 결과](experiments/K5/REPORT.md): **2026-10-25 완료.** 공개 1,000문제·별도 30문제, 13조건·확률 조건 5시드로 검증했다. 주 K5는 K3보다 pop 2.06배·계획기 시간 3.88배로 불리했고, DTS의 확장 절감과 관리 비용을 구분해 기록했다.
- [K6 적응형 경유점 가중치 구현·비교 결과](experiments/K6/REPORT.md): **2026-10-25 완료.** 19조건·공개 1,000문제·별도 30문제로 g+2h·K3·K4·K5와 비교했다. 주 K6는 K3보다 pop 22.28%·시간 64.98% 증가했다. 별도 5% 제한 조건의 장점, 안내 제거 대조, OPEN 재계산 비용과 악화 사례를 함께 기록했다.
- [K6 5% 제한 후속 대조](experiments/K6_w105/REPORT.md): **2026-10-25 완료.** LLM 제거·고정 α 및 K3 직선거리＋벽을 7조건으로 비교했다. 적응 안내는 no-guide 대비 pop 5.05% 감소, 실제 확장 2.97%·시간 20.37% 증가였고 길이는 사실상 같았다. 작은 고정 α 품질 이득과 비용을 함께 기록했다.
- [한국어 결과·해석](최신/재현_실험_결과.md)
- [V-Gate 이후 개선 방법 13가지 조사](최신/V-Gate_이후_개선_방법_조사.md): 2026-10-14 문헌 조사와 추천 조합. 새 실험을 진행하지 않은 제안 문서.
- [전체 14행 표](results/main/table1.md), [CSV](results/main/table1.csv), [집계 JSON](results/main/summary.json)
- [ACL 보고서 PDF](output/pdf/LLM_Astar_Reproduction_Report.pdf), [LaTeX](report/LLM_Astar_Reproduction_Report.tex)
- [코드·응답·문서 ZIP](output/LLM_Astar_CodeFile.zip)
- [실행 규약](reproduction/PROTOCOL.md), [구현 출처·검증·AI 지원 기록](reproduction/AUDIT.md)

GPT-4o mini Few-shot LLM-A*는 A* 대비 연산량 44.85%, 경로 길이 102.65%, 유효 경로 100%였다. V-Gate는 연산량 34.63%였지만 LLM 없는 g+2h도 34.32%였다. **LLM 안내가 단순 가중 탐색보다 유리하다는 결론은 얻지 못했다.** 감소의 원인을 구분하는 대조 분석을 보고서에 담았다.

## 설치

Python 3.10 이상. 탐색·API 호출·집계는 표준 라이브러리만 사용한다. 원본 비교·그림 생성에는 아래 패키지가 필요하다. 실제 버전은 [환경 기록](results/verification/environment.json)에 있다. torch나 모델 가중치는 필요 없다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

캐시 재생에는 API 키가 필요 없다. 새 응답을 생성할 때만 환경변수 `OPENAI_API_KEY` 또는 로컬 `.env`에 설정한다. `.env.example`은 값 없는 예시다. 실제 키는 Git·ZIP에 포함하지 않는다.

## 검증과 저장 응답 재생

```bash
python -m reproduction.summarize --output results/main
python -m reproduction.verify
python -m reproduction.verify_results
```

`verify`는 모델 패키지를 빈 모듈로 대체하고 고정 응답으로 원본 탐색·충돌 코드와 비교한다. `verify_results`는 6,000개 요청, 15,000개 평가 기록(A*/DWA/g+2h 포함), 모든 성공 경로, 전체 1,000개의 Dijkstra 최단거리, 실제 응답 62건의 원본 동등성을 검사한다.

처음부터 재계산하려면 새 폴더에 원문 응답만 복사한다. PowerShell 예시:

```powershell
New-Item -ItemType Directory -Force results/replay
Copy-Item results/main/generations.jsonl results/replay/generations.jsonl
python -m reproduction.experiment evaluate --output results/replay
python -m reproduction.summarize --output results/replay
python -m reproduction.diagnostics --output results/replay
```

외부 패키지를 비활성화한 별도 Python에서도 15,000개 지표·경로가 모두 일치했다. [API 호출 0회 재생 검사](results/verification/clean_replay.json).

## 새 응답 생성

아래는 새 폴더에 실제 API 요청을 만드는 명령이다. 같은 폴더를 재사용하면 기록된 요청을 건너뛴다. 본 실험 결과 확인에는 다시 호출할 필요가 없다.

```bash
python -m reproduction.experiment generate --limit 30 --output results/new_run --workers 12
python -m reproduction.experiment generate --output results/new_run --workers 24
python -m reproduction.experiment evaluate --output results/new_run
python -m reproduction.summarize --output results/new_run
python -m reproduction.diagnostics --output results/new_run
```

기본 모델은 `gpt-4o-mini-2024-07-18`, `gpt-3.5-turbo-0125`다. 논문의 Llama3를 GPT-4o mini로 대체했다. 모델 변경은 `--models`와 새 폴더로 명시한다. 종료된 모델을 자동 대체하지 않는다. 원 프롬프트 세 가지, temperature=0, max_tokens=1000, 제한된 전송 재시도를 사용한다. [실행 규약](reproduction/PROTOCOL.md).

## 과제 평가용 입력과 출력

현재 아래 `Proposed`는 기존 V-Gate 구현이다. 최종 선택한 K3의 현재 호출법은 [선정 메모](최신/최종_개선안_선정.md#확정한-설정)에 있으며 제출 진입점 연결은 남아 있다.

```python
from reproduction.planners import AStar, LLMAStar, Proposed
query = {
    "start": [5, 5], "goal": [20, 20],
    "range_x": [0, 51], "range_y": [0, 31],
    "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
    "vertical_barriers": [[25, 10, 22]],
}
baseline = AStar().searching(query, filepath="astar.png")
original = LLMAStar(model="gpt-4o-mini", prompt="standard").searching(query, filepath="llm_astar.png")
improved = Proposed(model="gpt-4o-mini", prompt="standard").searching(query, filepath="proposed.png")
print(improved["operation"], improved["storage"], improved["length"])
```

원본 `searching(query, filepath=...)`, `operation/storage/length`, hybrid의 `llm_output`을 제공하고 `path/valid/error` 등을 추가한다. `filepath=None`이면 그림을 생략한다. 자연어 입력은 원 parsing prompt로 dict로 변환하므로 모델 호출이 추가된다. Table 1은 dict 입력이다. 별도 TA 진입점이 공지되면 이 클래스를 연결한다. 새 맵마다 수동 수정할 필요가 없다.

## 추가 분석과 보고서

```bash
python -m reproduction.generalization             # 새 맵 저장 응답 재평가
python -m reproduction.generalization --generate  # 응답이 없으면 최대 30회 생성
python -m reproduction.report                     # 그림·LaTeX·한국어 설명 생성
python -m reproduction.package                    # 코드 ZIP 생성
```

컴파일은 [보고서 README](report/README.md)를 참고한다. `seconds`는 명시된 API/탐색 단계 값이며 원본 전체 프로그램의 실행시간 비교가 아니다. 예시 중복 5/3/3문제 제외 결과, LLM-only 공통 성공 집합, 비용·지연, 맵 bootstrap은 `results/main`에 있다. LLM-only 여섯 조건의 공통 성공 사례는 0개여서 그 집합의 평균 길이를 계산하지 않았다.

## 폴더와 출처

- `materials/`: [안내서](<materials/SWE3011_41 Assignment Notice (1).pdf>), [논문](materials/인지개과제_논문.pdf)
- `source_code/`: [공식 코드](https://github.com/SilinMeng0510/llm-astar)의 원본. [기준 커밋](UPSTREAM.md)과 [MIT 라이선스](source_code/LICENSE)
- `review/`: [논문 리뷰](review/PAPER_REVIEW.md), [코드 리뷰](review/CODE_REVIEW.md), [waypoint 문답](review/WAYPOINT_QA.md)
- `reproduction/`: 구현·실험·집계·검증·제안법
- `results/main/`: 최종 응답/결과. `results/development/`는 수정 전 개발 기록이며 최종 표에 사용하지 않는다.
- `results/generalization/`, `results/verification/`: 새 맵·검증 기록
- `report/`, `output/`: ACL 초안·그림·PDF·코드 ZIP
- `최신/`: 최신 설명과 [과제 지침 대응](최신/과제_지침_대응.md)

논문·과제 문서의 권리는 각 권리자에게 있다. 원본 이식과 새 구현은 [AUDIT](reproduction/AUDIT.md)에 구분했다. 팀 최종 버전 선택, 팀명·저자·제출 파일명 반영, 사용자의 설명·문장 검토, 개인 기여 보고서, 실제 제출·동료평가·답변은 별도다. Codex의 지원을 기록했으며 AI 탐지 회피나 결과 조작을 하지 않는다. 개인 기여 보고서는 공유 저장소에 넣지 않는다.
