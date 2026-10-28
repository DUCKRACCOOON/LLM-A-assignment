# LLM-A* GPT API / Llama Table 1 reproduction

## 세 가지 추가 탐색 실험

완료: 2026-10-14 주 실험 43,520행과 Euclidean/octile A* 보조 비교 1,200행을
서버와 로컬에서 검증했습니다. 실행 실패·경로 및 길이 상한 위반은 0건입니다.
[완료 보고서 (Markdown)](results/next_search_final/report.md)에 설정·전체 결과·
비용 분해·확대 및 반복 질의·한계·후속 방향을 정리했습니다.

기본 크기에서 R1은 대응 확장 수 69.9% 감소에도 준비 포함 시간은 A*의 2.033배였고,
R2는 1.871배, 결합형은 2.890배였습니다. R3 LLM은 검증에서 규칙보다 느려
최종 LLM 생성을 사전 기준에 따라 생략했습니다. 8배 확대에서는 R1 시간이
0.698배였으나 일부 질의의 큰 퇴행이 있어 신규 지도에서 조건을 재검증해야 합니다.
[연결 장애 당시 중간 기록](results/next_search_v1/interim_report.md)은 이력으로 보존합니다.

[실험 계획](results/next_experiments_plan/plan.md)에 따라 R1 정확한 공간 압축,
R2 장애물 단계적 반영, R3 위상 기반 안내를 순서대로 평가했습니다.
새 구현은 `reproduction/next_*.py`, 검증은 `reproduction/test_next_search.py`입니다.
Sirius 실행 폴더는 `/home/work/fanimal_hidream/next-search-20261014-v2`이며,
[실행 메타데이터](results/next_search_deployment.json)에 현재 실행과 수정 이력을 기록합니다.
`results/next_search_v1/verification.json`과 `local_verification.json`에서
검증 완료를 확인할 수 있습니다. 최단 길이·근사 상한·준비 포함 시간 및
5회 반복 측정을 검증했으며, 기존 Llama 모델 생성은 174회, 유료 API 호출은 0건입니다.
주 결과 ZIP은 `next-search-results-verified.zip`, 보조 ZIP은
`octile-supplement-results-verified.zip`입니다. 해시와 검증·재생성 명령은 완료 보고서에 있습니다.

## 네 가지 추가 후보 실험 (2026-10-14)

`reproduction/candidate_run.py`는 Conformal 후보 영역, CHOMP 계열 수리, Fast Marching,
TransPath 계열 학습을 같은 FOCAL(w=1.1) 탐색기에서 비교합니다. 기존 GPT 응답을 재사용하며
새 API 호출은 없습니다. CHOMP·TransPath는 이번 격자 문제에 맞춘 구현으로 원 논문의 정확
재현을 주장하지 않습니다. Conformal은 세 LLM 제안 주변의 **격자 후보 영역**을 보정합니다.

지도 단위로 학습 50개, 검증 10개, 보정 20개, 평가 20개를 분리합니다. 별도 미로와 확대
평가를 포함하며, 3개 학습 시드와 학습 전 네트워크·LLM 없는 대조군을 둡니다.
학습·정답 생성 비용, 수리·FMM 준비 시간, GPU 추론 시간을 기록합니다.
실제 완료 여부는 `results/candidates_v1/status.json` 및 `verification.json`으로 확인합니다.

```bash
python reproduction/test_candidates.py
python reproduction/candidate_run.py all --gpu 0
python reproduction/candidate_run.py verify
python reproduction/candidate_report.py
```

실행에는 GPU용 PyTorch와 NumPy가 필요합니다. Sirius의 `/usr/bin/python3`를 기반으로 만든
`.venv-candidates` 환경을 사용하고 Shapely 2.2.0을 별도 설치해 기본 이미지의 GEOS 누락을 해결했습니다.
서버 전용 폴더는 `/home/work/fanimal_hidream/candidates-20261014`입니다.
결과 ZIP을 내려받은 뒤 `reproduction/import_candidates.py ARCHIVE --sha256 SERVER_HASH`로
파일 해시·로컬 소스·4,100개 평가 경로·길이 상한을 다시 검증합니다.

완료: 4,100개 평가와 서버·로컬 검증을 통과했으며, 60문제의 최적 길이가 기존 Dijkstra와
일치했습니다. [최종 해석 보고서](results/candidates_final/report.html)와
[서버 상세 보고서](results/candidates_v1/report/report.html)에 결과를 정리했습니다.
주 평가에서 FMM+LLM의 탐색량은 기본 FOCAL 대비 25.58% 감소했지만 준비 포함 시간은
증가했습니다. 다른 후보도 이번 구현에서 일관된 시간 단축을 보이지 않았습니다.
가중치·원자료는 `candidate-results-verified.zip`이며 SHA-256은
`b811a910dfd5e51bdd71d73440aecaec3f95e1471c91137120159b0d28686509`입니다.

논문: LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning, arXiv:2407.02511v2.

Table 1의 GPT·기본 알고리즘 8개 조건과 Llama 6개 조건을 완료하고 로컬 검증까지 마쳤습니다. 최종 통합 보고서는 [results/llama/final_report/report.html](results/llama/final_report/report.html), 논문·재현·차이 표는 [CSV](results/llama/final_report/table1_comparison.csv)와 [JSON](results/llama/final_report/table1_comparison.json)입니다. 기존 GPT 보고서와 GPT로 변경한 확장성 실험은 `results/report/report.html`에 보존됩니다. Llama Figure 3·4는 이번 완료 범위에 포함하지 않습니다.

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

## Sirius Llama Table 1 실행

모델은 `meta-llama/Meta-Llama-3-8B-Instruct`, BF16, greedy, 배치 1입니다. Llama 전용 원문 프롬프트를 사용합니다. 원본의 `cuda:3`은 저자의 로컬 장치 번호이므로 서버에서는 할당된 GPU 0을 기본으로 사용합니다. 모델·토크나이저 revision, 가중치 해시, 실제 라이브러리 버전과 생성 토큰을 저장합니다. 저자는 Transformers 버전을 고정하지 않았으므로 이번 고정 버전 4.44.2와 역사적 실행 환경의 동일성은 보장되지 않습니다.

```powershell
.\.venv\Scripts\python.exe reproduction/test_llama.py
.\.venv\Scripts\python.exe reproduction/package_llama.py
```

생성된 `dist/llama-reproduction.zip`에는 명시적으로 허용한 코드·공개 데이터·기존 비교 통계만 포함됩니다. 인증 키는 포함하지 않습니다. 서버 영구 저장 폴더에 업로드·추출한 뒤 실행합니다. Sirius의 기본 conda `python`은 Python 3.14일 수 있으며, NGC CUDA PyTorch는 `/usr/bin/python3`의 Python 3.12에 설치되어 있습니다.

```bash
bash reproduction/server_setup.sh
.venv-llama/bin/python reproduction/server_smoke.py
.venv-llama/bin/python reproduction/llama_run.py download --prompt-token
bash reproduction/server_start.sh
```

다운로드는 모델 접근 승인을 받은 Hugging Face 읽기 토큰이 필요합니다. `--prompt-token` 입력은 화면에 표시하지 않고 저장하지 않습니다. `server_start.sh`는 다운로드 완료 후 예비실험 30응답, 본실험 3,000응답, 6,000개 평가, 검증 및 전체 Table 1 보고서를 순서대로 실행합니다. 진행은 `results/llama/status.json`과 `run.log`에 기록됩니다. nohup은 터미널 연결 종료를 견디지만, Backend.AI 세션 종료·만료 시에는 실행이 중단됩니다. 그 경우 영구 폴더의 결과를 사용해 같은 환경에서 재개합니다.

원본 생성 상한 8,000토큰은 입력을 포함한 모델의 총 문맥 한도를 넘지 않도록 제한합니다. 입력 프롬프트는 자르지 않습니다. 종료 토큰 전 길이 제한에 도달한 응답도 그대로 평가하며, `finish_reasons`에 기록합니다. 원문의 raw 프롬프트와 pipeline 기본 특수 토큰 처리를 유지하며 별도의 chat template을 다시 씌우지 않습니다.

이번 Llama 범위는 Table 1의 Few-shot·CoT·RePE × Llama-only·Llama-A*입니다. Llama Figure 3·4는 포함하지 않습니다. 기존 GPT 결과와 A* 기준값이 일치해야 통합 보고서를 생성합니다. 실패 응답 재생성, 부분 결과의 완료 표시, 서로 다른 실행 설정의 혼합을 막습니다.

2026-10-14 완료 상태: Sirius에서 본 실험을 00:54 KST에 완료했습니다. 공식 모델 revision은 `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`입니다. 예비 응답 30개, 본 응답 3,000개, 본 평가 6,000행을 확인했습니다. 정상 종료 2,994개와 길이 제한 종료 6개(RePE)를 그대로 평가했으며 실패 응답을 재생성하지 않았습니다. 추가 GPT 호출은 없습니다.

Jupyter 다운로드가 HTML 오류 파일을 반환하여 Sirius의 **데이터 → fanimal_hidream → llama-reproduction/results/llama → 파일 다운로드**로 원본 ZIP을 받았습니다. `llama-table1-verified.zip`은 7,882,469바이트이며 SHA-256은 `79d2b7b40db60197da63b4644e56850bc31802f035da60530d2f20afeb78fed9`입니다. 내부 4,078개 파일의 해시, 누락·중복, 로컬 프로토콜 소스 일치, 기존 GPT·A* 기준값 일치와 14조건 집계 재계산을 확인했습니다. 검증 기록은 `results/llama/local_verification.json`입니다. ZIP에는 키·인증 토큰·모델 가중치를 포함하지 않았습니다.

Llama-A*의 세 프롬프트 모두 유효 경로 100%를 달성했습니다. 연산량은 A*의 45.83%(Few-shot), 48.74%(CoT), 66.13%(RePE)로 논문보다 각각 +1.24, +1.14, +2.05%p 높았습니다. Llama-only RePE의 유효 경로 비율은 논문 16.4% 대비 8.4%로 낮아 프롬프트 간 순위까지 재현했다고 볼 수 없습니다. 정확한 모델·생성 조건과 비교 한계는 최종 보고서에 기록했습니다.

아래 명령은 저장 파일만 검증·집계하며 API 호출이나 모델 추론을 하지 않습니다.

```powershell
.\.venv\Scripts\python.exe reproduction/import_llama_results.py llama-table1-verified.zip --sha256 79d2b7b40db60197da63b4644e56850bc31802f035da60530d2f20afeb78fed9
.\.venv\Scripts\python.exe reproduction/final_llama_report.py
```
