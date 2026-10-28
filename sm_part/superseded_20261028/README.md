# 승민 S1~S6 실험 자료

원격 `main`의 [팀 실험 분담표](docs/팀_개선방법_실험_분담.md)에 따라 승민 담당 S1~S6의 기존 실험 자료를 모았다. 2026-10-25에 원격 `sm`을 `ea62233079a94ddd1a43d7a9ecb06d5b9077c7e6`까지 pull한 뒤 로컬의 2026-10-14 실험 자료를 추가했다. 분담 기준 `main` 커밋은 `bf4ae439094e0df003c02c0b4b4e607d5a330238`이다. GitHub의 실제 브랜치 이름은 소문자 `sm`이다.

기존 모음의 ‘S1~S6 자료 미확보’ 상태를 갱신했다. 코드, 고정 설정, 입력, 응답, 문제별 평가, 학습 가중치, 원래 계획서·결과 보고서, 중간·실패 기록 및 검증 증거를 포함한다. **자료 확보는 모든 방법의 성능 향상이나 팀 공통 비교 완료를 뜻하지 않는다.** 이번 정리에서 실험이나 API 요청을 새로 실행하지 않았다.

## 방법별 계획서와 결과 보고서

| ID | 방법 | 계획서 | 결과 보고서 | 원래 실행 이름 |
| --- | --- | --- | --- | --- |
| S1 | GPT 경유점 정제 | [계획](experiments/S1/plan.md) | [결과](experiments/S1/report.md) | E1 |
| S2 | 구체적인 오류 피드백으로 경로 수정 | [계획](experiments/S2/plan.md) | [결과](experiments/S2/report.md) | E2F / E2C |
| S3 | 탐색 정체 시에만 GPT 호출 | [계획](experiments/S3/plan.md) | [결과](experiments/S3/report.md) | E3 |
| S4 | TransPath 계열 경로 영역 예측 | [계획](experiments/S4/plan.md) | [결과](experiments/S4/report.md) | TRANSPATH_11 / 22 / 33 |
| S5 | 빈 공간을 압축하는 정확한 탐색 | [계획](experiments/S5/plan.md) | [결과](experiments/S5/report.md) | R1 |
| S6 | 필요한 장애물만 단계적으로 반영 | [계획](experiments/S6/plan.md) | [결과](experiments/S6/report.md) | R2 |

방법별 문서는 기존 실행 기록을 근거로 이번에 정리한 안내다. 원래 자료는 아래에 보존했다. 공통 묶음의 전체 평가 행 수를 각 방법의 독립 실험 수로 중복 계산하지 않는다.

| 자료 묶음 | 원래 계획·설정 | 원래 보고서와 검증 |
| --- | --- | --- |
| S1~S3 | [고정 프로토콜](results/improvements/protocol.json) | [결과](results/improvements_final/report.md), [통합 보고서](results/improvements_final/final_report.md), [경로 정확도 분석](results/path_accuracy/report.md), [로컬 검증](results/improvements/local_import_verification.json) |
| S4 | [고정 프로토콜](results/candidates_v1/protocol.json), [지도 분할](results/candidates_v1/data_manifest.json) | [최종 해석](results/candidates_final/report.html), [상세 보고서](results/candidates_v1/report/report.html), [가중치](results/candidates_v1/models/), [로컬 검증](results/candidates_v1/local_verification.json) |
| S5~S6 | [사전 계획](results/next_experiments_plan/plan.md), [계획 JSON](results/next_experiments_plan/protocol.json), [실행 고정 프로토콜](results/next_search_v1/protocol_frozen.json) | [완료 보고서](results/next_search_final/report.md), [서버 검증](results/next_search_v1/verification.json), [로컬 검증](results/next_search_v1/local_verification.json) |

## 모든 원자료 확인과 복원

[압축자료 안내](archives/README.md)에 원자료 ZIP 5개를 정리했다. 압축파일은 이 폴더 안에 있으며 Git으로 함께 내려받는다. 공통 대조군과 K1·K2·J1·J2 결과도 원래 실행 묶음에 보존했다. 이 네 방법의 최종 담당은 팀 분담표에 따른다.

보고서·계획·설정·검증 기록은 바로 열 수 있다. 많은 수의 응답·개별 평가와 대용량 시간 기록은 ZIP에 보관했다. 특히 `results/next_search_v1/per_case.jsonl`은 약 137 MB여서 압축 형태로 제공한다. 다음 명령을 **이 `sm_part` 폴더에서** 실행하면 선택 원자료를 원래 상대 경로로 복원한다. Python 3.12 이상과 표준 라이브러리만 있으면 된다.

```powershell
python tools/verify_collection.py
python tools/restore_results.py --extract
```

검증은 ZIP과 모든 원본 파일의 바이트 크기·SHA-256을 확인한다. 복원 시 동일 파일은 유지하고 내용이 다른 파일을 덮어쓰지 않는다. 복원 전에는 원래 보고서가 가리키는 개별 원자료 경로가 아직 없을 수 있다. 복원 생성 파일은 `.gitignore`에 기록해 중복 커밋을 방지했다.

## 코드와 환경

- [reproduction/](reproduction/): 실제 실행·분석·검증 코드와 실행 시점 소스 해시.
- [vendor/llm-astar/](vendor/llm-astar/): 원본 커밋 `9414ed894d774f71c36e8c0afd0e20e6033b95b1`의 코드·입력·프롬프트·MIT 라이선스 사본. `.git` 내부 자료는 제외했다.
- [requirements-lock.txt](requirements-lock.txt), [requirements-llama.txt](requirements-llama.txt): 기존 환경 명세. 실제 서버 환경은 결과 폴더의 `environment.json`과 패키지 목록을 확인한다. 학습 재실행에는 해당 PyTorch·GPU 환경이 필요하다.
- [dist/](dist/): 당시 소스를 확인할 기존 배포 패키지 5개.
- [README.local-experiments.md](README.local-experiments.md): 로컬 전체 실행 안내 원본.

기존 코드가 `sm_part`를 루트로 해석하도록 원래 구조를 유지했다. 원자료를 사용하는 검증·보고서 재생성 전에는 복원 명령을 먼저 실행한다. `improve_run.py all` 등 수집 명령은 새 API 요청을 발생시킬 수 있으므로 이번 자료 확인에는 필요하지 않다.

## 출처와 보존 범위

[s1_s6_manifest.json](s1_s6_manifest.json)은 로컬 원본 각각의 크기·해시 및 보관 파일/ZIP 멤버를 대응시킨다. [collection_manifest.json](collection_manifest.json)은 현재 수집 파일의 해시 목록이다. 기존 수집 목록은 [2026-10-24 이력](collection_manifest.2026-10-24.json)으로 보존했다.

원문의 `C:/인지개/` 링크는 배포용 Markdown·HTML 사본에서 상대 경로로 바꿨다. 해당 문서의 바이트 원본은 [original_documents/](original_documents/)에 따로 보존했다. 코드·원자료·원본 ZIP의 내용은 바꾸지 않았다. 로컬 후속 실험은 Git 저장소 밖에서 수행됐으므로 없는 실행 커밋을 만들어 적지 않고 SHA-256으로 소스를 식별한다.

기존 [GPT 재현 보고서](results/report/report.md), [과거 계획서](LLM-Astar_Project_Plan.md), [리뷰](review/), [gm 참고 자료](references/gm/)도 유지했다. gm 자료는 승민의 성과로 합산하지 않는다. 가상환경·캐시·인증 정보와 S1~S6에 직접 해당하지 않는 별도 Llama Table 1 실행 자료는 이번 추가 범위에서 제외했다.

로컬 최신본으로 갱신한 기존 `report.py`와 `requirements-lock.txt`의 이전 사본은 [previous_collection/](previous_collection/)에 보존했다. 과거 수집 목록의 해시는 당시 사본에 대한 기록이다.
