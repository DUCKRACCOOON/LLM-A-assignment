# 승민 담당 실험 자료 모음

승민 담당 S1~S6의 분담·검증 기준, 기존 GPT 재현 자료, 과거 계획서와 리뷰, gm 브랜치의 개선 방법 조사·V-Gate 비교 자료를 모았다. 원래 작업트리의 파일은 그대로 두고 이 폴더에 복사했다. 수집일은 2026-10-24다.

**S1~S6는 팀 분담표에서 기실험으로 전달된 항목이다. 현재 확인한 main·gm·sm의 최신 커밋에서는 이 여섯 방법의 개별 구현·실행 설정·결과 묶음을 찾지 못했다.** 아래 GPT 재현 및 V-Gate 보고서를 S1~S6의 완료 결과로 사용하지 않는다.

## 먼저 볼 문서

1. [승민 담당 실험과 자료 현황](승민_실험_자료현황.md): 여섯 실험의 범위, 비교 항목, 확보 자료와 추가로 필요한 자료.
2. [팀 개선 방법 실험 분담](docs/팀_개선방법_실험_분담.md): 최종 담당과 최초 실험자, 인계 대상.
3. [공통 실험과 결과 공유 기준](docs/공통_실험_기준.md): 공정한 비교와 결과 기록 기준.
4. [gm 개선 방법 조사](references/gm/최신/V-Gate_이후_개선_방법_조사.md): S4 관련 13번, S5 관련 6번을 포함한 연구·방법 설명. 이 문서의 후보는 미실험 제안이다.

## 계획서와 결과 보고서

| 자료 | 위치 | 읽을 때 구분할 내용 |
| --- | --- | --- |
| sm 과거 재현·개선 계획서 | [LLM-Astar_Project_Plan.md](LLM-Astar_Project_Plan.md) | main 정리 전에 있던 sm `fa6d8dc`에서 복원한 과거 계획. 현재 S1~S6 분담은 위 팀 분담표를 따른다. |
| sm 과제·논문 통합 정리 | [LLM-Astar_Codex_Context.md](LLM-Astar_Codex_Context.md) | 과제 조건, 알고리즘 설명과 초기 개선 후보. |
| sm GPT 재현 보고서 | [HTML 보고서](results/report/report.html), [Markdown 보고서](results/report/report.md) | GPT-3.5 Turbo 재현 8개 조건의 결과. |
| sm 성능 원인 분석 | [performance_analysis.md](results/report/performance_analysis.md) | 경유점 유무, g+2h 대조군과 집계 방식 분석. |
| sm 실행 절차 | [reproduction/README.md](reproduction/README.md) | 실제 GPT 재현 설정과 실행 명령. |
| gm 전체 실험 계획서 | [LLM-Astar_Project_Plan.md](references/gm/LLM-Astar_Project_Plan.md) | gm의 계획과 진행 기록. sm의 완료 상태와 구분한다. |
| gm 개선 방법 조사 | [V-Gate 이후 개선 방법 조사](references/gm/최신/V-Gate_이후_개선_방법_조사.md) | 13개 후속 후보의 원리·출처·비용·한계. |
| gm 결과 요약 | [재현 실험 결과](references/gm/최신/재현_실험_결과.md) | 모델 2개 재현 및 V-Gate 비교 결과. |
| gm 결과 보고서 | [PDF](references/gm/output/pdf/LLM_Astar_Reproduction_Report.pdf), [보고서 본문](references/gm/output/LLM_Astar_Report_KO.txt), [TeX 원본](references/gm/report/LLM_Astar_Reproduction_Report.tex) | gm의 보고서와 편집 가능한 원본. |
| gm 실행·평가 규약 | [PROTOCOL.md](references/gm/reproduction/PROTOCOL.md), [AUDIT.md](references/gm/reproduction/AUDIT.md) | 모델·이동·파싱·비용 조건과 원본 대비 차이. |

## 폴더 구성

| 위치 | 포함 내용 |
| --- | --- |
| `docs/` | main에서 가져온 공통 실험 기준과 팀 분담표. |
| `reproduction/` | sm GPT 재현·평가·분석·검증 코드와 실행 안내. |
| `results/` | sm 보고서, 표, 그림, 원문 응답, 문제별 평가, 실행·검증 기록. |
| `review/` | sm `fa6d8dc`의 논문·코드 리뷰, waypoint 문답과 검증 자료. |
| `materials/` | 과제 안내서와 대상 논문. |
| `source_code/` | 원본 연구 코드·데이터·라이선스. |
| `references/gm/` | gm `2e8bcc1`의 계획·조사·보고서·코드·원시 결과·검증 및 참고 자료. 원래 디렉터리 구조를 유지했다. |
| `README.original.md` | 수집 당시 현재 작업트리의 루트 README 원본. |
| `collection_manifest.json` | 파일별 원본 경로, 출처 커밋, 크기와 SHA-256. |

gm의 배포용 중복 압축파일과 임시 이미지 `temp.png`는 포함하지 않았다. 보고서와 그 근거 파일은 압축을 풀지 않고 바로 확인할 수 있도록 모았다.

## 자료 상태와 실행 경로

원본 파일과 Git에서 복원한 파일의 내용은 유지했다. `results/report/report.md`, `performance_analysis.md` 및 분석 스크립트처럼 아직 커밋되지 않은 로컬 자료도 포함했고, 파일 목록에서 `working_tree_untracked`로 구분했다. 이 모음은 수집 시점의 사본이므로 이후 원본 변경이 자동 반영되지는 않는다.

sm 코드를 실행할 때 이 폴더를, gm 코드를 실행할 때 `references/gm/`을 각 실행 안내서의 저장소 루트로 해석한다. sm 실행기는 별도의 고정 원본 체크아웃 `vendor/llm-astar/`를 요구하며, 현재 작업트리에 없으므로 이 모음에도 없다. 필요한 환경은 각 실행 안내서에 따른다. 이번 자료 정리에서는 실험이나 API 호출을 새로 실행하지 않았다.

## 출처

- 현재 sm 작업트리: `a7fc8a0583593178d1ce0f0a4dbf3c7f937c6e76`와 수집 시점의 미추적 자료.
- sm 과거 계획·리뷰: `fa6d8dc7cf6c1ca9278ab6b1bfdc3b89a5db3248`.
- 원격 main 확인 커밋: `bf4ae439094e0df003c02c0b4b4e607d5a330238`.
- 원격 gm 확인 커밋: `2e8bcc1538ff5ca9735e4f6c7c8dc534aa9a2bd8`.

gm 자료는 비교·참고 자료이며 승민의 구현 성과로 바꾸어 기록하지 않는다. 최초 구현·실험 기여와 최종 담당은 팀 분담표를 따른다.
