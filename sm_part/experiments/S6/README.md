# S6 · 필요한 장애물만 단계적으로 반영하는 탐색

최초 실험자·최종 담당자: 승민. 기존 식별자는 `R2`, 압축 결합은 `R1_R2`다.

- [계획서](plan.md), [결과 보고서](report.md), [분담 원문](../../docs/팀_개선방법_실험_분담.md).
- [원 사전 계획](../../results/next_experiments_plan/plan.md), [고정 프로토콜](../../results/next_search_v1/protocol_frozen.json), [선택 설정](../../results/next_search_v1/selection.json).
- [벽 선택·완화·충돌 검증·fallback](../../reproduction/next_refinement.py), [LPA 상태 갱신](../../reproduction/next_incremental.py), [공통 탐색](../../reproduction/next_search_core.py), [실행기](../../reproduction/next_run.py), [모델 호출](../../reproduction/next_model.py), [검증](../../reproduction/test_next_search.py).
- [최종 보고서](../../results/next_search_final/report.md), [주 통계](../../results/next_search_v1/summary.json), [후속 분석](../../results/next_search_v1/local_analysis.json), [모델 예산](../../results/next_search_v1/actual_budget.json).
- [환경](../../results/next_search_v1/environment.json), [사전검사](../../results/next_search_v1/server_preflight.json), [서버 검증](../../results/next_search_v1/verification.json), [로컬 검증](../../results/next_search_v1/local_verification.json).

[archives](../../archives)의 대형 자료 복원 후 `results/next_search_v1/per_case.jsonl`에서 method `R2`·`R1_R2`를 선택한다. 각 결과의 `rounds`에 활성 벽·새 충돌 벽·회차별 확장 수·길이가 있고 `timings.jsonl`은 5회 시간 원자료다. 같은 실행의 R3는 팀 분담 K1이므로 S6의 성능으로 합치지 않는다.

주 실행은 2026-10-14 완료됐고 문서는 2026-10-25 정리했다. 이번 작업에서는 실험·모델 생성을 새로 하지 않았다. 실행 소스는 고정 프로토콜 SHA-256으로 식별하며 기록에 없는 원 Git 커밋은 추정하지 않는다.
