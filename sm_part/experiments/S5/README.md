# S5 · 빈 공간을 압축하는 정확한 탐색

최초 실험자·최종 담당자: 승민. 기존 식별자는 `R1`이며 최종 선택은 `sparse` 직사각형 압축이다. JPS 또는 Subgoal Graphs 전체를 구현·검증한 결과가 아니다.

- [계획서](plan.md), [결과 보고서](report.md), [분담 원문](../../docs/팀_개선방법_실험_분담.md).
- [원 사전 계획](../../results/next_experiments_plan/plan.md), [고정 프로토콜](../../results/next_search_v1/protocol_frozen.json), [선택 설정](../../results/next_search_v1/selection.json).
- [영역 인증·압축 구현](../../reproduction/next_geometry.py), [탐색·복원·계측](../../reproduction/next_search_core.py), [실행기](../../reproduction/next_run.py), [재사용 평가](../../reproduction/next_reuse.py), [정확성 검사](../../reproduction/test_next_search.py).
- [최종 통합 보고서](../../results/next_search_final/report.md), [요약 통계](../../results/next_search_v1/summary.json), [후속 분석](../../results/next_search_v1/local_analysis.json), [서버 검증](../../results/next_search_v1/verification.json), [로컬 검증](../../results/next_search_v1/local_verification.json), [환경](../../results/next_search_v1/environment.json).

[archives](../../archives)를 복원하면 `results/next_search_v1/per_case.jsonl`, `timings.jsonl`, `data/`, 반복 질의·확대 결과에 접근할 수 있다. `R1_R2`는 S6와의 결합 조건으로 공유한다. 공통 실행 43,520행에는 K1에 해당하는 R3와 대조군도 포함되므로 S5 단독 규모로 세지 않는다.

주 실행은 2026-10-14 완료됐고 이 문서는 2026-10-25 정리했다. 새 실험은 하지 않았다. 원 실행의 코드 식별은 고정 프로토콜 `source_hashes`로 확인하며 기록에 없는 Git 커밋은 추정하지 않는다.
