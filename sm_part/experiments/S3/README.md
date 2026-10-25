# S3 · 탐색 정체 시에만 GPT 호출

최초 실험자·최종 담당자: 승민. 기존 식별자는 `E3`다.

- [계획서](plan.md), [결과 보고서](report.md), [분담 원문](../../docs/팀_개선방법_실험_분담.md).
- [조건부 탐색 코드](../../reproduction/improve_core.py), [실행·획득·재생 코드](../../reproduction/improve_run.py), [재생 감사 코드](../../reproduction/replay_improvement3_cached.py), [검증 코드](../../reproduction/test_improvements.py).
- [고정 프로토콜](../../results/improvements/protocol.json), [최종 보고서](../../results/improvements_final/report.md), [본실험 CSV](../../results/improvements/report/main_table.csv), [대응 비교 CSV](../../results/improvements/report/paired_comparisons.csv), [확대 CSV](../../results/improvements/report/scaling_table.csv).
- [실행 예외 기록](../../results/improvements/execution_deviation.json), [환경](../../results/improvements/environment.json), [서버 검증](../../results/improvements/verification.json), [로컬 검증](../../results/improvements/local_import_verification.json).

[archives](../../archives)를 복원하면 `results/improvements/{pilot,main,scaling}/e1/*.json`의 `e3_acquisition`에 최초 수집 당시 호출 시점·탐색 상태 결과·API 지연이 있다. `e3/*.json`은 이 기록과 대조한 최종 재생 결과이며 `raw/*-first.json`에 응답을 보존했다.

2026-10-25 정리본이다. 실험은 2026-10-14 완료됐고 이번 작업에서는 새 실험이나 API 호출을 하지 않았다. 실행 당시 소스 식별은 프로토콜 SHA-256을 사용하며 기록에 없는 원 Git 커밋을 추정하지 않는다.
