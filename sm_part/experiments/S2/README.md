# S2 · 구체적인 오류 피드백으로 경로 수정

최초 실험자·최종 담당자: 승민. 기존 실행의 실험군은 `E2F`, 동일 호출 대조군은 `E2C`다.

- [계획서](plan.md), [결과 보고서](report.md), [분담 원문](../../docs/팀_개선방법_실험_분담.md).
- [오류 진단·수정 프롬프트 코드](../../reproduction/improve_core.py), [실행기](../../reproduction/improve_run.py), [검증 코드](../../reproduction/test_improvements.py).
- [고정 프로토콜](../../results/improvements/protocol.json), [최종 종합 보고서](../../results/improvements_final/report.md).
- [본실험 CSV](../../results/improvements/report/main_table.csv), [대응 비교 CSV](../../results/improvements/report/paired_comparisons.csv), [확대 CSV](../../results/improvements/report/scaling_table.csv).
- [실행 환경](../../results/improvements/environment.json), [서버 검증](../../results/improvements/verification.json), [로컬 검증](../../results/improvements/local_import_verification.json), [실행 순서 예외](../../results/improvements/execution_deviation.json).

[archives](../../archives)의 대형 원자료를 복원하면 `results/improvements/{pilot,main,scaling}/e2/*.json`에서 수정 여부·첫 오류·두 응답의 해시·최종 결과를 확인한다. `raw/*-first.json`, `raw/*-E2C.json`, `raw/*-E2F.json`은 실제 요청과 응답이다. 처음부터 유효한 답에는 추가 수정 응답이 없다.

2026-10-25 정리본이며 기존 실험은 2026-10-14 완료됐다. 이번 문서화에서 재실험이나 API 호출은 하지 않았다. 소스 식별은 프로토콜의 SHA-256을 사용한다. 원 실행 Git 커밋은 기록에 없어 추정하지 않는다.
