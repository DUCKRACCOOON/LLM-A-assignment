# S1 · GPT 경유점 정제

최초 실험자·최종 담당자: 승민. 기존 실행의 식별자는 `E1`이다.

- [계획서](plan.md): 고정 프로토콜과 현재 분담의 확인 항목.
- [결과 보고서](report.md): 기본 LLM-A* 대비 개선과 해석의 한계.
- [분담 원문](../../docs/팀_개선방법_실험_분담.md), [기존 최종 보고서](../../results/improvements_final/report.md).
- [정제·오류 진단·조건부 호출 구현](../../reproduction/improve_core.py), [실행기](../../reproduction/improve_run.py), [검증 코드](../../reproduction/test_improvements.py).
- [고정 프로토콜](../../results/improvements/protocol.json), [본실험 표](../../results/improvements/report/main_table.csv), [대응 비교](../../results/improvements/report/paired_comparisons.csv), [확대 표](../../results/improvements/report/scaling_table.csv).
- [실행 환경](../../results/improvements/environment.json), [서버 검증](../../results/improvements/verification.json), [로컬 검증](../../results/improvements/local_import_verification.json).

대형 원자료는 [archives](../../archives)에 보관한다. 복원 후 `results/improvements/{pilot,main,scaling}/e1/*.json`에서 입력 지도, 경로, 정제 전후 경유점 수, 채택 여부와 결과를 확인한다. 같은 stage의 `raw/*-first.json`은 요청·응답 캐시다. S1~S3는 최초 응답과 대조군을 공유하므로 공통 자료를 중복 복제하지 않는다.

2026-10-25 자료 정리본이다. 실험은 2026-10-14 완료된 자료를 사용했으며, 이번 문서화에서 새 실험이나 API 호출은 하지 않았다. 기존 소스의 실행 당시 식별자는 프로토콜의 SHA-256으로 확인한다. 원 실행의 Git 커밋은 해당 기록에 없어 임의로 기재하지 않는다.
