# S4 · TransPath 계열 모델로 유망한 경로 영역 예측

최초 실험자·최종 담당자: 승민. 결과 식별자는 `TRANSPATH_11`, `TRANSPATH_22`, `TRANSPATH_33`이다. 자체 소형 모델이며 공식 TransPath 재현이 아니다.

- [계획서](plan.md), [결과 보고서](report.md), [분담 원문](../../docs/팀_개선방법_실험_분담.md).
- [모델·FOCAL 구현](../../reproduction/candidate_core.py), [학습·평가 실행기](../../reproduction/candidate_run.py), [검증 코드](../../reproduction/test_candidates.py), [집계 코드](../../reproduction/candidate_report.py).
- [고정 프로토콜·지도 분할](../../results/candidates_v1/protocol.json), [데이터 명세](../../results/candidates_v1/data_manifest.json), [학습 기록 폴더](../../results/candidates_v1/models).
- [요약 CSV](../../results/candidates_v1/report/summary.csv), [문제별 CSV](../../results/candidates_v1/report/per_case.csv), [최종 HTML 보고서](../../results/candidates_final/report.html), [통합 Markdown 보고서](../../results/improvements_final/final_report.md).
- [환경](../../results/candidates_v1/environment.json), [서버 검증](../../results/candidates_v1/verification.json), [로컬 검증](../../results/candidates_v1/local_verification.json), [환경 설정 예외](../../results/candidates_v1/setup_deviation.json).

[archives](../../archives)의 대형 자료를 복원하면 `results/candidates_v1/models/seed{11,22,33}.pt`, `training_data.npz`, `evaluation/{main,ood,scaling}/*.json`에 모델·학습 입력·경로 원자료가 있다. 공통 실행기는 K2·J1·J2도 함께 평가했으므로 총 4,100행을 S4 단독 수로 세지 않는다.

2026-10-25 자료 정리본이다. 기존 실행은 2026-10-14 완료됐고 이번 작업에서 학습이나 추가 API 호출은 하지 않았다. 소스 해시는 프로토콜에 기록돼 있다. 원 실행 Git 커밋은 기록에 없어 추정하지 않는다.
