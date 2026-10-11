# LLM-A* assignment

SWE3011 Introduction to Artificial Intelligence, Fall 2026 과제 자료와 논문 재현을 위한 팀 작업 저장소입니다. 초기 자료는 `gm` 브랜치에 정리했습니다.

## 자료 읽는 순서

1. [과제 안내서](<materials/SWE3011_41 Assignment Notice (1).pdf>)
2. [논문 PDF](materials/인지개과제_논문.pdf)
3. [과제와 논문 통합 정리](LLM-Astar_Codex_Context.md)
4. [논문 리뷰](review/PAPER_REVIEW.md)
5. [코드 리뷰](review/CODE_REVIEW.md)
6. [Waypoint 동작 문답](review/WAYPOINT_QA.md)
7. [재현과 개선 실험 계획](LLM-Astar_Project_Plan.md)

## 폴더 구성

- `materials/`: 제공받은 과제 안내서와 논문
- `source_code/`: 과제에서 안내한 공식 LLM-A* 저장소의 원본 코드, 데이터, 문서, 라이선스
- `review/`: 논문·코드 리뷰, 대화 중 개념 정리, 오프라인 동작 검증 코드와 결과
- `LLM-Astar_Codex_Context.md`: 기존 과제·논문 정리
- `LLM-Astar_Project_Plan.md`: 제출 요건, 재현 범위, 실험 설계와 일정
- `UPSTREAM.md`: 원본 출처와 기준 커밋

## 현재 진행 상태

2026년 10월 11일 기준으로 자료 읽기, 원본 확보, 코드 리뷰와 일부 오프라인 동작 검증까지 완료했습니다. 실제 LLM API를 사용한 Table 1 전체 재현과 제안 알고리즘 실험은 아직 수행하지 않았습니다.

팀원 3명 모두 각자 전체 재현을 수행한 뒤 결과와 보고서의 완성도를 비교해 팀 제출의 기반이 될 한 버전을 선택합니다. 선택 과정에서 결과의 정확성을 교차 검증합니다. `gm`에서는 사용자의 구현·실험·분석 전 과정을 진행합니다. [수정된 계획](LLM-Astar_Project_Plan.md)은 GPT-4o mini를 주 모델로 권장하고, 원 실험 재현에 더해 waypoint의 실제 기여·실패·비용을 분석하도록 구성했습니다. 최종 과제에는 재현 이후의 제안 알고리즘과 비교 실험도 필요합니다.

최종 재현 범위는 **모델 2개 × 프롬프트 3종 × 방식 2개와 공통 기준 알고리즘 2개, 총 14개 조건**입니다. 각 조건은 같은 공개 1,000개 사례로 평가합니다. 첫 모델은 GPT-4o mini이며, 두 번째 모델의 정확한 이름과 실행 환경은 접근성·예산 확인 후 확정합니다. 한 모델의 8개 조건은 중간 점검 단계입니다.

`review/baseline_evidence.json`은 원본의 동작을 확인한 기록입니다. LLM-A* 사례는 고정 waypoint를 넣었으며 논문 성능 재현 결과가 아닙니다. 검증 스크립트는 원본 의존성이 설치된 환경에서 다음 명령으로 실행합니다.

```bash
python review/check_baseline.py
```

실험 환경 설정과 전체 재현 명령은 구현 후 추가합니다. 이 README는 현재 자료 저장소 안내이며 최종 제출용 실행 문서가 아닙니다.

## 출처와 사용 범위

공식 저장소: [SilinMeng0510/llm-astar](https://github.com/SilinMeng0510/llm-astar)

프로젝트 페이지: [LLM-A*](https://silinmeng0510.github.io/llm-astar/)

`source_code/`의 저작권과 이용 조건은 [원본 LICENSE](source_code/LICENSE)를 따릅니다. 과제 안내서·논문 및 제3자 자료의 권리는 각 권리자에게 있으며, 원본 코드의 라이선스를 저장소 전체에 일괄 적용하지 않습니다.

리뷰와 계획은 과제 이해를 위한 작업 노트입니다. 제출 보고서는 실제 구현·실험을 바탕으로 팀이 검토하고 작성합니다. API 키·토큰·개인 `.env` 및 개인별 기여 보고서는 이 공유 저장소에 올리지 않습니다.
