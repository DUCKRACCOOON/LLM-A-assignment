# S1~S6 현재 결과 — 지도 단위 5-fold·Python 3.11

학습·준비는 2026-11-11, 새 응답 수집과 본 평가는 2026-11-12(Asia/Seoul)에 완료했다.

- [결과 보고서](results/s1_s6_cv1000_20261111/runs/run_20261111_01/report.md)
- [실험 계획](results/s1_s6_cv1000_20261111/plan.md) · [지도 분할](results/s1_s6_cv1000_20261111/fold_manifest.json)
- [독립 검산](results/s1_s6_cv1000_20261111/runs/run_20261111_01/verification.json)
- [문제별 결과](results/s1_s6_cv1000_20261111/runs/run_20261111_01/per_case.csv) · [직접 대조](results/s1_s6_cv1000_20261111/runs/run_20261111_01/direct_comparisons.json)
- [실행 코드·단계](experiments/s1_s6_cv1000_20261111/README.md)

100지도 × 10문제 × 22조건 × 5회 = 110,000회. 유효 경로 실패·선언 상한 위반 0회.
S4 모델 15개 재학습, 새 GPT-4o mini 응답 2,932건, RSS 보조 측정 1,320회.
원 LLM-A*를 주 분모로 사용했다. 과거 18개 방법의 다른 환경 결과와 합친 순위가 아니다.

## 해석

S1 계획기 시간은 원 LLM-A* 대비 25.44% 감소했다. S6는 최단 경로를 유지하며 8.72% 감소했지만 규칙 대조보다 0.94% 느려 LLM 계수의 추가 효용은 확인하지 못했다.
공개 지도는 과거 연구 노출이 있어 완전히 미관측인 hidden 평가가 아니다.
API 포함 시간은 새 응답의 지연과 오프라인 계획기 시간의 재구성 추정이다.

## 공개 사본 검증

이 폴더를 프로젝트 루트로 사용한다. 코드·입력·원문 응답·checkpoint·모든 반복 결과를 압축 없이 보관했다. 비밀키·가상환경·캐시는 제외했다.
`publication_manifest.json`은 사본의 SHA-256과 크기를 기록한다. `frozen.json`은 실행 당시 코드·모델·응답을 추적한다.
원본 바이트와 역사적 절대 경로 메타데이터를 보존했으며 `.gitattributes`로 줄바꿈 자동 변환을 막았다.

Python 3.11에 requirements.txt의 버전을 설치한다. CPU PyTorch는 `https://download.pytorch.org/whl/cpu` 인덱스를 사용한다.
저장 결과 검산은 API를 호출하지 않는다.

```powershell
python experiments/s1_s6_cv1000_20261111/report.py verify
```

이 명령은 검산 시각을 갱신하므로 사본 바이트 보존이 필요하면 별도 복사본에서 실행한다.
실제 collect 명령은 유료 API를 호출하며 로컬의 개인 키 설정이 필요하다. 공개 사본에는 키가 없다.
원 실험의 환경과 원문 응답이 보존됐으므로 기존 결과 검산에 새 API 호출은 필요하지 않다.
