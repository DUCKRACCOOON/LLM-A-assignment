# 원자료 압축본과 복원

보고서의 기존 상대 경로와 실행 명령을 유지하기 위해 원본 ZIP은 `sm_part` 바로 아래에 보관했다. 서버에서 회수하고 검증했던 파일 그대로이며, 모든 멤버의 내부 manifest SHA-256도 이번 정리에서 다시 확인했다.

| 묶음 | 파일 | 내용 |
| --- | --- | --- |
| S1~S3 최종 | [improvement-results-verified.zip](../improvement-results-verified.zip) | 12,166개 파일: 응답·수정·추적·평가·보고서·소스·검증 |
| S1~S3 중간 이력 | [improvement-results-partial.zip](../improvement-results-partial.zip) | 12,095개 파일: 중단 시점 증거·캐시 재생 |
| S4와 공유 후보 | [candidate-results-verified.zip](../candidate-results-verified.zip) | 344개 파일: 학습 입력·3개 시드 가중치·평가·공통 대조·다른 후보 |
| S5~S6와 공유 후속 실험 | [next-search-results-verified.zip](../next-search-results-verified.zip) | 44,597개 파일: 지도·경로·평가·시간·R3·확대·재사용·소스 |
| A* 보조 비교 | [octile-supplement-results-verified.zip](../octile-supplement-results-verified.zip) | 1,205개 파일: Euclidean/octile 1,200행·설정·검증 |

파일 수는 각 내부 manifest 자체 1개를 제외한다. ZIP에 없던 로컬 검증, 계획서, 최종 해석, 그래프, 서버 증거는 `results/`에 별도로 보존했다. 중간 결과는 최종 결과에 합산하지 않는다.

ZIP 전체 SHA-256과 모든 원본의 보관 위치는 [s1_s6_manifest.json](../s1_s6_manifest.json)에 있다. `sm_part`에서 Python 3.12 이상으로 실행한다.

```powershell
python tools/verify_collection.py
python tools/restore_results.py --extract
```

첫 명령은 파일과 압축 멤버를 검증한다. 두 번째는 검증 후 압축으로 보관한 원자료만 원래 `results/...` 경로로 복원한다. 같은 파일은 유지하고 다른 내용은 덮어쓰지 않는다. 직접 제공한 최신 문서·검증을 옛 ZIP 내용으로 덮어쓰지 않는다. 패키지 설치·네트워크·API 키는 필요하지 않다.

링크를 읽기 편하게 고친 문서는 [original_documents/](../original_documents/)에 바이트 원본도 있다. ZIP을 수동으로 풀 때는 별도 폴더를 사용하면 이후 추가된 문서·검증 사본을 보존할 수 있다.
