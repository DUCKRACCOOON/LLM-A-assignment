# 18개 방법 원 LLM-A* 기준 재검토

[비교 보고서](report.md), [CSV](comparison.csv), [전체 JSON](comparison.json), [재검산 검증](verification.json).

JY의 고정 커밋 `cd7d490eada5bd0af9c79a552c46694f06737f32`에서 COMMON18을 읽어 계산했다.
SM 공식 결과는 이 폴더와 별도의 `../gm_retest_20261028`이며 두 실행은 합치지 않는다.

원격 COMMON18 결과를 내려받은 후 Python과 NumPy를 사용해 재집계할 수 있다.

```powershell
python review.py --source PATH_TO_COMMON18_RESULTS/run_20261028 --output REVIEW_OUTPUT
```

검색 알고리즘·API는 실행하지 않는다. 기존 결과의 해시, 반복→문제별 집계,
경로 품질 꼬리, 원 LLM-A* 대응 비율을 재검산한다.
