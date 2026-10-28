# 동일 조건 실행 준비 중 확인한 사항

2026-10-28. 초기 수집 manifest 이후, 성능 평가 시작 전에 다음 어댑터 검토를 수행했다.
수집 중인 LLM 요청·프롬프트·입력·모델·예산은 변경하지 않았다.

- K4의 안내 제거 비교는 NumPy FOCAL과 다른 엔진을 공유하면 기여 분리가 불가능하다.
  따라서 기존 34조건에 **같은 K4 엔진의 no-guide 1.05/1.0 두 조건**을 추가했다.
  18개 primary는 동일하며 실제 timed 조건은 36개다. 실제 실행 소스·조건은 benchmark_manifest에 고정한다.
  이후 사용자 요구의 최단성 비교를 빠뜨리지 않도록 K1/K2/K5/K6/J1/J2/S4의 w=1과
  K1 RULE w=1을 추가했다. **최종 timed 조건은 44개**이며 추가 API 요청은 없다.
  K6 w=1에서는 α 안내 항이 사라지므로 이를 유효한 LLM 기여로 해석하지 않는다.
  S1의 정제와 단순 검증/거부 효과도 구분하도록 같은 parser/target filter/검증을 쓰되
  정제만 제거한 S1_gate를 추가했다. **최종 본평가 45조건**이며 추가 API 호출은 없다.
- 비LLM 방법에 LLM 텍스트 파싱 비용을 부과하지 않도록 파싱을 필요한 분기로 한정했다.
  S3는 callback이 호출된 경우에만 응답을 파싱한다. S2의 첫 응답 검증·수정 프롬프트 구성도
  local 계산에 포함한다. 원 응답/좌표는 수정하지 않는다.
- node expansion은 query 이웃 확장 수다. ALT landmark Dijkstra/FMM field/portal 생성은
  각자의 추가 카운터와 준비 시간으로 분리한다. 서로 다른 그래프/거리장 작업을 동등한
  1회 연산으로 더하지 않는다. 모든 준비는 로컬 총시간에 포함한다.
- 시간 반복 동안 cycle GC를 비활성화하고 반복 블록 사이에 수집한다. Python refcount는 유지한다.
  같은 정책을 모든 방법에 적용하며 메모리 샘플링은 별도 프로세스에서 수행한다.
  집계는 seed별 시간3회 중앙값을 먼저 구한 뒤 K5의3seed를 평균한다. 주 속도 판정은
  200문제 재구성 총지연의 산술평균이며 GM 대응 비율과 지도 bootstrap 구간도 함께 보인다.
  API는 문제별 한 번 수집한 응답을 공유하므로 3회 서비스 시간 반복으로 표현하지 않는다.
- K1 순위 파서는 원 next_model.py의 중괄호 JSON 추출과 완료 응답 검사에 맞췄다.
  좌표·순위를 수리하지 않고 schema/ID 순열 검사를 실패하면 RULE로 fallback한다.
- SM core의 ROOT/VENDOR 경로 및 package-relative import만 바꿔 vendoring했다.
  변경 전후 SHA와 출처 commit은 SM_PROVENANCE.json에 남겼다. 기존 source_code와
  gm/sm/jy의 과거 실험 파일은 변경하지 않았다.

## 본평가 전 발견한 incumbent 종료 회귀와 수정

공통 실행 준비 중 w=1 변형의 종료 조건을 점검했다. 가져온 SM candidate_core.focal은
유효한 incumbent로 가지치기한 뒤 OPEN이 소진되면 incumbent를 반환하지 않고 실패했다.
독립 최단 경로를 테스트 incumbent로 준 long_wall, crossing_walls, endpoint_contact,
same_line_detour에서 재현했다(예: long_wall 27회 선택 뒤 잘못된 실패).
이 정답 incumbent는 **검증에서만** 사용하며 온라인 알고리즘에는 전달하지 않는다.

공통 실행 사본에만 `not reached and incumbent and not opened`이면 검증된 incumbent를
반환하도록 고쳤다. 더 짧은 해가 있다면 admissible 하한의 frontier가 가지치기되지 않고
남아야 하므로, OPEN 소진은 상한 경로의 최단성을 인증한다. 네 경계 회귀를 추가했다.
기존 sm 코드·결과는 수정하지 않았고 원문을 original_candidate_core.py.txt로 보존했다.
패치는 J1의 정확 변형에 특히 필요하며 원 논문의 CHOMP 자체 오류로 일반화하지 않는다.
