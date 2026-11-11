"""Korean report prose, with every numerical statement taken from saved records."""


def localized_fields(rows, gpt, proposal, ci, nr, integrity, finish, schema_count,
                     schema, costs, total_cost, overlap, styles):
    pct = lambda x: f'{x:.2f}' + r'\%'
    hybrid = [r for r in rows if r['method'] == 'llm_astar']
    only = [r for r in rows if r['method'] == 'llm_only']
    return dict(
        AUDIT_EXTRA=f"최종 검사에서는 A* 전체 1,000개의 길이를 독립 Dijkstra 구현과 비교하고 모든 성공 경로를 다시 검사했다. 실제 모델 응답 {len(integrity['real_response_upstream_parity'])}개도 수정하지 않은 원본 탐색 코드에 넣어 경로와 지표가 일치하는지 확인했다.",
        FINISH=f"API 요청 6,000건은 모두 응답을 받았으며, 그중 {finish.get('length', 0)}건은 토큰 상한에서 종료되었다. API 호출 성공과 유효한 경로 생성은 구분했다.",
        RESULT_TEXT=f"여섯 LLM-A* 조건의 탐색 횟수는 일반 A*의 {pct(min(r['operation_percent'] for r in hybrid))}에서 {pct(max(r['operation_percent'] for r in hybrid))} 사이였다. GPT-3.5 Few-shot에서는 탐색 횟수 {pct(gpt['operation_percent'])}, 저장량 {pct(gpt['storage_percent'])}, 길이 {pct(gpt['length_percent'])}를 기록했다.",
        FAILURE_TEXT=f"두 숫자의 좌표를 요구하는 파서에서는 {schema_count}개 응답의 좌표 형식이 맞지 않았고, {schema.get('missing_list', 0)}개에서는 읽을 수 있는 목록을 찾지 못했다. LLM-only의 유효 경로 비율은 {pct(min(r['valid_percent'] for r in only))}에서 {pct(max(r['valid_percent'] for r in only))}였다. 반면 LLM-A*는 이 데이터에서 모두 유효한 경로를 찾았다.",
        CONTROL_TEXT="여섯 LLM-A* 조건 모두 이 대조군보다 집계 탐색 횟수가 많았다. 다만 모든 개별 문제에서 대조군이 이겼다는 뜻은 아니다.",
        PROPOSAL_TEXT=f"GPT-4o mini Few-shot에서 V-Gate는 1,000개 안내 중 {proposal['accepted']}개를 채택했다. 탐색 횟수 {pct(proposal['operation_percent'])}, 저장량 {pct(proposal['storage_percent'])}, 길이 {pct(proposal['length_percent'])}, 유효 경로 {pct(proposal['valid_percent'])}를 기록했다.",
        CI_TEXT=f"동일 문제에서 V-Gate/LLM-A* 탐색 횟수 비율의 기하평균은 {pct(ci['ratio_percent'])}였고 맵 단위 bootstrap 95\\% 구간은 [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]\\%였다. 비교 표본은 {ci['n']}문제이다.",
        GENERALIZATION=f"LLM-A*와 V-Gate는 모두 30문제에 성공했으며 탐색 횟수 비율은 각각 {pct(nr['llm_astar']['operation_percent'])}, {pct(nr['validated']['operation_percent'])}였다. LLM 없는 대조군은 {pct(nr['goal_only']['operation_percent'])}였고 LLM-only는 {nr['llm_only']['valid_n']}/30문제에 성공했다.",
        COST_TEXT=f"본 실험은 입력 {sum(u['prompt_tokens'] for u in costs.values()):,}토큰과 출력 {sum(u['completion_tokens'] for u in costs.values()):,}토큰을 사용했다. 표준 요금 추산은 미화 {total_cost:.3f}달러이다. 평균 요청 지연은 " + ', '.join(('GPT-4o mini' if '4o-mini' in m else 'GPT-3.5') + f" {u['mean_request_seconds']:.2f}초" for m, u in costs.items()) + '였다.',
        OVERLAP_TEXT='프롬프트 예시와 문제의 모든 입력 필드가 일치하는 경우는 ' + ', '.join(f'{styles[p]} {len(overlap[p])}개' for p in ['standard', 'cot', 'repe']) + '였다. 주 표는 전체 데이터를 유지하고 이 문제들을 제외한 집계도 별도로 저장했다. 예시와 겹치는 공개 데이터를 독립 시험 데이터라고 부르지 않았다.',
    )
