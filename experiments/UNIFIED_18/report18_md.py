"""Markdown tables for the 18-method comparison (reads summary18.json written by report18.py).

    python report18.py && python report18_md.py
"""
import json
from u18 import read
from u18 import RUN18 as R

NAME = {'A_GM': 'A*', 'B_llm': 'LLM-A* (통합 프롬프트)', 'B_llm_paper': 'LLM-A* (논문 원 프롬프트)', 'N_g2h': 'g+2h', 'A_OCT': 'A*_octile',
        'FOCAL_105': '무안내 FOCAL 1.05', 'K6_w105': 'K6_w105 (변형)'}
f2 = lambda x, d=2: '—' if x is None else f'{x:.{d}f}'
pct = lambda r: '—' if not r else f"{100*r['ratio']:.1f}%"
ci = lambda r: '—' if not r else f"{100*r['ratio']:.1f}% [{100*r['ci95'][0]:.1f}, {100*r['ci95'][1]:.1f}]"
gsd = lambda r: '—' if not r else f"×{r['gsd']:.2f}"
ms = lambda t, k=1, d=2: '—' if t[0] is None else f"{k*t[0]:.{d}f} ± {k*t[1]:.{d}f}"
dirs = lambda r: '—' if not r else '{improved}/{equal}/{worsened}'.format(**r['directions'])
bnd = lambda b: '없음' if not b else ('최단' if b == 1 else f'≤{b:g}L*')
EQ = {'SM': 'SM 기록과 결정 필드 1,000/1,000 일치', 'S4': 'SM 기록과 경로·길이 일치, 4,000회 중 14회 확장 수 1~3 차이(CPU 부동소수)',
      'JY': 'JY 기록과 1,000/1,000 일치(JY 원 응답 입력)', 'GM': 'gm 원 구현 그대로, 기준 7조건 SM 기록 7,000/7,000 일치'}

def head(L, title, cols):
    L += ['', f'## {title}', '', '| 방법 | 담당 | '+' | '.join(cols)+' |', '| --- | --- |'+' ---: |'*len(cols)]

def write_md(s):
    M = s['main']; ref = s['reference']; b = s['all']['B_llm']; order = list(M)
    frozen = read(R/'frozen.json')['frozen']; env = frozen['environment']; n_cond = len(frozen['methods'])
    L = ['# 18개 방법 vs 원 LLM-A* — 단일 측정 블록 비교 (정규화 전)', '',
         f"2026-11-12, gm. 공개 1,000문제(100지도×10쌍), SM 통합 규칙(지도 단위 5-fold OOF, 문제당 {n_cond}조건×5회, 반복마다 순서 섞음, 5회 중앙값, 네트워크 차단, 1스레드, 해시 고정).",
         f"S·J·K 18개 방법과 A*·LLM-A*를 **한 PC, 한 실행 블록**에서 측정했다(Python {env['python'].split()[0]}, {env['processor']}). "
         '5개 fold는 병렬 프로세스로 실행했다. 같은 문제 안에서 모든 조건이 섞여 실행되므로 비율 비교는 공정하지만 절대 ms는 단독 실행보다 클 수 있다.', '',
         '비율은 문제별 (방법 ÷ 원 LLM-A*)의 기하평균이다. 100% 미만이면 원 LLM-A*보다 작다(시간·연산·메모리는 좋음). '
         '[ ]는 fold 내 지도 단위 bootstrap 2,000회 95% 구간, ×는 문제별 비율의 기하 표준편차, ±는 문제별 원값의 표준편차.', '',
         '## 0. 원 LLM-A* 점수표 (논문 Table 1 형식, A* = 100)', '',
         '논문 정의: operation = 주 탐색 OPEN pop 수, storage = g가 부여된 상태 수(len(g)), 경로 길이는 A* 대비, 문제별 비율의 기하평균. 유효 경로는 단순 비율.', '',
         '| 조건 | Operation | Storage | Relative path length | Valid path | 계획기 ms | 총응답 s(추정) | 최단 경로 | 최대 초과율 |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in ('A_GM', 'B_llm', 'B_llm_paper'):
        d = ref[m]
        L.append(f"| {NAME[m]} | {f2(100*d['paper_operation_vs_astar']['ratio'])} | {f2(100*d['paper_storage_vs_astar']['ratio'])} | {f2(100*d['length_vs_astar']['ratio'])} | "
                 f"{100*d['valid']/d['cases']:.1f} | {ms(d['planner_s_mean_sd'], 1000)} | {ms(d['e2e_s_mean_sd'])} | {d['optimal']}/{d['quality_n']} | {f2(d['max_excess'])}% |")
    L += ['', '논문 보고값(LLAMA3 few-shot): 44.59 / 64.02 / 102.47 / 100. 통합 프롬프트는 공개 지도와 겹치지 않는 합성 예시 5개, '
          '논문 원 프롬프트는 공식 코드 예시(공개 지도 0·2·3·4·5번 pair0에서 가져옴)를 쓴다. 18개 방법의 비교 분모는 같은 응답을 공유하는 **통합 프롬프트 LLM-A***다.']
    head(L, '1. 속도', ['계획기 시간 비율 [95%]', '기하SD', '개선/동일/악화', '계획기 ms', '총응답시간 비율 [95%]', '총응답 s', '반복 CV 중앙값'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {ci(d['planner_s_vs_llm'])} | {gsd(d['planner_s_vs_llm'])} | {dirs(d['planner_s_vs_llm'])} | {ms(d['planner_s_mean_sd'], 1000)} | "
                 f"{ci(d['e2e_s_vs_llm'])} | {ms(d['e2e_s_mean_sd'])} | {100*d['planner_cv_median']:.1f}% |")
    L.append(f"| *원 LLM-A** | — | 100% | — | — | {ms(b['planner_s_mean_sd'], 1000)} | 100% | {ms(b['e2e_s_mean_sd'])} | {100*b['planner_cv_median']:.1f}% |")
    L.append('총응답시간 = 저장된 실제 API 대기(방법이 쓰는 응답의 합) + 계획기 시간. 재구성 추정이며 온라인 실측이 아니다.')
    head(L, '2. 연산량', ['이웃 확장 비율 [95%]', '확장 수', 'Operation(논문 정의) 비율', 'heap pop 비율', 'Storage(논문 정의) 비율', '저장 상태 수'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {ci(d['expansions_vs_llm'])} | {ms(d['expansions_mean_sd'], 1, 1)} | {pct(d['paper_operation_vs_llm'])} | {pct(d['heap_pops_vs_llm'])} | "
                 f"{pct(d['paper_storage_vs_llm'])} | {ms(d['paper_storage_mean_sd'], 1, 1)} |")
    L.append(f"| *원 LLM-A** | — | 100% | {ms(b['expansions_mean_sd'], 1, 1)} | 100% | 100% | 100% | {ms(b['paper_storage_mean_sd'], 1, 1)} |")
    L.append('엔진마다 OPEN 구현이 달라 heap pop은 직접 비교가 어렵다(배열 스캔 FOCAL은 pop이 거의 없고, 보조 탐색 pop이 섞이기도 함). '
             '엔진 간 비교는 이웃 확장 수를 우선 본다. "—"는 해당 엔진에서 논문 정의 값을 분리하지 못한 경우(K1·K2).')
    head(L, '3. 경로 품질', ['보장', '최단/유효', 'L/L* 기하평균', 'L/L* SD', 'P95 초과', '최대 초과', '5%↑', '10%↑', '길이 비율 vs LLM-A*'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {bnd(d['bound'])} | {d['optimal']}/{d['quality_n']} | {f2(d['lstar_gm'], 4)} | {f2(d['lstar_sd'], 4)} | {f2(d['p95_excess'])}% | "
                 f"{f2(d['max_excess'])}% | {d['over5']} | {d['over10']} | {pct(d['length_vs_llm'])} |")
    L.append(f"| *원 LLM-A** | — | 없음 | {b['optimal']}/{b['quality_n']} | {f2(b['lstar_gm'], 4)} | {f2(b['lstar_sd'], 4)} | {f2(b['p95_excess'])}% | {f2(b['max_excess'])}% | {b['over5']} | {b['over10']} | 100% |")
    L.append('S4·K5는 3시드 평균이다(품질 꼬리는 3,000개 시드별 경로).')
    head(L, '4. LLM 비용 (1,000문제)', ['질의 시 LLM', '논리 응답 수', '입력 토큰', '출력 토큰', '목록가 추정 $', 'API 대기 s/문제'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {'예' if d['online_llm'] else '아니오'} | {d['responses']:,.0f} | {d['prompt_tokens']:,.0f} | {d['completion_tokens']:,.0f} | "
                 f"{d['usd_list']:.3f} | {ms(d['api_s_mean_sd'])} |")
    L.append(f"| *원 LLM-A** | — | 예 | {b['responses']:,} | {b['prompt_tokens']:,} | {b['completion_tokens']:,} | {b['usd_list']:.3f} | {ms(b['api_s_mean_sd'])} |")
    L.append('S6의 LLM은 오프라인 계수 생성(질의 시 호출 없음). S4·S5·J4·J5·K3은 LLM을 쓰지 않는다. 토큰은 gpt-4o-mini 목록가($0.15/$0.60 per 1M)로 환산했다.')
    head(L, '5. 메모리 (논리 지표)', ['OPEN 최대 비율 [95%]', 'OPEN 최대', 'Storage 비율', '저장 상태 수'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {ci(d['peak_open_vs_llm'])} | {ms(d['peak_open_mean_sd'], 1, 1)} | {pct(d['paper_storage_vs_llm'])} | {ms(d['paper_storage_mean_sd'], 1, 1)} |")
    L.append(f"| *원 LLM-A** | — | 100% | {ms(b['peak_open_mean_sd'], 1, 1)} | 100% | {ms(b['paper_storage_mean_sd'], 1, 1)} |")
    L.append('OPEN 최대는 엔진의 논리 OPEN 크기 또는 heap 최대 길이(중복 포함)다. 프로세스 RSS는 이 블록에서 측정하지 않았다.')
    head(L, '6. 정확성·재현성', ['유효 경로', '선언 상한 위반', '반복 결정성', '원 구현 대비 동등성'])
    for k in order:
        d = M[k]
        L.append(f"| {k} | {d['owner']} | {d['valid']}/{d['cases']} | {d['bound_violations']} | 5회 동일 | {EQ.get(k, EQ[d['owner']])} |")
    head(L, '7. LLM 기여 (같은 엔진에서 LLM을 빼거나 대체한 대조군 대비)', ['대조군', '시간 비율 [95%]', '확장 비율', '길이 비율'])
    for k in order:
        d = M[k]; c = d['vs_control']
        L.append(f"| {k} | {d['owner']} | {NAME.get(d['control'], d['control'])} | {ci(c['planner_s'])} | {pct(c['expansions'])} | {pct(c['length'])} |")
    L.append('질의 시 LLM을 쓰지 않는 방법(S4·S5·S6·J4·J5·K3)의 대조는 학습·제안 요소를 뺀 효과이며 LLM 안내 효과가 아니다. 대조군 대부분은 LLM 호출이 없으므로 총응답시간 차이는 LLM 대기 그 자체다(4번 항목 참고).')
    L += ['', '## 참고 조건', '', '| 조건 | 계획기 시간 비율 [95%] | 확장 비율 | L/L* | 최단 | 보장 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for m, d in ref.items():
        L.append(f"| {NAME.get(m, m)} | {ci(d['planner_s_vs_llm'])} | {pct(d['expansions_vs_llm'])} | {f2(d['lstar_gm'], 4)} | {d['optimal']}/{d['quality_n']} | {bnd(d['bound'])} |")
    (R/'report18.md').write_text('\n'.join(L)+'\n', encoding='utf-8')

if __name__ == '__main__':
    write_md(read(R/'summary18.json'))
