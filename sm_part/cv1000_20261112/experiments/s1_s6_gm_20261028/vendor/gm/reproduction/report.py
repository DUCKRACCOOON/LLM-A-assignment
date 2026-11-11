"""Generate report source, figures and Korean analysis from final records."""
import ast
import json
import re
import runpy
from collections import Counter
from pathlib import Path
from .core import Geometry, parse_output, search
from .experiment import ROOT, MODELS, PROMPTS, PROMPT_FILE, cases, records
from .summarize import aggregate
from .report_ko import localized_fields


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=ROOT/'results/main'
    report=ROOT/'report'
    figures=report/'figures';figures.mkdir(exist_ok=True)
    summary=json.loads((out/'summary.json').read_text())
    diagnostics=json.loads((out/'diagnostics.json').read_text())
    integrity=json.loads((ROOT/'results/verification/final_integrity.json').read_text())
    assert summary['status']=='complete' and integrity['passed']
    rows=summary['rows']
    props=diagnostics['proposal']
    mainrows={(r['method'],r['model'],r['prompt_style']):r for r in rows}
    proprows={(r['variant'],r['model'],r['prompt']):r for r in props}
    model=MODELS[0]
    mini=mainrows[('llm_astar',model,'standard')]
    gpt=mainrows[('llm_astar',MODELS[1],'standard')]
    control=mainrows[('goal_only',None,None)]
    dwa=mainrows[('dwa',None,None)]
    proposal=proprows[('validated',model,'standard')]
    evaluations=records(out/'evaluations.jsonl')
    generations=records(out/'generations.jsonl')
    query_by_id={c['case_id']:c for c in cases()}
    baseline={r['case_id']:r for r in evaluations if r['method']=='astar'}
    control_cases={r['case_id']:r for r in evaluations if r['method']=='goal_only'}
    finish=Counter(g['response']['choices'][0]['finish_reason'] for g in generations)
    schema=Counter(str(parse_output(g['response']['choices'][0]['message'].get('content') or '')[1]) for g in generations)
    fmt=lambda x:'--' if x is None else f'{x:.2f}'
    pct=lambda x:fmt(x)+r'\%'
    names={'astar':'A*','dwa':'Dynamic WA*','llm_only':'LLM-only','llm_astar':'LLM-A*','goal_only':'g+2h',
           'validated':'V-Gate','parse_only':'Schema-only','astar_fallback':'A* fallback'}
    styles={'standard':'Few-shot','cot':'CoT','repe':'RePE',None:'--'}
    table=[]
    for r in rows[:14]:
        m='M' if r['model']==model else ('G' if r['model'] else '--')
        table.append(' & '.join([names[r['method']],m,styles[r['prompt_style']]]+
                     [fmt(r[k+'_percent']) for k in ['operation','storage','length','valid']]+[str(r['length_n'])])+r' \\')
    ablation=[]
    for name,r in [('A*',mainrows[('astar',None,None)]),('LLM-A*',mini),('V-Gate',proposal),
                   ('Schema-only',proprows[('parse_only',model,'standard')]),
                   ('A* fallback',proprows[('astar_fallback',model,'standard')]),('g+2h',control)]:
        ablation.append(' & '.join([name]+[fmt(r[k+'_percent']) for k in ['operation','storage','length','valid']])+r' \\')
    # Prompt-example overlap uses all supplied problem fields, not just endpoints.
    templates=runpy.run_path(str(PROMPT_FILE))['gpt_prompt']
    overlap={}
    pattern=r'Start Point: (\[[^\n]+\])\s*\nGoal Point: (\[[^\n]+\])\s*\nHorizontal Barriers: (\[[^\n]+\])\s*\nVertical Barriers: (\[[^\n]+\])'
    for style,template in templates.items():
        examples=[]
        for match in re.findall(pattern,template):
            try: examples.append(tuple(ast.literal_eval(x) for x in match))
            except (ValueError,SyntaxError): pass
        overlap[style]=[c['case_id'] for c in query_by_id.values() if tuple(c['query'][k] for k in ['start','goal','horizontal_barriers','vertical_barriers']) in examples]
    overlap_aggregate=[]
    for m in MODELS:
        for style in PROMPTS:
            group=[r for r in evaluations if r['method']=='llm_astar' and r['model']==m and r['prompt_style']==style and r['case_id'] not in overlap[style]]
            overlap_aggregate.append(dict(model=m,prompt=style,excluded=overlap[style],**aggregate(group,baseline)))
    (out/'prompt_overlap.json').write_text(json.dumps(dict(matches=overlap,excluded_hybrid=overlap_aggregate),indent=2),encoding='utf-8')
    # Published standard prices, not invoices. Cache discount applies only to M.
    costs={}
    for m,u in summary['usage'].items():
        input_rate,cache_rate,output_rate=(.15,.075,.6) if m==model else (.5,.5,1.5)
        estimate=((u['prompt_tokens']-u['cached_tokens'])*input_rate+u['cached_tokens']*cache_rate+u['completion_tokens']*output_rate)/1e6
        costs[m]=dict(**u,estimated_usd=estimate,input_per_million=input_rate,cached_per_million=cache_rate,output_per_million=output_rate)
    total_cost=sum(r['estimated_usd'] for r in costs.values())
    (out/'costs.json').write_text(json.dumps(dict(models=costs,total_estimated_usd=total_cost,
          sources=['https://developers.openai.com/api/docs/models/gpt-4o-mini','https://developers.openai.com/api/docs/models/gpt-3.5-turbo'],
          rates_checked='2026-10-13',scope='6000 primary requests only; excludes access probes, natural-language smoke, 30 new-map requests'),indent=2),encoding='utf-8')
    # The Korean report uses a system Korean font; figures retain model/prompt names.
    plt.rcParams.update({'font.family':['Times New Roman','Batang'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    labels=[('M' if m==model else 'G')+' / '+styles[p] for m in MODELS for p in PROMPTS]
    x=np.arange(6)
    for ax,metric,title in zip(axes,['operation','length'],['A* 대비 탐색 횟수 (%)','A* 대비 경로 길이 (%)']):
        a=[mainrows[('llm_astar',m,p)][metric+'_percent'] for m in MODELS for p in PROMPTS]
        b=[proprows[('validated',m,p)][metric+'_percent'] for m in MODELS for p in PROMPTS]
        if metric=='operation':
            ax.bar(x-.18,a,.36,label='LLM-A*',color='#2364aa')
            ax.bar(x+.18,b,.36,label='V-Gate',color='#ec9a29')
        else:
            ax.scatter(x-.12,a,label='LLM-A*',color='#2364aa',s=55,zorder=3)
            ax.scatter(x+.12,b,label='V-Gate',color='#ec9a29',s=55,zorder=3)
        ax.axhline(control[metric+'_percent'],color='#23794f',ls='--',label='LLM 없는 g+2h')
        ax.set_xticks(x,labels,rotation=28,ha='right');ax.set_title(title)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if metric=='length': ax.set_ylim(99.5,103.5)
    axes[0].legend(fontsize=9)
    fig.savefig(figures/'comparison.png',dpi=200);plt.close(fig)
    selected=sorted([r for r in evaluations if r['method']=='llm_astar' and r['model']==model and r['prompt_style']=='standard' and r['valid']],
                    key=lambda r:(r['operation']/control_cases[r['case_id']]['operation'],r['case_id']))
    choices=[selected[len(selected)//2],selected[-1]]
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    genlookup={(g['case_id'],g['model'],g['prompt_style']):g for g in generations}
    traces=[]
    for ax,r,label in zip(axes,choices,['탐색 횟수 비율 중앙값 사례','탐색 횟수 비율 최대 사례']):
        q=query_by_id[r['case_id']]['query'];geo=Geometry(q)
        raw=genlookup[(r['case_id'],model,'standard')]['response']['choices'][0]['message'].get('content') or ''
        traced=search(q,geo,'llm_astar',raw,trace=True)
        traces.append(dict(case_id=r['case_id'],selection=label,**traced))
        popped=[e['node'] for e in traced['trace'] if e['event']=='pop']
        if popped: ax.scatter(*zip(*popped),s=4,color='#c3c9ce',alpha=.6,zorder=1)
        for y,a,b in geo.horizontal:ax.plot([a,b],[y,y],color='#1b263b',lw=2)
        for x,a,b in geo.vertical:ax.plot([x,x],[a,b],color='#1b263b',lw=2)
        nodes,_=parse_output(raw,upstream=True)
        if nodes:ax.plot(*zip(*nodes),'o--',color='#d78521',ms=3,lw=1,label='생성한 waypoint')
        ax.plot(*zip(*r['path']),color='#2364aa',lw=2,label='LLM-A* 경로')
        ax.plot(*zip(*control_cases[r['case_id']]['path']),color='#23794f',lw=1.5,ls=':',label='g+2h 경로')
        ax.scatter(*q['start'],marker='s',s=55,color='black',zorder=5)
        ax.scatter(*q['goal'],marker='*',s=100,color='#aa2222',zorder=5)
        ax.set(xlim=(geo.x0,geo.x1),ylim=(geo.y0,geo.y1),aspect='equal',title=f"{label}: {r['case_id']}\nLLM-A* / g+2h 탐색 횟수 = {r['operation']/control_cases[r['case_id']]['operation']:.2f}")
    axes[0].legend(loc='lower left',fontsize=8)
    fig.savefig(figures/'cases.png',dpi=200);plt.close(fig)
    (out/'case_traces.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    new=json.loads((ROOT/'results/generalization/summary.json').read_text())
    nr={r['method']:r for r in new}
    ci=proposal['vs_llm_astar']['operation']
    schema_count=sum(v for k,v in schema.items() if k not in ('None','missing_list'))
    replacements=dict(TABLE1='\n'.join(table),ABLATION='\n'.join(ablation),
      MINI_OP=fmt(mini['operation_percent']),MINI_LENGTH=fmt(mini['length_percent']),MINI_VALID=fmt(mini['valid_percent']),
      CONTROL_OP=fmt(control['operation_percent']),CONTROL_STORAGE=fmt(control['storage_percent']),CONTROL_LENGTH=fmt(control['length_percent']),
      DWA_OP=fmt(dwa['operation_percent']),DWA_STORAGE=fmt(dwa['storage_percent']),DWA_LENGTH=fmt(dwa['length_percent']),
      AUDIT_EXTRA=f"Final auditing verifies all 1,000 A* lengths against Dijkstra, revalidates every successful recorded route, and checks {len(integrity['real_response_upstream_parity'])} actual model outputs against upstream search.",
      FINISH=f"All 6,000 requests succeeded at the API level; {finish.get('length',0)} ended at the token limit. A successful HTTP request is not equivalent to a valid generated path.",
      RESULT_TEXT=f"Across both models and three prompts, LLM-A* uses between {pct(min(r['operation_percent'] for r in rows if r['method']=='llm_astar'))} and {pct(max(r['operation_percent'] for r in rows if r['method']=='llm_astar'))} of A* operations. GPT-3.5 Few-shot yields {pct(gpt['operation_percent'])} operations, {pct(gpt['storage_percent'])} storage and {pct(gpt['length_percent'])} length.",
      FAILURE_TEXT=f"The strict two-coordinate parser flags {schema_count} schema-invalid responses and {schema.get('missing_list',0)} responses without a matching list. LLM-only validity ranges from {pct(min(r['valid_percent'] for r in rows if r['method']=='llm_only'))} to {pct(max(r['valid_percent'] for r in rows if r['method']=='llm_only'))}. In contrast, hybrid search validity is 100\\% in this dataset even when its model output is not a valid route.",
      CONTROL_TEXT='All six hybrid conditions have larger aggregate operation ratios than this no-LLM control. This does not imply the control wins on every individual query.',
      PROPOSAL_TEXT=f"For GPT-4o mini Few-shot, V-Gate accepts {proposal['accepted']} of 1,000 responses, uses {pct(proposal['operation_percent'])} operations and {pct(proposal['storage_percent'])} storage, and produces {pct(proposal['length_percent'])} relative length with {pct(proposal['valid_percent'])} validity.",
      CI_TEXT=f"The paired V-Gate/LLM-A* operation ratio is {pct(ci['ratio_percent'])}, with a map-bootstrap 95\\% interval [{fmt(ci['ci95'][0])}, {fmt(ci['ci95'][1])}]\\% over {ci['n']} common-success cases.",
      GENERALIZATION=f"LLM-A* and V-Gate both succeed on 30/30 queries. Their operation ratios are {pct(nr['llm_astar']['operation_percent'])} and {pct(nr['validated']['operation_percent'])}, respectively; the LLM-free control uses {pct(nr['goal_only']['operation_percent'])}. LLM-only succeeds on {nr['llm_only']['valid_n']}/30.",
      COST_TEXT=f"The main experiment consumed {sum(u['prompt_tokens'] for u in costs.values()):,} input tokens and {sum(u['completion_tokens'] for u in costs.values()):,} output tokens, with an estimated standard API cost of USD {total_cost:.3f}. Mean per-request latencies were "+', '.join(('GPT-4o mini' if m==model else 'GPT-3.5')+f" {u['mean_request_seconds']:.2f} seconds" for m,u in costs.items())+'.',
      OVERLAP_TEXT='Exact prompt-example overlaps were '+', '.join(f"{styles[p]}: {len(overlap[p])}" for p in PROMPTS)+'. We retain the full dataset for the primary table and supply separate overlap-excluded aggregates; these are not held-out data.',
      ORIGINAL='\n'.join(' & '.join([str(v) for v in r[:3]]+[fmt(v) for v in r[3:]])+r' \\' for r in json.loads((report/'paper_table1.json').read_text())['rows']))
    replacements.update(localized_fields(rows,gpt,proposal,ci,nr,integrity,finish,schema_count,
                                        schema,costs,total_cost,overlap,styles))
    audit=json.loads((ROOT/'results/verification/performance_audit.json').read_text(encoding='utf-8'))
    count=audit['against_original']['operation']
    excluded=audit['excluding_pilot_maps']
    replacements['PERFORMANCE_AUDIT']=(f"추가 확인에서 원 LLM-A*보다 탐색 횟수가 줄어든 문제는 {count['better']}개, 같은 문제는 {count['equal']}개, 늘어난 문제는 {count['worse']}개였다. "
        f"파일럿에 사용한 맵 전체를 제외한 {excluded['n']}문제에서도 탐색 횟수 비율은 {pct(excluded['comparison']['operation']['ratio_percent'])}였다. "
        "이는 저장된 공개 결과의 사후 부분집합 분석이며 새로 수집한 비공개 시험 결과는 아니다.")
    template=(report/'report_template.tex').read_text(encoding='utf-8')
    for key,value in replacements.items():template=template.replace('@@'+key+'@@',value)
    assert '@@' not in template
    (report/'LLM_Astar_Reproduction_Report.tex').write_text(template,encoding='utf-8')
    # Korean companion focuses on understanding, not unreviewed submission prose.
    ko=f'''# 2026-10-13 재현 결과와 해석

공개 1,000문제 × 모델 2개 × 프롬프트 3종의 **6,000개 실제 응답**을 수집했다. API 오류는 0건이다. Table 1의 14개 조건을 각 1,000문제로 평가했고, 제안 방법과 ablation 18,000건, 새 맵 30문제 비교를 추가했다. 원문 응답을 공유하므로 평가 건수만큼 LLM을 다시 호출한 것은 아니다.

## 먼저 볼 결과

| 방법 | A* 대비 연산량 | 저장량 | 경로 길이 | 유효 경로 |
| --- | ---: | ---: | ---: | ---: |
| A* | 100% | 100% | 100% | 100% |
| GPT-4o mini Few-shot LLM-A* | {mini['operation_percent']:.2f}% | {mini['storage_percent']:.2f}% | {mini['length_percent']:.2f}% | {mini['valid_percent']:.2f}% |
| V-Gate 개선안, 같은 응답 | {proposal['operation_percent']:.2f}% | {proposal['storage_percent']:.2f}% | {proposal['length_percent']:.2f}% | {proposal['valid_percent']:.2f}% |
| LLM 없는 g+2h 대조군 | {control['operation_percent']:.2f}% | {control['storage_percent']:.2f}% | {control['length_percent']:.2f}% | 100% |

비율은 문제별 방법/A*의 기하평균이다. 예를 들어 연산량 {mini['operation_percent']:.2f}%는 A*보다 {100-mini['operation_percent']:.2f}% 적다는 뜻이다. [전체 14행](../results/main/table1.md), [원문 Table 1 인용값](../report/paper_table1.json)을 구분해 확인한다.

## 이번 결과가 말해 주는 것

1. LLM이 직접 내놓은 선분 경로는 충돌이 많다. LLM-only 성공률은 여섯 조건에서 {min(r['valid_percent'] for r in rows if r['method']=='llm_only'):.1f}~{max(r['valid_percent'] for r in rows if r['method']=='llm_only'):.1f}%다. 탐색을 결합하면 이 데이터에서 모두 100% 경로를 얻었다.
2. A*보다 탐색이 줄었다고 해서 그 감소가 모두 LLM 덕분인 것은 아니다. LLM 없이 목표 거리의 가중치만 두 배로 한 대조군이 여섯 LLM-A* 조건보다 평균 연산량이 적었다. 이것이 우리 보고서의 핵심 대조 분석이다.
3. V-Gate는 생성 경로의 시작·끝, 경계, 충돌과 정수 좌표를 검사한다. 통과하면 원 안내를 쓰고, 실패하면 g+2h로 전환한다. 주 조건에서 {proposal['accepted']}/1,000개 안내만 채택했다. 원 LLM-A* 대비 연산량 비율은 {ci['ratio_percent']:.2f}%, 맵 단위 bootstrap 95% 구간은 {ci['ci95'][0]:.2f}~{ci['ci95'][1]:.2f}%다.
4. 개선안도 LLM 없는 대조군보다 좋다고 주장할 수 없다. 안내를 대부분 버리는 것이 좋아진 이유라면 LLM 호출의 필요성 자체가 약해진다. 개선안은 이미 응답을 받은 뒤 검증하므로 API 비용도 그대로 발생한다.

## 원본 구현에서 확인한 함정

- waypoint를 반드시 통과하지 않는다. 목표는 그대로이고 g+h(goal)+h(target)로 탐색 순서만 바뀐다.
- 목록이 없으면 start/goal로 fallback되어 g+2h가 된다. 이번에 {schema.get('missing_list',0)}개 응답이 해당했다.
- 마지막 `[[...]]`를 무조건 읽는 parser는 장애물 triple까지 waypoint로 해석한다. 실제 {schema_count}개 응답에서 2차원 schema 위반을 확인했다. 원본 재현에서는 앞 두 숫자를 쓰는 동작을 보존하고, LLM-only와 V-Gate에서는 거부한다.
- 1,000토큰 제한으로 {finish.get('length',0)}개 응답이 잘렸다. 이 응답도 숨기거나 다시 생성하지 않았다.
- operation은 중복 pop 포함, storage는 inf 비용 항목 포함이다. 실제 시간·메모리와 동일한 값이 아니다.

## 검증과 추가 실험

원본 대비 90개 고정 응답 검사에 더해 실제 응답 {len(integrity['real_response_upstream_parity'])}개를 원본 코드로 재생해 비교했다. A* 전체 1,000개 경로 길이를 독립 Dijkstra로 확인했고 저장된 모든 성공 경로를 재검사했다. [검증 기록](../results/verification/final_integrity.json).

새 맵 10개·30문제에서는 LLM-A*와 V-Gate 모두 30/30 성공했다. 연산량 비율은 각각 {nr['llm_astar']['operation_percent']:.2f}%, {nr['validated']['operation_percent']:.2f}%였고, g+2h는 {nr['goal_only']['operation_percent']:.2f}%였다. 이 작은 smoke 실험으로 광범위한 일반화를 보장하지 않는다.

본 실험 6,000회의 표준 요금 추산은 약 **US${total_cost:.3f}**다. 실제 청구액은 아니며 모델 접근 확인, 자연어 입력 검사, 새 맵 30회는 이 숫자에서 제외한다. 토큰과 요청 지연은 [비용 기록](../results/main/costs.json)에 있다. 요금 출처: [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), [GPT-3.5 Turbo](https://developers.openai.com/api/docs/models/gpt-3.5-turbo).

## 재현 범위와 남은 제출 준비

GPT-4o mini가 논문의 Llama3를 대체했다. GPT-3.5도 정확한 원 snapshot/API와 같다고 주장하지 않는다. 원 코드의 newline stop을 제거했고 DWA 감쇠 규칙 및 실패 경로 길이의 분모는 명시한 가정이다. 논문의 10단계 확장성 실험은 수행하지 않았다. 과제 안내서의 Table 1 재현과 제안 비교에 집중했다.

코드·실험·결과 분석·ACL 보고서 초안을 마련했다. 팀 최종 버전 선정, 저자/팀명 반영, 사용자의 설명·문장 검토, 개인별 기여 보고서, 실제 제출과 이후 동료평가/답변은 별도다. 다른 팀원 기여를 추측해 작성하지 않았다. AI 지원을 숨기거나 탐지 회피를 위한 문장 변경을 하지 않는다.

![전체 조건 비교](../report/figures/comparison.png)

![실제 사례](../report/figures/cases.png)
'''
    (ROOT/'최신/재현_실험_결과.md').write_text(ko,encoding='utf-8')
    print(json.dumps(dict(report_source=str(report/'LLM_Astar_Reproduction_Report.tex'),estimated_main_usd=total_cost,gate_accept=proposal['accepted'],overlap=overlap),ensure_ascii=False))


if __name__=='__main__':main()
