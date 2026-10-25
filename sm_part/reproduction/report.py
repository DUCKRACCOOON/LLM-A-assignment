"""Build CSV/JSON tables, publication-style plots, and a Korean HTML report."""
import collections
import csv
import html
import json
import math
import os
import statistics
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT, load_cases, World, search
from run import save

PAPER={('astar',None):[100,100,100,100],('dynamic_weighted',None):[60.91,78.53,100.24,100],
 ('llm_only','standard'):[None,None,119.38,12.8],('llm_only','cot'):[None,None,151.73,15.2],
 ('llm_only','repe'):[None,None,183.87,7.8],('llm_astar','standard'):[57.39,74.96,102.44,100],
 ('llm_astar','cot'):[69.5,83.65,102.54,100],('llm_astar','repe'):[85.47,96.53,102.41,100]}
LABEL={'astar':'A*','dynamic_weighted':'Dynamic weighted A* (reconstructed)',
       'llm_only':'GPT only','llm_astar':'GPT + LLM-A*'}
PROMPT={'standard':'Few-shot','cot':'CoT','repe':'RePE',None:''}

def gm(values):
    return 100*math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def summarize(rows,exclude=None):
    exclude=set(exclude or [])
    rows=[r for r in rows if r['case_id'] not in exclude]
    base={r['case_id']:r for r in rows if r['method']=='astar'}
    groups=collections.defaultdict(list)
    for r in rows: groups[r['method'],r['prompt']].append(r)
    table=[]
    for key in PAPER:
        values=groups.get(key,[])
        if not values: continue
        record={'method':key[0],'prompt':key[1],'n':len(values),'valid_n':sum(r['valid'] for r in values)}
        for metric in ('operation','storage','length'):
            eligible=[r for r in values if r['valid'] and r.get(metric) is not None
                      and r[metric]>0 and base[r['case_id']]['valid'] and base[r['case_id']][metric]>0]
            record[metric+'_ratio']=gm([r[metric]/base[r['case_id']][metric] for r in eligible])
            record[metric+'_n']=len(eligible)
        parsed=[r for r in values if r.get('length') is not None and r['length']>0 and base[r['case_id']]['length']>0]
        record['length_all_parseable_ratio']=gm([r['length']/base[r['case_id']]['length'] for r in parsed])
        record['length_all_parseable_n']=len(parsed)
        record['valid_ratio']=100*record['valid_n']/record['n']
        record['failures']=dict(collections.Counter(r.get('failure') for r in values if not r['valid']))
        record['parse_failures']=dict(collections.Counter(r.get('parse_error') for r in values if r.get('parse_error')))
        if key[0]=='llm_astar':
            record['only_start_goal_targets_n']=sum(len(r.get('targets',[]))==2 for r in values)
        table.append(record)
    return table

def usage(stage):
    files=list((ROOT/'results'/stage/'raw').glob('*.json'))
    records=[json.loads(p.read_text(encoding='utf-8')) for p in files]
    result={'requests':len(records),'prompt_tokens':0,'completion_tokens':0,
            'finish_reasons':dict(collections.Counter(r['response']['choices'][0]['finish_reason'] for r in records)),
            'returned_models':sorted({r['response']['model'] for r in records})}
    for r in records:
        for k in ('prompt_tokens','completion_tokens'): result[k]+=r['response']['usage'][k]
    result['estimated_usd']=result['prompt_tokens']*0.5/1e6+result['completion_tokens']*1.5/1e6
    return result

def number(v): return '—' if v is None else f'{v:.2f}'

def table_html(rows):
    body=''
    for r in rows:
        key=(r['method'],r['prompt']); paper=PAPER[key]
        values=[r['operation_ratio'],r['storage_ratio'],r['length_ratio'],r['valid_ratio']]
        cells=''.join(f'<td>{number(v)}<small>논문 {number(p)}</small></td>' for v,p in zip(values,paper))
        body+=f'<tr><th>{LABEL[key[0]]}<small>{PROMPT[key[1]]}</small></th>{cells}<td>{r["valid_n"]}/{r["n"]}</td></tr>'
    return '<table><thead><tr><th>조건</th><th>Operation % ↓</th><th>Storage % ↓</th><th>Length % ↓</th><th>Valid % ↑</th><th>유효 경로</th></tr></thead><tbody>'+body+'</tbody></table>'

def main_plot(summary,out):
    rows=[r for r in summary if r['method']=='llm_astar']
    if not rows:return
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,metric,title in zip(axes,['operation','storage','length'],['Operations / A* (%)','Storage / A* (%)','Path length / A* (%)']):
        xs=list(range(len(rows))); actual=[r[metric+'_ratio'] for r in rows]
        paper=[PAPER[r['method'],r['prompt']][['operation','storage','length'].index(metric)] for r in rows]
        ax.bar([x-.18 for x in xs],actual,.36,label='Reproduction',color='#287f9a')
        ax.bar([x+.18 for x in xs],paper,.36,label='Paper',color='#c7d3dd')
        ax.axhline(100,color='#666',ls=':',lw=1); ax.set_xticks(xs,[PROMPT[r['prompt']] for r in rows]);ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
        if metric=='length': ax.set_ylim(98,106)
    axes[0].legend(frameon=False)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)

def cases_plot(rows,out):
    qs={q['id']:q for q in load_cases()}; by=collections.defaultdict(dict)
    for r in rows:by[r['case_id']][r['method'],r['prompt']]=r
    candidates=[cid for cid,d in by.items() if ('llm_astar','standard') in d]
    if not candidates:return
    ranked=sorted(candidates,key=lambda cid:by[cid]['llm_astar','standard']['operation']/by[cid]['astar',None]['operation'])
    chosen=[ranked[0],ranked[-1]]
    failing=next((cid for cid in candidates if not by[cid].get(('llm_only','standard'),{}).get('valid',True)),None)
    if failing and failing not in chosen:chosen.append(failing)
    captions=[]
    for cid in chosen:
        q=qs[cid]; d=by[cid]; w=World(q)
        ref=search(q,'astar',world=w,keep_visited=True)
        hybrid=search(q,'llm_astar',[tuple(p) for p in d['llm_only','standard']['path']],w,True)
        fig,axes=plt.subplots(1,3,figsize=(13.5,3.7),layout='constrained')
        for ax,r,title in zip(axes,[ref,d['llm_only','standard'],hybrid],['A*','GPT only / Few-shot','GPT + LLM-A* / Few-shot']):
            for a,b in w.barriers:ax.plot([a[0],b[0]],[a[1],b[1]],color='#334155',lw=1.5)
            visits=r.get('visited') or []
            if visits:ax.scatter(*zip(*visits),s=4,color='#a9d2dd',alpha=.4)
            path=r.get('path') or []
            if path:ax.plot(*zip(*path),color='#d04e36',lw=1.5,marker='.' if len(path)<20 else None)
            ax.scatter(*q['start'],color='#059669',s=45,zorder=5);ax.scatter(*q['goal'],color='#7c3aed',s=45,zorder=5)
            ax.set_xlim(-1,51);ax.set_ylim(-1,31);ax.set_aspect('equal');ax.set_title(f'{title}\nvalid={r["valid"]}')
        fig.suptitle(cid);fig.savefig(out/f'case-{cid}.png',dpi=150);plt.close(fig)
        captions.append(cid)
    return captions

def scaling_plot(out):
    path=ROOT/'results/scaling/rows.json'
    if not path.exists():return None
    rows=json.loads(path.read_text(encoding='utf-8'))
    lookup={(r['case_id'],r['method']):r for r in rows if r['method']!='llm_only'}
    points=[]
    for scale in range(1,11):
        for method in ('astar','llm_astar'):
            for metric in ('operation','storage'):
                values=[]
                for (cid,m),r in lookup.items():
                    if m!=method or not cid.endswith(f'-s{scale:02d}'):continue
                    base=lookup[(cid[:-3]+'s01',method)]
                    values.append(r[metric]/base[metric])
                if values:points.append({'scale':scale,'method':method,'metric':metric,'growth':statistics.mean(values),'n':len(values)})
    save(out/'scaling.json',points)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,metric in zip(axes,['operation','storage']):
        for method,color in [('astar','#64748b'),('llm_astar','#287f9a')]:
            p=[p for p in points if p['method']==method and p['metric']==metric]
            ax.plot([r['scale'] for r in p],[r['growth'] for r in p],'-o',label=LABEL[method],color=color)
        ax.set_xlabel('Linear scale');ax.set_ylabel(f'{metric.title()} growth / scale 1');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('GPT adaptation: paired 10-case sample; Few-shot')
    fig.savefig(out/'scaling.png',dpi=180);fig.savefig(out/'scaling.pdf');plt.close(fig)
    return points

def diagnostics(rows, summary, out):
    baselines={r['case_id']:r for r in rows if r['method']=='astar'}
    diagnostic_rows=[]; worlds={}
    cache=out/'no_waypoint_ablation.json'
    if cache.exists():
        diagnostic_rows=json.loads(cache.read_text(encoding='utf-8'))
    else:
        for q in load_cases():
            if q['map_id'] not in worlds: worlds[q['map_id']]=World(q)
            r=search(q,'llm_astar',[],worlds[q['map_id']])
            diagnostic_rows.append({'case_id':q['id'],**{k:r[k] for k in ('operation','storage','length','valid')}})
        save(cache,diagnostic_rows)
    record={metric:gm([r[metric]/baselines[r['case_id']][metric] for r in diagnostic_rows if r['valid']])
            for metric in ('operation','storage','length')}
    record['valid_ratio']=100*sum(r['valid'] for r in diagnostic_rows)/len(diagnostic_rows)
    record['n']=len(diagnostic_rows)
    target_counts={PROMPT[r['prompt']]:r.get('only_start_goal_targets_n',0) for r in summary if r['method']=='llm_astar'}
    save(out/'diagnostics.json',{'no_waypoint_baseline':record,'only_start_goal_target_counts':target_counts})
    return record,target_counts

def main():
    out=ROOT/'results/report';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'results/main/rows.json').read_text(encoding='utf-8'))
    summary=summarize(rows)
    audit=json.loads((ROOT/'results/dataset_audit.json').read_text(encoding='utf-8'))
    overlap=sorted({cid for ids in audit['exact_demonstration_overlap'].values() for cid in ids})
    clean=summarize(rows,overlap)
    pilot=summarize(json.loads((ROOT/'results/pilot/rows.json').read_text(encoding='utf-8')))
    costs={stage:usage(stage) for stage in ['pilot','main','scaling']}
    audit_records=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'results/pilot/newline_audit').glob('*.json')]
    audit_cost=sum(r['response']['usage']['prompt_tokens']*.5/1e6+r['response']['usage']['completion_tokens']*1.5/1e6 for r in audit_records)
    total=sum(r['estimated_usd'] for r in costs.values())+audit_cost
    save(out/'summary.json',{'main':summary,'without_demo_overlap':clean,'pilot':pilot,'usage':costs,'estimated_total_usd':total})
    for name,data in [('table1',summary),('table1_without_demo_overlap',clean)]:
        with (out/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            fields=list(dict.fromkeys(k for row in data for k in row))
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    main_plot(summary,out); selected=cases_plot(rows,out) or []; scaling=scaling_plot(out)
    no_waypoint,target_counts=diagnostics(rows,summary,out)
    manifest=json.loads((ROOT/'results/main/manifest.json').read_text(encoding='utf-8'))
    complete=costs['main']['requests']==3000 and len(rows)==8000
    main_narrative=''
    if complete:
        few=next(r for r in summary if r['method']=='llm_astar' and r['prompt']=='standard')
        main_narrative=f'<p>Few-shot 결합은 A* 대비 연산량 {few["operation_ratio"]:.2f}%, 저장량 {few["storage_ratio"]:.2f}%, 경로 길이 {few["length_ratio"]:.2f}%, 유효 경로 {few["valid_ratio"]:.2f}%였습니다. 탐색량 감소와 경로 길이의 소폭 증가라는 경향은 관찰됐지만, 모델·생성 설정 차이가 있어 논문 수치의 완전한 일치로 해석하지 않습니다.</p>'
    scaling_narrative=''
    if scaling:
        a=next(p['growth'] for p in scaling if p['scale']==10 and p['method']=='astar' and p['metric']=='operation')
        l=next(p['growth'] for p in scaling if p['scale']==10 and p['method']=='llm_astar' and p['metric']=='operation')
        scaling_narrative=f'<p>규모 10에서 규모 1 대비 연산량 증가율은 A* {a:.2f}배, GPT LLM-A* {l:.2f}배였습니다. 이번 GPT 조건에서는 논문의 거의 선형적인 확장성 경향을 재현하지 못했습니다. 각 알고리즘 자신의 규모 1을 분모로 한 수치이므로 두 증가율의 비율을 같은 규모에서의 절대 연산량 비율로 해석하면 안 됩니다.</p>'
    raw_models=', '.join(costs['main']['returned_models'])
    response_diagnostics=[]
    for r in summary:
        if r['method']=='llm_only':
            response_diagnostics.append(f'{PROMPT[r["prompt"]]}: 경로 목록 추출 실패 {sum(r["parse_failures"].values())}/{r["n"]}건')
    sections=f'''<h1>LLM-A*: GPT API 재현 결과</h1>
    <p class="lead">상태: {'본 실험 1,000문제·8조건 완료' if complete else '진행 중 — 부분 결과'}. 이 보고서는 완료된 GPT 기반 8개 조건을 다룹니다.</p>
    <div class="cards"><div><b>{costs['main']['requests']:,}</b>본 실험 API 응답</div><div><b>{len(rows):,}</b>평가 결과 행</div><div><b>${total:.4f}</b>전체 토큰 요금 추정</div></div>
    <h2>실험 조건</h2><p>요청 모델 gpt-3.5-turbo-0125, 반환 모델 {html.escape(raw_models)}. temperature=0, max_tokens=1000, 프롬프트당 1회 생성. Few-shot 5-shot, CoT/RePE 3-shot. 동일 응답을 GPT 단독 및 LLM-A*에 공유했습니다. 학습·파인튜닝은 수행하지 않았습니다.</p>
    <p>저자 데이터 100개 지도 × 10개 시작·목표 쌍 = 1,000문제. 범위 [0,51]×[0,31]은 저자 코드에 따라 경계 좌표 0~50, 0~30으로 해석했습니다. 8방향 이동과 유클리드 비용, 경계 접촉·장애물 선분 교차를 충돌로 처리했습니다.</p>
    <p>저장소 고정 커밋: <code>{manifest['vendor_commit']}</code>. 요청·응답·사용량·요청 ID·프롬프트 해시를 저장했습니다.</p>
    <h2>Table 1 비교</h2><p>각 셀의 큰 숫자는 이번 결과, 작은 숫자는 논문 값입니다. 연산·저장·길이는 문제별 A* 대비 비율의 기하평균입니다. 길이의 주 통계는 유효 경로만 포함합니다. CSV에는 파싱 가능한 모든 경로의 길이 통계와 각 분모도 함께 제공합니다. 논문의 실패 경로 길이 집계 규칙은 명확하지 않아 직접 수치 비교에 한계가 있습니다.</p>{table_html(summary)}
    {main_narrative}<img src="comparison.png" alt="논문과 재현 결과 비교">
    <h2>논문·공개 코드와 달라진 점</h2><ul>
    <li>논문은 GPT-3.5-Turbo로 기재했지만 공개 코드의 경유점 생성은 Turbo-Instruct입니다. 이번에는 논문에 명시된 모델 계열의 고정 스냅샷과 Chat Completions를 사용했습니다. 당시 저자가 호출한 정확한 스냅샷은 확인되지 않았습니다.</li>
    <li>원문 프롬프트를 단일 user 메시지로 전달했습니다. 공개 코드의 stop=[줄바꿈] 설정은 개발 문제에서 CoT/RePE 최종 경로 출력을 차단하여 제외했습니다. 비교 응답 3건을 별도로 보존했습니다.</li>
    <li>공개 구현은 호출자 입력 범위를 변경하므로 독립적인 데이터 구조로 실행했습니다. 탐색 우선순위, 중복 힙 pop, 목표 전환, 경유점 필터, operation 및 storage 집계는 보존했습니다.</li>
    <li>공개 Dynamic Weighted A* 구현을 찾지 못했습니다. w=max(1,2×0.99^k), k=pop 횟수, 매 확장 후 OPEN 우선순위 재계산으로 구성한 비교군입니다. 논문과 완전히 같은 구현이라고 주장하지 않습니다.</li>
    <li>파싱 실패는 별도 기록하며, 빈 경유점은 저자 방식대로 시작·목표만 넣어 탐색합니다. 이 경우 우선순위가 g+2h가 되어 일반 A*와 다릅니다. 실패 응답을 성공할 때까지 재생성하지 않았습니다.</li>
    <li>연산량은 중복을 포함한 heap pop 횟수, 저장량은 g 딕셔너리 항목 수입니다. 실제 지연 시간·RAM 측정치가 아니며 실행 시간에는 충돌 캐시 영향이 있어 속도 우위를 주장하지 않습니다.</li></ul>
    <h2>경유점 실패와 추가 진단</h2><p>{'; '.join(response_diagnostics)}. GPT가 실제로 가능한 문제에 “No path found”라고 답한 경우도 포함합니다. 본 실험에서 출력 상한으로 종료된 응답은 {costs['main']['finish_reasons'].get('length',0)}건이며, 그대로 보존하고 다시 생성하지 않았습니다.</p>
    <p>시작·목표만 남은 경유점 목록의 수: {html.escape(json.dumps(target_counts,ensure_ascii=False))}. 파싱 실패 외에 필터링이나 GPT가 시작·목표만 생성한 경우도 포함합니다. GPT의 경유점 없이 g+2h만 사용한 추가 비교군은 연산량 {no_waypoint['operation']:.2f}%, 저장량 {no_waypoint['storage']:.2f}%, 경로 길이 {no_waypoint['length']:.2f}%, 유효 경로 {no_waypoint['valid_ratio']:.2f}%였습니다. 이 비교군은 논문 Table 1에 없는 진단용 실험이며 추가 API 호출 없이 실행했습니다. 따라서 A* 대비 효율 향상을 전부 GPT의 경유점 품질에 귀속하지 않습니다.</p>
    <h2>프롬프트 예시 중복 제외 분석</h2><p>저자 데이터에 예시와 정확히 겹치는 문제 {len(overlap)}개가 있습니다: {', '.join(overlap)}. 원본 1,000개를 유지한 주 실험 외에 세 프롬프트에서 이 문제들의 합집합을 공통으로 제외했습니다.</p>{table_html(clean)}
    <h2>예비 실험</h2><p>본 평가와 분리한 합성 문제 10개로 검증했습니다. 예비 실험 결과로 평가 데이터의 프롬프트 내용을 튜닝하지 않았습니다.</p>{table_html(pilot)}
    <h2>확장성</h2><p>논문 Figure 3의 Llama 실험을 GPT로 변경한 보조 실험입니다. 고정 시드 240702511로 원본 문제 10개를 선택하고 같은 문제를 1~10배 좌표 확대했습니다. 각 문제의 규모 1 대비 증가율을 계산해 산술평균했습니다. 원문의 정확한 샘플·확대 절차는 제공되지 않아 프로토콜을 명시한 재구성입니다.</p>
    {scaling_narrative}{'<img src="scaling.png" alt="GPT 확장성 실험">' if scaling else '<p>확장성 평가 진행 중.</p>'}
    <h2>경로 사례</h2><p>Few-shot 기준 연산량 비율의 최솟값·최댓값 사례 및 GPT 단독 실패 사례를 자동 선택했습니다. 파란 점은 탐색 상태, 빨간 선은 경로, 초록 점은 출발점, 보라 점은 목표점입니다. 대표성 있는 무작위 표본으로 해석하지 않습니다.</p>
    {''.join(f'<img src="case-{cid}.png" alt="{cid}">' for cid in selected)}
    <h2>사용량과 재실행</h2><pre>{html.escape(json.dumps(costs,ensure_ascii=False,indent=2))}</pre>
    <p>추정 비용은 입력 $0.50/백만 토큰, 출력 $1.50/백만 토큰으로 계산했습니다. 예비 실험·본 실험·확장성·줄바꿈 비교를 포함하며 실제 청구 내역은 아닙니다. 키는 산출물에 저장하지 않습니다.</p>
    <h2>검증</h2><p>저자 A*·LLM-A*와 15개 문제씩 연산·저장·경로 길이 일치, 20개 문제의 A*와 Dijkstra 최단거리 일치, 3,000회 기하 충돌 비교 및 퇴화 선분·파싱·경로 검증 테스트를 수행했습니다. 상세 실행 절차는 프로젝트 README를 참고하십시오.</p>
    <h2>출처</h2><p><a href="https://arxiv.org/abs/2407.02511v2">논문 v2</a> · <a href="https://github.com/SilinMeng0510/llm-astar/tree/{manifest['vendor_commit']}">고정된 저자 코드</a> · <a href="https://developers.openai.com/api/docs/pricing">OpenAI 요금</a></p>'''
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>LLM-A* GPT 재현 결과</title><style>
    body{font:16px/1.75 "Malgun Gothic",sans-serif;color:#183042;max-width:1120px;margin:40px auto;padding:0 24px;background:#f8fafc}
    h1{font-size:34px}h2{margin-top:44px;border-bottom:2px solid #d3e4ea;padding-bottom:8px}p,li{max-width:1000px}li{margin:10px 0}
    table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border-bottom:1px solid #dbe5ed;padding:12px;text-align:right}th:first-child{text-align:left}thead{background:#e5eff3}
    small{display:block;color:#667b8a;font-size:12px}img{width:100%;height:auto;margin-top:24px;background:white}pre{overflow:auto;font-size:13px;background:#edf2f7;padding:18px}
    .cards{display:flex;gap:20px;margin:30px 0}.cards div{background:#e5eff3;padding:18px 30px;border-radius:10px;flex:1}.cards b{display:block;font-size:29px;color:#15728b}
    a{color:#0d7496}.lead{font-size:19px}code{overflow-wrap:anywhere}
    </style><body>'''+sections+'</body></html>'
    (out/'report.html').write_text(document,encoding='utf-8')
    print(json.dumps({'main_requests':costs['main']['requests'],'rows':len(rows),'estimated_total_usd':total,'complete':complete},indent=2))

if __name__=='__main__':main()
