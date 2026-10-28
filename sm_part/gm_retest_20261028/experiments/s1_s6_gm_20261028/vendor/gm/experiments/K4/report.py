"""Render saved K4 measurements; only illustrative traces are replayed."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from reproduction.core import parse_output
from .search import METHODS, PRIMARY, run_once

BASE='g2h_euclidean'; K3='g2h_wall_octile'; NOLLM='focal_goal_w105'
LABELS={
    'astar_euclidean':'기존 A*', BASE:'기존 g+2h',K3:'K3: octile＋벽',
    'g2h_wall_euclidean':'K3: 직선거리＋벽','clean_wastar_wall':'새 엔진 g+2h0',
    NOLLM:'FOCAL 5%: LLM 없음','focal_straight_w105':'FOCAL 5%: 직선 안내',
    PRIMARY:'K4: LLM＋5% 제한','focal_llm_w100':'K4: LLM＋최단거리 제한',
    'focal_llm_w110':'K4: LLM＋10% 제한','focal_llm_w200':'K4: LLM＋100% 제한',
    'focal_llm_euclidean_w105':'K4: 벽 하한 제거',
}


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def gm(values): return math.exp(statistics.mean(math.log(v) for v in values))


def load_rows(path):
    first,times={},{}
    for line in path.open(encoding='utf-8'):
        r=json.loads(line); key=(r['case_id'],r['method'])
        times.setdefault(key,[]).append(r['total_seconds'])
        if r['repeat']==0:first.setdefault(r['case_id'],{})[r['method']]=r
    for cid,methods in first.items():
        for method,r in methods.items():r['median_seconds']=statistics.median(times[(cid,method)])
    return first


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    output=args.input/'analysis';output.mkdir(exist_ok=True)
    if 'Malgun Gothic' in {f.name for f in font_manager.fontManager.ttflist}:plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries={g:read(args.input/g/'summary.json') for g in ('pilot','public','transfer')}
    groups={g:load_rows(args.input/g/'evaluations.jsonl') for g in ('public','transfer')}
    replies={r['case_id']:r for r in (json.loads(line) for line in (args.input/'responses.jsonl').open(encoding='utf-8'))}
    lines=['# K4 자동 수치표','',f'실행: `{args.input.name}`. 주 방법: `{PRIMARY}`. 비율은 문제별 비율의 기하평균 ×100. 낮을수록 작다.',
           '', 'operation은 모든 heap pop, expansions는 유효 후보 선택·확장(목표 포함). 시간은 모델 응답이 이미 있는 계획기 시간이다.','']
    for group,s in summaries.items():
        lines += [f'## {group}', '', '기존 g+2h=100 기준:', '',
                  '| 방법 | 전체 heap pop | 실제 탐색 횟수 | 길이 | 계획기 시간 | 최단+5% 초과 | 최장/최단 비율 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            d=s['comparisons'][BASE][method];a=s['absolute'][method]
            ratios=' | '.join(f'{d[k]["geometric_ratio"]*100:.3f}%' for k in ('operation','expansions','length','total_seconds'))
            lines.append(f'| {LABELS[method]} | {ratios} | {a["over_5_percent"]}/{a["cases"]} | {a["max_optimal_length_ratio"]*100:.3f}% |')
        for control in (BASE,K3,'g2h_wall_euclidean',NOLLM):
            lines += ['',f'주 K4 대 {LABELS[control]}:', '', '| 지표 | 상대 비율 | 95% 구간 | 개선 / 동일 / 악화 |','| --- | ---: | ---: | ---: |']
            for metric in ('operation','expansions','storage','length','total_seconds'):
                d=s['comparisons'][control][PRIMARY][metric]
                lines.append(f'| {metric} | {d["geometric_ratio"]*100:.3f}% | {d["ci95"][0]*100:.3f}–{d["ci95"][1]*100:.3f}% | {d["better"]} / {d["equal"]} / {d["worse"]} |')
        lines += ['', '절대 지표의 산술평균(위 비율 기하평균과 다름):','',
                  '| 방법 | 전체 pop | 실제 탐색 | 길이 | 계획기 ms | 최단경로 수 | 유효/전체 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            a=s['absolute'][method];v=a['means']
            lines.append(f'| {LABELS[method]} | {v["operation"]:.2f} | {v["expansions"]:.2f} | {v["length"]:.4f} | {v["total_seconds"]*1000:.4f} | {a["optimal_paths"]} | {a["valid"]}/{a["cases"]} |')
        lines.append('')
    (output/'TABLES.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')

    display=(BASE,K3,'g2h_wall_euclidean',NOLLM,PRIMARY)
    colors=('#6b7280','#2e86b7','#2d936c','#9b77b8','#df7621')
    fig,axes=plt.subplots(2,3,figsize=(14.6,9),layout='constrained')
    for ri,group in enumerate(('public','transfer')):
        s=summaries[group]
        for ci,(metric,title) in enumerate((('operation','전체 heap pop'),('length','경로 길이'),('total_seconds','계획기 시간'))):
            ax=axes[ri,ci]
            for i,method in enumerate(display):
                d=s['comparisons'][BASE][method][metric];v=d['geometric_ratio']*100;lo,hi=[x*100 for x in d['ci95']]
                ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=colors[i],capsize=3,markersize=6)
                ax.annotate(f'{v:.2f}%',(v,i),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=9,color=colors[i])
            ax.set_yticks(range(5),[LABELS[m] for m in display] if ci==0 else [])
            ax.set_ylim(4.7,-.5);ax.axvline(100,color='#999999',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
            ax.set_xlim((30,270) if metric=='operation' else ((97.5,103) if metric=='length' else (40,350)))
            group_label='공개 1,000문제' if group=='public' else '기존 별도 지도 30문제'
            ax.set_title(f'{group_label} · {title}');ax.set_xlabel('기존 g+2h 대비 (%)')
    fig.suptitle('K4: 경로 비용 제한과 탐색 비용의 절충\n문제별 비율 기하평균 · 지도 bootstrap 95% 구간 · LLM 생성 시간 제외',fontsize=15)
    fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)

    # Tail quality, rather than only the average path length.
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,group in zip(axes,('public','transfer')):
        for method,color in zip(display,colors):
            excess=sorted(max(0.0,(r[method]['length']/r[method]['optimum']-1)*100) for r in groups[group].values())
            ax.step([0,*excess,45],[0,*[(i+1)/len(excess)*100 for i in range(len(excess))],100],where='post',label=LABELS[method],color=color)
        ax.axvline(5,color='#555555',ls='--',lw=1)
        ax.set(xlim=(0,45),ylim=(0,103),xlabel='실제 최단거리보다 긴 비율 (%)',ylabel='해당 비율 이내의 문제 (%)',title='공개 1,000문제' if group=='public' else '별도 30문제')
        ax.grid(alpha=.2)
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('K4는 긴 우회를 5% 이내로 제한했지만 항상 최단경로를 반환하지는 않음',fontsize=14)
    fig.savefig(output/'quality_tail.png',dpi=170);plt.close(fig)

    public=groups['public'];queries={c['case_id']:c['query'] for c in read(args.input/'public_cases.json')}
    selected=[('K3 길이/최단거리 비율이 가장 큰 공개 사례',max(public,key=lambda cid:public[cid][K3]['length']/public[cid][K3]['optimum'])),
              ('K4/K3 전체 pop 비율이 가장 큰 공개 사례',max(public,key=lambda cid:public[cid][PRIMARY]['operation']/public[cid][K3]['operation']))]
    fig,axes=plt.subplots(2,3,figsize=(15,8.6),layout='constrained');examples=[]
    for ri,(rule,cid) in enumerate(selected):
        q=queries[cid];example=dict(case_id=cid,selection_rule=rule,methods={})
        for ci,method in enumerate((BASE,K3,PRIMARY)):
            old=public[cid][method];replay=run_once(q,method,replies[cid]['text'],trace=True)
            for key in ('path','operation','length','expansions'):assert json.dumps(old[key])==json.dumps(replay[key]),(cid,method,key)
            ax=axes[ri,ci];popped=[e['node'] for e in replay['trace'] if e.get('event','pop')=='pop']
            ax.scatter(*zip(*popped),s=7,marker='s',color='#a8bfce',alpha=.4,linewidths=0)
            for y,a,b in q['horizontal_barriers']:ax.plot([a,b],[y,y],color='#222222',lw=2)
            for x,a,b in q['vertical_barriers']:ax.plot([x,x],[a,b],color='#222222',lw=2)
            if method==PRIMARY:
                points,error=parse_output(replies[cid]['text'])
                if not error:
                    points=[q['start'],*points,q['goal']]
                    ax.plot(*zip(*points),color='#9257b1',ls='--',lw=1.2,alpha=.7,label='LLM 안내 (충돌 가능)')
                    if ri==0:ax.legend(fontsize=8,loc='upper left')
            ax.plot(*zip(*old['path']),lw=2,color=('#6b7280','#2e86b7','#df7621')[ci])
            ax.scatter(*q['start'],c='#27836a',s=30,zorder=5);ax.scatter(*q['goal'],c='#b93840',s=70,marker='*',zorder=5)
            ax.set(xlim=(0,q['range_x'][1]-1),ylim=(0,q['range_y'][1]-1),aspect='equal')
            ax.set_title(f'{cid} · {LABELS[method]}\npop {old["operation"]}, 탐색 {old["expansions"]}, 길이 {old["length"]:.3f}',fontsize=10)
            ax.grid(alpha=.1)
            example['methods'][method]={k:old[k] for k in ('operation','expansions','length','optimum')}
        examples.append(example)
    fig.suptitle('위: 긴 우회를 줄인 사례 / 아래: 같은 최단경로를 더 비싸게 찾은 사례\n회색: 탐색 위치 · 초록: 시작 · 빨강: 목표',fontsize=14)
    fig.savefig(output/'case_paths.png',dpi=170);plt.close(fig);save(output/'case_examples.json',examples)

    diagnostics={}
    for group,rows in groups.items():
        fallback=[cid for cid,r in rows.items() if r[PRIMARY]['guide_error']]
        for cid in fallback:
            for key in ('path','operation','length','expansions'):assert rows[cid][PRIMARY][key]==rows[cid][NOLLM][key]
        guided=[r for cid,r in rows.items() if cid not in fallback]
        diagnostics[group]=dict(fallback_cases=len(fallback),fallback_exactly_matches_no_llm=True,
            guided_cases=len(guided),guided_subset_posthoc={m:gm(r[PRIMARY][m]/r[NOLLM][m] for r in guided) for m in ('operation','expansions','length','median_seconds')},
            historical_api_mean_seconds=statistics.mean(replies[cid]['historical_api_seconds'] for cid in rows),
            historical_api_median_seconds=statistics.median(replies[cid]['historical_api_seconds'] for cid in rows),
            historical_total_tokens=sum(replies[cid]['usage']['total_tokens'] for cid in rows),
            response_finish_reasons=dict(Counter(replies[cid]['finish_reason'] for cid in rows)))
    save(output/'diagnostics.json',diagnostics)
    inputs=[args.input/'manifest.json',args.input/'responses.jsonl',*[args.input/g/f for g in ('public','transfer','pilot') for f in ('evaluations.jsonl','summary.json')]]
    save(output/'manifest.json',dict(purpose='Post-experiment descriptive analysis, no retuning',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(args.input)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        matplotlib=matplotlib.__version__,numpy=np.__version__,illustrative_trace_replays_per_render=6,new_api_calls=0))
    print(json.dumps(dict(selected=selected,diagnostics=diagnostics),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
