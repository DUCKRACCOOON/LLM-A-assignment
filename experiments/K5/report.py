"""Tables, figures, diagnostics from saved K5 runs; no performance reruns."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from experiments.K3.run import sha,write_json
from .run import METRICS,read_rows,case_seed
from .search import METHODS,PRIMARY,STOCHASTIC,run_once

BASE='g2h_euclidean'; K3='g2h_wall_octile'; K4='focal_llm_w105'; TIGHT='mha_dts_w105'
LABELS={BASE:'기존 g+2h',K3:'K3: octile＋벽','g2h_wall_euclidean':'K3: 직선거리＋벽',
        'clean_wastar_wall':'단일 weighted A*',K4:'K4: LLM, 5% 제한',
        'focal_goal_w105':'K4: LLM 없음, 5% 제한','focal_llm_w200':'K4: LLM, 2배 제한',
        PRIMARY:'K5: DTS, 2배 제한','mha_rr_w200':'K5: 고정 순환',
        'mha_ts_w200':'K5: 일반 TS','mha_dts_nollm_w200':'K5: DTS, LLM 없음',
        'mha_wall_w200':'K5: 벽 안내 하나',TIGHT:'K5: DTS, 5% 제한'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;out=root/'analysis';out.mkdir(exist_ok=True)
    if 'Malgun Gothic' in {f.name for f in font_manager.fontManager.ttflist}:plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries={g:read(root/g/'summary.json') for g in ('pilot','public','transfer')}
    seeds={g:read(root/g/'per_seed.json') for g in ('public','transfer')}
    cases={g:read(root/f'{g}_cases.json') for g in ('public','transfer')}
    text=['# K5 자동 수치표','','문제별 5시드 산술평균 → 대조군 대비 비율 → 문제 간 기하평균. 시간은 시드별 3회 중앙값을 먼저 취한다.',
          '전체 pop에는 bandit 보상 heap도 포함한다. 이웃 확장은 목표를 제외한다. 시간은 LLM 생성 지연을 제외한 계획기 시간이다.','']
    for group,s in summaries.items():
        text += [f'## {group}','','g+2h=100, 낮을수록 작은 값:','',
                 '| 방법 | 전체 pop | 이웃 확장 | 길이 | 시간 | 최단+5% 초과/시드 시행 | 최악/최단 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            d=s['comparisons'][BASE][m];a=s['absolute'][m]
            ratios=' | '.join(f'{100*d[k]["geometric_ratio"]:.3f}%' for k in ('operation','successor_expansions','length','total_seconds'))
            text.append(f'| {LABELS[m]} | {ratios} | {a["over_5_percent"]}/{a["seed_trials"]} | {100*a["max_optimal_length_ratio"]:.3f}% |')
        text += ['','절대값 산술평균:','','| 방법 | pop | 이웃 확장 | 길이 | 계획기 ms | 최적 경로/시행 | 성공/시행 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            a=s['absolute'][m];v=a['means']
            text.append(f'| {LABELS[m]} | {v["operation"]:.3f} | {v["successor_expansions"]:.3f} | {v["length"]:.5f} | {1000*v["total_seconds"]:.4f} | {a["optimal_paths"]}/{a["seed_trials"]} | {a["valid"]}/{a["seed_trials"]} |')
        contrasts=[(PRIMARY,c) for c in (BASE,K3,'g2h_wall_euclidean','mha_rr_w200','mha_ts_w200','mha_dts_nollm_w200','mha_wall_w200','focal_llm_w200')]
        contrasts += [(TIGHT,K4),(TIGHT,'focal_goal_w105')]
        for m,c in contrasts:
            text += ['',f'### {LABELS[m]} / {LABELS[c]}','','| 지표 | 비율 | 지도 bootstrap 95% 구간 | 개선 / 동일 / 악화 문제 |',
                     '| --- | ---: | ---: | ---: |']
            for k in METRICS:
                d=s['comparisons'][c][m][k]
                text.append(f'| {k} | {100*d["geometric_ratio"]:.3f}% | {100*d["ci95"][0]:.3f}–{100*d["ci95"][1]:.3f}% | {d["better"]} / {d["equal"]} / {d["worse"]} |')
        text += ['','### 시드별 집계 범위','','각 seed_index로 전 문제를 한 번씩 풀었을 때의 g+2h 대비 비율 범위. 확률적 방법만 표기한다.','',
                 '| 방법 | pop 범위 | 길이 범위 | 시간 범위 |','| --- | ---: | ---: | ---: |']
        for m in STOCHASTIC:
            d=s['seed_spread'][m]
            cells=' | '.join(f'{100*d[k]["min"]:.3f}–{100*d[k]["max"]:.3f}%' for k in ('operation','length','total_seconds'))
            text.append(f'| {LABELS[m]} | {cells} |')
        text.append('')
    (out/'TABLES.md').write_text('\n'.join(text),encoding='utf-8')

    display=(BASE,K3,'g2h_wall_euclidean',K4,PRIMARY,TIGHT,'mha_dts_nollm_w200')
    colors=('#737b86','#277eb5','#329068','#9b77b8','#de7622','#c64449','#9d8b32')
    fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
    for ri,group in enumerate(('public','transfer')):
        for ci,(metric,title) in enumerate((('operation','전체 heap pop'),('length','경로 길이'),('total_seconds','계획기 시간'))):
            ax=axes[ri,ci];bounds=[]
            for i,(m,color) in enumerate(zip(display,colors)):
                d=summaries[group]['comparisons'][BASE][m][metric]
                v=100*d['geometric_ratio'];lo,hi=[100*x for x in d['ci95']];bounds.extend([lo,hi])
                ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,capsize=3)
                ax.annotate(f'{v:.2f}%',(v,i),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=9,color=color)
            ax.set_yticks(range(len(display)),[LABELS[m] for m in display] if ci==0 else [])
            ax.set_ylim(len(display)-.3,-.5);ax.grid(axis='x',alpha=.2);ax.axvline(100,color='#999',ls='--')
            span=max(bounds)-min(bounds);ax.set_xlim(min(bounds)-max(.25,span*.12),max(bounds)+max(.25,span*.12))
            ax.set_xlabel('g+2h 대비 (%)');ax.set_title(f'{"공개 1,000문제" if group=="public" else "별도 지도 30문제"} · {title}')
    fig.suptitle('K5와 기존 방법 비교\n5시드 평균 · 지도 bootstrap 95% 구간 · K4 주 설정은 5%, K5 주 설정은 2배 비용 상한',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,group in zip(axes,('public','transfer')):
        for m,color in zip(display,colors):
            excess=sorted(100*(r['length']/r['optimum']-1) for r in seeds[group] if r['method']==m)
            ax.plot(np.linspace(0,100,len(excess)),excess,label=LABELS[m],color=color)
        ax.axhline(5,color='black',ls='--',lw=1);ax.set_xlabel('문제·시드 시행 백분위');ax.set_ylabel('최단거리보다 긴 비율 (%)')
        ax.set_title('공개 1,000문제' if group=='public' else '별도 지도 30문제');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left');fig.suptitle('평균과 별개로 보는 경로 길이의 나쁜 사례')
    fig.savefig(out/'quality_tail.png',dpi=160);plt.close(fig)

    diagnostics={}
    for group in ('public','transfer'):
        sub=seeds[group];d={}
        for m in METHODS:
            rows=[r for r in sub if r['method']==m]
            entry={}
            if m.startswith('mha'):
                for field in ('queue_expansions','bandit_choices','bandit_rewards'):
                    entry[field]=np.mean([r[field] for r in rows],axis=0).tolist()
                expanded=sum(sum(r['queue_expansions']) for r in rows)
                entry['anchor_expansion_fraction']=sum(r['queue_expansions'][0] for r in rows)/expanded
                if m not in ('mha_dts_nollm_w200','mha_wall_w200'):
                    entry['llm_expansion_fraction']=sum(r['queue_expansions'][-1] for r in rows)/expanded
            d[m]=entry
        parsed={r['case_id'] for r in sub if r['method']==PRIMARY and r['guide_error'] is None}
        selected={}
        for m in (PRIMARY,'mha_dts_nollm_w200','mha_rr_w200'):
            entries={}
            for r in sub:
                if r['method']==m and r['case_id'] in parsed:entries.setdefault(r['case_id'],[]).append(r)
            selected[m]={cid:{k:statistics.mean(r[k] for r in rows) for k in METRICS} for cid,rows in entries.items()}
        d['parsed_only_posthoc']={c:{k:math.exp(statistics.mean(math.log(selected[PRIMARY][cid][k]/selected[c][cid][k]) for cid in parsed))
                                   for k in METRICS} for c in ('mha_dts_nollm_w200','mha_rr_w200')}
        d['parsed_only_posthoc']['cases']=len(parsed)
        diagnostics[group]=d
    responses={r['case_id']:r for r in read_rows(root/'responses.jsonl.gz')}
    diagnostics['api']={g:dict(cases=len(cases[g]),mean_historical_seconds=statistics.mean(responses[c['case_id']]['historical_api_seconds'] for c in cases[g]),
        tokens=sum(responses[c['case_id']]['usage']['total_tokens'] for c in cases[g])) for g in ('public','transfer')}

    # Choose problem examples by the five-seed mean, but always show seed 0.
    averaged={}
    for r in seeds['public']:
        averaged.setdefault((r['case_id'],r['method']),[]).append(r)
    ratios={cid:statistics.mean(r['operation'] for r in averaged[cid,PRIMARY])/averaged[cid,K3][0]['operation']
            for cid,m in averaged if m==PRIMARY}
    best,worst=min(ratios,key=ratios.get),max(ratios,key=ratios.get)
    quality=max((r for r in seeds['public'] if r['method']==PRIMARY and r['seed_index']==0),key=lambda r:r['length']/r['optimum'])['case_id']
    examples=list(dict.fromkeys([best,worst,quality]));plot_methods=(BASE,K3,K4,PRIMARY)
    raw={}
    for r in read_rows(root/'public/evaluations.jsonl.gz'):
        if r['case_id'] in examples and r['method'] in plot_methods and r['repeat']==0 and r['seed_index'] in (-1,0):
            raw[r['case_id'],r['method']]=r
    queries={c['case_id']:c['query'] for c in cases['public']}
    fig,axes=plt.subplots(len(examples),len(plot_methods),figsize=(16,3.9*len(examples)),squeeze=False,layout='constrained')
    details=[]
    for ri,cid in enumerate(examples):
        q=queries[cid];detail=dict(case_id=cid,mean_pop_ratio_to_K3=ratios[cid],seed_index=0,methods={})
        for ci,m in enumerate(plot_methods):
            saved=raw[cid,m]
            replay=run_once(q,m,responses[cid]['text'],seed=case_seed(cid,0) if m in STOCHASTIC else 0,trace=True)
            assert json.dumps(replay['path'])==json.dumps(saved['path']) and replay['operation']==saved['operation']
            detail['methods'][m]={k:saved[k] for k in ('operation','successor_expansions','length','optimum')}
            ax=axes[ri,ci]
            nodes=[e['node'] for e in replay['trace'] if 'node' in e]
            if nodes:ax.scatter(*zip(*nodes),s=5,color='#8d99ae',alpha=.16)
            for y,a,b in q['horizontal_barriers']:ax.plot([a,b],[y,y],color='#30343b',lw=2)
            for x,a,b in q['vertical_barriers']:ax.plot([x,x],[a,b],color='#30343b',lw=2)
            ax.plot(*zip(*saved['path']),color=colors[0 if ci==0 else (1 if ci==1 else (3 if ci==2 else 4))],lw=2)
            ax.scatter(*q['start'],marker='o',color='#16865a',s=35,zorder=5);ax.scatter(*q['goal'],marker='*',color='#c0393c',s=70,zorder=5)
            ax.set_xlim(q['range_x'][0],q['range_x'][1]-1);ax.set_ylim(q['range_y'][0],q['range_y'][1]-1);ax.set_aspect('equal')
            ax.set_title(f'{cid} · {LABELS[m]}\npop {saved["operation"]}, 이웃 확장 {saved["successor_expansions"]}\n길이 {saved["length"]:.2f} / 최단 {saved["optimum"]:.2f}',fontsize=9)
        details.append(detail)
    diagnostics['examples']=details
    fig.suptitle('사례: K3 대비 평균 pop 비율 최선·최악, seed 0의 길이 비율 최악\nK5 그림은 사전 고정 seed_index=0 · 회색 점은 탐색한 위치',fontsize=13)
    fig.savefig(out/'case_paths.png',dpi=160);plt.close(fig)
    write_json(out/'diagnostics.json',diagnostics)
    inputs=[root/g/f for g in ('public','transfer','pilot') for f in ('summary.json','evaluations.jsonl.gz')]
    inputs += [root/'responses.jsonl.gz',Path(__file__)]
    write_json(out/'manifest.json',dict(status='completed',input_sha256={str(p):sha(p) for p in inputs},
                                       illustrative_replays=len(examples)*len(plot_methods),performance_reruns=0))
    print(json.dumps(dict(status='completed',examples=examples),ensure_ascii=True))


if __name__=='__main__':main()
