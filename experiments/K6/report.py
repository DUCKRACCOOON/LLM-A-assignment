"""Generate K6 tables/figures and diagnostics from immutable measurements."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from experiments.K3.run import write_json,sha
from experiments.K5.run import read_rows,case_seed
from .run import METRICS
from .search import METHODS,NEW_METHODS,STOCHASTIC,PRIMARY,run_once

BASE='g2h_euclidean';K3='g2h_wall_octile';K4='focal_llm_w105';K5='mha_dts_w200';TIGHT='adaptive_wall_w105'
LABELS={BASE:'기존 g+2h',K3:'K3: octile＋벽','g2h_wall_euclidean':'K3: 직선거리＋벽',
        'clean_wastar_wall':'K4 단일 weighted A*',K4:'K4: LLM, 5% 제한','focal_goal_w105':'K4: LLM 없음, 5%',
        'focal_llm_w200':'K4: LLM, 2배 제한',K5:'K5: DTS, 2배 제한','mha_dts_nollm_w200':'K5: DTS, LLM 없음',
        'mha_dts_w105':'K5: DTS, 5% 제한',PRIMARY:'K6: 적응 α, 2배 제한','fixed_half_wall_w200':'K6: 고정 α=0.5',
        'fixed_full_wall_w200':'K6: 고정 α=1','adaptive_literal_wall':'K6: 초안식, 적응 α',
        'fixed_literal_wall':'K6: 초안식, 고정 α=1','no_guide_wall_w200':'K6: 안내 없음',
        'adaptive_euclidean_w200':'K6: 벽 하한 제거',TIGHT:'K6: 적응 α, 5% 제한',
        'adaptive_window16_wall_w200':'K6: 관측 구간 16회'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def gm(values):return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;out=root/'analysis';out.mkdir(exist_ok=True)
    if 'Malgun Gothic' in {f.name for f in font_manager.fontManager.ttflist}:plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries={g:read(root/g/'summary.json') for g in ('pilot','public','transfer')}
    seeds={g:list(read_rows(root/g/'per_seed.jsonl.gz')) for g in ('public','transfer')}
    cases={g:read(root/f'{g}_cases.json') for g in ('public','transfer')}
    text=['# K6 자동 비교표','','기준 g+2h=100. 문제별 비율의 기하평균. K5는 문제별 5시드 평균, 시간은 시드별 3회 중앙값을 먼저 사용했다.',
          '전체 pop은 heapify/전체 재계산 횟수를 포함하지 않는다. 이 비용은 별도 표와 계획기 시간에 포함한다. LLM 생성 지연은 제외했다.','']
    for group,s in summaries.items():
        text += [f'## {group}','','| 방법 | 전체 pop | 이웃 확장 | 경로 길이 | 시간 | 최단+5% 초과/시행 | 최악/최단 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            d=s['comparisons'][BASE][m];a=s['absolute'][m]
            vals=' | '.join(f'{100*d[k]["geometric_ratio"]:.3f}%' for k in ('operation','successor_expansions','length','total_seconds'))
            text.append(f'| {LABELS[m]} | {vals} | {a["over_5_percent"]}/{a["seed_trials"]} | {100*a["max_optimal_length_ratio"]:.3f}% |')
        text += ['','절대 산술평균:','','| 방법 | pop | 이웃 확장 | 길이 | ms | 최적/전체 시행 | 성공/전체 시행 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            a=s['absolute'][m];v=a['means']
            text.append(f'| {LABELS[m]} | {v["operation"]:.3f} | {v["successor_expansions"]:.3f} | {v["length"]:.5f} | {1000*v["total_seconds"]:.4f} | {a["optimal_paths"]}/{a["seed_trials"]} | {a["valid"]}/{a["seed_trials"]} |')
        contrasts=[(PRIMARY,c) for c in (BASE,K3,'g2h_wall_euclidean',K5,'no_guide_wall_w200','fixed_half_wall_w200','fixed_full_wall_w200','adaptive_literal_wall')]
        contrasts += [(TIGHT,K4),(TIGHT,'mha_dts_w105'),('adaptive_literal_wall','fixed_literal_wall')]
        for m,c in contrasts:
            text += ['',f'### {LABELS[m]} / {LABELS[c]}','','| 지표 | 비율 | 지도 bootstrap 95% 구간 | 개선/동일/악화 문제 |','| --- | ---: | ---: | ---: |']
            for k in METRICS:
                d=s['comparisons'][c][m][k]
                text.append(f'| {k} | {100*d["geometric_ratio"]:.3f}% | {100*d["ci95"][0]:.3f}–{100*d["ci95"][1]:.3f}% | {d["better"]}/{d["equal"]}/{d["worse"]} |')
        text += ['','### K6 재정렬과 제어 비용','','시행당 산술평균:','','| 조건 | 재구성 횟수 | 재계산 스캔 | 버린 heap 항목 | α 변경 | target 변경 | 안내 거리 계산 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in NEW_METHODS:
            d=s['absolute'][m]['extra_means']
            vals=' | '.join(f'{d[k]:.3f}' for k in ('heap_rebuilds','rekey_scans','rekey_discarded','alpha_updates','target_changes','guide_distance_checks'))
            text.append(f'| {LABELS[m]} | {vals} |')
        text.append('')
    (out/'TABLES.md').write_text('\n'.join(text),encoding='utf-8')
    display=(BASE,K3,'g2h_wall_euclidean',K4,K5,PRIMARY,TIGHT)
    colors=('#737b86','#277eb5','#329068','#9b77b8','#b18b2a','#dc612b','#c64464')
    fig,axes=plt.subplots(2,3,figsize=(16,11),layout='constrained')
    for ri,group in enumerate(('public','transfer')):
        for ci,(metric,title) in enumerate((('operation','전체 heap pop'),('length','경로 길이'),('total_seconds','계획기 시간'))):
            ax=axes[ri,ci];bounds=[]
            for i,(m,color) in enumerate(zip(display,colors)):
                d=summaries[group]['comparisons'][BASE][m][metric];v=100*d['geometric_ratio'];lo,hi=[100*x for x in d['ci95']];bounds.extend([lo,hi])
                ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,capsize=3)
                ax.annotate(f'{v:.2f}%',(v,i),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=9,color=color)
            ax.set_yticks(range(len(display)),[LABELS[m] for m in display] if ci==0 else []);ax.set_ylim(len(display)-.3,-.5)
            span=max(bounds)-min(bounds);ax.set_xlim(min(bounds)-max(.25,span*.12),max(bounds)+max(.25,span*.12))
            ax.axvline(100,color='#999',ls='--');ax.grid(axis='x',alpha=.2);ax.set_xlabel('g+2h 대비 (%)')
            ax.set_title(f'{"공개 1,000문제" if group=="public" else "별도 지도 30문제"} · {title}')
    fig.suptitle('K6 적응형 경유점 가중치 비교\n지도 bootstrap 95% 구간 · pop에 OPEN 전체 재계산은 미포함, 시간에는 포함',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,group in zip(axes,('public','transfer')):
        for m,color in zip(display,colors):
            excess=sorted(100*(r['length']/r['optimum']-1) for r in seeds[group] if r['method']==m)
            ax.plot(np.linspace(0,100,len(excess)),excess,label=LABELS[m],color=color)
        ax.axhline(5,color='black',ls='--',lw=1);ax.set_xlabel('문제·시드 시행 백분위');ax.set_ylabel('최단거리 초과 (%)')
        ax.set_title('공개 1,000문제' if group=='public' else '별도 30문제');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left');fig.savefig(out/'quality_tail.png',dpi=160);plt.close(fig)

    diagnostics={};lookup={}
    for group in ('public','transfer'):
        selected={m:{r['case_id']:r for r in seeds[group] if r['method']==m} for m in NEW_METHODS}
        lookup[group]=selected
        d={}
        for m,entries in selected.items():
            rows=list(entries.values());hist=Counter();rejected=Counter()
            for r in rows:
                hist.update(r['alpha_histogram'] or {});rejected.update(r['guide_rejected'] or {})
            total=sum(hist.values())
            d[m]=dict(cases_with_intermediate=sum(r['guide_kept_points']>0 for r in rows),
                      alpha_histogram=dict(hist),alpha_zero_fraction=hist['0.0']/total,
                      weighted_mean_alpha=sum(float(k)*v for k,v in hist.items())/total,
                      cases_with_disable=sum(r['guide_disables']>0 for r in rows),
                      cases_with_target_change=sum(r['target_changes']>0 for r in rows),
                      cases_with_rekey=sum(r['heap_rebuilds']>0 for r in rows),rejected_points=dict(rejected))
        usable=[cid for cid,r in selected[PRIMARY].items() if r['guide_kept_points']>0]
        d['usable_only_posthoc']={control:{k:gm(selected[PRIMARY][cid][k]/selected[control][cid][k] for cid in usable) for k in METRICS}
                                  for control in ('no_guide_wall_w200','fixed_half_wall_w200')}
        d['usable_only_posthoc']['cases']=len(usable)
        # Expose work hidden by the pop metric, without treating operations as equal cost.
        d['primary_work']=dict(mean_pop_plus_rekey_scans=statistics.mean(r['operation']+r['rekey_scans'] for r in selected[PRIMARY].values()),
                               note='Descriptive count only; pop and scan have different CPU costs.')
        diagnostics[group]=d
    replies={r['case_id']:r for r in read_rows(root/'responses.jsonl.gz')}
    diagnostics['api']={group:dict(cases=len(cases[group]),mean_historical_seconds=statistics.mean(replies[c['case_id']]['historical_api_seconds'] for c in cases[group]),
        tokens=sum(replies[c['case_id']]['usage']['total_tokens'] for c in cases[group])) for group in ('public','transfer')}

    primary=lookup['public'][PRIMARY]
    k3={r['case_id']:r for r in seeds['public'] if r['method']==K3}
    usable=[cid for cid,r in primary.items() if r['guide_kept_points']>0]
    best=min(usable,key=lambda cid:primary[cid]['operation']/k3[cid]['operation'])
    worst=max(usable,key=lambda cid:primary[cid]['operation']/k3[cid]['operation'])
    benefit=min(usable,key=lambda cid:primary[cid]['operation']/lookup['public']['fixed_half_wall_w200'][cid]['operation'])
    example_ids=list(dict.fromkeys([best,worst,benefit]));plot_methods=(BASE,K3,K4,K5,PRIMARY)
    raw={}
    for r in read_rows(root/'public/evaluations.jsonl.gz'):
        if r['case_id'] in example_ids and r['method'] in plot_methods and r['repeat']==0 and r['seed_index'] in (-1,0):raw[r['case_id'],r['method']]=r
    queries={c['case_id']:c['query'] for c in cases['public']};traces={};details=[]
    fig,axes=plt.subplots(len(example_ids),5,figsize=(19,3.9*len(example_ids)),squeeze=False,layout='constrained')
    for ri,cid in enumerate(example_ids):
        q=queries[cid];detail=dict(case_id=cid,methods={},primary_vs_fixed_half_pop=primary[cid]['operation']/lookup['public']['fixed_half_wall_w200'][cid]['operation'])
        for ci,m in enumerate(plot_methods):
            saved=raw[cid,m];r=run_once(q,m,replies[cid]['text'],seed=case_seed(cid,0) if m in STOCHASTIC else 0,trace=True)
            assert json.dumps(r['path'])==json.dumps(saved['path']) and r['operation']==saved['operation']
            detail['methods'][m]={k:saved[k] for k in ('operation','successor_expansions','length','optimum')}
            if m==PRIMARY:traces[cid]=r['trace'];detail['controller']={k:saved[k] for k in ('heap_rebuilds','rekey_scans','alpha_histogram','guide_disables','target_changes')}
            ax=axes[ri,ci];nodes=[e['node'] for e in r['trace'] if 'node' in e]
            if nodes:ax.scatter(*zip(*nodes),s=5,color='#8d99ae',alpha=.18)
            for y,a,b in q['horizontal_barriers']:ax.plot([a,b],[y,y],color='#30343b',lw=2)
            for x,a,b in q['vertical_barriers']:ax.plot([x,x],[a,b],color='#30343b',lw=2)
            ax.plot(*zip(*saved['path']),color=colors[{0:0,1:1,2:3,3:4,4:5}[ci]],lw=2)
            ax.scatter(*q['start'],color='#16865a',s=35,zorder=5);ax.scatter(*q['goal'],marker='*',color='#c0393c',s=70,zorder=5)
            ax.set_xlim(q['range_x'][0],q['range_x'][1]-1);ax.set_ylim(q['range_y'][0],q['range_y'][1]-1);ax.set_aspect('equal')
            ax.set_title(f'{cid} · {LABELS[m]}\npop {saved["operation"]}, 확장 {saved["successor_expansions"]}\n길이 {saved["length"]:.2f} / 최단 {saved["optimum"]:.2f}',fontsize=9)
        details.append(detail)
    fig.suptitle('안내가 남은 문제 중 K3 대비 pop 최선·최악 / 고정 α=0.5 대비 pop 최선\nK5만 사전 고정 seed_index=0 · 회색 점은 확장한 위치',fontsize=13)
    fig.savefig(out/'case_paths.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(example_ids),figsize=(5.2*len(example_ids),4.4),squeeze=False,layout='constrained')
    for ax,cid in zip(axes[0],example_ids):
        trace=traces[cid];ax.step(range(len(trace)),[r['alpha'] for r in trace],where='post',color='#dc612b',label='α')
        ax.set_ylim(-.05,1.05);ax.set_xlabel('유효 노드 선택 순서');ax.set_ylabel('경유점 영향력 α');ax.grid(alpha=.2)
        other=ax.twinx();other.step(range(len(trace)),[r['target_index'] for r in trace],where='post',color='#277eb5',alpha=.5,ls='--')
        last=max(1,max(r['target_index'] for r in trace));other.set_ylim(-.1,last+.1);other.set_yticks(range(last+1))
        other.set_ylabel('현재 target 번호',color='#277eb5');ax.set_title(cid)
    fig.suptitle('K6 제어 기록: 주황은 α, 파란 점선은 경유점 순서')
    fig.savefig(out/'alpha_trace.png',dpi=160);plt.close(fig)
    diagnostics['examples']=details;write_json(out/'diagnostics.json',diagnostics)
    write_json(out/'trace_examples.json',traces)
    paths=[root/g/f for g in ('pilot','public','transfer') for f in ('summary.json','per_seed.jsonl.gz','evaluations.jsonl.gz')]
    paths += [root/'responses.jsonl.gz',Path(__file__)]
    write_json(out/'manifest.json',dict(status='completed',input_sha256={str(p):sha(p) for p in paths},
                                      illustrative_replays=len(example_ids)*5,performance_reruns=0))
    print(json.dumps(dict(status='completed',examples=example_ids)))


if __name__=='__main__':main()
