"""Tables, paired contrasts, guidance diagnostics, and illustrative replays."""
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

from experiments.K3.run import write_json,sha
from experiments.K5.run import read_rows
from .search import METHODS,TIGHT,PRIMARY,run_once
from .run import METRICS

BASE='g2h_euclidean';K3='g2h_wall_euclidean';NO='no_guide_wall_w105';HALF='fixed_half_wall_w105';FULL='fixed_full_wall_w105'
LABELS={BASE:'기존 g+2h','g2h_wall_octile':'K3 octile＋벽',K3:'K3 직선거리＋벽',PRIMARY:'K6 적응 α',
        NO:'K6 LLM 제거',HALF:'K6 고정 α=0.5',FULL:'K6 고정 α=1'}
COLORS={BASE:'#757d88','g2h_wall_octile':'#4e98c4',K3:'#177847',NO:'#2563ad',HALF:'#9c66b9',FULL:'#af872b',PRIMARY:'#db582d'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def gm(values):return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;out=root/'analysis';out.mkdir(exist_ok=True)
    if 'Malgun Gothic' in {f.name for f in font_manager.fontManager.ttflist}:plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries={g:read(root/g/'summary.json') for g in ('pilot','public','transfer')}
    rows={g:list(read_rows(root/g/'per_case.jsonl.gz')) for g in ('public','transfer')}
    by={g:{m:{r['case_id']:r for r in rows[g] if r['method']==m} for m in METHODS} for g in rows}
    text=['# K6 5% 제한 대조 실험: 자동 비교표','','문제별 비율 기하평균. 시간은 같은 문제·조건의 5회 중앙값. K6 네 조건은 모두 w=1.05, K3는 w=2.',
          'pop은 OPEN 전체 재계산을 포함하지 않는다. 시간에는 포함하며 LLM 생성은 제외한다.','']
    for group,s in summaries.items():
        text += [f'## {group}','','기준 g+2h=100.','','| 방법 | pop | 이웃 확장 | 저장량 | 길이 | 시간 | 최적/전체 | 최단+5% 초과 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            a=s['absolute'][m];d=s['comparisons'][BASE][m]
            vals=' | '.join(f'{100*d[k]["geometric_ratio"]:.3f}%' for k in METRICS)
            text.append(f'| {LABELS[m]} | {vals} | {a["optimal_paths"]}/{a["cases"]} | {a["over_5_percent"]}/{a["cases"]} |')
        text += ['','아래는 절대 산술평균이다.','',
                 '| 방법 | pop | 이웃 확장 | 길이 | ms | 최악 길이/최단 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for m in METHODS:
            a=s['absolute'][m];v=a['means']
            text.append(f'| {LABELS[m]} | {v["operation"]:.3f} | {v["successor_expansions"]:.3f} | {v["length"]:.5f} | {1000*v["total_seconds"]:.4f} | {100*a["max_optimal_length_ratio"]:.4f}% |')
        contrasts=[(PRIMARY,NO),(HALF,NO),(FULL,NO),(PRIMARY,HALF),(PRIMARY,FULL),(PRIMARY,K3),(NO,K3)]
        for method,control in contrasts:
            text += ['',f'### {LABELS[method]} / {LABELS[control]}','','| 지표 | 비율 | 지도 bootstrap 95% 구간 | 개선/동일/악화 |','| --- | ---: | ---: | ---: |']
            for metric in METRICS:
                d=s['comparisons'][control][method][metric]
                text.append(f'| {metric} | {100*d["geometric_ratio"]:.3f}% | {100*d["ci95"][0]:.3f}–{100*d["ci95"][1]:.3f}% | {d["better"]}/{d["equal"]}/{d["worse"]} |')
        text += ['','### 별도 관리 비용, 시행당 평균','','| 조건 | heap 재구성 | OPEN 재계산 | 버린 heap 항목 | 안내 거리 계산 | α 변경 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for m in TIGHT:
            d=s['absolute'][m]['extra_means']
            vals=' | '.join(f'{d[k]:.3f}' for k in ('heap_rebuilds','rekey_scans','rekey_discarded','guide_distance_checks','alpha_updates'))
            text.append(f'| {LABELS[m]} | {vals} |')
        text.append('')
    (out/'TABLES.md').write_text('\n'.join(text),encoding='utf-8')

    for filename,control,methods,title in [('comparison.png',BASE,METHODS,'K3 및 기존 g+2h와 비교'),
                                         ('ablation.png',NO,(NO,PRIMARY,HALF,FULL),'같은 5% 제한·같은 엔진에서 LLM 안내 비교')]:
        fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
        for ri,group in enumerate(('public','transfer')):
            for ci,(metric,label) in enumerate((('operation','전체 heap pop'),('length','경로 길이'),('total_seconds','계획기 시간'))):
                ax=axes[ri,ci];bounds=[]
                for i,m in enumerate(methods):
                    d=summaries[group]['comparisons'][control][m][metric];v=100*d['geometric_ratio'];lo,hi=[100*x for x in d['ci95']];bounds.extend([lo,hi])
                    ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=COLORS[m],capsize=3)
                    value_label=f'{v:.3f}%' if metric=='length' and control==NO else f'{v:.2f}%'
                    ax.annotate(value_label,(v,i),xytext=(0,-15),textcoords='offset points',ha='center',fontsize=9,color=COLORS[m])
                ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods] if ci==0 else []);ax.set_ylim(len(methods)-.3,-.5)
                margin=max(.08,(max(bounds)-min(bounds))*.12);ax.set_xlim(min(bounds)-margin,max(bounds)+margin)
                ax.axvline(100,color='#999',ls='--');ax.grid(axis='x',alpha=.2);ax.set_xlabel(f'{LABELS[control]} 대비 (%)')
                ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                ax.set_title(f'{"공개 1,000문제" if group=="public" else "기존 별도 30문제"} · {label}')
        fig.suptitle(title+'\n지도 bootstrap 95% 구간 · pop에 재계산은 미포함, 시간에는 포함',fontsize=14)
        fig.savefig(out/filename,dpi=160);plt.close(fig)

    diagnostics={}
    for group,lookup in by.items():
        usable=[cid for cid,r in lookup[PRIMARY].items() if r['guide_kept_points']>0]
        empty=[cid for cid in lookup[PRIMARY] if cid not in usable]
        d={}
        for m in TIGHT:
            hist=Counter();rejected=Counter()
            for r in lookup[m].values():hist.update(r['alpha_histogram']);rejected.update(r['guide_rejected'])
            total=sum(hist.values())
            d[m]=dict(cases_with_intermediate=sum(r['guide_kept_points']>0 for r in lookup[m].values()),
                alpha_histogram=dict(hist),alpha_zero_fraction=hist['0.0']/total,
                weighted_mean_alpha=sum(float(k)*v for k,v in hist.items())/total,
                cases_with_disable=sum(r['guide_disables']>0 for r in lookup[m].values()),
                cases_with_rekey=sum(r['heap_rebuilds']>0 for r in lookup[m].values()),rejected_points=dict(rejected),
                mean_pop_plus_rekey=statistics.mean(r['operation']+r['rekey_scans'] for r in lookup[m].values()))
        subsets={}
        for m,c in ((PRIMARY,NO),(HALF,NO),(FULL,NO),(PRIMARY,HALF)):
            subsets[m+'/'+c]={k:gm(lookup[m][cid][k]/lookup[c][cid][k] for cid in usable) for k in METRICS}
        fallback_checks=0
        for cid in empty:
            for m in TIGHT:
                for k in ('operation','storage','length','successor_expansions'):
                    assert lookup[m][cid][k]==lookup[NO][cid][k]
                fallback_checks+=1
        d['usable_only_posthoc']=dict(cases=len(usable),contrasts=subsets)
        d['no_intermediate']=dict(cases=len(empty),metric_parity_checks=fallback_checks)
        diagnostics[group]=d

    replies={r['case_id']:r for r in read_rows(root/'responses.jsonl.gz')}
    cases={g:read(root/f'{g}_cases.json') for g in ('public','transfer')}
    diagnostics['api']={g:dict(mean_historical_seconds=statistics.mean(replies[c['case_id']]['historical_api_seconds'] for c in cases[g]),
        tokens=sum(replies[c['case_id']]['usage']['total_tokens'] for c in cases[g])) for g in cases}
    lookup=by['public'];usable=[cid for cid,r in lookup[PRIMARY].items() if r['guide_kept_points']>0]
    ids=list(dict.fromkeys([min(usable,key=lambda cid:lookup[PRIMARY][cid]['operation']/lookup[NO][cid]['operation']),
        max(usable,key=lambda cid:lookup[PRIMARY][cid]['operation']/lookup[NO][cid]['operation']),
        max(usable,key=lambda cid:lookup[PRIMARY][cid]['operation']/lookup[HALF][cid]['operation'])]))
    display=(K3,NO,HALF,FULL,PRIMARY);raw={}
    for r in read_rows(root/'public/evaluations.jsonl.gz'):
        if r['case_id'] in ids and r['repeat']==0 and r['method'] in display:raw[r['case_id'],r['method']]=r
    queries={c['case_id']:c['query'] for c in cases['public']};details=[];traces={}
    fig,axes=plt.subplots(len(ids),len(display),figsize=(19,3.9*len(ids)),squeeze=False,layout='constrained')
    for ri,cid in enumerate(ids):
        q=queries[cid];detail=dict(case_id=cid,methods={})
        for ci,m in enumerate(display):
            saved=raw[cid,m];r=run_once(q,m,replies[cid]['text'],trace=True)
            for key,value in r.items():
                if key not in ('total_seconds','seconds','trace'):assert json.dumps(value,sort_keys=True)==json.dumps(saved[key],sort_keys=True),(cid,m,key)
            detail['methods'][m]={k:saved.get(k) for k in ('operation','successor_expansions','length','optimum','heap_rebuilds','rekey_scans','reexpansions')}
            if m==PRIMARY:traces[cid]=r['trace']
            ax=axes[ri,ci];nodes=[e['node'] for e in r['trace'] if 'node' in e]
            if nodes:ax.scatter(*zip(*nodes),s=5,color='#8996a9',alpha=.18)
            for y,a,b in q['horizontal_barriers']:ax.plot([a,b],[y,y],color='#30343b',lw=2)
            for x,a,b in q['vertical_barriers']:ax.plot([x,x],[a,b],color='#30343b',lw=2)
            ax.plot(*zip(*saved['path']),color=COLORS[m],lw=2)
            ax.scatter(*q['start'],color='#16865a',s=35,zorder=5);ax.scatter(*q['goal'],marker='*',color='#c0393c',s=70,zorder=5)
            ax.set_xlim(q['range_x'][0],q['range_x'][1]-1);ax.set_ylim(q['range_y'][0],q['range_y'][1]-1);ax.set_aspect('equal')
            ax.set_title(f'{cid} · {LABELS[m]}\npop {saved["operation"]}, 확장 {saved["successor_expansions"]}\n길이 {saved["length"]:.2f} / 최단 {saved["optimum"]:.2f}',fontsize=9)
        details.append(detail)
    fig.suptitle('적응/LLM 제거 pop 최선·최악, 적응/고정 0.5 pop 최악\n유효 안내가 있는 문제의 사후 극단 사례 · K3만 w=2, 나머지는 w=1.05',fontsize=13)
    fig.savefig(out/'case_paths.png',dpi=150);plt.close(fig)
    diagnostics['examples']=details;write_json(out/'diagnostics.json',diagnostics);write_json(out/'trace_examples.json',traces)
    paths=[root/g/f for g in ('pilot','public','transfer') for f in ('summary.json','per_case.jsonl.gz','evaluations.jsonl.gz')]
    paths += [root/'responses.jsonl.gz',root/'public_cases.json',root/'transfer_cases.json',Path(__file__)]
    write_json(out/'manifest.json',dict(status='completed',input_sha256={str(p):sha(p) for p in paths},illustrative_replays=len(ids)*5,performance_reruns=0))
    print(json.dumps(dict(status='completed',examples=ids)))


if __name__=='__main__':main()
