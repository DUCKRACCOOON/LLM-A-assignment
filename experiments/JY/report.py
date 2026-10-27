"""Map-cluster summaries and figures. Run after the offline benchmark."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from experiments.K3.run import write_json,sha
from .run import read_rows,ROOT,SEED

METRICS=('operation','cold_operation','expansions','cold_expansions','storage','length','query_seconds','total_seconds')
PRIMARIES=('j3_weight_prune','j3_llm_prune','j4_alt4','j4_alt8','j5_bidijkstra','j5_nbs_euclidean','j5_nbs_octile','j6_lazy')


def paired(rows,reference,metric):
    common=sorted(set(rows)&set(reference));logs=[];by_map=defaultdict(list);ratios=[];better=same=worse=0
    for cid in common:
        a,b=rows[cid][metric],reference[cid][metric]
        # Zero-work searches are genuine zeros; floor only when taking logs.
        ratio=max(a,1)/max(b,1) if metric not in ('total_seconds','query_seconds','length') else a/b
        val=math.log(ratio);logs.append(val);by_map[rows[cid]['map_index']].append(val);ratios.append(ratio)
        if math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9 if 'seconds' not in metric else 0):same+=1
        elif a<b:better+=1
        else:worse+=1
    means=np.array([statistics.mean(v) for _,v in sorted(by_map.items())]);rng=np.random.default_rng(SEED)
    boots=np.exp(means[rng.integers(0,len(means),size=(2000,len(means)))].mean(axis=1))
    return dict(n=len(common),maps=len(means),ratio=math.exp(statistics.mean(logs)),
        ci95=list(map(float,np.quantile(boots,[.025,.975]))),map_equal_ratio=float(np.exp(means.mean())),
        improved=better,same=same,worsened=worse,p95_ratio=float(np.quantile(ratios,.95)),max_ratio=max(ratios),
        total_numerator=sum(rows[c][metric] for c in common),total_denominator=sum(reference[c][metric] for c in common))


def summarize(folder):
    buckets=defaultdict(list)
    for r in read_rows(folder/'evaluations.jsonl.gz'):buckets[r['case_id'],r['method']].append(r)
    methods=defaultdict(dict);csv_rows=[]
    for (cid,m),trials in sorted(buckets.items()):
        assert len(trials)==3
        row={k:v for k,v in trials[0].items() if k not in ('path','order','repeat') and not isinstance(v,(list,dict))}
        for k in ('query_seconds','total_seconds','preprocessing_seconds','seed_seconds'):
            if k in row:row[k]=statistics.median(t[k] for t in trials)
        row['repeat_seconds']=[r['total_seconds'] for r in trials]
        row['repeat_query_seconds']=[r['query_seconds'] for r in trials]
        methods[m][cid]=row;csv_rows.append(row)
    fields=sorted(set().union(*(r.keys() for r in csv_rows)))
    with (folder/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csv_rows)
    summaries={}
    for m,rows in methods.items():
        vals=list(rows.values());ratio=[r['length']/r['optimum'] for r in vals]
        summaries[m]=dict(n=len(vals),valid=sum(r['valid'] for r in vals),
            optimal=sum(abs(r['length']-r['optimum'])<1e-8 for r in vals),
            length_over_optimum=math.exp(statistics.mean(map(math.log,ratio))),
            max_length_over_optimum=max(ratio),p95_length_over_optimum=float(np.quantile(ratio,.95)),
            means={k:statistics.mean(r[k] for r in vals) for k in METRICS},
            totals={k:sum(r[k] for r in vals) for k in METRICS})
    comparisons={}
    for control in ('astar_octile','astar_wall','legacy_astar','legacy_g2h','k3_wall','j6_full','j3_weight_noprune','j3_llm_noprune'):
        if control not in methods:continue
        comparisons[control]={m:{k:paired(rows,methods[control],k) for k in METRICS} for m,rows in methods.items()}
    # Actual map batch: measured preparation exactly ONCE, then fixed first Q
    # queries. A map's all query times are summed within each repeat first.
    pre={(r['map_index'],r['repeat'],r['landmark_count']):r for r in read_rows(folder/'preprocessing.jsonl.gz')}
    reuse=[]
    baseline=methods['astar_octile'];maps=defaultdict(list)
    for cid,r in baseline.items():maps[r['map_index']].append(cid)
    for mi,cids in sorted(maps.items()):
        cids.sort();qs=sorted({1,min(5,len(cids)),len(cids)})
        for k in (4,8):
            m=f'j4_alt{k}'
            for q in qs:
                selected=cids[:q];costs=[];bases=[]
                for repeat in range(3):
                    costs.append(pre[mi,repeat,k]['seconds']+sum(methods[m][c]['repeat_query_seconds'][repeat] for c in selected))
                    bases.append(sum(baseline[c]['repeat_query_seconds'][repeat] for c in selected))
                reuse.append(dict(map_index=mi,landmarks=k,queries=q,total_seconds=statistics.median(costs),
                    baseline_seconds=statistics.median(bases),ratio=statistics.median(costs)/statistics.median(bases),
                    repeats=costs,baseline_repeats=bases))
    write_json(folder/'reuse.json',reuse)
    summary=dict(methods=summaries,comparisons=comparisons,zero_count_policy='max(counter,1) for log ratios only',
                 bootstrap=dict(unit='map',replicates=2000,seed=SEED,interval='exploratory 95%, no multiplicity correction'))
    write_json(folder/'summary.json',summary)
    return methods,summary


def table(summary,control='astar_octile',metric='total_seconds'):
    lines=['| 방법 | N | 최단 일치 | pop % | 확장 % | 길이/최적 % | 단일질의 시간 % [95% CI] |',
           '|---|---:|---:|---:|---:|---:|---|']
    for m,r in summary['methods'].items():
        comp=summary['comparisons'][control][m];t=comp[metric]
        lines.append(f'| {m} | {r["n"]} | {r["optimal"]} | {100*comp["operation"]["ratio"]:.2f} | '
          f'{100*comp["expansions"]["ratio"]:.2f} | {100*r["length_over_optimum"]:.3f} | '
          f'{100*t["ratio"]:.2f} [{100*t["ci95"][0]:.2f}, {100*t["ci95"][1]:.2f}] |')
    return '\n'.join(lines)


def figures(out,groups):
    # Optional plotting dependency installed outside tracked files.
    local=ROOT/'local_private/plot_deps'
    if local.exists():sys.path.insert(0,str(local))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, FormatStrFormatter, NullFormatter
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    m,s=groups['public'];names=list(PRIMARIES)
    fig,axes=plt.subplots(1,3,figsize=(17,6),constrained_layout=True)
    for ax,metric,label in zip(axes,('expansions','total_seconds','cold_operation'),('실제 이웃 확장','전처리 포함 시간','전처리 포함 전체 pop')):
        for i,n in enumerate(names):
            r=s['comparisons']['j6_full' if n=='j6_lazy' else 'astar_octile'][n][metric]
            val=r['ratio'];lo,hi=r['ci95']
            ax.errorbar(val,i,xerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',capsize=3,
                        color='#3478a8' if val<1 else '#be6553')
        ax.set_yticks(range(len(names)),names);ax.axvline(1,color='black',lw=1)
        ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'));ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('비교군 대비 비율 (로그 축, 95% 구간)');ax.set_title(label)
    fig.suptitle('공개 1,000문제 · J3~J5: clean octile A* 대비 / J6: full 갱신 대비')
    fig.savefig(out/'analysis/comparison.png',dpi=160);plt.close(fig)
    # Representative best/worst total-time ratios are chosen automatically.
    examples=[];cases={c['case_id']:c for c in json.loads((out/'public_cases.json').read_text(encoding='utf-8'))}
    need={}
    for method in ('j3_weight_prune','j4_alt4','j5_nbs_octile','j6_lazy'):
        control='j6_full' if method=='j6_lazy' else 'astar_octile'
        ordered=sorted(m[method],key=lambda cid:m[method][cid]['total_seconds']/m[control][cid]['total_seconds'])
        for side,cid in [('best',ordered[0]),('worst',ordered[-1])]:
            need[cid,method]=None;need[cid,control]=None
            examples.append(dict(method=method,control=control,side=side,case_id=cid,
                time_ratio=m[method][cid]['total_seconds']/m[control][cid]['total_seconds']))
    for r in read_rows(out/'public/evaluations.jsonl.gz'):
        if r['repeat']==0 and (r['case_id'],r['method']) in need:need[r['case_id'],r['method']]=r
    fig,axes=plt.subplots(4,2,figsize=(13,14),constrained_layout=True)
    for ax,ex in zip(axes.flat,examples):
        q=cases[ex['case_id']]['query']
        for y,a,b in q['horizontal_barriers']:ax.plot([a,b],[y,y],color='black',lw=2)
        for x,a,b in q['vertical_barriers']:ax.plot([x,x],[a,b],color='black',lw=2)
        for method,color in [(ex['control'],'#888888'),(ex['method'],'#d35400')]:
            path=need[ex['case_id'],method]['path'];ax.plot(*zip(*path),label=method,color=color,lw=1.5)
        ax.scatter(*q['start'],marker='o',color='green');ax.scatter(*q['goal'],marker='*',color='red')
        ax.set_xlim(q['range_x']);ax.set_ylim(q['range_y']);ax.set_aspect('equal')
        ax.set_title(f'{ex["method"]} {ex["side"]}: {ex["case_id"]}\n전체 시간 {ex["time_ratio"]:.2f}배');ax.legend(fontsize=7)
    fig.savefig(out/'analysis/case_paths.png',dpi=140);plt.close(fig)
    write_json(out/'analysis/case_examples.json',examples)
    return dict(matplotlib=matplotlib.__version__,font='Malgun Gothic',backend='Agg')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert json.loads((out/'manifest.json').read_text())['status']=='completed'
    (out/'analysis').mkdir(exist_ok=True);groups={};lines=['# J3~J6 대응 비교표','',
        '비율은 문제별 방법/clean octile A*의 기하평균. 100 미만이면 감소. 시간은 문제별 3회 중앙값.',
        'ALT pop/확장 열은 query만; cold 전체 비용은 summary의 cold_operation/cold_expansions와 별도 표 참조.',
        'NBS pop은 waiting→ready 이동까지 포함. J3는 초기 해+정확성 확인의 총합. 새 API 호출 0회.','']
    for group in ('public','transfer','synthetic'):
        groups[group]=summarize(out/group);lines.extend([f'## {group}','',table(groups[group][1]),''])
    for group,(_,summary) in groups.items():
        lines.extend([f'## {group}: J6 및 J3 구성요소 비교','',
            '| 비교 | pop 비율 | 확장 비율 | 전체 시간 비율 [95% CI] | 시간 개선/동일/악화 |',
            '|---|---:|---:|---|---|'])
        pairs=[('j3_weight_prune','j3_weight_noprune'),('j4_alt4','astar_octile'),('j5_nbs_octile','astar_octile')]
        if 'j6_lazy' in summary['methods']:pairs += [('j3_llm_prune','j3_llm_noprune'),('j6_lazy','j6_full')]
        for m,b in pairs:
            c=summary['comparisons'][b][m];t=c['total_seconds']
            lines.append(f'| {m}/{b} | {c["cold_operation"]["ratio"]:.4f} | {c["cold_expansions"]["ratio"]:.4f} | '
                f'{t["ratio"]:.4f} [{t["ci95"][0]:.4f}, {t["ci95"][1]:.4f}] | {t["improved"]}/{t["same"]}/{t["worsened"]} |')
    (out/'analysis/TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plotting=figures(out,groups)
    write_json(out/'analysis/manifest.json',dict(report_code_sha256=sha(Path(__file__)),numpy=np.__version__,plotting=plotting,new_api_calls=0))
    print('Summaries, tables, reuse analysis and figures completed.')


if __name__=='__main__':main()
