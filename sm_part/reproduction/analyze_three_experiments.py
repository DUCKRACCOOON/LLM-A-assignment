"""Local, post-freeze descriptive analysis of the completed server experiment."""
import json
import math
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/next_search_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def aggregate(rows, baseline='A_OCT'):
    groups = defaultdict(list)
    base = {r['id']: r for r in rows if r['method'] == baseline}
    for r in rows:
        groups[r['method']].append(r)
    result = []
    for method, rs in sorted(groups.items()):
        ok = [r for r in rs if r['status'] == 'ok']
        pair = [(r, base[r['id']]) for r in ok
                if r['id'] in base and base[r['id']]['status'] == 'ok']
        item = dict(method=method, n=len(rs), failures=len(rs)-len(ok))
        if not ok:
            result.append(item)
            continue
        lengths = [r['length']/r['optimal'] for r in ok if r.get('optimal')]
        item['exact_fraction'] = sum(abs(r['length']-r['optimal'])<=1e-8+1e-10*r['optimal'] for r in ok)/len(rs)
        item['max_length_ratio'] = max(lengths) if lengths else None
        item['median_total_ms'] = 1000*statistics.median(r['total_seconds'] for r in ok)
        item['component_medians_ms'] = {
            k: 1000*statistics.median(statistics.median(t.get(k, 0) for t in r['timings']) for r in ok)
            for k in ('preparation_seconds', 'inference_seconds', 'search_seconds',
                      'required_validation_seconds', 'reconstruction_seconds')}
        item['peak_sampled_rss_mib'] = max((r.get('sampled_peak_rss') or 0) for r in ok)/1024**2
        if pair:
            logs = defaultdict(list)
            for r, b in pair:
                logs[r['map_id']].append(math.log(r['total_seconds']/b['total_seconds']))
            item['map_paired_time_ratio'] = math.exp(statistics.mean(statistics.mean(v) for v in logs.values()))
            item['p95_case_time_ratio'] = float(np.quantile([r['total_seconds']/b['total_seconds'] for r,b in pair],.95))
            counters = sorted({k for r,b in pair for k in r['counters']} | {k for r,b in pair for k in b['counters']})
            item['counter_totals'] = {k: dict(candidate=sum(r['counters'].get(k,0) for r,b in pair),
                                                baseline=sum(b['counters'].get(k,0) for r,b in pair)) for k in counters}
            exlogs = defaultdict(list)
            for r,b in pair:
                exlogs[r['map_id']].append(math.log(max(r['counters']['expansions'],1e-12)/max(b['counters']['expansions'],1e-12)))
            item['map_paired_expansion_ratio'] = math.exp(statistics.mean(statistics.mean(v) for v in exlogs.values()))
            ordered = sorted(pair,key=lambda z:z[0]['total_seconds']/z[1]['total_seconds'])
            item['diagnostic_cases'] = [dict(id=r['id'],family=r['family'],time_ratio=r['total_seconds']/b['total_seconds'],
                expansions=r['counters']['expansions'],baseline_expansions=b['counters']['expansions'],
                rounds=r.get('rounds'),fallback=r.get('fallback'),counters=r['counters']) for r,b in (ordered[0],ordered[len(ordered)//2],ordered[-1])]
        if any('rounds' in r for r in ok):
            item['refinement'] = dict(mean_rounds=statistics.mean(len(r['rounds']) for r in ok),
                fallback_fraction=statistics.mean(bool(r['fallback']) for r in ok),
                repeat_fraction=statistics.mean(len(r['rounds'])>1 for r in ok))
        result.append(item)
    return result


def main():
    assert read(OUT/'local_verification.json')['verified']
    rows = [json.loads(line) for line in (OUT/'per_case.jsonl').open(encoding='utf-8')]
    final = [r for r in rows if r['stage']=='final']
    result = dict(main=aggregate(final), by_family={f:aggregate([r for r in final if r['family']==f])
                 for f in sorted({r['family'] for r in final})})
    result['validation'] = {stage:aggregate([r for r in rows if r['stage']==stage],baseline)
        for stage,baseline in [('r1_validation','A_OCT'),('r2_validation','A_OCT'),
            ('combination_validation','R1'),('r3_development','TOPO:rule:1.02:0'),('r3_validation','TOPO:rule:1.02:0')]}
    result['development_r1'] = aggregate([r for r in rows if r['stage']=='r1_development'])
    result['development_r2'] = aggregate([r for r in rows if r['stage'] in
        ('r2_development','r2_program_selected_dev','r2_incremental_development')])
    result['r3_against_standard'] = {
        w:aggregate([r for r in rows if r['stage']=='r3_final' and r['method'].split(':')[2]==w]
                    + [r for r in final if r['method']==base],base)
        for w,base in [('1.0','A_OCT'),('1.01','FOCAL_101'),('1.02','FOCAL_102')]}
    scale_lookup = {q['id']:q['scale'] for q in read(OUT/'data/scaling.json')}
    result['scaling'] = {str(s):aggregate([r for r in rows if r['stage']=='scaling' and scale_lookup[r['id']]==s])
                         for s in sorted(set(scale_lookup.values()))}
    from next_report import comparison
    result['scaling_intervals'] = {str(s):{m:comparison(
        [r for r in rows if r['stage']=='scaling' and scale_lookup[r['id']]==s],m,'A_OCT')
        for m in ('R1','R2','R1_R2')} for s in sorted(set(scale_lookup.values()))}
    result['scaling_8_by_family']={family:aggregate([r for r in rows if r['stage']=='scaling'
        and scale_lookup[r['id']]==8 and r['family']==family]) for family in sorted(result['by_family'])}
    rules={r['id']:r for r in rows if r['stage']=='r2_development' and r['method']=='R2:rule'}
    programs={r['id']:r for r in rows if r['stage']=='r2_program_selected_dev' and r['method']=='R2:program'}
    result['r2_program_vs_rule_behaviour']=dict(pairs=len(programs),
        identical_paths=sum(r['path']==rules[key]['path'] for key,r in programs.items()),
        identical_expansions=sum(r['counters']['expansions']==rules[key]['counters']['expansions'] for key,r in programs.items()),
        identical_rounds=sum(r['rounds']==rules[key]['rounds'] for key,r in programs.items()))
    result['reuse'] = []
    reuse = [read(p) for p in (OUT/'reuse').glob('*/*/*.json')]
    for workload in ('varying','fixed'):
        for n in (1,5,10):
            subset = [r for r in reuse if r['workload']==workload and r['queries']==n]
            base = {r['map_id']:r['total_seconds'] for r in subset if r['method']=='A_OCT'}
            for m in sorted({r['method'] for r in subset}):
                vals = [r for r in subset if r['method']==m]
                result['reuse'].append(dict(workload=workload, queries=n, method=m, maps=len(vals),
                    time_ratio=math.exp(statistics.mean(math.log(r['total_seconds']/base[r['map_id']]) for r in vals)),
                    median_ms_per_query=1000*statistics.median(r['mean_query_seconds'] for r in vals)))
    (OUT/'local_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {r['method']:r for r in read(OUT/'summary.json')['summary']}
    methods = ['R1','R2','R1_R2','N_CLEAN','WA_102','FOCAL_102']
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    colors = ['#2563eb','#ea580c','#7c3aed','#475569','#0d9488','#65a30d']
    for ax,key,title in zip(axes,['expansions','time'],['Node expansions','Total time including preparation']):
        values = [summary[m][key]['ratio'] for m in methods]
        lows = [v-summary[m][key]['low'] for m,v in zip(methods,values)]
        highs = [summary[m][key]['high']-v for m,v in zip(methods,values)]
        ax.barh(methods,values,color=colors,xerr=[lows,highs],capsize=3)
        ax.axvline(1,color='#111827',linestyle='--',linewidth=1)
        ax.set_title(title);ax.set_xlabel('Ratio to A* (lower is better)')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Held-out evaluation: paired map ratios and descriptive 95% intervals')
    fig.savefig(OUT/'comparison.png',dpi=200)
    plt.close(fig)
    print(json.dumps(dict(main_rows=len(final), total_rows=len(rows), reuse_batches=len(reuse))))


if __name__ == '__main__':
    main()
