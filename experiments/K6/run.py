"""Offline K6 evaluation. Raw records include all repeats and K5 seeds."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import gzip
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np

from experiments.K3.run import public_cases,write_json,sha,git
from experiments.K3.verification import Oracle
from experiments.K4.run import load_responses,MODEL
from experiments.K5.run import read_rows,case_seed
from .search import BASELINES,NEW_METHODS,METHODS,STOCHASTIC,PRIMARY,run_once
from .verify import verify

ROOT=Path(__file__).resolve().parents[2]
SEED=20261028
METRICS=('operation','successor_expansions','storage','length','total_seconds')
CONTROLS=('g2h_euclidean','g2h_wall_octile','g2h_wall_euclidean','focal_llm_w105',
          'focal_goal_w105','focal_llm_w200','mha_dts_w200','mha_dts_w105',
          'mha_dts_nollm_w200','no_guide_wall_w200','fixed_half_wall_w200',
          'fixed_full_wall_w200','adaptive_literal_wall','fixed_literal_wall','clean_wastar_wall')
EXTRAS=('heap_pushes','stale_pops','peak_heap_entries','reexpansions','heap_rebuilds','rekey_scans',
        'rekey_entries','rekey_discarded','guide_distance_checks','wall_checks','controller_distance_checks',
        'alpha_updates','target_changes','good_windows','bad_windows','guide_disables',
        'search_pops','reward_pops','beta_samples','guide_kept_points')


def previous(group):
    result={}
    path=ROOT/f'experiments/K5/results/run_20261025/{group}/evaluations.jsonl.gz'
    for row in read_rows(path):
        if row['repeat']==0 and row['method'] in BASELINES:
            result[row['case_id'],row['method'],row['seed_index']]={k:row[k] for k in ('path','operation','storage','length','valid','seed')}
    return result


def benchmark(cases,replies,output,old):
    output.mkdir();rng=random.Random(SEED)
    checks=dict(cases=len(cases),runs=0,independent_path_checks=0,baseline_parity_checks=0,no_guide_parity_checks=0,
                bound_checks=0,bound_violations=0,deterministic_repeat_checks=0)
    oracle,map_index,optima=None,None,[]
    with gzip.open(output/'evaluations.jsonl.gz','xt',encoding='utf-8') as stream:
        for position,case in enumerate(cases):
            q,cid=case['query'],case['case_id']
            if map_index!=case['map_index']:map_index=case['map_index'];oracle=Oracle(q)
            optimum=oracle.distances(q['goal'])[tuple(q['start'])]
            optima.append(dict(case_id=cid,optimum=optimum));stable={}
            for repeat in range(3):
                jobs=[(m,s) for m in METHODS for s in (range(5) if m in STOCHASTIC else [-1])];rng.shuffle(jobs)
                for order,(method,si) in enumerate(jobs):
                    seed=case_seed(cid,si) if si>=0 else 0
                    r=run_once(q,method,replies[cid]['text'],seed=seed)
                    llm=(method in NEW_METHODS and method!='no_guide_wall_w200') or method in ('focal_llm_w105','focal_llm_w200','mha_dts_w200','mha_dts_w105')
                    r.update(case_id=cid,map_index=map_index,method=method,seed_index=si,seed=seed,repeat=repeat,order=order,
                             optimum=optimum,uses_llm_guidance=llm,historical_api_seconds=replies[cid]['historical_api_seconds'] if llm else 0,
                             response_sha256=replies[cid]['response_text_sha256'] if llm else None)
                    stream.write(json.dumps(r,allow_nan=False)+'\n');checks['runs']+=1
                    assert r['valid'],(cid,method,si,r['error'])
                    oracle.check_path(q,r);checks['independent_path_checks']+=1
                    assert r['length']+1e-8>=optimum
                    if r.get('bound_factor') is not None:
                        checks['bound_checks']+=1
                        passed=r['length']<=r['bound_factor']*optimum+1e-8
                        checks['bound_violations']+=not passed
                        assert passed,(cid,method,si,'bound violation')
                    signature={k:r.get(k) for k in ('path','operation','storage','length','successor_expansions',
                        'heap_rebuilds','rekey_scans','alpha_histogram','final_alpha','queue_expansions','bandit_rewards','heap_pushes')}
                    if (method,si) in stable:
                        assert signature==stable[method,si],(cid,method,si,'repeat changed');checks['deterministic_repeat_checks']+=1
                    stable[method,si]=signature
                    if method in BASELINES or method=='no_guide_wall_w200':
                        prev=old[cid,'clean_wastar_wall' if method=='no_guide_wall_w200' else method,si]
                        for key in ('path','operation','storage','length','valid','seed'):
                            assert json.dumps(r[key])==json.dumps(prev[key]),(cid,method,key,'old parity')
                        checks['no_guide_parity_checks' if method=='no_guide_wall_w200' else 'baseline_parity_checks']+=1
            if (position+1)%25==0 or position+1==len(cases):
                stream.flush();print(f'{output.name}: {position+1}/{len(cases)} ({checks["runs"]} runs)',flush=True)
    write_json(output/'checks.json',checks);write_json(output/'optima.json',optima)


def summarize(output):
    buckets=defaultdict(list)
    fields=('case_id','map_index','method','seed_index','seed','optimum','valid','error','bound_factor',
            'guide_error','guide_rejected','guide_raw_points','guide_kept_points','alpha_histogram','final_alpha',
            'historical_api_seconds','uses_llm_guidance')+METRICS+EXTRAS
    for r in read_rows(output/'evaluations.jsonl.gz'):
        buckets[r['case_id'],r['method'],r['seed_index']].append({k:r.get(k) for k in fields})
    seeds=[]
    for key,rows in buckets.items():
        assert len(rows)==3
        r=dict(rows[0]);r['total_seconds']=statistics.median(x['total_seconds'] for x in rows)
        r['repeat_seconds']=[x['total_seconds'] for x in rows];seeds.append(r)
    del buckets
    with gzip.open(output/'per_seed.jsonl.gz','xt',encoding='utf-8') as stream:
        for r in seeds:stream.write(json.dumps(r,allow_nan=False)+'\n')
    grouped=defaultdict(list)
    for r in seeds:grouped[r['case_id'],r['method']].append(r)
    cases=[];by_method={m:{} for m in METHODS}
    for (cid,m),rows in grouped.items():
        assert len(rows)==(5 if m in STOCHASTIC else 1)
        r=dict(case_id=cid,method=m,map_index=rows[0]['map_index'],optimum=rows[0]['optimum'],seeds=len(rows),valid=sum(x['valid'] for x in rows))
        r.update({k:statistics.mean(x[k] for x in rows) for k in METRICS});cases.append(r);by_method[m][cid]=r
    with (output/'per_case.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    comparisons={}
    for control in CONTROLS:
        comparisons[control]={}
        for method in METHODS:
            pairs=[(r,by_method[control][cid]) for cid,r in by_method[method].items()]
            maps=sorted({r['map_index'] for r,b in pairs});idx=np.random.default_rng(SEED).integers(0,len(maps),(2000,len(maps)))
            out={}
            for metric in METRICS:
                logs=np.array([math.log(r[metric]/b[metric]) for r,b in pairs])
                grouped_logs=[[v for v,(r,b) in zip(logs,pairs) if r['map_index']==mi] for mi in maps]
                sums=np.array([sum(x) for x in grouped_logs]);counts=np.array([len(x) for x in grouped_logs])
                boot=np.exp(sums[idx].sum(axis=1)/counts[idx].sum(axis=1))
                eps=1e-8 if metric=='length' else 0
                changes=[r[metric]-b[metric] for r,b in pairs]
                better=sum(d < -eps for d in changes);worse=sum(d > eps for d in changes)
                out[metric]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                 pairs=len(pairs),better=better,equal=len(pairs)-better-worse,worse=worse)
            out['joint']=dict(operation_better_length_not_worse=sum(r['operation']<b['operation'] and r['length']<=b['length']+1e-8 for r,b in pairs))
            comparisons[control][method]=out
    absolute={};seed_spread={}
    for m in METHODS:
        rows=[r for r in seeds if r['method']==m]
        absolute[m]=dict(cases=len(by_method[m]),seed_trials=len(rows),valid=sum(r['valid'] for r in rows),
            means={k:statistics.mean(r[k] for r in rows) for k in METRICS},
            optimal_paths=sum(abs(r['length']-r['optimum'])<1e-8 for r in rows),
            over_5_percent=sum(r['length']>1.05*r['optimum']+1e-8 for r in rows),
            max_optimal_length_ratio=max(r['length']/r['optimum'] for r in rows),
            guide_errors=dict(Counter(r['guide_error'] for r in rows if r['guide_error'])),
            extra_means={k:statistics.mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None for k in EXTRAS})
        if m in STOCHASTIC:
            seed_spread[m]={}
            for k in METRICS:
                vals=[math.exp(statistics.mean(math.log(r[k]/by_method['g2h_euclidean'][r['case_id']][k]) for r in rows if r['seed_index']==si)) for si in range(5)]
                seed_spread[m][k]=dict(per_seed_ratio=vals,min=min(vals),max=max(vals))
    write_json(output/'summary.json',dict(primary=PRIMARY,absolute=absolute,comparisons=comparisons,seed_spread=seed_spread,
        aggregation='3 timing medians; K5 five-seed arithmetic means per case; geometric ratios across cases; map bootstrap'))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    assert git('branch','--show-current')=='gm'
    args.output.mkdir(parents=True);started=time.perf_counter()
    inputs=[ROOT/'reproduction/core.py',*[ROOT/f'experiments/K3/{n}.py' for n in ('heuristic','verification','run')],
            *[ROOT/f'experiments/K4/{n}.py' for n in ('search','run')],*[ROOT/f'experiments/K5/{n}.py' for n in ('search','run')],
            ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py',ROOT/'source_code/dataset/environment_50_30.json',
            ROOT/'results/main/generations.jsonl',ROOT/'results/generalization/generations.jsonl',ROOT/'results/generalization/cases.json',
            *[ROOT/f'experiments/K5/results/run_20261025/{g}/evaluations.jsonl.gz' for g in ('public','transfer')],
            ROOT/'experiments/K6/PROTOCOL.md',*[ROOT/f'experiments/K6/{n}.py' for n in ('__init__','search','verify','run')]]
    manifest=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        git_branch=git('branch','--show-current'),git_base_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),
        primary=PRIMARY,methods=METHODS,stochastic=STOCHASTIC,seeds=5,timing_repeats=3,order_bootstrap_seed=SEED,
        stochastic_seed_scheme='unchanged K5 SHA256 scheme',model=MODEL,prompt_style='standard',new_api_calls=0,
        python=sys.version,platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),numpy=np.__version__,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    write_json(args.output/'manifest.json',manifest);write_json(args.output/'verification.json',verify())
    print('1080 reference cases and special checks passed',flush=True)
    public=public_cases();transfer=json.loads((ROOT/'results/generalization/cases.json').read_text(encoding='utf-8'))
    replies={**load_responses(public,ROOT/'results/main/generations.jsonl'),**load_responses(transfer,ROOT/'results/generalization/generations.jsonl')}
    write_json(args.output/'public_cases.json',public);write_json(args.output/'transfer_cases.json',transfer)
    with gzip.open(args.output/'responses.jsonl.gz','xt',encoding='utf-8') as stream:
        for r in replies.values():stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest['fixed_data_sha256']={n:sha(args.output/n) for n in ('public_cases.json','transfer_cases.json','responses.jsonl.gz')}
    write_json(args.output/'manifest.json',manifest)
    old_public,old_transfer=previous('public'),previous('transfer')
    pilot=[c for c in public if c['map_index']%10==0 and c['pair_index']<3]
    for group,cases,old in [('pilot',pilot,old_public),('public',public,old_public),('transfer',transfer,old_transfer)]:
        benchmark(cases,replies,args.output/group,old);summarize(args.output/group)
    for p in inputs:assert sha(p)==manifest['input_sha256'][str(p.relative_to(ROOT))],p
    manifest.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'manifest.json',manifest);print(json.dumps(dict(status='completed',seconds=manifest['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
