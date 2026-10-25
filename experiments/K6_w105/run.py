"""Frozen seven-condition offline ablation with five timing repeats."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
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
from experiments.K5.run import read_rows
from .search import METHODS,BASELINES,TIGHT,PRIMARY,run_once,no_guide_reference
from .verify import verify

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/K6/results/run_20261025'
SEED=20261029
REPEATS=5
METRICS=('operation','successor_expansions','storage','length','total_seconds')
EXTRAS=('heap_pushes','stale_pops','reexpansions','heap_rebuilds','rekey_scans','rekey_discarded',
        'peak_heap_entries','guide_distance_checks','wall_checks','alpha_updates','target_changes','guide_disables')


def previous(group):
    return {(r['case_id'],r['method']):r for r in read_rows(OLD/group/'evaluations.jsonl.gz')
            if r['repeat']==0 and r['method'] in BASELINES+(PRIMARY,)}


def benchmark(cases,replies,output,old):
    output.mkdir();rng=random.Random(SEED);oracle=None;map_index=None;optima=[]
    checks=dict(cases=len(cases),runs=0,independent_paths=0,bound_checks=0,bound_violations=0,
                previous_parity=0,no_guide_reference_checks=0,no_guide_reference_runs=0,repeat_checks=0)
    with gzip.open(output/'evaluations.jsonl.gz','xt',encoding='utf-8') as stream:
        for position,case in enumerate(cases):
            q,cid=case['query'],case['case_id']
            if map_index!=case['map_index']:map_index=case['map_index'];oracle=Oracle(q)
            optimum=oracle.distances(q['goal'])[tuple(q['start'])]
            optima.append(dict(case_id=cid,optimum=optimum))
            ref=no_guide_reference(q);checks['no_guide_reference_runs']+=1;stable={}
            for repeat in range(REPEATS):
                jobs=list(METHODS);rng.shuffle(jobs)
                for order,method in enumerate(jobs):
                    r=run_once(q,method,replies[cid]['text'])
                    # Freeze all non-time fields, including controller/heap counters.
                    signature=json.dumps({k:v for k,v in r.items() if k not in ('total_seconds','seconds')},sort_keys=True,allow_nan=False)
                    uses_llm=method in TIGHT and method!='no_guide_wall_w105'
                    r.update(case_id=cid,map_index=map_index,method=method,repeat=repeat,order=order,optimum=optimum,
                             uses_llm_guidance=uses_llm,response_sha256=replies[cid]['response_text_sha256'] if uses_llm else None,
                             historical_api_seconds=replies[cid]['historical_api_seconds'] if uses_llm else 0)
                    stream.write(json.dumps(r,allow_nan=False)+'\n');checks['runs']+=1
                    assert r['valid'],(cid,method,r['error']);oracle.check_path(q,r);checks['independent_paths']+=1
                    assert r['length']+1e-8>=optimum
                    if method in TIGHT:
                        checks['bound_checks']+=1;passed=r['length']<=1.05*optimum+1e-8
                        checks['bound_violations']+=not passed;assert passed,(cid,method,'bound')
                    if method in stable:assert stable[method]==signature;checks['repeat_checks']+=1
                    stable[method]=signature
                    if method in BASELINES+(PRIMARY,):
                        prev=old[cid,method]
                        for key,value in json.loads(signature).items():
                            assert value==prev[key],(cid,method,key,'old mismatch')
                        checks['previous_parity']+=1
                    if method=='no_guide_wall_w105':
                        for key in ('path','operation','storage','goal_g'):
                            assert json.dumps(r[key])==json.dumps(ref[key]),(cid,key,'separate weighted engine')
                        checks['no_guide_reference_checks']+=1
            if (position+1)%100==0 or position+1==len(cases):
                stream.flush();print(f'{output.name}: {position+1}/{len(cases)} ({checks["runs"]} runs)',flush=True)
    write_json(output/'checks.json',checks);write_json(output/'optima.json',optima)


def summarize(output):
    buckets=defaultdict(list)
    for r in read_rows(output/'evaluations.jsonl.gz'):buckets[r['case_id'],r['method']].append(r)
    rows=[]
    for values in buckets.values():
        assert len(values)==REPEATS
        row={k:v for k,v in values[0].items() if k not in ('path','order','repeat')}
        row['total_seconds']=statistics.median(v['total_seconds'] for v in values)
        row['repeat_seconds']=[v['total_seconds'] for v in values];rows.append(row)
    with gzip.open(output/'per_case.jsonl.gz','xt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    fields=('case_id','map_index','method','optimum','valid')+METRICS+EXTRAS
    with (output/'per_case.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in rows)
    by={m:{r['case_id']:r for r in rows if r['method']==m} for m in METHODS}
    comparisons={}
    for control in METHODS:
        comparisons[control]={}
        for method in METHODS:
            pairs=[(r,by[control][cid]) for cid,r in by[method].items()]
            maps=sorted({r['map_index'] for r,b in pairs});indices=np.random.default_rng(SEED).integers(0,len(maps),(2000,len(maps)))
            stats={}
            for metric in METRICS:
                logs=[math.log(r[metric]/b[metric]) for r,b in pairs]
                groups=[[v for v,(r,b) in zip(logs,pairs) if r['map_index']==mi] for mi in maps]
                sums=np.array([sum(g) for g in groups]);counts=np.array([len(g) for g in groups])
                boot=np.exp(sums[indices].sum(axis=1)/counts[indices].sum(axis=1))
                eps=1e-8 if metric=='length' else 0;diff=[r[metric]-b[metric] for r,b in pairs]
                better=sum(d < -eps for d in diff);worse=sum(d > eps for d in diff)
                stats[metric]=dict(geometric_ratio=math.exp(statistics.mean(logs)),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                   better=better,equal=len(pairs)-better-worse,worse=worse,pairs=len(pairs))
            stats['joint']=dict(pop_lower_length_not_worse=sum(r['operation']<b['operation'] and r['length']<=b['length']+1e-8 for r,b in pairs),
                                time_lower_length_not_worse=sum(r['total_seconds']<b['total_seconds'] and r['length']<=b['length']+1e-8 for r,b in pairs))
            comparisons[control][method]=stats
    absolute={}
    for method in METHODS:
        values=list(by[method].values())
        absolute[method]=dict(cases=len(values),valid=sum(r['valid'] for r in values),
            means={k:statistics.mean(r[k] for r in values) for k in METRICS},
            optimal_paths=sum(abs(r['length']-r['optimum'])<1e-8 for r in values),
            over_5_percent=sum(r['length']>1.05*r['optimum']+1e-8 for r in values),
            max_optimal_length_ratio=max(r['length']/r['optimum'] for r in values),
            guide_errors=dict(Counter(r['guide_error'] for r in values if r.get('guide_error'))),
            extra_means={k:statistics.mean(r[k] for r in values) if all(r.get(k) is not None for r in values) else None for k in EXTRAS})
    write_json(output/'summary.json',dict(primary=PRIMARY,absolute=absolute,comparisons=comparisons,
        aggregation='Five timing repeats -> median per case; geometric mean of paired case ratios; 2000 map bootstraps'))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new directory')
    assert git('branch','--show-current')=='gm'
    args.output.mkdir(parents=True);started=time.perf_counter()
    inputs=[ROOT/'reproduction/core.py',*[ROOT/f'experiments/K3/{n}.py' for n in ('heuristic','verification','run')],
            *[ROOT/f'experiments/K4/{n}.py' for n in ('search','run')],*[ROOT/f'experiments/K5/{n}.py' for n in ('search','run')],
            *[ROOT/f'experiments/K6/{n}.py' for n in ('search','verify')],
            ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py',ROOT/'source_code/dataset/environment_50_30.json',
            ROOT/'results/main/generations.jsonl',ROOT/'results/generalization/generations.jsonl',ROOT/'results/generalization/cases.json',
            *[OLD/g/'evaluations.jsonl.gz' for g in ('public','transfer')],ROOT/'experiments/K6_w105/PROTOCOL.md',
            *[ROOT/f'experiments/K6_w105/{n}.py' for n in ('__init__','search','verify','run')]]
    manifest=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        git_branch=git('branch','--show-current'),git_base_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),
        methods=METHODS,primary=PRIMARY,repeats=REPEATS,order_bootstrap_seed=SEED,new_api_calls=0,
        model=MODEL,prompt_style='standard',python=sys.version,platform=platform.platform(),processor=platform.processor(),
        cpu_count=os.cpu_count(),numpy=np.__version__,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    write_json(args.output/'manifest.json',manifest);write_json(args.output/'verification.json',verify())
    print('480 full-scan reference configurations passed',flush=True)
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
