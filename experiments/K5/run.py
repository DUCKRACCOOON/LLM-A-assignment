"""Frozen offline K5 benchmark. Never overwrite an existing output directory."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
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

from experiments.K3.run import public_cases, write_json, sha, git
from experiments.K3.verification import Oracle
from experiments.K4.run import load_responses, MODEL
from .search import BASELINES, METHODS, PRIMARY, STOCHASTIC, run_once
from .verify import verify

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261027
REPEATS = 3
SEEDS = 5
CONTROLS = ('g2h_euclidean','g2h_wall_octile','g2h_wall_euclidean','focal_llm_w105',
            'focal_llm_w200','focal_goal_w105','mha_rr_w200','mha_ts_w200',
            'mha_dts_nollm_w200','mha_wall_w200','clean_wastar_wall')
METRICS = ('operation','successor_expansions','storage','length','total_seconds')


def case_seed(cid,index):
    return int.from_bytes(hashlib.sha256(f'K5|{SEED}|{cid}|{index}'.encode()).digest()[:8],'big')


def read_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def previous(group):
    path = ROOT/f'experiments/K4/results/run_20261025/{group}/evaluations.jsonl'
    result = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['repeat']==0 and row['method'] in BASELINES:
                result[(row['case_id'],row['method'])] = row
    return result


def benchmark(cases,replies,output,old):
    output.mkdir()
    rng = random.Random(SEED)
    checks = dict(cases=len(cases),runs=0,independent_path_checks=0,baseline_parity_checks=0,
                  bound_checks=0,bound_violations=0,deterministic_repeat_checks=0)
    oracle,map_id,optima = None,None,[]
    with gzip.open(output/'evaluations.jsonl.gz','xt',encoding='utf-8') as stream:
        for position,case in enumerate(cases):
            q,cid = case['query'],case['case_id']
            if map_id != case['map_index']:
                map_id = case['map_index']; oracle = Oracle(q)
            optimum = oracle.distances(q['goal'])[tuple(q['start'])]
            optima.append(dict(case_id=cid,optimum=optimum))
            stable = {}
            for repeat in range(REPEATS):
                jobs = [(m,s) for m in METHODS for s in (range(SEEDS) if m in STOCHASTIC else [-1])]
                rng.shuffle(jobs)
                for order,(method,si) in enumerate(jobs):
                    seed = case_seed(cid,si) if si>=0 else 0
                    result = run_once(q,method,replies[cid]['text'],seed=seed)
                    llm = 'llm' in method and 'nollm' not in method
                    llm = llm or method in (PRIMARY,'mha_rr_w200','mha_ts_w200','mha_dts_w105')
                    result.update(case_id=cid,map_index=map_id,method=method,seed_index=si,seed=seed,
                                  repeat=repeat,order=order,optimum=optimum,uses_llm_guidance=llm,
                                  historical_api_seconds=replies[cid]['historical_api_seconds'] if llm else 0,
                                  response_sha256=replies[cid]['response_text_sha256'] if llm else None)
                    stream.write(json.dumps(result,allow_nan=False)+'\n')
                    checks['runs'] += 1
                    assert result['valid'],(cid,method,si,result['error'])
                    oracle.check_path(q,result); checks['independent_path_checks'] += 1
                    assert result['length']+1e-8 >= optimum
                    if result.get('bound_factor') is not None:
                        checks['bound_checks'] += 1
                        passed = result['length'] <= result['bound_factor']*optimum+1e-8
                        checks['bound_violations'] += not passed
                        assert passed,(cid,method,si,'bound')
                        anchor = result.get('anchor_key_at_goal',result.get('lower_bound_at_goal'))
                        assert anchor <= optimum+1e-8
                    signature = {k:result.get(k) for k in ('path','operation','storage','length','successor_expansions',
                                  'queue_expansions','bandit_rewards','bandit_alpha','bandit_beta','heap_pushes')}
                    if (method,si) in stable:
                        assert signature == stable[method,si],(cid,method,si,'repeat mismatch')
                        checks['deterministic_repeat_checks'] += 1
                    stable[method,si] = signature
                    if method in BASELINES:
                        prev = old[cid,method]
                        for field in ('path','operation','storage','length','valid'):
                            assert json.dumps(result[field])==json.dumps(prev[field]),(cid,method,field,'old parity')
                        checks['baseline_parity_checks'] += 1
            if (position+1)%25==0 or position+1==len(cases):
                stream.flush()
                print(f'{output.name}: {position+1}/{len(cases)} cases ({checks["runs"]} runs)',flush=True)
    write_json(output/'checks.json',checks)
    write_json(output/'optima.json',optima)


def summarize(output):
    buckets = {}
    for r in read_rows(output/'evaluations.jsonl.gz'):
        buckets.setdefault((r['case_id'],r['method'],r['seed_index']),[]).append(r)
    seeds = []
    fields = ('case_id','map_index','method','seed_index','seed','optimum','valid','error',
              'operation','successor_expansions','storage','length','bound_factor','guide_error',
              'historical_api_seconds','search_pops','reward_pops','stale_pops','heap_pushes',
              'peak_heap_entries','beta_samples','anchor_fallbacks','reexpansions',
              'guide_segment_checks','wall_checks','queue_expansions','bandit_choices','bandit_rewards')
    for key,rows in buckets.items():
        assert len(rows)==REPEATS
        row = {k:rows[0].get(k) for k in fields}
        row['total_seconds'] = statistics.median(r['total_seconds'] for r in rows)
        row['repeat_seconds'] = [r['total_seconds'] for r in rows]
        seeds.append(row)
    del buckets
    write_json(output/'per_seed.json',seeds)
    grouped = {}
    for row in seeds:
        grouped.setdefault((row['case_id'],row['method']),[]).append(row)
    cases,by_method = [],{m:{} for m in METHODS}
    for (cid,method),rows in grouped.items():
        assert len(rows)==(SEEDS if method in STOCHASTIC else 1)
        row = dict(case_id=cid,method=method,map_index=rows[0]['map_index'],optimum=rows[0]['optimum'],
                   seeds=len(rows),valid=sum(r['valid'] for r in rows))
        for metric in METRICS:
            row[metric] = statistics.mean(r[metric] for r in rows)
        cases.append(row);by_method[method][cid] = row
    with (output/'per_case.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    comparisons = {}
    for control in CONTROLS:
        comparisons[control] = {}
        for method in METHODS:
            pairs=[(r,by_method[control][cid]) for cid,r in by_method[method].items()]
            maps=sorted({r['map_index'] for r,_ in pairs})
            idx=np.random.default_rng(SEED).integers(0,len(maps),(2000,len(maps)))
            out={}
            for metric in METRICS:
                logs=np.array([math.log(r[metric]/b[metric]) for r,b in pairs])
                sums=np.array([sum(v for v,(r,b) in zip(logs,pairs) if r['map_index']==mi) for mi in maps])
                counts=np.array([sum(r['map_index']==mi for r,b in pairs) for mi in maps])
                boot=np.exp(sums[idx].sum(axis=1)/counts[idx].sum(axis=1))
                tolerance=1e-8 if metric=='length' else 0
                delta=[r[metric]-b[metric] for r,b in pairs]
                better=sum(d < -tolerance for d in delta); worse=sum(d > tolerance for d in delta)
                out[metric]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                 pairs=len(pairs),better=better,equal=len(pairs)-better-worse,worse=worse)
            out['joint']=dict(operation_better_length_not_worse=sum(r['operation']<b['operation'] and r['length']<=b['length']+1e-8 for r,b in pairs))
            comparisons[control][method]=out
    absolute,seed_spread = {},{}
    for method in METHODS:
        rows=[r for r in seeds if r['method']==method]
        absolute[method]=dict(cases=len(by_method[method]),seed_trials=len(rows),valid=sum(r['valid'] for r in rows),
            means={k:statistics.mean(r[k] for r in rows) for k in METRICS},
            optimal_paths=sum(abs(r['length']-r['optimum'])<1e-8 for r in rows),
            over_5_percent=sum(r['length']>1.05*r['optimum']+1e-8 for r in rows),
            max_optimal_length_ratio=max(r['length']/r['optimum'] for r in rows),
            guide_errors=dict(Counter(r['guide_error'] for r in rows if r['guide_error'])),
            extra_means={k:statistics.mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                for k in ('search_pops','reward_pops','stale_pops','heap_pushes','peak_heap_entries','beta_samples',
                          'anchor_fallbacks','reexpansions','guide_segment_checks','wall_checks')})
        if method in STOCHASTIC:
            seed_spread[method]={}
            for metric in METRICS:
                ratios=[]
                for si in range(SEEDS):
                    subset=[r for r in rows if r['seed_index']==si]
                    ratios.append(math.exp(statistics.mean(math.log(r[metric]/by_method['g2h_euclidean'][r['case_id']][metric]) for r in subset)))
                seed_spread[method][metric]=dict(per_seed_ratio=ratios,min=min(ratios),max=max(ratios))
    write_json(output/'summary.json',dict(primary=PRIMARY,absolute=absolute,comparisons=comparisons,
        seed_spread=seed_spread,aggregation='mean over 5 seeds per case; median over 3 timing repeats per seed; map-cluster bootstrap'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output exists; choose a new directory')
    assert git('branch','--show-current')=='gm'
    args.output.mkdir(parents=True)
    started=time.perf_counter()
    inputs=[ROOT/'reproduction/core.py',ROOT/'experiments/K3/heuristic.py',ROOT/'experiments/K3/verification.py',ROOT/'experiments/K3/run.py',
            ROOT/'experiments/K4/search.py',ROOT/'experiments/K4/run.py',
            ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py',ROOT/'source_code/dataset/environment_50_30.json',
            ROOT/'results/main/generations.jsonl',ROOT/'results/generalization/generations.jsonl',ROOT/'results/generalization/cases.json',
            *[ROOT/f'experiments/K4/results/run_20261025/{g}/evaluations.jsonl' for g in ('public','transfer')],
            ROOT/'experiments/K5/PROTOCOL.md',*[ROOT/f'experiments/K5/{n}.py' for n in ('__init__','search','verify','run')]]
    manifest=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        git_branch=git('branch','--show-current'),git_base_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),
        primary=PRIMARY,methods=METHODS,stochastic=STOCHASTIC,seeds=SEEDS,timing_repeats=REPEATS,seed=SEED,
        model=MODEL,prompt_style='standard',new_api_calls=0,python=sys.version,platform=platform.platform(),
        processor=platform.processor(),cpu_count=os.cpu_count(),numpy=np.__version__,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'verification.json',verify())
    print('2160 independent reference cases and special checks passed',flush=True)
    public=public_cases(); transfer=json.loads((ROOT/'results/generalization/cases.json').read_text(encoding='utf-8'))
    responses={**load_responses(public,ROOT/'results/main/generations.jsonl'),
               **load_responses(transfer,ROOT/'results/generalization/generations.jsonl')}
    write_json(args.output/'public_cases.json',public);write_json(args.output/'transfer_cases.json',transfer)
    with gzip.open(args.output/'responses.jsonl.gz','xt',encoding='utf-8') as stream:
        for row in responses.values():stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest['fixed_data_sha256']={n:sha(args.output/n) for n in ('public_cases.json','transfer_cases.json','responses.jsonl.gz')}
    write_json(args.output/'manifest.json',manifest)
    old_public,old_transfer=previous('public'),previous('transfer')
    pilot=[c for c in public if c['map_index']%10==0 and c['pair_index']<3]
    for name,cases,old in [('pilot',pilot,old_public),('public',public,old_public),('transfer',transfer,old_transfer)]:
        benchmark(cases,responses,args.output/name,old)
        summarize(args.output/name)
    for path in inputs:assert sha(path)==manifest['input_sha256'][str(path.relative_to(ROOT))],path
    manifest.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(status='completed',seconds=manifest['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
