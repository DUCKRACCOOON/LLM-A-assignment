"""Offline K4 experiment. Existing output directories are never overwritten."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import runpy
import statistics
import subprocess
import sys
import time

import numpy as np

from experiments.K3.run import public_cases, write_json, sha, git
from experiments.K3.verification import Oracle
from .search import METHODS, PRIMARY, run_once
from .verify import verify

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261026
MODEL = 'gpt-4o-mini-2024-07-18'
PROMPT = ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'
K3 = ROOT/'experiments/K3/results/run_20261025'
CONTROLS = ('g2h_euclidean','g2h_wall_octile','g2h_wall_euclidean','focal_goal_w105','focal_straight_w105','astar_euclidean')


def load_responses(cases, path):
    selected = {}
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        if row['model'] == MODEL and row['prompt_style'] == 'standard':
            assert row['case_id'] not in selected
            selected[row['case_id']] = row
    template = runpy.run_path(str(PROMPT))['gpt_prompt']['standard']
    replies = {}
    for case in cases:
        row = selected[case['case_id']]
        assert row['status'] == 'ok'
        q = case['query']
        text = template.format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
        expected = dict(model=MODEL, messages=[dict(role='user',content=text)], temperature=0,max_tokens=1000,store=False)
        assert row['request'] == expected, (case['case_id'],'request/input mismatch')
        assert hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest() == row['request_sha256']
        content = row['response']['choices'][0]['message'].get('content') or ''
        replies[case['case_id']] = dict(case_id=case['case_id'], model=MODEL,prompt_style='standard',
            request_sha256=row['request_sha256'], response_text_sha256=hashlib.sha256(content.encode()).hexdigest(),
            text=content, historical_api_seconds=row['seconds'],usage=row['response'].get('usage'),
            finish_reason=row['response']['choices'][0].get('finish_reason'), source_file=str(path.relative_to(ROOT)))
    return replies


def old_rows(group):
    values = {}
    if group == 'public':
        for line in (K3/'public/evaluations.jsonl').open(encoding='utf-8'):
            row = json.loads(line)
            if row['repeat'] == 0 and row['method'] in METHODS[:4]:
                values[(row['case_id'],row['method'])] = row
    else:
        for line in (ROOT/'results/generalization/evaluations.jsonl').open(encoding='utf-8'):
            row = json.loads(line)
            method = {'astar':'astar_euclidean','goal_only':'g2h_euclidean'}.get(row['method'])
            if method:
                values[(row['case_id'],method)] = row
    return values


def benchmark(cases, replies, output, old):
    output.mkdir()
    rng = random.Random(SEED)
    counts = dict(cases=len(cases), runs=0, independent_path_checks=0, baseline_parity_checks=0,
                  bound_checks=0, bound_violations=0, deterministic_repeat_checks=0)
    optimum_records, map_index, oracle = [], None, None
    with (output/'evaluations.jsonl').open('x',encoding='utf-8') as stream:
        for i,case in enumerate(cases):
            query,cid = case['query'],case['case_id']
            if case['map_index'] != map_index:
                map_index = case['map_index']; oracle = Oracle(query)
            optimum = oracle.distances(query['goal'])[tuple(query['start'])]
            optimum_records.append(dict(case_id=cid,optimum=optimum))
            stable = {}
            for repeat in range(3):
                order=list(METHODS); rng.shuffle(order)
                for position,method in enumerate(order):
                    result = run_once(query,method,replies[cid]['text'])
                    counts['runs'] += 1
                    result.update(case_id=cid,map_index=case['map_index'],method=method,repeat=repeat,order=position,optimum=optimum)
                    llm = 'llm' in method
                    result.update(uses_llm_guidance=llm, historical_api_seconds=replies[cid]['historical_api_seconds'] if llm else 0,
                                  response_sha256=replies[cid]['response_text_sha256'] if llm else None)
                    # Persist even an unexpected failure before raising, for diagnosis.
                    stream.write(json.dumps(result,allow_nan=False)+'\n')
                    assert result['valid'], (cid,method,result['error'])
                    oracle.check_path(query,result)
                    counts['independent_path_checks'] += 1
                    assert result['length']+1e-8 >= optimum
                    if method.startswith('focal'):
                        counts['bound_checks'] += 1
                        passed = result['length'] <= result['bound_factor']*optimum+1e-8
                        counts['bound_violations'] += not passed
                        assert passed, (cid,method,'cost bound violated')
                        assert result['lower_bound_at_goal'] <= optimum+1e-8
                    signature={k:result[k] for k in ('path','operation','storage','length','expansions')}
                    if method in stable:
                        assert signature == stable[method]
                        counts['deterministic_repeat_checks'] += 1
                    stable[method]=signature
                    if (cid,method) in old:
                        previous=old[(cid,method)]
                        for key in ('operation','storage','path','length','valid'):
                            assert json.dumps(result[key]) == json.dumps(previous[key]), (cid,method,key,'baseline mismatch')
                        counts['baseline_parity_checks'] += 1
            if (i+1)%25 == 0 or i+1==len(cases):
                stream.flush()
                print(f'{output.name}: {i+1}/{len(cases)} cases',flush=True)
    write_json(output/'checks.json',counts)
    write_json(output/'optima.json',optimum_records)


def summarize(output):
    buckets={}
    for line in (output/'evaluations.jsonl').open(encoding='utf-8'):
        row=json.loads(line)
        buckets.setdefault((row['case_id'],row['method']),[]).append(row)
    by_method={m:{} for m in METHODS}
    selected_fields=('case_id','map_index','method','valid','error','operation','expansions','storage','length','unique_pops',
                     'optimum','bound_factor','guide_error','guide_segment_checks','wall_checks','reopenings','reexpansions',
                     'open_pops','waiting_pops','focal_pops','heap_pushes','stale_pops','peak_heap_entries',
                     'eligibility_rebuilds','rebuild_scans','historical_api_seconds','uses_llm_guidance')
    rows=[]
    for (cid,method),trials in buckets.items():
        assert len(trials)==3
        row={k:trials[0].get(k) for k in selected_fields}
        row['total_seconds']=statistics.median(r['total_seconds'] for r in trials)
        row['estimated_fresh_seconds']=row['total_seconds']+row['historical_api_seconds']
        row['repeat_seconds']=[r['total_seconds'] for r in trials]
        rows.append(row); by_method[method][cid]=row
    with (output/'per_case.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics=('operation','expansions','storage','length','total_seconds','estimated_fresh_seconds')
    comparisons={}
    for control in CONTROLS:
        comparisons[control]={}
        for method in METHODS:
            pairs=[(r,by_method[control][cid]) for cid,r in by_method[method].items() if r['valid'] and by_method[control][cid]['valid']]
            maps=sorted({r['map_index'] for r,_ in pairs})
            index=np.random.default_rng(SEED).integers(0,len(maps),(2000,len(maps)))
            out={}
            for metric in metrics:
                logs=np.array([math.log(r[metric]/b[metric]) for r,b in pairs])
                sums=np.array([sum(v for v,(r,b) in zip(logs,pairs) if r['map_index']==mi) for mi in maps])
                counts=np.array([sum(r['map_index']==mi for r,b in pairs) for mi in maps])
                boot=np.exp(sums[index].sum(axis=1)/counts[index].sum(axis=1))
                tolerance=1e-8 if metric=='length' else 0
                delta=[r[metric]-b[metric] for r,b in pairs]
                better=sum(v < -tolerance for v in delta);worse=sum(v > tolerance for v in delta)
                out[metric]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                 pairs=len(pairs),better=better,equal=len(pairs)-better-worse,worse=worse)
            out['joint']=dict(operation_better_length_not_worse=sum(r['operation']<b['operation'] and r['length']<=b['length']+1e-8 for r,b in pairs),
                              expansions_better_length_not_worse=sum(r['expansions']<b['expansions'] and r['length']<=b['length']+1e-8 for r,b in pairs))
            comparisons[control][method]=out
    absolute={}
    for method,entries in by_method.items():
        valid=[r for r in entries.values() if r['valid']]
        absolute[method]=dict(cases=len(entries),valid=len(valid),
            means={k:statistics.mean(r[k] for r in valid) for k in metrics},
            optimal_paths=sum(abs(r['length']-r['optimum'])<1e-8 for r in valid),
            max_optimal_length_ratio=max(r['length']/r['optimum'] for r in valid),
            over_5_percent=sum(r['length']>1.05*r['optimum']+1e-8 for r in valid),
            guide_errors=dict(Counter(r['guide_error'] for r in valid if r['guide_error'])),
            extra_means={k:statistics.mean(r[k] for r in valid if r[k] is not None) if any(r[k] is not None for r in valid) else None
                         for k in ('open_pops','waiting_pops','focal_pops','heap_pushes','peak_heap_entries','reopenings','reexpansions',
                                   'eligibility_rebuilds','rebuild_scans','guide_segment_checks','wall_checks')})
    write_json(output/'summary.json',dict(primary=PRIMARY,absolute=absolute,comparisons=comparisons,
        timing='Planner only; estimated_fresh_seconds uses historical response latency, not a new live measurement'))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output exists; choose a new directory')
    assert git('branch','--show-current')=='gm'
    args.output.mkdir(parents=True)
    start=time.perf_counter()
    inputs=[ROOT/'reproduction/core.py',ROOT/'experiments/K3/heuristic.py',ROOT/'experiments/K3/verification.py',ROOT/'experiments/K3/run.py',
            PROMPT,ROOT/'source_code/dataset/environment_50_30.json',K3/'public/evaluations.jsonl',
            ROOT/'results/main/generations.jsonl',ROOT/'results/generalization/generations.jsonl',
            ROOT/'results/generalization/cases.json',ROOT/'results/generalization/evaluations.jsonl',
            ROOT/'experiments/K4/PROTOCOL.md',*sorted((ROOT/'experiments/K4').glob('*.py'))]
    manifest=dict(status='started',started_utc=datetime.now(timezone.utc).isoformat(),git_branch=git('branch','--show-current'),
        git_base_commit=git('rev-parse','HEAD'),git_status=git('status','--short'),command=sys.argv,primary=PRIMARY,methods=METHODS,
        repeats=3,seed=SEED,model=MODEL,prompt_style='standard',new_api_calls=0,python=sys.version,platform=platform.platform(),
        processor=platform.processor(),cpu_count=os.cpu_count(),numpy=np.__version__,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    write_json(args.output/'manifest.json',manifest)
    verification=verify();write_json(args.output/'verification.json',verification)
    print('520 correctness cases plus reopening/numeric/endpoint checks passed',flush=True)
    public=public_cases()
    transfer=json.loads((ROOT/'results/generalization/cases.json').read_text(encoding='utf-8'))
    public_replies=load_responses(public,ROOT/'results/main/generations.jsonl')
    transfer_replies=load_responses(transfer,ROOT/'results/generalization/generations.jsonl')
    assert len(public)==1000 and len(transfer)==30
    write_json(args.output/'public_cases.json',public);write_json(args.output/'transfer_cases.json',transfer)
    with (args.output/'responses.jsonl').open('x',encoding='utf-8') as stream:
        for reply in [*public_replies.values(),*transfer_replies.values()]: stream.write(json.dumps(reply,ensure_ascii=False)+'\n')
    manifest['fixed_data_sha256']={name:sha(args.output/name) for name in ('public_cases.json','transfer_cases.json','responses.jsonl')}
    write_json(args.output/'manifest.json',manifest)
    public_old,transfer_old=old_rows('public'),old_rows('transfer')
    pilot=[c for c in public if c['map_index']%10==0 and c['pair_index']<3]
    for name,cases,replies,old in [('pilot',pilot,public_replies,public_old),('public',public,public_replies,public_old),
                                   ('transfer',transfer,transfer_replies,transfer_old)]:
        benchmark(cases,replies,args.output/name,old)
        summarize(args.output/name)
    for path in inputs: assert sha(path)==manifest['input_sha256'][str(path.relative_to(ROOT))],path
    manifest.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(status='completed',seconds=manifest['elapsed_seconds'],output=str(args.output)),ensure_ascii=True),flush=True)


if __name__=='__main__': main()
