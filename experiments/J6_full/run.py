"""Freeze and execute seven J6 conditions once on all 1,000 public queries."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys

import numpy as np
import scipy
import shapely

from experiments.COMMON18.backends import ROOT,cc
from .adapter import METHODS,run
from experiments.COMMON18.prepare import save,digest,MODEL,split
from experiments.K3.run import public_cases,sha
from experiments.K3.verification import Oracle,fixtures
from experiments.K4.run import load_responses

SEED=20261111
SOURCE=ROOT/'results/main/generations.jsonl'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def replies(out):
    with gzip.open(out/'responses.jsonl.gz','rt',encoding='utf-8') as f:
        return {r['case_id']:{'standard':r} for r in map(json.loads,f)}


def sources():
    patterns=('experiments/J6_full/*.py','experiments/COMMON18/*.py','experiments/COMMON18/vendor_sm/*.py',
              'experiments/K3/*.py','experiments/K4/*.py','experiments/K5/*.py','experiments/K6/*.py',
              'experiments/JY/*.py','reproduction/*.py')
    analysis={Path(__file__).with_name('report.py'),Path(__file__).with_name('audit.py')}
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for pat in patterns for p in ROOT.glob(pat) if p not in analysis}


def deny(*args,**kwargs):raise RuntimeError('Offline J6 study: network/API disabled')


def no_network():
    socket.create_connection=deny;socket.socket.connect=deny
    import reproduction.api as api
    import experiments.COMMON18.prepare as preparation
    api.completion=preparation.completion=deny


def freeze(out):
    assert not out.exists(),'Use a fresh output directory; never overwrite earlier experiments'
    cases=public_cases();assert len(cases)==1000 and len({c['map_index'] for c in cases})==100
    # Existing loader independently matches every request body to the official template and query.
    normalized=load_responses(cases,SOURCE);selected={}
    for line in SOURCE.open(encoding='utf-8'):
        r=json.loads(line)
        if r['model']==MODEL and r['prompt_style']=='standard':
            assert r['case_id'] not in selected;selected[r['case_id']]=r
    assert set(selected)=={c['case_id'] for c in cases}
    assert all(r['request_sha256']==digest(r['request']) and r['status']=='ok' for r in selected.values())
    out.mkdir(parents=True);save(out/'cases.json',cases);save(out/'response_provenance.json',list(normalized.values()))
    with gzip.open(out/'responses.jsonl.gz','xt',encoding='utf-8') as f:
        for c in cases:f.write(json.dumps(selected[c['case_id']],ensure_ascii=False)+'\n')
    save(out/'protocol.json',dict(client_date='2026-11-11',created_utc=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,cases=1000,maps=100,repeats=1,seed=SEED,new_api_calls=0,model=MODEL,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_response_file=str(SOURCE.relative_to(ROOT)),source_response_sha256=sha(SOURCE),
        frozen_response_sha256=sha(out/'responses.jsonl.gz'),data_sha256=sha(ROOT/'source_code/dataset/environment_50_30.json'),
        protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),source_sha256=sources(),
        common18_test_maps=split()['test'],python=sys.version,platform=platform.platform(),
        processor=platform.processor(),packages=dict(numpy=np.__version__,scipy=scipy.__version__,shapely=shapely.__version__),
        native_threads=1,api_latency='historical shared latency, reconstructed not new live service measurement'))


def preflight(out,cache):
    records=[]
    for name in ('empty','long_wall','crossing_walls','endpoint_contact','degenerate_walls'):
        q=fixtures()[name];o=Oracle(q,crosscheck=True);opt=o.distances(q['goal'])[tuple(q['start'])]
        for method in METHODS:
            r=run(method,q,cache['m000_p00'],{})
            assert r['valid'],(name,method,r.get('error'));o.check_path(q,r)
            if r['bound']:assert r['length']<=opt*r['bound']+1e-7
            records.append(dict(fixture=name,method=method,valid=r['valid']))
    import io,unittest
    name='experiments.JY.test_search.SearchTests.test_lazy_exact_same_pop_sequence_on_target_changes'
    suite=unittest.defaultTestLoader.loadTestsFromName(name);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream).run(suite)
    save(out/'unit_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,name=name,log=stream.getvalue(),
        source_sha256=sha(ROOT/'experiments/JY/test_search.py'),fixtures=10,target_draws_per_fixture=10,conditions=2))
    assert result.wasSuccessful(),stream.getvalue()
    save(out/'preflight.json',dict(passed=True,fixture_runs=len(records),records=records,forced_target_trace_regression_passed=True))


def execute(out):
    no_network();cache=replies(out);preflight(out,cache)
    cases=load(out/'cases.json');rng=random.Random(SEED);counts=Counter();failed=[];optima={};geometry=[]
    assert sources()==load(out/'protocol.json')['source_sha256'],'Source changed after freeze'
    with gzip.open(out/'evaluations.jsonl.gz','xt',encoding='utf-8') as f:
        for i,c in enumerate(cases):
            q=c['query'];cid=c['case_id']
            if c['pair_index']==0:
                oracle=Oracle(q,crosscheck=True);grid=cc.Grid(q);edges=0
                for p,links in oracle.graph.items():
                    actual={(tuple(map(int,grid.xy[v])),float(grid.costs[k])) for k,v in enumerate(grid.edges[grid.index(p)]) if v>=0}
                    assert actual==set(links),(c['map_index'],p);edges+=len(links)
                geometry.append(dict(map_index=c['map_index'],directed_edges=edges,crosschecks=oracle.collision_checks))
            opt=oracle.distances(q['goal'])[tuple(q['start'])];optima[cid]=opt
            order=list(METHODS);rng.shuffle(order);gc.collect();gc.disable();paired={}
            for position,method in enumerate(order):
                r=run(method,q,cache[cid],{},seed=SEED)
                r.update(case_id=cid,map_index=c['map_index'],pair_index=c['pair_index'],repeat=0,order=position,optimum=opt,
                    response_sha256=cache[cid]['standard']['request_sha256'] if r['llm_calls'] else None)
                if method in ('J6_full','J6'):
                    r['effective_trace_sha256']=hashlib.sha256(json.dumps(r['pop_trace'],separators=(',',':')).encode()).hexdigest();paired[method]=r
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');counts['runs']+=1
                if r['valid']:
                    oracle.check_path(q,r);counts['path_checks']+=1
                    assert r['length']>=opt-1e-7
                    if r['bound']:
                        assert r['length']<=opt*r['bound']+1e-7,(cid,method,'bound violation');counts['bound_checks']+=1
                    if r['bound']==1:counts['exact_checks']+=1
                else:failed.append(dict(case_id=cid,method=method,error=r['error']))
            for key in ('valid','path','length','expansions','storage','target_changes','relax_attempts','pop_trace','effective_trace_sha256'):
                assert paired['J6_full'][key]==paired['J6'][key],(cid,key,'full/lazy mismatch')
            for method,r in paired.items():
                assert r['operation']==r['expansions']+int(r['valid'])+r['stale_pops']+r['lazy_refreshes']
                assert len(r['pop_trace'])==r['expansions']+int(r['valid'])
                if method=='J6_full':assert r['lazy_refreshes']==0
            counts['full_lazy_parity_cases']+=1
            gc.enable()
            if (i+1)%25==0:
                f.flush();save(out/'status.json',dict(completed_cases=i+1,total_cases=1000,runs=counts['runs'],failed=len(failed)))
                print(f'J6 full {i+1}/1000 cases; {counts["runs"]} runs; {len(failed)} failures',flush=True)
    save(out/'optima.json',optima);save(out/'geometry.json',geometry);save(out/'failures.json',failed)
    assert counts['runs']==7000 and len(geometry)==100
    save(out/'complete.json',dict(counts,cases=1000,maps=100,failed=len(failed),complete=True,new_api_calls=0,
        finished_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    freeze(out);execute(out)


if __name__=='__main__':main()
