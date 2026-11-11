"""Freeze and execute seven J2 conditions once on all 1,000 public queries."""
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

from experiments.COMMON18.backends import ROOT,run,cc
from experiments.COMMON18.prepare import save,digest,MODEL,split
from experiments.K3.run import public_cases,sha
from experiments.K3.verification import Oracle,fixtures
import runpy
STYLES=('standard','cot','repe')

METHODS=('Astar','LLMA','G2H','FOCAL0','J2_uniform','J2','J2_exact')
SEED=20261111
SOURCE=ROOT/'results/main/generations.jsonl'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def replies(out):
    with gzip.open(out/'responses.jsonl.gz','rt',encoding='utf-8') as f:
        result={}
        for r in map(json.loads,f):result.setdefault(r['case_id'],{})[r['prompt_style']]=r
        return result


def sources():
    patterns=('experiments/J2_full/*.py','experiments/COMMON18/*.py','experiments/COMMON18/vendor_sm/*.py',
              'experiments/K3/*.py','experiments/K4/*.py','experiments/K5/*.py','experiments/K6/*.py',
              'experiments/JY/*.py','reproduction/*.py')
    analysis={Path(__file__).with_name('report.py'),Path(__file__).with_name('audit.py')}
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for pat in patterns for p in ROOT.glob(pat) if p not in analysis}


def deny(*args,**kwargs):raise RuntimeError('Offline J2 study: network/API disabled')


def no_network():
    socket.create_connection=deny;socket.socket.connect=deny
    import reproduction.api as api
    import experiments.COMMON18.prepare as preparation
    api.completion=preparation.completion=deny


def freeze(out):
    assert not out.exists(),'Use a fresh output directory; never overwrite earlier experiments'
    cases=public_cases();assert len(cases)==1000 and len({c['map_index'] for c in cases})==100
    templates=runpy.run_path(str(ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']
    selected={}
    for line in SOURCE.open(encoding='utf-8'):
        r=json.loads(line)
        if r['model']==MODEL and r['prompt_style'] in STYLES:
            key=(r['case_id'],r['prompt_style']);assert key not in selected;selected[key]=r
    assert set(selected)=={(c['case_id'],style) for c in cases for style in STYLES}
    provenance=[]
    for c in cases:
        q=c['query']
        for style in STYLES:
            r=selected[c['case_id'],style]
            text=templates[style].format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
            expected=dict(model=MODEL,messages=[dict(role='user',content=text)],temperature=0,max_tokens=1000,store=False)
            assert r['request']==expected and r['request_sha256']==digest(expected) and r['status']=='ok'
            assert r['response']['model']==MODEL
            content=r['response']['choices'][0]['message'].get('content') or ''
            provenance.append(dict(case_id=c['case_id'],style=style,request_sha256=r['request_sha256'],
                response_text_sha256=hashlib.sha256(content.encode()).hexdigest(),historical_api_seconds=r['seconds'],
                usage=r['response']['usage'],finish_reason=r['response']['choices'][0]['finish_reason']))
    out.mkdir(parents=True);save(out/'cases.json',cases);save(out/'response_provenance.json',provenance)
    with gzip.open(out/'responses.jsonl.gz','xt',encoding='utf-8') as f:
        for c in cases:
            for style in STYLES:f.write(json.dumps(selected[c['case_id'],style],ensure_ascii=False)+'\n')
    save(out/'protocol.json',dict(client_date='2026-11-11',created_utc=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,response_styles=STYLES,cases=1000,maps=100,repeats=1,seed=SEED,new_api_calls=0,model=MODEL,
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
    # Numeric regression: first-order Eikonal is not 8-neighbour Dijkstra.
    q=dict(range_x=[0,8],range_y=[0,8],horizontal_barriers=[],vertical_barriers=[],start=[2,2],goal=[4,4])
    grid=cc.Grid(q);unit,meta=cc.fast_marching(grid,q['goal']);fast,_=cc.fast_marching(grid,q['goal'],np.full(grid.n,2.))
    assert unit[grid.index([4,4])]==0 and abs(unit[grid.index([5,4])]-1)<1e-10
    assert abs(unit[grid.index([5,5])]-(1+math.sqrt(.5)))<1e-10
    finite=np.isfinite(unit);assert np.array_equal(finite,np.isfinite(fast))
    assert np.allclose(unit[finite],2*fast[finite]) and np.all(unit[finite]>=0)
    save(out/'preflight.json',dict(passed=True,fixture_runs=len(records),records=records,
        fmm_unit_cardinal=1.,fmm_unit_diagonal=float(unit[grid.index([5,5])]),uniform_speed_scaling_passed=True))


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
            order=list(METHODS);rng.shuffle(order);gc.collect();gc.disable()
            for position,method in enumerate(order):
                r=run(method,q,cache[cid],{},seed=SEED)
                r.update(case_id=cid,map_index=c['map_index'],pair_index=c['pair_index'],repeat=0,order=position,optimum=opt,
                    response_sha256={k:cache[cid][k]['request_sha256'] for k in r['used_responses']})
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');counts['runs']+=1
                if r['valid']:
                    oracle.check_path(q,r);counts['path_checks']+=1
                    assert r['length']>=opt-1e-7
                    if r['bound']:
                        assert r['length']<=opt*r['bound']+1e-7,(cid,method,'bound violation');counts['bound_checks']+=1
                    if r['bound']==1:counts['exact_checks']+=1
                else:failed.append(dict(case_id=cid,method=method,error=r['error']))
            gc.enable()
            if (i+1)%25==0:
                f.flush();save(out/'status.json',dict(completed_cases=i+1,total_cases=1000,runs=counts['runs'],failed=len(failed)))
                print(f'J2 full {i+1}/1000 cases; {counts["runs"]} runs; {len(failed)} failures',flush=True)
    save(out/'optima.json',optima);save(out/'geometry.json',geometry);save(out/'failures.json',failed)
    assert counts['runs']==7000 and len(geometry)==100
    save(out/'complete.json',dict(counts,cases=1000,maps=100,failed=len(failed),complete=True,new_api_calls=0,
        finished_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    freeze(out);execute(out)


if __name__=='__main__':main()
