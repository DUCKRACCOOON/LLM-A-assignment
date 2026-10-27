"""Offline J3--J6 benchmark. Refuses existing output. No API implementation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import platform
from pathlib import Path
import random
import runpy
import subprocess
import sys

from experiments.K3.run import public_cases, sha, write_json
from experiments.K3.verification import Oracle
from reproduction.core import Geometry
from .search import METHODS, EXACT, LandmarkIndex, run_once

ROOT=Path(__file__).resolve().parents[2]
SEED=20261027
MODEL='gpt-4o-mini-2024-07-18'
LLM_METHODS={'legacy_llm','j3_llm_noprune','j3_llm_prune','j6_full','j6_lazy'}
SOURCES=('source_code/dataset/environment_50_30.json','results/main/generations.jsonl',
         'results/main/evaluations.jsonl','results/generalization/cases.json',
         'results/generalization/generations.jsonl','results/generalization/evaluations.jsonl')


def read_rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def responses(cases,path):
    lookup={r['case_id']:r for r in read_rows(path) if r['model']==MODEL and r['prompt_style']=='standard'}
    template=runpy.run_path(str(ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']['standard']
    result={}
    for c in cases:
        r=lookup[c['case_id']];q=c['query'];assert r['status']=='ok'
        content=template.format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
        expected=dict(model=MODEL,messages=[dict(role='user',content=content)],temperature=0,max_tokens=1000,store=False)
        assert r['request']==expected
        digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        assert r['request_sha256']==digest
        text=r['response']['choices'][0]['message'].get('content') or ''
        result[c['case_id']]=dict(text=text,request_sha256=digest,
            response_sha256=hashlib.sha256(text.encode()).hexdigest(),historical_api_seconds=r['seconds'],
            model=MODEL,prompt_style='standard',usage=r['response'].get('usage'))
    return result


def synthetic_cases():
    rng=random.Random(SEED);cases=[];sampling=[]
    for mi in range(30):
        width,height=((32,22),(52,32),(72,42))[mi%3]; count=(4,6,8)[(mi//3)%3]
        q=dict(range_x=[0,width+1],range_y=[0,height+1],horizontal_barriers=[],vertical_barriers=[])
        for _ in range(count):
            low=rng.randint(0,width-8)
            q['horizontal_barriers'].append([rng.randint(3,height-3),low,min(width,low+rng.randint(5,width//2))])
        for _ in range(count):
            low=rng.randint(0,height-6)
            q['vertical_barriers'].append([rng.randint(3,width-3),low,min(height,low+rng.randint(3,height//2))])
        oracle=Oracle(q);used=set();rejected=defaultdict(int)
        for pi in range(4):
            for _ in range(100000):
                s=(rng.randint(1,width-1),rng.randint(1,height-1));g=(rng.randint(1,width-1),rng.randint(1,height-1))
                if s==g:rejected['same_point']+=1
                elif s not in oracle.graph or g not in oracle.graph:rejected['blocked_endpoint']+=1
                elif (s,g) in used:rejected['duplicate_pair']+=1
                elif s not in oracle.distances(g):rejected['disconnected']+=1
                else:
                    used.add((s,g));cases.append(dict(case_id=f'jy_s{mi:03d}_p{pi:02d}',map_index=mi,pair_index=pi,
                        query=dict(q,start=list(s),goal=list(g))));break
            else:raise RuntimeError('Sampler exhausted')
        sampling.append(dict(map_index=mi,size=[width,height],walls_per_axis=count,rejected=dict(rejected)))
    return cases,sampling


def baseline_rows(group):
    path=ROOT/('results/main/evaluations.jsonl' if group=='public' else 'results/generalization/evaluations.jsonl')
    old={}
    if group=='synthetic':return old
    for r in read_rows(path):
        m={'astar':'legacy_astar','goal_only':'legacy_g2h','llm_astar':'legacy_llm'}.get(r['method'])
        if m and (m!='legacy_llm' or (r.get('model',MODEL)==MODEL and r.get('prompt_style','standard')=='standard')):
            old[(r['case_id'],m)]=r
    return old


def benchmark(group,cases,replies,out,repeats=3):
    folder=out/group;folder.mkdir();rng=random.Random(SEED)
    old=baseline_rows(group);last_map=None;oracle=None;indexes={};counts=defaultdict(int);optima={}
    with gzip.open(folder/'evaluations.jsonl.gz','xt',encoding='utf-8') as stream, \
         gzip.open(folder/'preprocessing.jsonl.gz','xt',encoding='utf-8') as prestream:
        for i,c in enumerate(cases):
            q=c['query'];cid=c['case_id'];mi=c['map_index']
            if mi!=last_map:
                oracle=Oracle(q,crosscheck=True);last_map=mi;indexes={};counts['maps']+=1
                counts['collision_checks']+=oracle.collision_checks
                for repeat in range(repeats):
                    order=[4,8];rng.shuffle(order)
                    for k in order:
                        idx=LandmarkIndex(q,k);indexes[repeat,k]=idx
                        prestream.write(json.dumps(dict(map_index=mi,repeat=repeat,landmark_count=k,
                            seconds=idx.seconds,pops=idx.pops,expansions=idx.expansions,pushes=idx.pushes,
                            relax_attempts=idx.relax,landmarks=idx.landmarks,table_entries=idx.table_entries,
                            graph_nodes=len(idx.graph),graph_edges=idx.edge_entries))+'\n')
                        # Validate every ALT edge/vertex for a goal in this map.
                        hs=idx.heuristic(tuple(q['goal']));actual=oracle.distances(q['goal'])
                        for n,edges in idx.graph.items():
                            assert n in oracle.graph
                            if n in actual:
                                assert hs(n)<=actual[n]+1e-8;counts['alt_admissibility_checks']+=1
                            for other,step in edges:
                                assert (other,step) in oracle.graph[n]
                                assert hs(n)<=step+hs(other)+1e-8;counts['alt_consistency_checks']+=1
            optimum=oracle.distances(q['goal'])[tuple(q['start'])];optima[cid]=optimum
            text=replies.get(cid,{}).get('text','');stable={}
            methods=[m for m in METHODS if cid in replies or m not in LLM_METHODS]
            for repeat in range(repeats):
                order=list(methods);rng.shuffle(order);paired={}
                for position,m in enumerate(order):
                    idx=indexes[repeat,4 if m=='j4_alt4' else 8] if m.startswith('j4_') else None
                    r=run_once(q,m,text,index=idx,trace=m.startswith('j6_'))
                    r.update(case_id=cid,map_index=mi,pair_index=c.get('pair_index',i%10),method=m,
                             repeat=repeat,order=position,optimum=optimum,
                             historical_api_seconds=replies[cid]['historical_api_seconds'] if m in LLM_METHODS else 0,
                             response_sha256=replies[cid]['response_sha256'] if m in LLM_METHODS else None)
                    trace=r.pop('pop_trace',[])
                    r['pop_trace_sha256']=hashlib.sha256(json.dumps(trace).encode()).hexdigest() if m.startswith('j6_') else None
                    stream.write(json.dumps(r,allow_nan=False)+'\n');counts['runs']+=1
                    assert r['valid'],(cid,m,r['error'])
                    oracle.check_path(q,r);counts['path_checks']+=1
                    assert r['length']+1e-8>=optimum
                    if m in EXACT:
                        assert abs(r['length']-optimum)<1e-8,(cid,m,r['length'],optimum)
                        counts['optimality_checks']+=1
                    signature={k:r.get(k) for k in ('path','length','valid','operation','expansions','storage','pop_trace_sha256','target_changes')}
                    if m in stable:
                        assert stable[m]==signature,(cid,m,'nondeterministic');counts['repeat_checks']+=1
                    stable[m]=signature;paired[m]=r
                    if (cid,m) in old:
                        for k in ('valid','path','operation','storage','length'):
                            assert json.dumps(r[k])==json.dumps(old[cid,m][k]),(cid,m,k,'baseline parity')
                        counts['baseline_parity_checks']+=1
                if 'j6_full' in paired:
                    for k in ('path','length','expansions','storage','pop_trace_sha256','target_changes'):
                        assert paired['j6_full'][k]==paired['j6_lazy'][k],(cid,k,'J6 parity')
                    counts['j6_parity_checks']+=1
            if (i+1)%10==0 or i+1==len(cases):
                stream.flush();prestream.flush();print(f'{group}: {i+1}/{len(cases)} cases, {counts["runs"]} runs',flush=True)
    write_json(folder/'checks.json',dict(counts,case_count=len(cases),repeats=repeats,methods=methods,passed=True))
    write_json(folder/'optima.json',optima)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source_files=list((ROOT/'experiments/JY').glob('*.py'))+[ROOT/'experiments/JY/PROTOCOL.md',ROOT/'reproduction/core.py',ROOT/'experiments/K3/heuristic.py',ROOT/'experiments/K3/verification.py']
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        command=sys.argv,python=sys.version,platform=platform.platform(),processor=platform.processor(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),new_api_calls=0,
        repeats=3,methods=list(METHODS),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_files},
        input_sha256={f:sha(ROOT/f) for f in SOURCES},smoke=args.smoke,
        timing_note='ALT cold=measured map preparation + query; prep physically once/map/k/repeat; historical API latency separate')
    assert manifest['git_branch']=='jy';write_json(out/'manifest.json',manifest)
    try:
        public=public_cases();transfer=json.loads((ROOT/'results/generalization/cases.json').read_text(encoding='utf-8'))
        synthetic,sampling=synthetic_cases()
        if args.smoke:public=public[:3];transfer=transfer[:2];synthetic=synthetic[:2]
        for group,cases in [('public',public),('transfer',transfer),('synthetic',synthetic)]:
            write_json(out/(group+'_cases.json'),cases)
        write_json(out/'synthetic_sampling.json',sampling)
        pub_resp=responses(public,ROOT/'results/main/generations.jsonl');trans_resp=responses(transfer,ROOT/'results/generalization/generations.jsonl')
        with gzip.open(out/'responses.jsonl.gz','xt',encoding='utf-8') as f:
            for group,replies in [('public',pub_resp),('transfer',trans_resp)]:
                for cid,r in replies.items():f.write(json.dumps(dict(stage=group,case_id=cid,**r),ensure_ascii=False)+'\n')
        for group,cases,replies in [('public',public,pub_resp),('transfer',transfer,trans_resp),('synthetic',synthetic,{})]:
            benchmark(group,cases,replies,out)
        manifest.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(status='failed',error_class=type(exc).__name__,error=str(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
