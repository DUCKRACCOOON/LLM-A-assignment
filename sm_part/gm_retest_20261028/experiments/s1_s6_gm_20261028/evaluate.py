import argparse
import copy
import math
import random
import time
from collections import defaultdict
from common import *
from algorithms import run, BASE, MODEL_METHODS, models_load, candidate
from experiments.K3.verification import Oracle

def oracle_all():
    path=OUT/'oracle.json';answers=read(path) if path.exists() else {};by_map=defaultdict(list);started=time.perf_counter()
    for q in cases():by_map[q['map_id']].append(q)
    for i,(mid,qs) in enumerate(by_map.items()):
        if all(q['id'] in answers for q in qs):continue
        oracle=Oracle(qs[0])
        for q in qs:
            distance=oracle.distances(q['goal']).get(tuple(q['start']));assert distance is not None
            answers[q['id']]=dict(optimal=distance,map_id=mid)
        save(path,answers)
        if (i+1)%10==0:print(json.dumps(dict(phase='oracle',maps=i+1,total=len(by_map))),flush=True)
    save(OUT/'oracle_metadata.json',dict(oracle='GM independent integer segment graph and Dijkstra',cases=len(answers),latest_run_seconds=time.perf_counter()-started))
    return answers

def bounds(method):
    if method in ('A_GM','A_OCT','S5') or method.startswith('S6_'):return 1.
    if method=='FOCAL_105' or method in MODEL_METHODS:return 1.05
    return None

def verify_result(q,method,r,optimal):
    if r['valid']:
        assert r['path'];assert all(all(type(v) in (int,float) and float(v).is_integer() for v in p) for p in r['path'])
        length=sum(math.dist(a,b) for a,b in zip(r['path'],r['path'][1:]));assert math.isclose(length,r['length'],abs_tol=1e-8,rel_tol=1e-10)
        assert r['length']>=optimal-1e-8-1e-10*optimal
        bound=bounds(method)
        r['bound_violation']=bool(bound and r['length']>bound*optimal+1e-8+1e-10*optimal)
    else:r['bound_violation']=False

def source_manifest():
    files=[HERE/n for n in ('common.py','collect.py','algorithms.py','train.py','programs.py','evaluate.py')]
    files += [ROOT/'reproduction'/n for n in ('core.py','candidate_core.py','next_search_core.py','next_geometry.py','next_refinement.py')]
    files += [GM/n for n in read(OUT/'source_snapshot.json')['files'] if n.endswith('.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def evaluate(stage,pilot=False):
    qs=cases(stage);truth=oracle_all() if not (OUT/'oracle.json').exists() else read(OUT/'oracle.json')
    methods=BASE if pilot else BASE+MODEL_METHODS
    selection=read(OUT/'program_selection.json') if (OUT/'program_selection.json').exists() else dict(weights=[.5,.3,.1,.1],pilot_fallback=True)
    if not pilot:assert not selection.get('pilot_fallback')
    models={};model_load_seconds=0
    if not pilot:models,model_load_seconds=models_load()
    repeat_count=1 if pilot else 5
    frozen=dict(methods=methods,repeats=repeat_count,program=selection,code=source_manifest(),protocol=sha(OUT/'protocol.json'),
                model_files={k:sha(OUT/'models'/f'seed{k}.pt') for k in (11,22,33)} if not pilot else {})
    signature=digest(frozen);run_name='pilot' if pilot else 'main'
    fpath=OUT/f'{run_name}_frozen.json'
    if fpath.exists():assert read(fpath)['signature']==signature,'Frozen code/settings changed; use a new run output'
    else:save(fpath,dict(signature=signature,frozen=frozen,created_at=now()))
    save(OUT/f'model_loading_{run_name}.json',dict(seconds=model_load_seconds,device='cpu'))
    warm=set();warmups=[]
    for qi,q in enumerate(qs):
        target=OUT/'evaluations'/run_name/q['stage']/f'{q["id"]}.json'
        if target.exists():assert read(target)['signature']==signature;continue
        first=raw(q);text=text_of(first);initial=diagnosis(q,text);revisions={};rawhash={'first':first['response_sha256']}
        if not initial['valid']:
            for variant in ('control','feedback'):
                record=raw(q,variant);revisions[variant]=text_of(record);rawhash[variant]=record['response_sha256']
        if models:
            size=(q['range_x'][1],q['range_y'][1])
            if size not in warm:
                import torch
                grid=candidate.Grid(q);x=torch.from_numpy(candidate.network_inputs(grid,q)[None])
                for name,model in models.items():
                    t=time.perf_counter()
                    with torch.no_grad():model(x)
                    warmups.append(dict(size=size,method=name,seconds=time.perf_counter()-t))
                warm.add(size)
        rows=[];paths={};signatures={}
        for repeat in range(repeat_count):
            order=methods[:];random.Random(f'20261029:{q["id"]}:{repeat}').shuffle(order)
            for method in order:
                r=run(q,method,text,revisions,models,selection['weights']);verify_result(q,method,r,truth[q['id']]['optimal'])
                path=r.pop('path');pathhash=digest(path);paths[pathhash]=path
                fixed={k:r.get(k) for k in ('valid','error','length','operation','successor_expansions','storage','api_calls')};fixed['path_sha256']=pathhash
                if method in signatures:assert signatures[method]==fixed,('repeat_changed_output',q['id'],method)
                signatures[method]=fixed
                rows.append(dict(method=method,repeat=repeat,path_sha256=pathhash,**r))
        save(target,dict(signature=signature,case=q,optimal=truth[q['id']]['optimal'],initial_proposal=initial,raw_hashes=rawhash,paths=paths,rows=rows))
        if (qi+1)%10==0 or qi==0 or qi+1==len(qs):
            status=dict(phase='evaluate',run=run_name,stage=stage,done=qi+1,total=len(qs),complete=False,updated_at=now())
            save(OUT/'evaluation_status.json',status);print(json.dumps(status),flush=True)
    save(OUT/f'warmups_{run_name}_{stage}.json',warmups)
    save(OUT/'evaluation_status.json',dict(phase='evaluated',run=run_name,stage=stage,done=len(qs),complete=True,updated_at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['oracle','pilot','all']);a=p.parse_args()
    if a.action=='oracle':oracle_all()
    elif a.action=='pilot':evaluate('pilot',pilot=True)
    else:
        for stage in ('public','transfer','synthetic'):evaluate(stage)
