"""Fixed OOF evaluation, five sequential repeats, no API during measurement."""
import math
import random
import socket
import time
from collections import defaultdict
from common import *
from algorithms import run, BASE, MODEL_METHODS, models_load, candidate
from experiments.K3.verification import Oracle

def bounds(method):
    if method in ('A_GM','A_OCT','S5') or method.startswith('S6_'):return 1.
    if method=='FOCAL_105' or method in MODEL_METHODS:return 1.05
    return None

def oracle_all():
    target=RUN/'oracle.json'
    if target.exists():return read(target)
    ans={};groups=defaultdict(list)
    for q in cases():groups[q['map_id']].append(q)
    for i,(mid,qs) in enumerate(groups.items()):
        oracle=Oracle(qs[0])
        for q in qs:
            d=oracle.distances(q['goal']).get(tuple(q['start']));assert d is not None
            ans[q['id']]=dict(optimal=d,map_id=mid)
        if (i+1)%10==0:print(json.dumps(dict(phase='oracle',maps=i+1)),flush=True)
    save(target,ans);return ans

def pilot():
    models,seconds=models_load();refs=read(RUN/'data/pilot_reference.json');n=0
    weights=read(OUT/'program_selection.json')['weights']
    for item in refs:
        q=item['query'];oracle=Oracle(q);text='Generated Path: '+json.dumps(item['waypoints'])
        for method in BASE+MODEL_METHODS:
            r=run(q,method,text,{'control':text,'feedback':text},models,weights)
            assert r['valid'],(q['id'],method,r['error']);oracle.check_path(q,r)
            w=bounds(method)
            if w:assert r['length']<=w*item['optimal']+1e-8,(method,q['id'])
            n+=1
    save(OUT/'pilot.json',dict(passed=True,checks=n,maps=30,model_loading_seconds=seconds,updated_at=now()))
    print(json.dumps(dict(fold=FOLD,pilot_checks=n,passed=True)),flush=True)

def freeze():
    assert read(RUN/'training_status.json')['complete']
    assert read(RUN/'collection_status.json')['complete']
    for fold in range(5):assert read(RUN/'folds'/f'fold{fold}'/'pilot.json')['passed']
    files=[HERE/n for n in ('common.py','algorithms.py','cv_candidate_core.py','train.py','collect.py','collect_all.py','collect_tail.py','programs.py','evaluate.py','prepare.py')]
    files+=list((ROOT/'reproduction').glob('*.py'))+list(GM.rglob('*.py'))
    files.append(HERE/'collect_middle.py')
    code={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    artifacts=list(RUN.glob('folds/fold*/models/seed*.pt'))+list(RUN.glob('folds/fold*/models/seed*.json'))
    artifacts+=list(RUN.glob('folds/fold*/models/training_data.json'))+list(RUN.glob('folds/fold*/programs/*.json'))
    artifacts+=list(RUN.glob('folds/fold*/program_selection.json'))+list(RUN.glob('raw/public/*.json'))
    artifacts+=list(RUN.glob('folds/fold*/raw/s6_programs/*.json'))
    frozen=dict(code=code,artifacts={p.relative_to(RUN).as_posix():sha(p) for p in artifacts},
                methods=BASE+MODEL_METHODS,repeats=5,protocol_sha256=sha(RUN/'protocol.json'),prompt_sha256=sha(PROMPT_FILE),
                fold_manifest_sha256=sha(RUN/'fold_manifest.json'))
    record=dict(signature=digest(frozen),frozen=frozen,created_at=now())
    dest=RUN/'frozen.json'
    if dest.exists():assert read(dest)['signature']==record['signature'],'Frozen settings changed'
    else:save(dest,record)

def evaluate():
    frozen=read(RUN/'frozen.json');signature=frozen['signature'];truth=read(RUN/'oracle.json')
    for rel,h in frozen['frozen']['code'].items():assert sha(ROOT/rel)==h,rel
    qs=cases('test');methods=BASE+MODEL_METHODS;selection=read(OUT/'program_selection.json');models,loading=models_load()
    save(OUT/'model_loading.json',dict(seconds=loading,device='cpu'));warmups=[];warmed=set()
    for qi,q in enumerate(qs):
        target=OUT/'evaluations'/f'{q["id"]}.json'
        if target.exists():assert read(target)['signature']==signature;continue
        first=raw(q);text=text_of(first);initial=diagnosis(q,text);revisions={};rawhash={'first':first['response_sha256']}
        if not initial['valid']:
            for v in ('control','feedback'):
                record=raw(q,v);revisions[v]=text_of(record);rawhash[v]=record['response_sha256']
        size=tuple(q['range_x']+q['range_y'])
        if size not in warmed:
            import torch
            graph=candidate.Grid(q);x=torch.from_numpy(candidate.network_inputs(graph,q)[None])
            for name,model in models.items():
                t=time.perf_counter()
                with torch.no_grad():model(x)
                warmups.append(dict(method=name,seconds=time.perf_counter()-t))
            warmed.add(size)
        rows=[];paths={};fixed={}
        for repeat in range(5):
            order=methods[:];random.Random(f'20261111:{FOLD}:{q["id"]}:{repeat}').shuffle(order)
            for method in order:
                r=run(q,method,text,revisions,models,selection['weights']);path=r.pop('path');ph=digest(path);paths[ph]=path
                w=bounds(method);r['bound_violation']=bool(r['valid'] and w and r['length']>w*truth[q['id']]['optimal']+1e-8)
                deterministic={k:r.get(k) for k in ('valid','error','length','operation','successor_expansions','storage','api_calls')};deterministic['path_sha256']=ph
                if method in fixed:assert fixed[method]==deterministic,(q['id'],method,'repeat changed')
                fixed[method]=deterministic;rows.append(dict(method=method,repeat=repeat,path_sha256=ph,**r))
        save(target,dict(signature=signature,case=q,optimal=truth[q['id']]['optimal'],initial_proposal=initial,raw_hashes=rawhash,paths=paths,rows=rows))
        if qi%10==0 or qi==len(qs)-1:
            st=dict(fold=FOLD,done=qi+1,total=len(qs),complete=False,updated_at=now());save(RUN/'evaluation_status.json',st);print(json.dumps(st),flush=True)
    save(OUT/'warmups.json',warmups);save(OUT/'evaluation_status.json',dict(complete=True,cases=len(qs),trials=len(qs)*110,updated_at=now()))

def deny_network(*a,**kw):raise RuntimeError('Network disabled during offline evaluation')

if __name__=='__main__':
    socket.socket.connect=deny_network;socket.create_connection=deny_network
    {'oracle':oracle_all,'pilot':pilot,'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
