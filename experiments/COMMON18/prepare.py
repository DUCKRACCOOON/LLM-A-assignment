"""Freeze data, validate backends, and collect bounded shared LLM responses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import subprocess
import sys
import time

import numpy as np
from reproduction.api import completion
from reproduction.core import Geometry,parse_output,validate_path
from experiments.K3.run import public_cases,sha
from experiments.K3.verification import Oracle
from .backends import ROOT,PRIMARY,METHODS,load_model,run,content,cc,topo,nc

MODEL='gpt-4o-mini-2024-07-18';SEED=20261028


def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def rows(path):
    if not path.exists():return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def split():
    demos=[0,2,3,4,5];other=[i for i in range(100) if i not in demos];random.Random(261014).shuffle(other)
    return dict(train=demos+other[:45],validation=other[45:55],calibration=other[55:75],test=other[75:])


def freeze(out):
    if (out/'protocol.json').exists():return
    out.mkdir(parents=True,exist_ok=False);division=split();allcases=public_cases()
    selected={stage:[c for c in allcases if c['map_index'] in division[stage]] for stage in ('calibration','test')}
    for stage,cases in selected.items():save(out/(stage+'_cases.json'),cases)
    save(out/'split.json',division)
    sources=list(Path(__file__).parent.rglob('*.py'))+[Path(__file__).parent/'PROTOCOL.md']
    save(out/'protocol.json',dict(split=division,model=MODEL,seed=SEED,temperature=0,max_tokens=1000,
        ranking_max_tokens=256,request_limit=1600,input_token_limit=5000000,output_token_limit=1600000,
        concurrent_requests=4,primary=list(PRIMARY),methods=list(METHODS),repeats=3,
        baseline='LLMA',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        data_sha256=sha(ROOT/'source_code/dataset/environment_50_30.json'),
        checkpoint_sha256=sha(Path(__file__).parent/'assets/seed11.pt'),
        created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
        numpy=np.__version__,timing='local fresh, shared same-collection LLM latency; total is reconstructed'))


def read_key():
    value=os.environ.get('OPENAI_API_KEY')
    if value:return value
    # Original dirty worktree remains untouched. No credential is copied.
    path=ROOT.parent/'LLM-A-assignment/.env'
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('OPENAI_API_KEY='):return line.split('=',1)[1].strip().strip('\"\'')
    raise RuntimeError('No configured API credential; no requests made')


def body(prompt,max_tokens=1000):
    return dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=max_tokens,store=False)


def check_geometry(q):
    oracle=Oracle(q,crosscheck=True);grid=cc.Grid(q);native=nc.Grid(q);counts=0
    for n,edges in oracle.graph.items():
        a={(tuple(map(int,grid.xy[v])),float(grid.costs[k])) for k,v in enumerate(grid.edges[grid.index(n)]) if v>=0}
        b={(native.point(v),c) for v,c in native.neighbors(native.index(n))}
        assert a==set(edges)==b,(n,a,set(edges),b);counts+=len(edges)
    return oracle,counts


def preflight(out):
    if (out/'preflight.json').exists():return
    test=json.loads((out/'test_cases.json').read_text(encoding='utf-8'))
    original={r['case_id']:r for r in rows(ROOT/'results/main/generations.jsonl') if r['model']==MODEL and r['prompt_style']=='standard'}
    ctx=dict(quantile=6.5,model=load_model(),untrained=load_model(True));checks=0;geometry_checks=0
    for c in test:
        if c['pair_index']!=0:continue
        oracle,n=check_geometry(c['query']);geometry_checks+=n
    for c in test[:3]:
        q=c['query'];oracle=Oracle(q);opt=oracle.distances(q['goal'])[tuple(q['start'])]
        r=original[c['case_id']];reply={k:r for k in ('standard','cot','repe','correction')}
        reply['ranking']=dict(r,response={'choices':[{'message':{'content':'{"ranking":[0,1]}'}}]})
        for method in METHODS:
            result=run(method,q,reply,ctx)
            assert result['valid'],(c['case_id'],method,result['error']);oracle.check_path(q,result)
            if result['bound']:assert result['length']<=opt*result['bound']+1e-7
            checks+=1
    save(out/'preflight.json',dict(passed=True,methods=len(METHODS),smoke_results=checks,
        independently_checked_directed_edges=geometry_checks,geometry_test_maps=20,new_api_calls=0))
    print('Preflight passed: common geometry, all method paths and cost bounds.',flush=True)


def collect_batch(tasks,out,key):
    existing=rows(out/'generations.jsonl');cache={r['request_sha256']:r for r in existing}
    requested=sum(1 for r in existing);input_used=sum(r.get('response',{}).get('usage',{}).get('prompt_tokens',0) for r in existing)
    output_used=sum(r.get('response',{}).get('usage',{}).get('completion_tokens',0) for r in existing)
    pending=[t for t in tasks if digest(t['request']) not in cache]
    random.Random(SEED).shuffle(pending)
    print(f'Collection: {len(pending)} pending, {requested} stored requests',flush=True)
    def invoke(task):
        started=datetime.now(timezone.utc).isoformat();value=completion(task['request'],key=key,attempts=1)
        return dict(task,request_sha256=digest(task['request']),started_utc=started,**value)
    with ThreadPoolExecutor(max_workers=4) as pool,(out/'generations.jsonl').open('a',encoding='utf-8',newline='\n') as f:
        for offset in range(0,len(pending),4):
            batch=pending[offset:offset+4]
            assert requested+len(batch)<=1600,'Request budget would be exceeded'
            # Conservative reservation before dispatch: UTF-8 bytes bound the
            # prompt token count; completion reservation uses max_tokens.
            reserve=sum(len(json.dumps(t['request']).encode())+100 for t in batch)
            assert input_used+reserve<=5000000 and output_used+sum(t['request']['max_tokens'] for t in batch)<=1600000
            results=list(pool.map(invoke,batch));failure=False
            for r in results:
                f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());requested+=1
                cache[r['request_sha256']]=r
                if r['status']!='ok':failure=True
                usage=r.get('response',{}).get('usage',{});input_used+=usage.get('prompt_tokens',0);output_used+=usage.get('completion_tokens',0)
            save(out/'collection_status.json',dict(requests=requested,prompt_tokens=input_used,completion_tokens=output_used,
                phase=batch[0]['stage'],finished_tasks=offset+len(batch),pending_phase_total=len(pending),failed=failure))
            if failure:raise RuntimeError('API request failed; saved without exposing credentials; no further requests dispatched')
            if (offset+len(batch))%20==0 or offset+len(batch)==len(pending):
                print(f'Collected {requested} total requests; phase {offset+len(batch)}/{len(pending)}',flush=True)
    return cache


def correction_prompt(q,text,template):
    parsed,error=parse_output(text);geo=Geometry(q)
    reason=error or validate_path(geo,parsed,q['start'],q['goal'])
    details=dict(valid=reason is None,error=reason)
    if reason=='collision':
        details['first_segment']=next([list(a),list(b)] for a,b in zip(parsed,parsed[1:]) if geo.collision(a,b))
    original=template.format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
    return reason,(original+'\n\nPrevious response:\n'+text+'\n\nExternal validation report:\n'+json.dumps(details,ensure_ascii=False,sort_keys=True)+
        '\n\nReview the previous response and provide one revised answer for the same original problem. Follow the original output instructions and include the final coordinate list. Do not change the start, goal, map boundaries, or barriers.')


def collect(out):
    if (out/'collection_complete.json').exists():return
    templates=runpy.run_path(str(ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt'];tasks=[]
    stages={s:json.loads((out/(s+'_cases.json')).read_text(encoding='utf-8')) for s in ('calibration','test')}
    for stage,cases in stages.items():
        for c in cases:
            q=c['query']
            for style in ('standard','cot','repe'):
                prompt=templates[style].format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
                tasks.append(dict(stage=stage,case_id=c['case_id'],kind=style,request=body(prompt)))
    key=read_key();cache=collect_batch(tasks,out,key)
    by_case={}
    for task in tasks:by_case.setdefault(task['case_id'],{})[task['kind']]=cache[digest(task['request'])]
    extra=[];topologies=[]
    for c in stages['test']:
        q=c['query'];text=content(by_case[c['case_id']]['standard']);reason,prompt=correction_prompt(q,text,templates['standard'])
        if reason:extra.append(dict(stage='test',case_id=c['case_id'],kind='correction',request=body(prompt)))
        proposal,graph=topo.routes(q,k=2)
        if len(proposal['candidates'])>=2:
            extra.append(dict(stage='test',case_id=c['case_id'],kind='ranking',request=body(topo.prompt(q,proposal),256)))
        topologies.append(dict(case_id=c['case_id'],proposal=proposal))
    save(out/'topology_tasks.json',topologies);cache=collect_batch(extra,out,key)
    for task in extra:by_case[task['case_id']][task['kind']]=cache[digest(task['request'])]
    save(out/'response_index.json',{cid:{kind:r['request_sha256'] for kind,r in rs.items()} for cid,rs in by_case.items()})
    allrows=rows(out/'generations.jsonl');assert all(r['status']=='ok' for r in allrows)
    save(out/'collection_complete.json',dict(requests=len(allrows),new_api_calls=len(allrows),
        prompt_tokens=sum(r['response'].get('usage',{}).get('prompt_tokens',0) for r in allrows),
        completion_tokens=sum(r['response'].get('usage',{}).get('completion_tokens',0) for r in allrows),
        actual_model_names=sorted({r['response'].get('model','unknown') for r in allrows}),
        system_fingerprints=sorted({str(r['response'].get('system_fingerprint')) for r in allrows}),
        completed_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--collect',action='store_true');a=p.parse_args()
    out=a.output.resolve();freeze(out);preflight(out)
    if a.collect:collect(out)


if __name__=='__main__':main()
