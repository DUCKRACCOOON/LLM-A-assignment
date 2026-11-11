"""Small prespecified new-map smoke evaluation; not a replacement for Table 1."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import runpy
from pathlib import Path
from .api import completion, load_key
from .core import Geometry, search, llm_only
from .experiment import PROMPT_FILE, append, records, manifest
from .proposed import proposed_search
from .summarize import aggregate
from .verify import dijkstra


def make_cases():
    rng=random.Random(20261013)
    result=[]
    for mi in range(10):
        width,height=[(32,22),(52,32),(72,42)][mi%3]
        q=dict(range_x=[0,width+1],range_y=[0,height+1],horizontal_barriers=[],vertical_barriers=[])
        for _ in range(4):
            left=rng.randint(0,width-8)
            q['horizontal_barriers'].append([rng.randint(3,height-3),left,min(width,left+rng.randint(5,width//2))])
        for _ in range(4):
            low=rng.randint(0,height-6)
            q['vertical_barriers'].append([rng.randint(3,width-3),low,min(height,low+rng.randint(3,height//2))])
        geometry=Geometry(q)
        pi=0
        for _ in range(10000):
            start=[rng.randint(1,width-1),rng.randint(1,height-1)]
            goal=[rng.randint(1,width-1),rng.randint(1,height-1)]
            if start==goal or geometry.collision(start,start) or geometry.collision(goal,goal): continue
            candidate=dict(q,start=start,goal=goal)
            optimal=dijkstra(candidate,geometry)
            if optimal is None: continue
            result.append(dict(case_id=f'new_m{mi:02d}_p{pi}',map_index=mi,query=candidate,dijkstra_length=optimal))
            pi+=1
            if pi==3: break
        if pi!=3: raise RuntimeError('Could not generate three reachable queries')
    return result


def run(generate=False):
    out=Path('results/generalization')
    out.mkdir(parents=True,exist_ok=True)
    cases=make_cases()
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    model='gpt-4o-mini-2024-07-18'
    manifest_name=manifest(out,dict(action='new_map_smoke',model=model,seed=20261013,maps=10,pairs_per_map=3,
                                     sizes=[[32,22],[52,32],[72,42]],prompt='standard'))
    template=runpy.run_path(str(PROMPT_FILE))['gpt_prompt']['standard']
    existing=records(out/'generations.jsonl')
    known={r['case_id'] for r in existing}
    if generate:
        key=load_key()
        def execute(case):
            q=case['query']
            prompt=template.format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
            body=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=1000,store=False)
            return dict(case_id=case['case_id'],model=model,prompt_style='standard',request=body,
                        request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
                        manifest=manifest_name,**completion(body,key))
        with (out/'generations.jsonl').open('a',encoding='utf-8') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                for value in pool.map(execute,[c for c in cases if c['case_id'] not in known]):
                    append(f,value)
    responses={r['case_id']:r for r in records(out/'generations.jsonl')}
    evaluations=[]
    for case in cases:
        q,geo=case['query'],Geometry(case['query'])
        for method in ['astar','dwa','goal_only','llm_only','llm_astar','validated']:
            if method in ['astar','dwa','goal_only']:
                r=search(q,geo,method)
            else:
                response=responses.get(case['case_id'])
                if not response or response['status']!='ok': continue
                content=response['response']['choices'][0]['message'].get('content') or ''
                r=llm_only(q,geo,content) if method=='llm_only' else (
                    proposed_search(q,geo,content) if method=='validated' else search(q,geo,method,content))
            evaluations.append(dict(case_id=case['case_id'],map_index=case['map_index'],method=method,**r))
    baseline={r['case_id']:r for r in evaluations if r['method']=='astar'}
    for c in cases:
        assert abs(baseline[c['case_id']]['length']-c['dijkstra_length'])<1e-9
    (out/'evaluations.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in evaluations),encoding='utf-8')
    summary=[dict(method=m,**aggregate([r for r in evaluations if r['method']==m],baseline))
             for m in ['astar','dwa','goal_only','llm_only','llm_astar','validated']]
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--generate',action='store_true')
    run(p.parse_args().generate)
