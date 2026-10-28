import hashlib
import json
import os
import platform
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results' / 's1_s6_gm_20261028'
GM = HERE / 'vendor' / 'gm'
sys.path.insert(0, str(ROOT / 'reproduction'))
sys.path.insert(0, str(GM))
from reproduction import core as gm
from experiments.K3.heuristic import LowerBound
from reproduction.proposed import proposed_search

MODEL = 'gpt-4o-mini-2024-07-18'
PROMPT_FILE = GM / 'source_code/llmastar/pather/llm_a_star/prompt.py'
TEMPLATE = runpy.run_path(str(PROMPT_FILE))['gpt_prompt']['standard']
SPLITS = json.loads((ROOT/'results/candidates_v1/protocol.json').read_text())['splits']

def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def cases(stage=None):
    sources = {
        'public': GM/'experiments/K3/results/run_20261025/public_cases.json',
        'transfer': GM/'experiments/K4/results/run_20261025/transfer_cases.json',
        'synthetic': GM/'experiments/K3/results/run_20261025/synthetic_cases.json',
    }
    answer=[]
    for name, path in sources.items():
        for item in read(path):
            q=dict(item['query']); mid=item['map_index']
            q.update(id=item['case_id'], map_id=f'{name}:{mid}', map_index=mid,
                     pair_id=item.get('pair_index',int(item['case_id'].split('p')[-1])),stage=name,
                     split=next((k for k,v in SPLITS.items() if name=='public' and mid in v),'external'))
            answer.append(q)
    if stage=='pilot': return [q for q in answer if q['stage']=='public' and q['map_index']%10==0 and q['pair_id']<3]
    return [q for q in answer if stage is None or q['stage']==stage]

def payload(q):
    return dict(model=MODEL,messages=[dict(role='user',content=TEMPLATE.format(**q))],
                temperature=0,max_tokens=1000,store=False)

def diagnosis(q,text):
    nodes,error=gm.parse_output(text); geo=gm.Geometry(q)
    error=error or gm.validate_path(geo,nodes,q['start'],q['goal'])
    detail={"valid":error is None,"error":error}
    if error=='wrong_endpoints': detail.update(observed_start=nodes[0] if nodes else None,observed_goal=nodes[-1] if nodes else None,required_start=q['start'],required_goal=q['goal'])
    elif error=='outside_boundary': detail['first_bad_point']=next(dict(index=i,point=p) for i,p in enumerate(nodes) if not geo.inside(p))
    elif error=='collision':
        for i,(a,b) in enumerate(zip(nodes,nodes[1:])):
            if geo.collision(a,b):
                detail['first_bad_segment']=dict(index=i,segment=[a,b]); break
        # Identify an actual first offending obstacle using the same GM geometry.
        if 'first_bad_segment' in detail:
            a,b=detail['first_bad_segment']['segment']
            for kind in ('horizontal_barriers','vertical_barriers'):
                for j,wall in enumerate(q[kind]):
                    single=dict(q,horizontal_barriers=[],vertical_barriers=[]); single[kind]=[wall]
                    if gm.Geometry(single).collision(a,b):
                        detail['first_bad_wall']=dict(kind=kind,index=j,wall=wall); break
                if 'first_bad_wall' in detail: break
    return detail

def correction(q,text,feedback):
    body=payload(q)
    report=json.dumps(diagnosis(q,text),ensure_ascii=False,sort_keys=True) if feedback else 'Not provided.'
    body['messages'][0]['content'] += ('\n\nPrevious response:\n'+text+'\n\nExternal validation report:\n'+report+
      '\n\nReview the previous response and provide one revised answer for the same original problem. '
      'Follow the original output instructions and include the final coordinate list. '
      'Do not change the start, goal, map boundaries, or barriers.')
    return body

def text_of(record): return record['response']['choices'][0]['message'].get('content') or ''
def raw_path(q,variant): return OUT/'raw'/q['stage']/f'{q["id"]}-{variant}.json'
def raw(q,variant='first'):
    record=read(raw_path(q,variant))
    assert record['request_sha256']==digest(record['request'])
    assert record['response_sha256']==digest(record['response'])
    return record

def environment():
    import numpy, scipy, shapely, torch
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                processor=platform.processor(),cpu_count=os.cpu_count(),numpy=numpy.__version__,
                scipy=scipy.__version__,shapely=shapely.__version__,torch=torch.__version__,device='cpu')

def prepare():
    assert sys.version_info[:2]==(3,11), 'Python 3.11 required'
    OUT.mkdir(parents=True,exist_ok=True)
    allq=cases(); hashes={}; bysplit={}
    for q in allq:
        geo={k:q[k] for k in ('range_x','range_y','horizontal_barriers','vertical_barriers')}
        # Canonical geometry checks barrier-order-independent duplicates.
        for k in ('horizontal_barriers','vertical_barriers'): geo[k]=sorted(geo[k])
        h=digest(geo); hashes[q['map_id']]=h; bysplit.setdefault(q['split'],set()).add(h)
    for a,b in [('train','test'),('train','external'),('validation','test'),('validation','external')]:
        assert not bysplit[a]&bysplit[b], f'Map overlap: {a}/{b}'
    for s in ('public','transfer','synthetic'): save(OUT/'data'/f'{s}.json',cases(s))
    save(OUT/'data/splits.json',SPLITS)
    save(OUT/'data/geometry_hashes.json',hashes)
    protocol=dict(version='gm-py311-v1',created_at=now(),model=MODEL,prompt='standard',temperature=0,max_tokens=1000,
                  prompt_file_sha256=sha(PROMPT_FILE),template_sha256=hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                  timing_repeats=5,order_bootstrap_seed=20261029,bootstrap_repeats=2000,
                  s4_seeds=[11,22,33],s4_weight=1.05,s5_variant='sparse',s6_proposals=12,
                  stages={s:len(cases(s)) for s in ('public','transfer','synthetic')},
                  input_sha256={s:sha(OUT/'data'/f'{s}.json') for s in ('public','transfer','synthetic')},
                  source_gm_commit='c56eb83b7df4be06d402daba36e0a2b45afa950b',environment=environment())
    dest=OUT/'protocol.json'
    if dest.exists():
        previous=read(dest)
        for k in ('model','template_sha256','input_sha256','timing_repeats','s4_weight'): assert previous[k]==protocol[k]
    else: save(dest,protocol)
    print(json.dumps({'prepared':len(allq),'python':platform.python_version(),'overlap_check':'passed'}),flush=True)

if __name__=='__main__': prepare()
