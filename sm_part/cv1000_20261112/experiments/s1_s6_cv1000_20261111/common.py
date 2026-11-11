"""Isolated five-fold experiment; no writes to the October experiment."""
import hashlib
import json
import os
import platform
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = ROOT/'results/s1_s6_cv1000_20261111'
RUN = PLAN/'runs/run_20261111_01'
FOLD = int(os.environ.get('CV_FOLD', '0'))
OUT = RUN/'folds'/f'fold{FOLD}'
GM = ROOT/'experiments/s1_s6_gm_20261028/vendor/gm'
sys.path.insert(0, str(ROOT/'reproduction'))
sys.path.insert(0, str(GM))
from reproduction import core as gm
from experiments.K3.heuristic import LowerBound
from reproduction.proposed import proposed_search

MODEL = 'gpt-4o-mini-2024-07-18'
PROMPT_FILE = RUN/'prompts/standard.txt'
TEMPLATE = PROMPT_FILE.read_text(encoding='utf-8') if PROMPT_FILE.exists() else ''
MANIFEST = json.loads((PLAN/'fold_manifest.json').read_text(encoding='utf-8'))
PART = MANIFEST['folds'][FOLD]
SPLITS = {k:PART[k+'_maps'] for k in ('train','validation','test')}

def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try:temp.replace(path);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def cases(stage=None):
    if stage=='pilot': return read(RUN/'data/pilot.json')
    ans=[]
    for item in read(GM/'experiments/K3/results/run_20261025/public_cases.json'):
        mid=item['map_index'];q=dict(item['query'])
        q.update(id=item['case_id'],map_id=f'public:{mid}',map_index=mid,pair_id=item['pair_index'],
                 stage='public',fold_id=FOLD,split=next(k for k,v in SPLITS.items() if mid in v))
        ans.append(q)
    return [q for q in ans if q['split']=='test'] if stage=='test' else ans

def payload(q):
    assert TEMPLATE, 'Prepare the independent examples first'
    return dict(model=MODEL,messages=[dict(role='user',content=TEMPLATE.format(**q))],temperature=0,max_tokens=1000,store=False)

def diagnosis(q,text):
    nodes,error=gm.parse_output(text);geo=gm.Geometry(q)
    error=error or gm.validate_path(geo,nodes,q['start'],q['goal'])
    detail=dict(valid=error is None,error=error)
    if error=='wrong_endpoints': detail.update(observed_start=nodes[0] if nodes else None,observed_goal=nodes[-1] if nodes else None,required_start=q['start'],required_goal=q['goal'])
    elif error=='outside_boundary': detail['first_bad_point']=next(dict(index=i,point=p) for i,p in enumerate(nodes) if not geo.inside(p))
    elif error=='collision':
        for i,(a,b) in enumerate(zip(nodes,nodes[1:])):
            if geo.collision(a,b): detail['first_bad_segment']=dict(index=i,segment=[a,b]);break
        if 'first_bad_segment' in detail:
            a,b=detail['first_bad_segment']['segment']
            for kind in ('horizontal_barriers','vertical_barriers'):
                for j,wall in enumerate(q[kind]):
                    single=dict(q,horizontal_barriers=[],vertical_barriers=[]);single[kind]=[wall]
                    if gm.Geometry(single).collision(a,b): detail['first_bad_wall']=dict(kind=kind,index=j,wall=wall);break
                if 'first_bad_wall' in detail:break
    return detail

def correction(q,text,feedback):
    body=payload(q);report=json.dumps(diagnosis(q,text),ensure_ascii=False,sort_keys=True) if feedback else 'Not provided.'
    body['messages'][0]['content'] += ('\n\nPrevious response:\n'+text+'\n\nExternal validation report:\n'+report+
      '\n\nReview the previous response and provide one revised answer for the same original problem. '
      'Follow the original output instructions and include the final coordinate list. '
      'Do not change the start, goal, map boundaries, or barriers.')
    return body

def text_of(record):return record['response']['choices'][0]['message'].get('content') or ''
def raw_path(q,variant):
    base=RUN if q['stage']=='public' else OUT
    return base/'raw'/q['stage']/f'{q["id"]}-{variant}.json'
def raw(q,variant='first'):
    record=read(raw_path(q,variant))
    assert record['request_sha256']==digest(record['request'])
    assert record['response_sha256']==digest(record['response'])
    return record

def environment():
    import numpy,scipy,shapely,torch
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),
                numpy=numpy.__version__,scipy=scipy.__version__,shapely=shapely.__version__,torch=torch.__version__,device='cpu')
