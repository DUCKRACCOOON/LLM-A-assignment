"""Resumable GPT paper reproduction; python reproduction/run.py --help."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from api_client import read_key, request
from core import ROOT, VENDOR, PROMPTS, load_cases, prompt_for, parse_path, World, search, validate, path_length, scale_case

def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

def payload(q,kind,stop_newline=False):
    body={'model':'gpt-3.5-turbo-0125','messages':[{'role':'user','content':prompt_for(q,kind)}],
          'temperature':0,'max_tokens':1000}
    if stop_newline: body['stop']=['\n']
    return body

def cases_for(stage):
    cases=load_cases()
    if stage=='pilot':
        # Separate synthetic development cases on held-out simple geometries.
        return [dict(id=f'dev-{i:02d}',map_id=-1-i,pair_id=0,range_x=[0,51],range_y=[0,31],
                     start=[3+i,4],goal=[43-i,25],horizontal_barriers=[[12,0,24+i]],
                     vertical_barriers=[[33,17,30]]) for i in range(10)]
    if stage=='scaling':
        # Paired sample: same ten problems at all scales, sampled before any results.
        chosen=random.Random(240702511).sample(cases,10)
        return [scale_case(q,s) for q in chosen for s in range(1,11)]
    return cases

def collect(args):
    key=read_key(args.key_file)
    cases=cases_for(args.stage)
    if args.limit: cases=cases[:args.limit]
    kinds=['standard'] if args.stage=='scaling' else ['standard','cot','repe']
    out=ROOT/'results'/args.stage/('newline_audit' if args.stop_newline else 'raw')
    tasks=[(q,k) for q in cases for k in kinds]
    manifest={'stage':args.stage,'case_count':len(cases),'kinds':kinds,'stop_newline':args.stop_newline,
              'model':'gpt-3.5-turbo-0125','temperature':0,'max_tokens':1000,
              'vendor_commit':subprocess.check_output(['git','-c',f'safe.directory={VENDOR.as_posix()}',
                              '-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),
              'data_sha256':hashlib.sha256((VENDOR/'dataset/environment_50_30.json').read_bytes()).hexdigest(),
              'prompts_sha256':digest(PROMPTS),'cases':cases}
    save(out.parent/('newline_manifest.json' if args.stop_newline else 'manifest.json'),manifest)
    def one(task):
        q,kind=task; body=payload(q,kind,args.stop_newline); path=out/f'{q["id"]}-{kind}.json'
        if path.exists():
            stored=json.loads(path.read_text(encoding='utf-8'))
            if stored['payload_sha256']!=digest(body): raise RuntimeError('Cached request mismatch')
            return 'cached'
        result=request(key,'chat/completions',body)
        record={'case_id':q['id'],'prompt_kind':kind,'created_utc':datetime.now(timezone.utc).isoformat(),
                'payload':body,'payload_sha256':digest(body),**result}
        save(path,record)
        return 'new'
    # Bounded pool: only workers outstanding; stop scheduling after permanent error.
    counts={'new':0,'cached':0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator=iter(tasks); pending={}
        for _ in range(args.workers):
            task=next(iterator,None)
            if task is not None: pending[pool.submit(one,task)]=task
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                try: counts[f.result()]+=1
                except Exception:
                    for p in pending: p.cancel()
                    raise
                if sum(counts.values())%25==0: print(json.dumps({'completed':sum(counts.values()),'total':len(tasks),**counts}),flush=True)
                task=next(iterator,None)
                if task is not None: pending[pool.submit(one,task)]=task
    print(json.dumps(counts),flush=True)

def evaluate(args):
    cases=cases_for(args.stage)
    if args.limit: cases=cases[:args.limit]
    out=ROOT/'results'/args.stage
    worlds={}; rows=[]
    kinds=['standard'] if args.stage=='scaling' else ['standard','cot','repe']
    for index,q in enumerate(cases):
        world_key=(q['map_id'],q.get('scale',1))
        if world_key not in worlds: worlds[world_key]=World(q)
        world=worlds[world_key]
        p=out/'evaluation'/f'{q["id"]}.json'
        raw_records={}
        for kind in kinds:
            raw=out/'raw'/f'{q["id"]}-{kind}.json'
            if raw.exists(): raw_records[kind]=json.loads(raw.read_text(encoding='utf-8'))
        signature=digest({'q':q,'raw':raw_records,'core_sha256':hashlib.sha256((ROOT/'reproduction/core.py').read_bytes()).hexdigest()})
        if p.exists():
            old=json.loads(p.read_text(encoding='utf-8'))
            if old.get('signature')==signature:
                rows.extend(old['rows']); continue
        results=[]
        methods=['astar'] if args.stage=='scaling' else ['astar','dynamic_weighted']
        for method in methods:
            r=search(q,method,world=world)
            results.append({'case_id':q['id'],'method':method,'prompt':None,**r})
        for kind,record in raw_records.items():
            content=record['response']['choices'][0]['message']['content'] or ''
            path,parse_error=parse_path(content)
            valid,reason=validate(q,path,world)
            results.append({'case_id':q['id'],'method':'llm_only','prompt':kind,'operation':None,'storage':None,
                            'length':path_length(path) if path else None,'valid':valid,'failure':parse_error or reason,
                            'path':path,'parse_error':parse_error})
            r=search(q,'llm_astar',path,world)
            results.append({'case_id':q['id'],'method':'llm_astar','prompt':kind,'parse_error':parse_error,**r})
        save(p,{'signature':signature,'case':q,'rows':results}); rows.extend(results)
        if (index+1)%25==0: print(f'evaluated {index+1}/{len(cases)}',flush=True)
        # Larger scaling graphs are used once; avoid retaining hundreds of MB.
        if args.stage=='scaling': worlds.clear()
    save(out/'rows.json',rows)
    print(f'Saved {len(rows)} result rows',flush=True)

def audit():
    cases=load_cases()
    overlaps={kind:[q['id'] for q in cases if
        f'Start Point: {q["start"]}\nGoal Point: {q["goal"]}\nHorizontal Barriers: {q["horizontal_barriers"]}\nVertical Barriers: {q["vertical_barriers"]}' in PROMPTS[kind]] for kind in PROMPTS}
    result={'maps':len({q['map_id'] for q in cases}),'cases':len(cases),
            'ranges':sorted({(tuple(q['range_x']),tuple(q['range_y'])) for q in cases}),
            'exact_demonstration_overlap':overlaps,'unique_case_payloads':len({digest({k:v for k,v in q.items() if k not in ('id','map_id','pair_id')}) for q in cases})}
    save(ROOT/'results/dataset_audit.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','collect','evaluate'])
    parser.add_argument('--stage',choices=['pilot','main','scaling'],default='pilot')
    parser.add_argument('--key-file',default=r'C:\api\gpt.txt')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stop-newline',action='store_true')
    args=parser.parse_args()
    try:
        if args.action=='audit': audit()
        elif args.action=='collect': collect(args)
        else: evaluate(args)
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr); raise SystemExit(1)
