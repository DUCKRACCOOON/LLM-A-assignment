"""Fresh, bounded, resumable API collection; never prints credential contents."""
import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from common import *

def key_value():
    from api_client import read_key
    return read_key(r'C:\api\gpt.txt')

def request(body=None,route='chat/completions'):
    key=key_value(); attempts=[]
    for attempt in range(3):
        req=urllib.request.Request('https://api.openai.com/v1/'+route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        t=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=120) as res:
                data=json.load(res); seconds=time.perf_counter()-t
                attempts.append(dict(attempt=attempt+1,status=200,seconds=seconds))
                return dict(response=data,latency_seconds=seconds,request_seconds=sum(a['seconds'] for a in attempts),
                            attempts=attempts,request_id=res.headers.get('x-request-id'))
        except urllib.error.HTTPError as exc:
            try: error=json.loads(exc.read()).get('error',{})
            except Exception: error={}
            code=error.get('code'); attempts.append(dict(attempt=attempt+1,status=exc.code,code=code,seconds=time.perf_counter()-t))
            if (exc.code==429 and code!='insufficient_quota' or exc.code>=500) and attempt<2:
                time.sleep(2**(attempt+1)); continue
            raise RuntimeError(f'OpenAI HTTP {exc.code}; code={code}; type={error.get("type")}') from None
        except urllib.error.URLError:
            raise RuntimeError('OpenAI connection failed; credential details suppressed') from None

def obtain(q,variant,body):
    dest=raw_path(q,variant)
    if dest.exists():
        record=read(dest); assert record['request_sha256']==digest(body)
        assert record['response_sha256']==digest(record['response']); return record,False
    record=dict(case_id=q['id'],stage=q['stage'],variant=variant,created_at=now(),request=body,request_sha256=digest(body),**request(body))
    record['response_sha256']=digest(record['response']); save(dest,record)
    return record,True

def one(q):
    first,new=obtain(q,'first',payload(q)); calls=int(new); diag=diagnosis(q,text_of(first))
    if not diag['valid']:
        variants=[('control',False),('feedback',True)]
        if int(hashlib.sha256(q['id'].encode()).hexdigest()[:8],16)%2: variants.reverse()
        for variant,feedback in variants:
            _,new=obtain(q,variant,correction(q,text_of(first),feedback)); calls+=int(new)
    return dict(id=q['id'],new_calls=calls,initial_valid=diag['valid'])

def collect(stage,workers):
    qs=cases(stage); started=now(); done=[]
    save(OUT/'collection_status.json',dict(phase='collect',stage=stage,started_at=started,complete=False))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            it=iter(qs); pending={}
            for _ in range(workers):
                q=next(it,None)
                if q: pending[pool.submit(one,q)]=q
            while pending:
                ready,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in ready:
                    q=pending.pop(f)
                    try: done.append(f.result())
                    except Exception:
                        for other in pending: other.cancel()
                        raise
                    if len(done)%10==0 or len(done)==len(qs):
                        status=dict(phase='collect',stage=stage,done=len(done),total=len(qs),new_calls=sum(r['new_calls'] for r in done),updated_at=now(),complete=False)
                        save(OUT/'collection_status.json',status); print(json.dumps(status),flush=True)
                    q=next(it,None)
                    if q: pending[pool.submit(one,q)]=q
    except Exception as e:
        save(OUT/'collection_status.json',dict(phase='failed',stage=stage,error=str(e),done=len(done),complete=False,updated_at=now())); raise
    save(OUT/'collection_status.json',dict(phase='collected',stage=stage,done=len(done),complete=True,updated_at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['probe','collect']);p.add_argument('--stage',default='pilot');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.action=='probe':
        result=request(route='models/'+MODEL)['response']
        out=dict(model_id=result['id'],accessible=True,paid_generation_performed=False,checked_at=now());save(OUT/'model_access.json',out);print(json.dumps(out))
    else: collect(None if a.stage=='all' else a.stage,a.workers)
