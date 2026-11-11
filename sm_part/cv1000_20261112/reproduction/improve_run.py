"""Sequential GPT improvement experiment. Secrets remain only in memory.

Run: python reproduction/improve_run.py all --key-file PATH (local), or
     python reproduction/improve_run.py all --detach --gpu 1 (server getpass).
"""
import argparse
import concurrent.futures
import getpass
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from core import ROOT, World, search, parse_path, validate, path_length
from run import cases_for, payload, save, digest
from api_client import read_key, request
from improve_core import refine, diagnosis, correction_payload, delayed_search

OUT = ROOT/'results'/'improvements'
MODEL='gpt-3.5-turbo-0125'
PROTOCOL={
 'version':'2026-10-14-v1', 'model':MODEL, 'temperature':0, 'max_tokens':1000,
 'primary_metrics':['operation','storage','length','valid'],
 'data':'environment_50_30.json', 'main_cases':1000, 'pilot_cases':10,
 'kinds':['standard','cot','repe'], 'scaling_kinds':['standard'],
 'scaling_sample_seed':240702511, 'scaling_factors':list(range(1,11)),
 'e1':'original targets -> chronological loop erasure -> farthest collision-free shortcut -> validate or no internal targets',
 'e2':'if first raw path invalid: control and feedback each get exactly one revision; second answer adopted unconditionally',
 'e2_feedback':'first original-validator error; first offending coordinate/segment/barrier in original order',
 'e3':'check after each max(10,ceil(start-goal distance)) pops; progress of best popped distance <0.05 initial distance -> call once; preserve search state',
 'e3_prompt':'original start/goal/map and original prompt; no current-state additions',
 'sequence':'pilot verification; E1 main+scaling; E2 main+scaling; E3 main+scaling; verify and report',
 'shared_responses':'new first responses shared across all arms; E3 acquisition trace prepared during first-response collection, final E3 replay evaluated last',
 'environment_change':'user requested Linux GPU server instead of previous Windows PC; rerun all controls and verify legacy classical metrics',
 'gpu_role':'GPT inference uses OpenAI API; unchanged Python search runs on CPU; selected idle GPU is visible but no GPU kernel is introduced',
 'timing':'secondary only; full E3 acquisition trace recorded; other total times are observed shared API latency plus local processing, not independent online timing trials',
 'failed_outputs':'never regenerate for quality; existing transport retry rules only; no best-of selection',
}

def now(): return datetime.now(timezone.utc).isoformat()
def text_of(record): return record['response']['choices'][0]['message']['content'] or ''
def kinds(stage): return ['standard'] if stage=='scaling' else PROTOCOL['kinds']
def source_hashes():
    names=['core.py','run.py','api_client.py','improve_core.py','improve_run.py','improve_report.py','test_improvements.py']
    return {n:hashlib.sha256((ROOT/'reproduction'/n).read_bytes()).hexdigest() for n in names}
def protocol_signature(): return digest({'protocol':PROTOCOL,'source':source_hashes()})
def row(q,method,kind,result,**extra): return {'case_id':q['id'],'map_id':q['map_id'],'method':method,'prompt':kind,**result,**extra}
def raw_row(q,kind,record,method):
    path,error=parse_path(text_of(record)); good,reason=validate(q,path)
    return row(q,method,kind,{'operation':None,'storage':None,'length':path_length(path) if path else None,
        'valid':good,'failure':error or reason,'path':path,'parse_error':error})

class Runner:
    def __init__(self,key,workers=8):
        self.key=key; self.workers=workers; self.lock=threading.Lock()
        self.state={'started_at':now(),'phase':'initializing','complete':False,'new_requests':0,'cached_requests':0}
    def status(self,**kwargs):
        with self.lock:
            self.state.update(kwargs); self.state['updated_at']=now(); save(OUT/'status.json',self.state)
    def api(self,stage,q,kind,variant,body):
        path=OUT/stage/'raw'/f'{q["id"]}-{kind}-{variant}.json'
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            if old['payload_sha256']!=digest(body): raise RuntimeError('Cached payload mismatch')
            if old['response_sha256']!=digest(old['response']): raise RuntimeError('Cached response hash mismatch')
            with self.lock: self.state['cached_requests']+=1
            return old
        result=request(self.key,'chat/completions',body)
        if result['response'].get('model')!=MODEL: raise RuntimeError('Unexpected returned model')
        record={'case_id':q['id'],'prompt_kind':kind,'variant':variant,'created_utc':now(),
            'payload':body,'payload_sha256':digest(body),**result,'response_sha256':digest(result['response'])}
        save(path,record)
        with self.lock: self.state['new_requests']+=1
        return record
    def batch(self,stage,phase,fn):
        qs=cases_for(stage); done=0
        self.status(phase=phase,stage=stage,completed_cases=0,total_cases=len(qs))
        workers=min(self.workers,3) if stage in ('pilot','scaling') else self.workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            tasks=iter(qs); pending=set()
            for _ in range(workers):
                q=next(tasks,None)
                if q is not None: pending.add(pool.submit(fn,stage,q))
            while pending:
                ready,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                try:
                    for future in ready:
                        future.result(); done+=1
                        self.status(completed_cases=done)
                        if done%10==0 or done==len(qs): print(json.dumps({'phase':phase,'stage':stage,'done':done,'n':len(qs)}),flush=True)
                        q=next(tasks,None)
                        if q is not None: pending.add(pool.submit(fn,stage,q))
                except Exception:
                    for future in pending: future.cancel()
                    raise
    def valid_saved(self,path):
        if not path.exists(): return False
        value=json.loads(path.read_text(encoding='utf-8'))
        if value['protocol_signature']!=protocol_signature(): raise RuntimeError('Saved evaluation protocol mismatch')
        return True
    def common(self,stage,q):
        dest=OUT/stage/'e1'/f'{q["id"]}.json'
        if self.valid_saved(dest): return
        rows=[]; trace={}; references=[]
        for name,method in [('A','astar'),('D','dynamic_weighted'),('N','llm_astar')]:
            if stage=='scaling' and name=='D': continue
            rows.append(row(q,name,None,search(q,method)))
        for kind in kinds(stage):
            holder={}
            def fetch():
                holder['record']=self.api(stage,q,kind,'first',payload(q,kind))
                return parse_path(text_of(holder['record']))[0]
            e3=delayed_search(q,fetch)
            called=e3['api_calls']>0
            record=holder.get('record') or self.api(stage,q,kind,'first',payload(q,kind))
            raw,error=parse_path(text_of(record))
            references.append(record['response_sha256'])
            trace[kind]={'result':e3,'first_response_sha256':record['response_sha256'],
                         'first_request_for_e3':called,'callback_used_cache':record['created_utc'] < self.state['started_at'],
                         'recorded_api_latency_seconds':record['latency_seconds'] if called else 0.0}
            rows.append(raw_row(q,kind,record,'GPT'))
            base=search(q,'llm_astar',raw)
            rows.append(row(q,'B',kind,base,parse_error=error,api_calls=1))
            t=time.perf_counter(); cleaned,meta=refine(q,raw); overhead=time.perf_counter()-t
            improved=search(q,'llm_astar',cleaned)
            rows.append(row(q,'E1',kind,improved,parse_error=error,api_calls=1,refinement=meta,preprocess_seconds=overhead))
        save(dest,{'case':q,'protocol_signature':protocol_signature(),'response_hashes':references,
                   'rows':rows,'e3_acquisition':trace})
    def feedback(self,stage,q):
        dest=OUT/stage/'e2'/f'{q["id"]}.json'
        if self.valid_saved(dest): return
        rows=[]; evidence=[]
        for kind in kinds(stage):
            first=self.api(stage,q,kind,'first',payload(q,kind)); firsttext=text_of(first)
            report=diagnosis(q,firsttext); needs=not report['valid']
            # Alternate API order deterministically so treatment is not always later.
            order=[('E2C',False),('E2F',True)]
            if (q['map_id']+q['pair_id']+kinds(stage).index(kind))%2: order.reverse()
            for name,feedback in order:
                record=self.api(stage,q,kind,name,correction_payload(q,kind,firsttext,feedback)) if needs else first
                raw,error=parse_path(text_of(record))
                t=time.perf_counter(); result=search(q,'llm_astar',raw); elapsed=time.perf_counter()-t
                rows.append(row(q,name,kind,result,parse_error=error,api_calls=1+int(needs),
                                revised=needs,first_failure=report['error']))
                rows.append(raw_row(q,kind,record,'GPT_'+name))
                evidence.append({'prompt':kind,'arm':name,'first_response_sha256':first['response_sha256'],
                                 'final_response_sha256':record['response_sha256'],'revised':needs,
                                 'validation':report if feedback else None})
        save(dest,{'case':q,'protocol_signature':protocol_signature(),'rows':rows,'evidence':evidence})
    def delayed(self,stage,q):
        dest=OUT/stage/'e3'/f'{q["id"]}.json'
        if self.valid_saved(dest): return
        firstphase=json.loads((OUT/stage/'e1'/f'{q["id"]}.json').read_text(encoding='utf-8'))
        rows=[]
        for kind in kinds(stage):
            record=self.api(stage,q,kind,'first',payload(q,kind))
            raw,error=parse_path(text_of(record))
            r=delayed_search(q,lambda:raw)
            trace=firstphase['e3_acquisition'][kind]
            for metric in ('operation','storage','length','valid','path','targets','trigger','api_calls'):
                if digest(r[metric])!=digest(trace['result'][metric]): raise RuntimeError('Delayed acquisition/replay mismatch: '+metric)
            rows.append(row(q,'E3',kind,r,parse_error=error,
                acquisition_elapsed_seconds=trace['result']['elapsed_with_callback_seconds'],
                acquisition_callback_seconds=trace['result']['callback_seconds'],
                recorded_api_latency_seconds=trace['recorded_api_latency_seconds']))
        save(dest,{'case':q,'protocol_signature':protocol_signature(),'rows':rows})
    def all(self):
        OUT.mkdir(parents=True,exist_ok=True)
        manifest=OUT/'protocol.json'
        specification={'protocol':PROTOCOL,'source_sha256':source_hashes(),'signature':protocol_signature(),
          'data_sha256':hashlib.sha256((ROOT/'vendor/llm-astar/dataset/environment_50_30.json').read_bytes()).hexdigest(),
          'first_payloads_sha256':{stage:digest([payload(q,k) for q in cases_for(stage) for k in kinds(stage)]) for stage in ('pilot','main','scaling')}}
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=specification:
            raise RuntimeError('Frozen protocol mismatch; do not mix settings')
        save(manifest,specification)
        if not (OUT/'environment.json').exists():
            save(OUT/'environment.json',{'created_at':now(),'platform':platform.platform(),'python':sys.version,
                 'hostname':platform.node(),'processor':platform.processor(), 'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'gpu_used_for_inference':False,'packages':{p:importlib.metadata.version(p) for p in ['shapely','numpy','matplotlib']}})
        for command in [['reproduction/test_reproduction.py'],['reproduction/test_improvements.py']]:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,capture_output=True,text=True)
            (OUT/(Path(command[0]).stem+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode: raise RuntimeError('Preflight test failure: '+command[0])
        available=request(self.key,'models')['response']
        if MODEL not in {m['id'] for m in available.get('data',[])}: raise RuntimeError('Required model unavailable; no substitution')
        save(OUT/'model_access.json',{'checked_at':now(),'required_model':MODEL,'available':True})
        for fn,phase in [(self.common,'E1'),(self.feedback,'E2'),(self.delayed,'E3')]: self.batch('pilot','pilot_'+phase,fn)
        self.status(phase='pilot_verified')
        for fn,phase in [(self.common,'E1'),(self.feedback,'E2'),(self.delayed,'E3')]:
            for stage in ('main','scaling'): self.batch(stage,phase,fn)
        self.status(phase='verification')
        from improve_report import build
        build()
        self.status(phase='complete',complete=True,finished_at=now())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['all','report'])
    ap.add_argument('--key-file'); ap.add_argument('--detach',action='store_true'); ap.add_argument('--gpu',default='1')
    ap.add_argument('--workers',type=int,default=8); args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if args.action=='report':
        from improve_report import build
        build(); return
    OUT.mkdir(parents=True,exist_ok=True)
    pidfile=OUT/'runner.pid'
    if os.name=='posix' and pidfile.exists():
        oldpid=int(pidfile.read_text().strip())
        try: os.kill(oldpid,0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Recorded runner still alive; refusing duplicate')
    key=read_key(args.key_file) if args.key_file else getpass.getpass('OpenAI key (memory only): ')
    if not key.startswith('sk-'): raise RuntimeError('Invalid credential format; content suppressed')
    if args.detach:
        if os.name!='posix': raise RuntimeError('Detach requires Linux')
        pid=os.fork()
        if pid:
            print('Runner launched PID '+str(pid),flush=True); return
        os.setsid()
        log=os.open(str(OUT/'run.log'),os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        null=os.open(os.devnull,os.O_RDONLY)
        os.dup2(null,0);os.dup2(log,1);os.dup2(log,2)
        os.close(null);os.close(log)
        sys.stdout.reconfigure(line_buffering=True);sys.stderr.reconfigure(line_buffering=True)
    pidfile.write_text(str(os.getpid()),encoding='utf-8')
    runner=Runner(key,args.workers)
    try: runner.all()
    except Exception as exc:
        # Never include arbitrary provider messages or request credentials.
        message=str(exc) if isinstance(exc,RuntimeError) else type(exc).__name__
        runner.status(phase='failed',error=message,complete=False)
        print('Experiment stopped: '+message,flush=True)
        raise SystemExit(1)

if __name__=='__main__': main()
