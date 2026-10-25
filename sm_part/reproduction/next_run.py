"""Sequential, resumable R1 -> R2 -> R3 experiment execution on Sirius.

Development/validation and final are separate. Final generation follows the
frozen protocol; no candidate tuning consumes final outcomes.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from core import ROOT,search as legacy_search,World
from next_search_core import Grid,search,audit_path
from next_geometry import Rectangles
from next_refinement import solve as refine
from next_topology import routes,Guide,rank,prompt
from next_data import generate,digest,r3_subset,auxiliary,FAMILIES
from next_oracle import compute as oracle_compute

OUT=ROOT/'results/next_search_v1'; PLAN=ROOT/'results/next_experiments_plan/protocol.json'
BASE=['A_OCT','N_LEGACY','N_CLEAN','WA_101','WA_102','WA_200','FOCAL_101','FOCAL_102']
MODEL_ROOT='/home/work/fanimal_hidream/llama-reproduction'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def status(phase,**kw):
    d=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw);save(OUT/'status.json',d);print(json.dumps(d),flush=True)
def sources():return {p.name:file_sha(p) for p in sorted((ROOT/'reproduction').glob('next_*.py'))}|{'test_next_search.py':file_sha(ROOT/'reproduction/test_next_search.py'),'core.py':file_sha(ROOT/'reproduction/core.py')}


@contextlib.contextmanager
def time_limit(seconds):
    if hasattr(signal,'SIGALRM'):
        old=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('whole query timeout')))
        signal.setitimer(signal.ITIMER_REAL,seconds)
        try:yield
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    else:yield


def rss():
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return None


class MemorySampler:
    def __enter__(self):
        self.peak=rss();self.stop=threading.Event()
        def run():
            while not self.stop.wait(.005):
                v=rss()
                if v is not None:self.peak=max(self.peak or 0,v)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*exc):
        self.stop.set();self.thread.join();v=rss()
        if v is not None:self.peak=max(self.peak or 0,v)


def execute(q,method,instrument=True,config=None,response=None,reuse=None):
    config=config or {};t=time.perf_counter();prep=0.;inference=0.;extra={};g=None;deadline=t+60
    if method=='N_LEGACY':
        world=World(q);prep=time.perf_counter()-t;r0=legacy_search(q,'llm_astar',world=world)
        r=dict(path=r0['path'],length=r0['length'],reachable=r0['valid'],search_seconds=r0['seconds'],reconstruction_seconds=0.,
               counters={'legacy_heap_pops':r0['operation'],'expansions':r0['operation']},peak_open=None,discovered=r0['storage'])
    elif method.startswith('R2') or method=='R1_R2':
        mode=method.split(':')[1] if ':' in method else config.get('r2','rule')
        if method=='R1_R2':mode=config.get('combination_r2',mode)
        r=refine(q,mode,compressed=method=='R1_R2',instrument=instrument,program=config.get('program'),deadline=deadline)
        r['inference_seconds']=0.;return r
    elif method.startswith('TOPO'):
        _,mode,w,seed=method.split(':');prop,rect=routes(q,k=config.get('k',3),instrument=instrument)
        prep=time.perf_counter()-t;ranking=rank(prop,mode,int(seed));used=False;fallback=False
        if mode=='llm' and len(prop['candidates'])>=2:
            assert response is not None,'missing collected inference'
            inference=response.get('seconds',0.);fallback=not response.get('valid',False)
            if not fallback:ranking=response['ranking'];used=True
        guide=Guide(rect,prop,ranking) if len(prop['candidates'])>=2 else None
        # Rectangles are used only to form guidance. The search remains the full grid.
        g=Grid(q,instrument);prep+=g.seconds;r=search(g,q,float(w),True,guide=guide,instrument=instrument,deadline=deadline)
        r['counters']['guidance_grid_cells_scanned']=prop['grid_cells_scanned'];r['counters']['guidance_portals']=prop['portal_edges']
        extra=dict(candidate_count=len(prop['candidates']),ranking=ranking,llm_used=used,fallback=fallback)
    else:
        if method.startswith('R1'):
            variant=method.split(':')[1] if ':' in method else config.get('r1','sparse')
            g=reuse if reuse is not None else Rectangles(q,instrument,variant)
            if reuse is not None:g.set_query(q);g.counts=Counter()
        else:
            g=reuse if reuse is not None else Grid(q,instrument)
            if reuse is not None:g.counts=Counter()
        prep=0. if reuse is not None else g.seconds
        w={'N_CLEAN':2.,'WA_101':1.01,'WA_102':1.02,'WA_200':2.,'FOCAL_101':1.01,'FOCAL_102':1.02}.get(method,1.)
        r=search(g,q,w,method.startswith('FOCAL'),euclidean=method=='N_CLEAN',instrument=instrument,deadline=deadline)
    r.update(preparation_seconds=prep,inference_seconds=inference,required_validation_seconds=r.get('required_validation_seconds',0.),
             total_seconds=time.perf_counter()-t+inference,**extra)
    return r


def bound(method):
    if method.startswith('TOPO'):return float(method.split(':')[2])
    return {'WA_101':1.01,'WA_102':1.02,'WA_200':2.,'N_CLEAN':2.,'N_LEGACY':math.inf,'FOCAL_101':1.01,'FOCAL_102':1.02}.get(method,1.)


def dataset(stage,n):
    p=OUT/'data'/f'{stage}.json'
    if p.exists():return read(p)
    qs,manifest=generate(stage,n);save(p,qs);save(OUT/'data'/f'{stage}_manifest.json',manifest)
    # Geometry disjointness is checked, not merely different query IDs.
    hashes={q['geometry_hash'] for q in qs}
    for f in (OUT/'data').glob('*.json'):
        if f.stem in ('development','validation','final') and f!=p:
            assert not hashes.intersection(q['geometry_hash'] for q in read(f)), 'duplicate map across splits'
    return qs


def get_oracles(stage,qs):
    p=OUT/'oracle'/f'{stage}.json'
    if p.exists():return read(p)
    cache_path=OUT/'oracle/cache.json';cache=read(cache_path) if cache_path.exists() else {}
    missing=[q for q in qs if digest(q) not in cache]
    if missing:
        status('oracle_'+stage,cases=len(missing))
        new=oracle_compute(missing,lambda i,n,sec:status('oracle_'+stage,maps_done=i,maps_total=n,last_map_seconds=sec) if i%10==0 or i==n else None)
        for q in missing:cache[digest(q)]=new[q['id']]
        save(cache_path,cache)
    r={q['id']:cache[digest(q)] for q in qs}
    save(p,r);return r


def evaluate(stage,qs,methods,config=None,repeats=5,responses=None):
    oracle=get_oracles(stage,qs);files=[];rng=random.Random(26101420)
    for i,q in enumerate(qs):
        ordered=list(methods);rng.shuffle(ordered);pending={}
        for method in ordered:
            key=method.replace(':','_');p=OUT/'evaluation'/stage/key/(q['id']+'.json');files.append(p)
            if p.exists():continue
            opt=oracle[q['id']]['optimal'];opt=math.inf if opt is None else opt
            response=(responses or {}).get(q['id'])
            try:
                with MemorySampler() as mem,time_limit(60):
                    r=execute(q,method,True,config,response)
                audit_path(q,r,opt,bound(method))
                r.update(timings=[],
                         sampled_peak_rss=mem.peak,memory_sample_interval_ms=5,status='ok',failure=None)
                if mem.peak is not None and mem.peak>8*1024**3:raise MemoryError('RSS above protocol limit')
            except (TimeoutError,MemoryError) as e:
                r=dict(status='failed',failure=type(e).__name__,path=[],reachable=False,length=None,total_seconds=None,timings=[])
            r.update(id=q['id'],map_id=q['base_map_id'],family=q['family'],method=method,stage=stage,
                     optimal=opt if math.isfinite(opt) else None,bound=bound(method) if math.isfinite(bound(method)) else None,
                     input_hash=digest(q),source_hash=digest(sources()))
            pending[method]=(p,r)
        for repeat in range(repeats):
            order=list(pending);rng.shuffle(order)
            for method in order:
                p,r=pending[method]
                if r['status']!='ok':continue
                try:
                    with time_limit(60):z=execute(q,method,False,config,(responses or {}).get(q['id']))
                    assert z['path']==r['path'],'timing and counter mode path differ'
                    r['timings'].append({k:z.get(k,0.) for k in ('preparation_seconds','inference_seconds','search_seconds','required_validation_seconds','reconstruction_seconds','total_seconds')})
                except (TimeoutError,MemoryError) as e:r.update(status='failed',failure=type(e).__name__)
        for p,r in pending.values():
            r['total_seconds']=statistics.median(v['total_seconds'] for v in r['timings']) if r['status']=='ok' else None
            save(p,r)
        if i%20==0 or i+1==len(qs):status(stage,cases_done=i+1,cases_total=len(qs),methods=len(methods))
    return [read(p) for p in files]


def ratios(rows,method,baseline='A_OCT'):
    a={r['id']:r for r in rows if r['method']==method};b={r['id']:r for r in rows if r['method']==baseline}
    times=[];exp=[]
    for k,r in a.items():
        if k not in b:continue
        if r['status']!='ok' or b[k]['status']!='ok':return dict(time=math.inf,expansions=math.inf,failures=True)
        times.append(r['total_seconds']/b[k]['total_seconds']);exp.append(r['counters']['expansions']/b[k]['counters']['expansions'])
    assert times,(method,baseline)
    return dict(time=math.exp(statistics.mean(math.log(v) for v in times)),expansions=math.exp(statistics.mean(math.log(v) for v in exp)),
                p95_time=float(np.quantile(times,.95)),n=len(times))


def model_tasks(tasks,name):
    p=OUT/'model_tasks'/f'{name}.json';save(p,tasks)
    if tasks:
        subprocess.run([sys.executable,str(ROOT/'reproduction/next_model.py'),'--tasks',str(p),'--out',str(OUT/'raw'/name),
                        '--model-root',MODEL_ROOT,'--device','0'],check=True)
    return {t['id']:read(OUT/'raw'/name/(t['id']+'.json')) for t in tasks}


def collect_rankings(qs,stage,k):
    tasks=[]
    for q in qs:
        prop,_=routes(q,k)
        if len(prop['candidates'])>=2:
            tasks.append(dict(id=q['id'],kind='ranking',prompt=prompt(q,prop),candidate_count=len(prop['candidates']),max_new_tokens=256))
    return model_tasks(tasks,'r3_'+stage)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'execution_revision.json',dict(attempt=2,previous_directory='/home/work/fanimal_hidream/next-search-20261014',
         previous_stopped_before_final_generation=True,reason='LPA equal-f floating-point plateau could hide an underconsistent predecessor; drain the full goal-f plateau and add 600 medium-map repair regressions',
         previous_results_not_reused=True))
    save(OUT/'plan_snapshot.json',read(PLAN));save(OUT/'source_manifest.json',sources())
    subprocess.run([sys.executable,str(ROOT/'reproduction/test_next_search.py'),'--cases','2000','--out',str(OUT/'server_preflight.json')],check=True)
    from next_model import model_manifest
    model_manifest(MODEL_ROOT,OUT/'model_manifest.json')
    import torch,scipy,shapely,transformers
    save(OUT/'environment.json',dict(host=platform.node(),python=sys.version,platform=platform.platform(),numpy=np.__version__,
         scipy=scipy.__version__,shapely=shapely.__version__,torch=torch.__version__,transformers=transformers.__version__,
         gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cpu_threads=os.environ.get('OMP_NUM_THREADS'),cpu_count=os.cpu_count()))
    subprocess.run([sys.executable,'-m','pip','list','--format=json'],stdout=(OUT/'packages.json').open('w'),check=True)


def run_all():
    if not(OUT/'server_preflight.json').exists():prepare()
    assert read(OUT/'source_manifest.json')==sources(),'source changed after run started; create new run version'
    dev=dataset('development',15);val=dataset('validation',15)
    selection=read(OUT/'selection.json') if (OUT/'selection.json').exists() else {}
    if 'r1' not in selection:
        status('R1_development')
        dr=evaluate('r1_development',dev,['A_OCT','R1:full','R1:sparse'])
        chosen=min(('full','sparse'),key=lambda v:ratios(dr,'R1:'+v)['time'])
        vr=evaluate('r1_validation',val,['A_OCT','R1:'+chosen]);metric=ratios(vr,'R1:'+chosen)
        selection.update(r1=chosen,r1_validation=metric,r1_gate=metric['time']<=.9 and metric['expansions']<=.8)
        save(OUT/'selection.json',selection)
    if 'r2' not in selection:
        status('R2_development')
        methods=['A_OCT','R2:first','R2:batch','R2:rule','R2:random']
        dr=evaluate('r2_development',dev,methods)
        # Fixed four-feature linear programs: no arbitrary generated Python.
        tasks=[]
        for i in range(12):
            tasks.append(dict(id=f'program-{i:02d}',kind='program',max_new_tokens=512,
                prompt='Design a cheap wall-priority score for exact lazy grid pathfinding. Return only JSON {"weights":[a,b,c,d]}. '
                'Each coefficient must be finite between -10 and 10. Features: [straight start-goal intersection flag, '
                'normalized wall length, normalized minimum start-endpoint-goal detour distance, boundary attachment flag]. '
                'Higher score activates a wall earlier. Incorrect priorities only add computation; exact validation protects correctness. '
                f'Explore design {i+1} of 12 with a distinct tradeoff; no explanations.'))
        generated=model_tasks(tasks,'r2_programs');programs=[]
        for r in generated.values():
            if r.get('valid') and r['weights'] not in programs:programs.append(r['weights'])
        fitness=[]
        for j,program in enumerate(programs):
            rows=evaluate(f'r2_program_dev{j}',dev,['A_OCT','R2:program'],dict(program=program))
            fitness.append((ratios(rows,'R2:program')['time'],program))
        if fitness:
            selection['program']=min(fitness,key=lambda x:x[0])[1]
            dr+=evaluate('r2_program_selected_dev',dev,['R2:program'],selection)
        choices=['first','batch','rule']+(['program'] if fitness else [])
        selection['combination_r2']=min(choices,key=lambda v:ratios(dr,'R2:'+v)['time'])
        chosen_rows=[r for r in dr if r['method']=='R2:'+selection['combination_r2'] and r['status']=='ok']
        repeat_fraction=statistics.mean(len(r['rounds'])>1 for r in chosen_rows)
        selection['incremental_trigger_fraction']=repeat_fraction
        if repeat_fraction>=.1:
            dr+=evaluate('r2_incremental_development',dev,['R2:incremental'],selection)
            choices.append('incremental')
        chosen=min(choices,key=lambda v:ratios(dr,'R2:'+v)['time'])
        vr=evaluate('r2_validation',val,['A_OCT','R2:'+chosen],selection);metric=ratios(vr,'R2:'+chosen)
        selection.update(r2=chosen,r2_validation=metric,r2_gate=metric['time']<=.9)
        # Costed combination pilot even when the standalone gate failed.
        cr=evaluate('combination_validation',val,['A_OCT','R1','R2','R1_R2'],selection)
        selection['combination_validation']=ratios(cr,'R1_R2','R1')
        selection['combination_gate']=selection['combination_validation']['time']<=.9
        save(OUT/'selection.json',selection)
    if 'r3_gate' not in selection:
        status('R3_development');subdev=r3_subset(dev,'development');subval=r3_subset(val,'validation')
        scores=[]
        for k in (2,3):
            rows=evaluate('r3_k'+str(k),subdev,['FOCAL_102','TOPO:rule:1.02:0'],dict(selection,k=k))
            scores.append((ratios(rows,'TOPO:rule:1.02:0','FOCAL_102')['time'],k))
        selection['k']=min(scores)[1];save(OUT/'selection.json',selection)
        response=collect_rankings(subdev,'development',selection['k'])
        methods=['FOCAL_102','TOPO:rule:1.02:0','TOPO:llm:1.02:0']+['TOPO:random:1.02:'+str(s) for s in (26101431,26101432,26101433)]
        dr=evaluate('r3_development',subdev,methods,selection,responses=response)
        metric=ratios(dr,'TOPO:llm:1.02:0','TOPO:rule:1.02:0');selection['r3_development']=metric
        # Validation still measures the third experiment even if expansion target fails;
        # final LLM generation is gated to avoid expanding an unpromising branch.
        response=collect_rankings(subval,'validation',selection['k'])
        vr=evaluate('r3_validation',subval,methods,selection,responses=response)
        vm=ratios(vr,'TOPO:llm:1.02:0','TOPO:rule:1.02:0')
        selection.update(r3_validation=vm,r3_gate=metric['expansions']<=.9 and vm['expansions']<=.9)
        save(OUT/'selection.json',selection)
    freeze_and_final(selection,val)


def freeze_and_final(selection,val):
    path=OUT/'protocol_frozen.json'
    if not path.exists():
        # Power review uses validation map log-time ratios, independent of final maps.
        rr=list((OUT/'evaluation/r1_validation').glob('*/*.json'));rows=[read(p) for p in rr]
        a={r['id']:r for r in rows if r['method']=='R1:'+selection['r1']};b={r['id']:r for r in rows if r['method']=='A_OCT'}
        logs=defaultdict(list)
        for key,r in a.items():
            if r['status']=='ok' and b[key]['status']=='ok':logs[r['map_id']].append(math.log(r['total_seconds']/b[key]['total_seconds']))
        vals=np.array([statistics.mean(v) for v in logs.values()]);rng=np.random.default_rng(26101422)
        centered=vals-vals.mean();powers={}
        for n in (200,400):
            samples=rng.choice(centered,(5000,n),replace=True)+math.log(.9)
            powers[n]=float(np.mean(samples.mean(axis=1)+2.50*samples.std(axis=1,ddof=1)/math.sqrt(n)<0))
        count=200 if powers[200]>=.8 else 400
        frozen=dict(plan=read(PLAN),selected=selection,source_hashes=sources(),model_revision=read(OUT/'model_manifest.json')['revision'],
           final_maps=count,power_review=dict(method='validation-centered log ratio simulation; normal critical value 2.50',powers=powers),
           refinements=dict(generator='integer positive-length axis-aligned segments; manual selection removed',
             memory='5 ms sampled RSS, not an exact allocation peak',
             r2_programs='12 bounded linear score proposals, within the maximum 60 generation budget',
             r2_incremental='LPA evaluated if at least 10% of best restart-policy development queries require repeat solves; compressed combination uses best restart policy',
             r1_adaptive='conditional extension not included; conservative full and sparse evaluated',
             inference='Llama chat template for new ranking tasks; earlier reproduction prompts unchanged'),
           frozen_utc=datetime.now(timezone.utc).isoformat(),signature=None)
        frozen['signature']=digest(frozen);save(path,frozen)
    frozen=read(path);assert frozen['source_hashes']==sources();final=dataset('final',frozen['final_maps']//4)
    # Exact candidates retained as diagnostic confirmations if practical targets fail.
    # Such failed-gate results are explicitly not promoted to confirmatory claims.
    evaluate('final',final,BASE+['R1'],selection)
    evaluate('final',final,['R2','R1_R2'],selection)
    sub=r3_subset(final,'final');response=collect_rankings(sub,'final',selection['k']) if selection['r3_gate'] else {}
    modes=['rule','random']+(['llm'] if selection['r3_gate'] else [])
    topo=[f'TOPO:{mode}:{w}:{seed}' for w in (1.,1.01,1.02) for mode in modes for seed in ((26101431,26101432,26101433) if mode=='random' else (0,))]
    evaluate('r3_final',sub,topo,selection,responses=response)
    scaled,varying,fixed=auxiliary(final)
    save(OUT/'data/scaling.json',scaled);save(OUT/'data/reuse_varying.json',varying);save(OUT/'data/reuse_fixed.json',fixed)
    evaluate('scaling',scaled,['A_OCT','N_CLEAN','R1','R2','R1_R2'],selection)
    from next_reuse import run as reuse_run
    reuse_run(OUT,varying,fixed,selection)
    from next_report import build
    build(OUT);status('complete',complete=True,selection=selection)
    export()


def export():
    import zipfile
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('export.json','run.log','runner.pid')]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):file_sha(p) for p in files}
    sourcefiles=list((ROOT/'reproduction').glob('next_*.py'))+[ROOT/'reproduction/test_next_search.py',ROOT/'reproduction/core.py']
    for p in sourcefiles:manifest[p.relative_to(ROOT).as_posix()]=file_sha(p)
    archive=ROOT/'next-search-results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
    save(OUT/'export.json',dict(path=str(archive),sha256=file_sha(archive),bytes=archive.stat().st_size,files=len(manifest)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['all','prepare','report','export']);args=parser.parse_args()
    try:
        if args.action=='all':run_all()
        elif args.action=='prepare':prepare()
        elif args.action=='export':export()
        else:
            from next_report import build
            build(OUT)
    except Exception as e:
        status('failed',error=type(e).__name__,detail=str(e),complete=False);raise
