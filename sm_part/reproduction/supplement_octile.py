"""Post-freeze, planned development-only Euclidean/octile A* comparison.

Does not select settings, alter the frozen runner, or consume final maps.
Run only after the main runner has finished to avoid timing interference.
"""
import hashlib
import json
import os
import random
import statistics
import sys
import time
import zipfile
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='reproduction' else HERE
sys.path.insert(0,str(ROOT/'reproduction'))
from next_run import sources,time_limit
from next_search_core import Grid,search,audit_path
from next_data import digest
from next_report import comparison

MAIN=ROOT/'results/next_search_v1'
OUT=ROOT/'results/next_octile_supplement'
def read(p):return json.loads(p.read_text())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(phase,**kw):
    s=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw)
    save(OUT/'status.json',s);print(json.dumps(s),flush=True)

def execute(q,euclidean,instrument):
    start=time.perf_counter();g=Grid(q,instrument)
    r=search(g,q,weight=1.,euclidean=euclidean,instrument=instrument)
    r.update(preparation_seconds=g.seconds,total_seconds=time.perf_counter()-start)
    return r

def main():
    assert read(MAIN/'status.json')['phase']=='complete','main experiment must finish first'
    assert read(MAIN/'source_manifest.json')==sources(),'frozen solver changed'
    qs=read(MAIN/'data/development.json');assert len(qs)==600
    oracle=read(MAIN/'oracle/r1_development.json');rng=random.Random(26101440)
    save(OUT/'protocol.json',dict(role='planned auxiliary measurement executed after main freeze and completion',
        split='development',queries=600,maps=60,repeats=5,weight=1,source_hashes=sources(),
        script_sha256=sha(Path(__file__)),main_signature=read(MAIN/'protocol_frozen.json')['signature'],
        order_seed=26101440,threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        no_retuning=True,no_final_data_used=True))
    status('running',cases_done=0,cases_total=600)
    rows=[]
    for i,q in enumerate(qs):
        pending={};order=['A_OCT','A_EUC'];rng.shuffle(order)
        for method in order:
            path=OUT/'evaluation'/method/(q['id']+'.json')
            if path.exists():rows.append(read(path));continue
            with time_limit(60):r=execute(q,method=='A_EUC',True)
            audit_path(q,r,oracle[q['id']]['optimal'],1.)
            r.update(id=q['id'],map_id=q['base_map_id'],family=q['family'],method=method,
                input_hash=digest(q),optimal=oracle[q['id']]['optimal'],timings=[],status='ok')
            pending[method]=(path,r)
        for repeat in range(5):
            order=list(pending);rng.shuffle(order)
            for method in order:
                path,r=pending[method]
                with time_limit(60):z=execute(q,method=='A_EUC',False)
                assert z['path']==r['path'],'timing and instrumented path disagree'
                r['timings'].append({k:z[k] for k in ('total_seconds','preparation_seconds','search_seconds','reconstruction_seconds')})
        for path,r in pending.values():
            r['total_seconds']=statistics.median(t['total_seconds'] for t in r['timings']);save(path,r);rows.append(r)
        if (i+1)%20==0:status('running',cases_done=i+1,cases_total=600)
    result=dict(verified=True,rows=len(rows),time=comparison(rows,'A_EUC','A_OCT'),
        expansions=comparison(rows,'A_EUC','A_OCT','expansions'),
        by_family={family:{k:comparison([r for r in rows if r['family']==family],'A_EUC','A_OCT',metric)
                   for k,metric in [('time','total_seconds'),('expansions','expansions')]}
                   for family in sorted({r['family'] for r in rows})})
    save(OUT/'summary.json',result)
    status('complete',complete=True,rows=len(rows))
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('run.log','export.json')}
    archive=ROOT/'octile-supplement-results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.write(Path(__file__),'supplement_octile.py')
        manifest['supplement_octile.py']=sha(Path(__file__))
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
    save(OUT/'export.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest)))
    print(json.dumps(read(OUT/'export.json')),flush=True)

if __name__=='__main__':main()
