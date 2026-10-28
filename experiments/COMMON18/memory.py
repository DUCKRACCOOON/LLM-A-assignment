"""Isolated process working-set sample; separate from timed benchmark."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from .prepare import save
from .benchmark import shared,context
from .backends import METHODS,run,load_model


def worker(out,method):
    replies=shared(out);ctx=context(out,need_models=False)
    if method in ('S4','S4_untrained','S4_exact'):
        key='untrained' if method=='S4_untrained' else 'model';ctx[key]=load_model(method=='S4_untrained')
        # Warm resident inference, not model initialization/compilation peak.
        from .backends import cc
        import torch
        q=json.loads((out/'test_cases.json').read_text(encoding='utf-8'))[0]['query'];g=cc.Grid(q)
        with torch.inference_mode():
            for _ in range(3):ctx[key](torch.from_numpy(cc.network_inputs(g,q)[None]))
    cases=[c for c in json.loads((out/'test_cases.json').read_text(encoding='utf-8')) if c['pair_index']==0][:10]
    process=psutil.Process();print(json.dumps(dict(ready=True,baseline_rss=process.memory_info().rss)),flush=True)
    records=[]
    for c in cases:
        gc.collect();r=run(method,c['query'],replies[c['case_id']],ctx,seed=20261028+c['map_index']*10)
        records.append(dict(case_id=c['case_id'],valid=r['valid'],length=r['length']))
    save(out/'memory'/f'{method}_paths.json',records)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--worker');a=p.parse_args();out=a.input.resolve()
    if a.worker:worker(out,a.worker);return
    folder=out/'memory';folder.mkdir(exist_ok=True);results=[]
    for method in METHODS:
        dest=folder/(method+'.json')
        if dest.exists():results.append(json.loads(dest.read_text(encoding='utf-8')));continue
        child=subprocess.Popen([sys.executable,'-m','experiments.COMMON18.memory','--input',str(out),'--worker',method],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        first=child.stdout.readline()
        if not first:
            _,err=child.communicate();raise RuntimeError(f'Memory worker {method} failed: '+err[:500])
        info=json.loads(first);assert info['ready'];baseline=info['baseline_rss'];peak=baseline;sample_count=0;started=time.perf_counter()
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:peak=max(peak,process.memory_info().rss);sample_count+=1
            except psutil.NoSuchProcess:break
            if time.perf_counter()-started>120:
                child.terminate();raise RuntimeError('Isolated memory worker exceeded 120 s')
            time.sleep(.001)
        stdout,stderr=child.communicate();assert child.returncode==0,(method,stderr[:300])
        result=dict(method=method,cases=10,baseline_rss_bytes=baseline,sampled_peak_rss_bytes=peak,
                    sampled_increment_bytes=max(0,peak-baseline),samples=sample_count,
                    nominal_interval_ms=1,seconds=time.perf_counter()-started,
                    definition='fresh process after imports/model warmup; includes resident process; not pure allocations')
        save(dest,result);results.append(result);print(f'Memory {method}: {peak/1024**2:.2f} MiB',flush=True)
    save(out/'memory_summary.json',results)


if __name__=='__main__':main()
