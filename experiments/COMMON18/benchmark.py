"""Common-host timed replay after same-collection shared responses exist."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import time

import numpy as np

from experiments.K3.verification import Oracle
from experiments.K3.run import sha
from .prepare import save,rows,SEED
from .backends import ROOT,METHODS,PRIMARY,run,content,load_model,cc


def shared(out):
    assert (out/'collection_complete.json').exists(),'Shared collection incomplete; no mixed-model fallback'
    cache={r['request_sha256']:r for r in rows(out/'generations.jsonl')}
    index=json.loads((out/'response_index.json').read_text(encoding='utf-8'))
    return {cid:{kind:cache[h] for kind,h in kinds.items()} for cid,kinds in index.items()}


def calibrate(out,replies):
    if (out/'calibration.json').exists():return json.loads((out/'calibration.json').read_text(encoding='utf-8'))
    cases=json.loads((out/'calibration_cases.json').read_text(encoding='utf-8'));scores=defaultdict(list);details=[];last=None
    began=time.perf_counter()
    for c in cases:
        q=c['query']
        if c['map_index']!=last:grid=cc.Grid(q);last=c['map_index']
        paths=[cc.normalize_proposal(q,cc_path(replies[c['case_id']][k])) for k in ('standard','cot','repe')]
        field,scale=cc.llm_field(grid,q,paths)
        distances,parents,_=grid.distances(q['start']);path=cc.reconstruct(grid,parents,q['start'],q['goal'])
        score=max(float(field[grid.index(p)]) for p in path)/scale;scores[c['map_index']].append(score)
        details.append(dict(case_id=c['case_id'],score=score,canonical_path=path,optimum=float(distances[grid.index(q['goal'])])))
    maxima={str(k):max(v) for k,v in scores.items()};quantile=cc.conformal_quantile(maxima.values(),.1)
    assert np.isfinite(quantile)
    value=dict(quantile=quantile,alpha=.1,map_scores=maxima,details=details,seconds=time.perf_counter()-began,
               calibration_maps=len(maxima),cases=len(cases),fresh_model='gpt-4o-mini-2024-07-18')
    save(out/'calibration.json',value);print(f'Calibration complete: qhat={quantile:.6f}',flush=True)
    return value


def cc_path(reply):
    from .vendor_sm.core import parse_path
    return parse_path(content(reply))[0]


def context(out,need_models=True):
    qhat=json.loads((out/'calibration.json').read_text(encoding='utf-8'))['quantile']
    value=dict(quantile=qhat,standard_template=runpy.run_path(str(ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']['standard'])
    if need_models:
        began=time.perf_counter();value['model']=load_model();value['untrained']=load_model(True)
        loading=time.perf_counter()-began
        q=json.loads((out/'test_cases.json').read_text(encoding='utf-8'))[0]['query'];grid=cc.Grid(q)
        import torch
        with torch.inference_mode():
            x=torch.from_numpy(cc.network_inputs(grid,q)[None])
            for _ in range(3):value['model'](x);value['untrained'](x)
        save(out/'model_loading.json',dict(seconds=loading,threads=torch.get_num_threads(),device='cpu',torch=torch.__version__,warmup_passes=3))
    return value


def signature(r):
    return {k:r.get(k) for k in ('valid','path','length','operation','storage','successor_expansions','llm_calls')}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();out=a.input.resolve()
    if (out/'benchmark_complete.json').exists():raise RuntimeError('Completed benchmark exists; refusing to overwrite')
    assert not (out/'evaluations.jsonl.gz').exists(),'Use a new benchmark output after an interrupted timed run'
    replies=shared(out)
    # Replaying a benchmark must never silently generate missing responses.
    from . import prepare as preparation
    import reproduction.api as api
    def deny_network(*args,**kwargs):raise RuntimeError('Offline benchmark: API calls are disabled')
    preparation.completion=api.completion=deny_network
    calibrate(out,replies);ctx=context(out)
    cases=json.loads((out/'test_cases.json').read_text(encoding='utf-8'));rng=random.Random(SEED)
    analysis_only={'report.py','audit.py','memory.py','verify.py'}
    source={str(f.relative_to(ROOT)):sha(f) for f in Path(__file__).parent.rglob('*.py') if f.name not in analysis_only}
    save(out/'benchmark_manifest.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),methods=list(METHODS),
        primary=list(PRIMARY),cases=len(cases),repeats=3,stochastic_seeds=3,source_sha256=source,
        numpy=np.__version__,python=sys.version,collection_sha256=sha(out/'generations.jsonl'),
        checkpoint_sha256=sha(Path(__file__).parent/'assets/seed11.pt'),calibration_sha256=sha(out/'calibration.json')))
    counts=defaultdict(int);last=None;oracle=None;optima={};failures=[]
    with gzip.open(out/'evaluations.jsonl.gz','xt',encoding='utf-8') as f:
        for i,c in enumerate(cases):
            q=c['query'];cid=c['case_id']
            if c['map_index']!=last:oracle=Oracle(q);last=c['map_index']
            optimum=oracle.distances(q['goal'])[tuple(q['start'])];optima[cid]=optimum;stable={}
            for repeat in range(3):
                gc.collect();gc.disable()
                order=list(METHODS);rng.shuffle(order)
                for position,m in enumerate(order):
                    seeds=range(3) if m in ('K5','K5_noLLM','K5_exact') else [-1]
                    for si in seeds:
                        seed=SEED+c['map_index']*10+c['pair_index']+max(si,0)*10000
                        r=run(m,q,replies[cid],ctx,seed=seed)
                        r.update(case_id=cid,map_index=c['map_index'],pair_index=c['pair_index'],repeat=repeat,
                                 order=position,seed_index=si,seed=seed,optimum=optimum)
                        f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');counts['runs']+=1
                        if r['valid']:
                            oracle.check_path(q,r);counts['path_checks']+=1
                            assert r['length']>=optimum-1e-7,(cid,m,'shorter than optimum')
                            if r['bound'] is not None:
                                assert r['length']<=r['bound']*optimum+1e-7,(cid,m,'bound violated')
                                counts['bound_checks']+=1
                            if r['bound']==1:counts['exact_checks']+=1
                        else:failures.append(dict(case_id=cid,method=m,repeat=repeat,seed_index=si,error=r['error']))
                        key=(m,si);sig=signature(r)
                        if key in stable:assert stable[key]==sig,(cid,key,'repeat changed');counts['repeat_checks']+=1
                        stable[key]=sig
                gc.enable()
            if (i+1)%5==0 or i+1==len(cases):
                f.flush();save(out/'benchmark_status.json',dict(done=i+1,total=len(cases),runs=counts['runs'],failures=len(failures)))
                print(f'Common benchmark {i+1}/{len(cases)} cases; {counts["runs"]} runs; {len(failures)} failures',flush=True)
    save(out/'optima.json',optima);save(out/'failures.json',failures)
    save(out/'benchmark_complete.json',dict(counts,passed=True,cases=len(cases),failures=len(failures),
        finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
