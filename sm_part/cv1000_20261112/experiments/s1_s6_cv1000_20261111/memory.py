"""Auxiliary Windows RSS samples; not part of timed trials."""
import concurrent.futures
import ctypes
from ctypes import wintypes
import subprocess
import time
from common import *
from algorithms import BASE, MODEL_METHODS, run, candidate

class MemoryCounters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

def process_handle(pid):
    fn=ctypes.windll.kernel32.OpenProcess;fn.restype=wintypes.HANDLE
    h=fn(0x0410,False,pid)
    if not h:raise ctypes.WinError()
    return h

def rss(handle):
    c=MemoryCounters();c.cb=ctypes.sizeof(c)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(handle),ctypes.byref(c),c.cb):raise ctypes.WinError()
    return c.WorkingSetSize

def worker(case_id,method):
    from evaluate import deny_network
    import socket
    socket.socket.connect=deny_network;socket.create_connection=deny_network
    q=next(q for q in cases('test') if q['id']==case_id);record=raw(q);text=text_of(record);revisions={}
    if not diagnosis(q,text)['valid']:
        revisions={v:text_of(raw(q,v)) for v in ('control','feedback')}
    models={}
    if method in MODEL_METHODS:
        import torch
        torch.set_num_threads(1);torch.manual_seed(20261111+FOLD);model=candidate.make_network()
        if method!='S4_untrained':model.load_state_dict(torch.load(OUT/'models'/f'{method[3:]}.pt',weights_only=True,map_location='cpu'))
        models[method]=model.eval()
        dq=cases('pilot')[0];grid=candidate.Grid(dq)
        with torch.no_grad():model(torch.from_numpy(candidate.network_inputs(grid,dq)[None]))
    print('READY',flush=True);sys.stdin.readline()
    r=run(q,method,text,revisions,models,read(OUT/'program_selection.json')['weights'])
    print(json.dumps(dict(valid=r['valid'],path_sha256=digest(r['path']),operation=r['operation'])),flush=True)
    sys.stdin.readline() # Retain allocations until the parent takes its final sample.

def measure(task):
    fold,case_id,method,repeat=task;target=RUN/'memory'/f'f{fold}_{case_id}_{method}_{repeat}.json'
    if target.exists():return read(target)
    env=dict(os.environ,CV_FOLD=str(fold),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([sys.executable,str(HERE/'memory.py'),'worker',case_id,method],cwd=ROOT,env=env,
                       stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    line=p.stdout.readline().strip()
    if line!='READY':raise RuntimeError(f'Memory worker startup failed: {case_id}/{method}: '+p.stderr.read()[-1000:])
    handle=process_handle(p.pid);samples=[];t=time.perf_counter();base=rss(handle)
    p.stdin.write('GO\n');p.stdin.flush()
    # Read completion in another thread so the sampler is not blocked on stdout.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(p.stdout.readline)
        while not future.done():
            samples.append((time.perf_counter()-t,rss(handle)));time.sleep(.001)
        result=json.loads(future.result());samples.append((time.perf_counter()-t,rss(handle)))
    p.stdin.write('DONE\n');p.stdin.flush();p.wait(timeout=30)
    ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle))
    if p.returncode:raise RuntimeError(p.stderr.read()[-1000:])
    expected=read(RUN/'folds'/f'fold{fold}'/'evaluations'/f'{case_id}.json')
    ref=next(r for r in expected['rows'] if r['method']==method)
    assert result['path_sha256']==ref['path_sha256'] and result['operation']==ref['operation']
    value=dict(fold=fold,case_id=case_id,method=method,repeat=repeat,baseline_rss_bytes=base,
               sampled_peak_rss_bytes=max([base]+[x[1] for x in samples]),samples=len(samples),
               observed_seconds=time.perf_counter()-t,nominal_interval_seconds=.001,**result)
    value['sampled_increment_bytes']=value['sampled_peak_rss_bytes']-base;save(target,value);return value

def main():
    tasks=[];selected=[]
    for fold,p in enumerate(MANIFEST['folds']):
        for mid in sorted(p['test_maps'])[:4]:
            cid=f'm{mid:03d}_p00';selected.append(dict(fold=fold,map_index=mid,case_id=cid))
            tasks += [(fold,cid,m,r) for m in BASE+MODEL_METHODS for r in range(3)]
    save(RUN/'memory_protocol.json',dict(selected=selected,methods=BASE+MODEL_METHODS,repeats=3,processes=len(tasks),concurrent_workers=4,
        selection_rule='smallest four test map IDs in each outer fold, pair00',timing_use=False,
        interpretation='Process RSS includes interpreter, libraries, and loaded model; 1ms nominal sampling can miss instantaneous peaks'))
    done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(measure,tasks):
            done+=1
            if done%44==0:save(RUN/'memory_status.json',dict(done=done,total=len(tasks),complete=False,updated_at=now()));print(json.dumps(dict(memory_done=done,total=len(tasks))),flush=True)
    save(RUN/'memory_status.json',dict(done=done,total=len(tasks),complete=True,updated_at=now()))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
    else:main()
