"""Auxiliary Windows RSS samples (adapted from sm_src/memory.py); not part of timed trials.

Selection rule identical to SM: per outer fold the four smallest test map ids, pair00
(20 cases); every condition; 3 fresh processes each; 4 concurrent workers. The worker
builds ctx with evaluate.context() and asserts that path digest and heap pops equal the
stored evaluation row (repeat 0).

    python memory.py [--run DIR] [--repeats 3] [--workers 4] [--folds 0,1,2,3,4] [--per-fold 4]
Output: <run>/memory/f{fold}_{case}_{method}_{repeat}.json, <run>/memory_status.json, <run>/memory_protocol.json
"""
import argparse
import concurrent.futures
import ctypes
import subprocess
from ctypes import wintypes
from pathlib import Path
from common import *

def local_evaluate():
    """Load UNIFIED_K/evaluate.py explicitly: common puts sm_src (which has its own evaluate.py) first on sys.path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('unified_evaluate', HERE/'evaluate.py'); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod

class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[(n, ctypes.c_size_t) for n in
        ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
         'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]

def process_handle(pid):
    fn = ctypes.windll.kernel32.OpenProcess; fn.restype = wintypes.HANDLE
    h = fn(0x0410, False, pid)
    if not h: raise ctypes.WinError()
    return h

def rss(handle):
    c = MemoryCounters(); c.cb = ctypes.sizeof(c)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(handle), ctypes.byref(c), c.cb): raise ctypes.WinError()
    return c.WorkingSetSize

def worker(case_id, method):
    import socket
    evaluate = local_evaluate()
    socket.socket.connect = evaluate.deny_network; socket.create_connection = evaluate.deny_network
    from harness import run
    q = next(q for q in cases('test') if q['id'] == case_id)
    ctx, _ = evaluate.context(q, evaluate.calibration())
    print('READY', flush=True); sys.stdin.readline()
    r = run(q, method, ctx)
    print(json.dumps(dict(valid=r['valid'], path_sha256=digest(r['path']), operation=r['operation'])), flush=True)
    sys.stdin.readline()  # keep allocations alive until the parent's final sample

def measure(task):
    R, fold, case_id, method, repeat = task; target = R/'memory'/f'f{fold}_{case_id}_{method}_{repeat}.json'
    if target.exists(): return read(target)
    env = dict(os.environ, CV_FOLD=str(fold), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    p = subprocess.Popen([sys.executable, str(HERE/'memory.py'), 'worker', case_id, method], cwd=HERE, env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    line = p.stdout.readline().strip()
    if line != 'READY': raise RuntimeError(f'Memory worker startup failed: {case_id}/{method}: '+p.stderr.read()[-1500:])
    # A Windows venv python.exe is a launcher that runs the base interpreter as a child process;
    # sample the actual interpreter (the launcher's only child) when present.
    import psutil
    kids = psutil.Process(p.pid).children(recursive=True); target_pid = kids[-1].pid if kids else p.pid
    handle = process_handle(target_pid); samples = []; t = time.perf_counter(); base = rss(handle)
    p.stdin.write('GO\n'); p.stdin.flush()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(p.stdout.readline)
        while not future.done():
            samples.append((time.perf_counter()-t, rss(handle))); time.sleep(.001)
        result = json.loads(future.result()); samples.append((time.perf_counter()-t, rss(handle)))
    p.stdin.write('DONE\n'); p.stdin.flush(); p.wait(timeout=60)
    ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle))
    if p.returncode: raise RuntimeError(p.stderr.read()[-1500:])
    expected = read(R/'folds'/f'fold{fold}'/'evaluations'/f'{case_id}.json')
    ref = next(r for r in expected['rows'] if r['method'] == method and r['repeat'] == 0)
    assert result['path_sha256'] == ref['path_sha256'] and result['operation'] == ref['operation'], (case_id, method, result, ref['operation'])
    value = dict(fold=fold, case_id=case_id, method=method, repeat=repeat, baseline_rss_bytes=base,
                 sampled_peak_rss_bytes=max([base]+[x[1] for x in samples]), samples=len(samples),
                 observed_seconds=time.perf_counter()-t, nominal_interval_seconds=.001,
                 sampled_pid='interpreter_child' if kids else 'direct', **result)
    value['sampled_increment_bytes'] = value['sampled_peak_rss_bytes']-base; save(target, value); return value

def main(R, repeats, workers, folds, per_fold):
    fz = R/'frozen.json'
    if fz.exists(): methods = read(fz)['frozen']['methods']
    else:
        from harness import METHODS as methods
    tasks = []; selected = []
    for fold in folds:
        part = MANIFEST['folds'][fold]
        for mid in sorted(part['test_maps'])[:per_fold]:
            cid = f'm{mid:03d}_p00'
            if not (R/'folds'/f'fold{fold}'/'evaluations'/f'{cid}.json').exists(): continue
            selected.append(dict(fold=fold, map_index=mid, case_id=cid))
            tasks += [(R, fold, cid, m, r) for m in methods for r in range(repeats)]
    save(R/'memory_protocol.json', dict(selected=selected, methods=methods, repeats=repeats, processes=len(tasks), concurrent_workers=workers,
         selection_rule='smallest four test map IDs in each outer fold, pair00 (same as SM)', timing_use=False,
         interpretation='Process RSS includes interpreter, libraries, and loaded data; 1ms nominal sampling can miss instantaneous peaks'))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(measure, tasks):
            done += 1
            if done % 44 == 0 or done == len(tasks):
                save(R/'memory_status.json', dict(done=done, total=len(tasks), complete=done == len(tasks), updated_at=now()))
                print(json.dumps(dict(memory_done=done, total=len(tasks))), flush=True)
    save(R/'memory_status.json', dict(done=done, total=len(tasks), complete=True, updated_at=now()))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker': worker(sys.argv[2], sys.argv[3])
    else:
        p = argparse.ArgumentParser(); p.add_argument('--run'); p.add_argument('--repeats', type=int, default=3); p.add_argument('--workers', type=int, default=4)
        p.add_argument('--folds', default='0,1,2,3,4'); p.add_argument('--per-fold', type=int, default=4); a = p.parse_args()
        main(Path(a.run or os.environ.get('UNIFIED_RUN') or RUN).resolve(), a.repeats, a.workers, [int(x) for x in a.folds.split(',')], a.per_fold)
