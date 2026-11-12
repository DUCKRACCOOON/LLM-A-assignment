"""OOF evaluation of K1-K6 under the SM unified protocol (adapted from sm_src/evaluate.py).

Same policy as SM: fixed fold manifest, five repeats per case with a seeded random method
order per repeat, deterministic fields must agree across repeats, network disabled during
measurement, oracle-checked bound violations, all code/artifact hashes frozen before timing.

    python evaluate.py pilot            # 30 disjoint synthetic cases, fake guidance, all methods
    python evaluate.py freeze           # hash code, prompts, raw responses, calibrations
    CV_FOLD=k python evaluate.py evaluate
"""
import json
import random
import socket
import sys
from common import *
from harness import run, METHODS, bound, family
from experiments.K3.verification import Oracle

REPEATS = 5
VARIANTS = ('k1_ranking', 'k2_cot', 'k2_repe')

def plain(o):
    """numpy scalars/arrays and tuples in engine extras -> JSON values."""
    if hasattr(o, 'item') and getattr(o, 'ndim', 0) == 0: return o.item()
    if hasattr(o, 'tolist'): return o.tolist()
    if isinstance(o, (set, frozenset)): return sorted(o)
    raise TypeError(type(o).__name__)

def calibration():
    p = OUT/'k2_calibration.json'
    return read(p) if p.exists() else None

def context(q, calib):
    first = raw(q); ctx = dict(text=text_of(first), replies={}, finish={}, fold=FOLD, calibration=calib, seed=f'20261112:{FOLD}:{q["id"]}')
    hashes = {'first': first['response_sha256']}
    for v in VARIANTS:
        if raw_path(q, v).exists():
            record = raw(q, v); ctx['replies'][v] = text_of(record); hashes[v] = record['response_sha256']
            ctx['finish'][v] = record['response']['choices'][0].get('finish_reason')
    return ctx, hashes

def pilot():
    refs = pilot_reference(); calib = calibration() or dict(quantile=0.0, note='pilot placeholder')
    n = 0; per = {}
    for item in refs:
        q = item['query']; oracle = Oracle(q); text = 'Generated Path: '+json.dumps(item['waypoints'])
        ctx = dict(text=text, replies={'k1_ranking': '{"ranking":[1,0]}', 'k2_cot': text, 'k2_repe': text},
                   finish={'k1_ranking': 'stop', 'k2_cot': 'stop', 'k2_repe': 'stop'}, fold=FOLD, calibration=calib, seed=f'pilot:{q["id"]}')
        for method in METHODS:
            r = run(q, method, ctx)
            assert r['valid'], (q['id'], method, r['error']); oracle.check_path(q, r)
            w = bound(method)
            if w: assert r['length'] <= w*item['optimal']+1e-8, (method, q['id'], r['length'], item['optimal'])
            per.setdefault(method, []).append(r['total_seconds']); n += 1
    save(RUN/'pilot.json', dict(passed=True, checks=n, cases=len(refs), methods=METHODS,
                                mean_ms={m: 1000*sum(v)/len(v) for m, v in per.items()}, updated_at=now()))
    print(json.dumps(dict(pilot_checks=n, passed=True)), flush=True)

def freeze():
    assert read(RUN/'pilot.json')['passed']
    status = RUN/'collection/collection_status.json'; assert read(status)['complete']
    for f in range(5): assert (RUN/'folds'/f'fold{f}'/'k2_calibration.json').exists(), f
    code = sorted(list(HERE.glob('*.py'))+list((HERE/'sm_src').glob('*.py'))+[REPO/'reproduction/core.py', REPO/'reproduction/proposed.py']
                  +list((REPO/'experiments').glob('K[3-6]/*.py')))
    artifacts = sorted(list((INPUTS/'raw/public').glob('*.json'))+list((RUN/'raw').glob('*/*.json'))
                       +list(RUN.glob('folds/fold*/k2_calibration.json'))+list((RUN/'prompts').glob('*'))+list(PROMPTS.glob('*')))
    frozen = dict(code={p.relative_to(REPO).as_posix(): sha(p) for p in code},
                  artifacts={p.relative_to(HERE).as_posix(): sha(p) for p in artifacts},
                  methods=METHODS, bounds={m: bound(m) for m in METHODS}, families={m: family(m) for m in METHODS},
                  repeats=REPEATS, fold_manifest_sha256=sha(INPUTS/'fold_manifest.json'), environment=environment())
    record = dict(signature=digest({k: v for k, v in frozen.items() if k != 'environment'}), frozen=frozen, created_at=now())
    dest = RUN/'frozen.json'
    if dest.exists(): assert read(dest)['signature'] == record['signature'], 'Frozen settings changed'
    else: save(dest, record)
    print(json.dumps(dict(frozen=record['signature'], methods=len(METHODS), code=len(code), artifacts=len(artifacts))), flush=True)

def evaluate():
    frozen = read(RUN/'frozen.json'); signature = frozen['signature']; truth = read(INPUTS/'oracle.json')
    for rel, h in frozen['frozen']['code'].items(): assert sha(REPO/rel) == h, rel
    assert frozen['frozen']['methods'] == METHODS
    calib = calibration(); assert calib is not None
    qs = cases('test')
    for qi, q in enumerate(qs):
        target = OUT/'evaluations'/f'{q["id"]}.json'
        if target.exists(): assert read(target)['signature'] == signature; continue
        ctx, hashes = context(q, calib); rows = []; paths = {}; fixed = {}
        for repeat in range(REPEATS):
            order = METHODS[:]; random.Random(f'20261112:{FOLD}:{q["id"]}:{repeat}').shuffle(order)
            for method in order:
                r = run(q, method, ctx); path = r.pop('path'); ph = digest(path); paths[ph] = path
                w = bound(method); r['bound_violation'] = bool(r['valid'] and w and r['length'] > w*truth[q['id']]['optimal']+1e-8)
                det = {k: r.get(k) for k in ('valid', 'error', 'length', 'operation', 'successor_expansions', 'storage', 'api_calls')}; det['path_sha256'] = ph
                if method in fixed: assert fixed[method] == det, (q['id'], method, 'repeat changed')
                fixed[method] = det
                rows.append(dict(method=method, family=family(method), repeat=repeat, path_sha256=ph,
                                 **{k: v for k, v in r.items() if k not in ('targets', 'final_target')}))
        rows = json.loads(json.dumps(rows, default=plain))
        save(target, dict(signature=signature, case=q, optimal=truth[q['id']]['optimal'], raw_hashes=hashes, paths=paths, rows=rows))
        if qi % 20 == 0 or qi == len(qs)-1:
            st = dict(fold=FOLD, done=qi+1, total=len(qs), complete=False, updated_at=now()); save(OUT/'evaluation_status.json', st); print(json.dumps(st), flush=True)
    save(OUT/'evaluation_status.json', dict(fold=FOLD, complete=True, cases=len(qs), trials=len(qs)*len(METHODS)*REPEATS, updated_at=now()))

def deny_network(*a, **kw): raise RuntimeError('Network disabled during offline evaluation')

if __name__ == '__main__':
    socket.socket.connect = deny_network; socket.create_connection = deny_network
    {'pilot': pilot, 'freeze': freeze, 'evaluate': evaluate}[sys.argv[1]]()
