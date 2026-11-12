"""Single-block OOF evaluation of all 18 team methods + baselines (SM unified protocol).

Policy copied from UNIFIED_K/evaluate.py (itself SM's): fixed fold manifest, 5 repeats per case
with a seeded random condition order per repeat, deterministic fields must agree across repeats,
network disabled, oracle-checked bound violations, all code/artifact hashes frozen before timing.

    python evaluate18.py pilot      # fake-guidance smoke run on SM pilot cases, all conditions
    python evaluate18.py freeze     # hash code + inputs, write results/run_20261112/frozen.json
    CV_FOLD=k python evaluate18.py evaluate
"""
import json
import random
import socket
import sys
from u18 import *
from u18 import RUN18, OUT18, UK_RUN, ALL, U18, UK, run, bound, family
from experiments.K3.verification import Oracle

REPEATS = 5
VARIANTS = ('k1_ranking', 'k2_cot', 'k2_repe')
PAPER = REPO/'results/main/generations.jsonl'
CONDITIONS = ALL          # narrowed in freeze() only if CONDITIONS_FILE exists
CONDITIONS_FILE = U18/'conditions.json'
if CONDITIONS_FILE.exists(): CONDITIONS = read(CONDITIONS_FILE)['conditions']

def plain(o):
    if hasattr(o, 'item') and getattr(o, 'ndim', 0) == 0: return o.item()
    if hasattr(o, 'tolist'): return o.tolist()
    if isinstance(o, (set, frozenset)): return sorted(o)
    raise TypeError(type(o).__name__)

_paper = None
def paper_record(q):
    """Original-paper-prompt GPT-4o mini standard response (GM collection 2026-10-13)."""
    global _paper
    if _paper is None:
        _paper = {}
        for line in PAPER.open(encoding='utf-8'):
            d = json.loads(line)
            if d['model'] == 'gpt-4o-mini-2024-07-18' and d['prompt_style'] == 'standard' and d['status'] == 'ok': _paper[d['case_id']] = d
    return _paper.get(q['id'])

def paper_text(d):
    resp = d['response'] if isinstance(d['response'], dict) else json.loads(d['response'])
    return resp['choices'][0]['message'].get('content') or '', resp['choices'][0].get('finish_reason')

def calibration():
    p = UK_RUN/'folds'/f'fold{FOLD}'/'k2_calibration.json'
    return read(p) if p.exists() else None

def context(q, calib):
    first = raw(q); ctx = dict(text=text_of(first), replies={}, finish={}, fold=FOLD, calibration=calib, seed=f'20261112:{FOLD}:{q["id"]}')
    hashes = {'first': first['response_sha256']}
    for v in VARIANTS:
        if raw_path(q, v).exists():
            record = raw(q, v); ctx['replies'][v] = text_of(record); hashes[v] = record['response_sha256']
            ctx['finish'][v] = record['response']['choices'][0].get('finish_reason')
    p = paper_record(q)
    if p is not None:
        t, fin = paper_text(p); ctx['replies']['paper_standard'] = t; ctx['finish']['paper_standard'] = fin; hashes['paper_standard'] = digest(t)
    import methods_s
    if hasattr(methods_s, 'replies'):
        for v, (t, fin, h) in methods_s.replies(q).items():
            ctx['replies'][v] = t; ctx['finish'][v] = fin; hashes[v] = h
    return ctx, hashes

def pilot():
    refs = pilot_reference(); calib = calibration() or dict(quantile=0.0, note='pilot placeholder')
    n = 0; per = {}
    for item in refs:
        q = item['query']; oracle = Oracle(q); text = 'Generated Path: '+json.dumps(item['waypoints'])
        import methods_s; methods_s.prepare(FOLD); methods_s.warmup(q, FOLD)
        keys = ('k1_ranking', 'k2_cot', 'k2_repe', 'paper_standard', 's2_feedback', 's2_control')
        ctx = dict(text=text, replies={k: ('{"ranking":[1,0]}' if k == 'k1_ranking' else text) for k in keys},
                   finish={k: 'stop' for k in keys}, fold=FOLD, calibration=calib, seed=f'pilot:{q["id"]}')
        for method in CONDITIONS:
            r = run(q, method, ctx)
            assert r['valid'], (q['id'], method, r['error']); oracle.check_path(q, r)
            w = bound(method)
            if w: assert r['length'] <= w*item['optimal']+1e-8, (method, q['id'], r['length'], item['optimal'])
            per.setdefault(method, []).append(r['total_seconds']); n += 1
    save(RUN18/'pilot.json', dict(passed=True, checks=n, cases=len(refs), methods=CONDITIONS,
                                  mean_ms={m: 1000*sum(v)/len(v) for m, v in per.items()}, updated_at=now()))
    print(json.dumps(dict(pilot_checks=n, passed=True, slowest=sorted(((1000*sum(v)/len(v), m) for m, v in per.items()), reverse=True)[:8])), flush=True)

def code_files():
    return sorted(set(list(U18.glob('*.py'))+list(U18.glob('jy_src/**/*.py'))+list(UK.glob('*.py'))+list((UK/'sm_src').glob('*.py'))
                      +[REPO/'reproduction/core.py', REPO/'reproduction/proposed.py']+list((REPO/'experiments').glob('K[3-6]/*.py'))))

def freeze():
    assert read(RUN18/'pilot.json')['passed']
    for f in range(5): assert (UK_RUN/'folds'/f'fold{f}'/'k2_calibration.json').exists(), f
    code = code_files()
    artifacts = sorted(list((UK/'inputs/raw/public').glob('*.json'))+list((UK_RUN/'raw').glob('*/*.json'))+[PAPER]
                       +list(UK_RUN.glob('folds/fold*/k2_calibration.json'))+list((UK_RUN/'prompts').glob('*'))+list(PROMPTS.glob('*'))
                       +[p for p in (U18/'inputs_s').rglob('*') if p.is_file()])
    frozen = dict(code={p.relative_to(REPO).as_posix(): sha(p) for p in code},
                  artifacts={p.relative_to(REPO).as_posix(): sha(p) for p in artifacts},
                  methods=CONDITIONS, bounds={m: bound(m) for m in CONDITIONS}, families={m: family(m) for m in CONDITIONS},
                  repeats=REPEATS, fold_manifest_sha256=sha(INPUTS/'fold_manifest.json'), environment=environment())
    record = dict(signature=digest({k: v for k, v in frozen.items() if k != 'environment'}), frozen=frozen, created_at=now())
    dest = RUN18/'frozen.json'
    if dest.exists(): assert read(dest)['signature'] == record['signature'], 'Frozen settings changed'
    else: save(dest, record)
    print(json.dumps(dict(frozen=record['signature'], methods=len(CONDITIONS), code=len(code), artifacts=len(artifacts))), flush=True)

def evaluate():
    frozen = read(RUN18/'frozen.json'); signature = frozen['signature']; truth = read(INPUTS/'oracle.json')
    for rel, h in frozen['frozen']['code'].items(): assert sha(REPO/rel) == h, rel
    assert frozen['frozen']['methods'] == CONDITIONS
    calib = calibration(); assert calib is not None
    qs = cases('test')
    for qi, q in enumerate(qs):
        target = OUT18/'evaluations'/f'{q["id"]}.json'
        if target.exists(): assert read(target)['signature'] == signature; continue
        ctx, hashes = context(q, calib); rows = []; paths = {}; fixed = {}
        for repeat in range(REPEATS):
            order = CONDITIONS[:]; random.Random(f'20261112:{FOLD}:{q["id"]}:{repeat}').shuffle(order)
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
        if qi % 10 == 0 or qi == len(qs)-1:
            st = dict(fold=FOLD, done=qi+1, total=len(qs), complete=False, updated_at=now()); save(OUT18/'evaluation_status.json', st); print(json.dumps(st), flush=True)
    save(OUT18/'evaluation_status.json', dict(fold=FOLD, complete=True, cases=len(qs), trials=len(qs)*len(CONDITIONS)*REPEATS, updated_at=now()))

def deny_network(*a, **kw): raise RuntimeError('Network disabled during offline evaluation')

if __name__ == '__main__':
    socket.socket.connect = deny_network; socket.create_connection = deny_network
    {'pilot': pilot, 'freeze': freeze, 'evaluate': evaluate}[sys.argv[1]]()
