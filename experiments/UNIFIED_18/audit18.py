"""Independent audit of the 18-method single block (does not import report18.py).

Checks frozen signature + code hashes, coverage (1000 cases, each in its OOF fold), rows =
cases x conditions x 5, path digests, repeat determinism, independent Oracle path check +
length recomputation, optimal distance recomputation, bound re-judgement, and cross-checks
the shared/K conditions against the earlier UNIFIED_K run and the S conditions against SM's
recorded rows (deterministic fields, repeat 0).

    python audit18.py    ->  results/run_20261112/verification18.json
"""
import math
from collections import defaultdict
from u18 import *
from u18 import RUN18 as R, UK_RUN, U18
from experiments.K3.verification import Oracle

FIELDS = ('valid', 'length', 'operation', 'successor_expansions', 'storage', 'path_sha256')

def main():
    frozen = read(R/'frozen.json'); f = frozen['frozen']; methods = f['methods']; bounds = f['bounds']
    assert digest({k: v for k, v in f.items() if k != 'environment'}) == frozen['signature']
    code_ok = all(sha(REPO/rel) == h for rel, h in f['code'].items())
    truth = read(INPUTS/'oracle.json'); seen = {}; groups = defaultdict(list); trials = invalid = viol = 0; problems = []
    for p in sorted(R.glob('folds/fold*/evaluations/*.json')):
        d = read(p); q = d['case']; cid = q['id']; fold = q['fold_id']
        assert cid not in seen; seen[cid] = fold
        assert cid in MANIFEST['folds'][fold]['test_case_ids'] and d['signature'] == frozen['signature']
        assert len(d['rows']) == len(methods)*5
        ref = {}
        for r in d['rows']:
            assert digest(d['paths'][r['path_sha256']]) == r['path_sha256']
            fx = {k: r.get(k) for k in FIELDS+('error', 'api_calls')}
            if r['method'] in ref: assert fx == ref[r['method']], (cid, r['method'])
            ref[r['method']] = fx; trials += 1; invalid += int(not r['valid']); viol += int(r['bound_violation'])
        groups[q['map_id']].append(d)
    path_checks = 0
    for mid, ds in groups.items():
        oracle = Oracle(ds[0]['case'])
        for d in ds:
            q = d['case']; opt = oracle.distances(q['goal'])[tuple(q['start'])]
            assert math.isclose(opt, truth[q['id']]['optimal'], abs_tol=1e-8)
            done = set()
            for r in d['rows']:
                path = d['paths'][r['path_sha256']]
                if r['valid']:
                    L = sum(math.dist(a, b) for a, b in zip(path, path[1:])); assert math.isclose(L, r['length'], abs_tol=1e-8)
                    if r['path_sha256'] not in done: oracle.check_path(q, dict(valid=True, path=path, length=L)); done.add(r['path_sha256']); path_checks += 1
                w = bounds.get(r['method']); bad = bool(r['valid'] and w and r['length'] > w*opt+1e-8)
                if bad != r['bound_violation']: problems.append(('bound_flag', q['id'], r['method']))
    # cross-checks (deterministic fields, repeat 0)
    def cross(other_dir, names):
        ok = bad = miss = 0; diffs = []
        for cid, fold in seen.items():
            op = other_dir/'folds'/f'fold{fold}'/'evaluations'/f'{cid}.json'
            if not op.exists(): miss += 1; continue
            o = {r['method']: r for r in read(op)['rows'] if r['repeat'] == 0}
            u = {r['method']: r for r in read(R/'folds'/f'fold{fold}'/'evaluations'/f'{cid}.json')['rows'] if r['repeat'] == 0}
            for m in names:
                if m not in o or m not in u: continue
                if all(o[m].get(k) == u[m].get(k) for k in FIELDS): ok += 1
                else: bad += 1; diffs.append((cid, m)) if len(diffs) < 20 else None
        return dict(matched=ok, mismatched=bad, missing_cases=miss, examples=diffs)
    uk_names = [m for m in read(UK_RUN/'frozen.json')['frozen']['methods'] if m in methods]
    out = dict(signature=frozen['signature'], code_hashes_ok=code_ok, cases=len(seen), conditions=len(methods), timed_trials=trials,
               invalid_trials=invalid, bound_violation_trials=viol, oracle_path_checks=path_checks, problems=problems,
               repeat_determinism='all 5 repeats identical on deterministic fields',
               crosscheck_vs_unified_k_run=cross(UK_RUN, uk_names), created_at=now())
    save(R/'verification18.json', out); print(json.dumps({k: v for k, v in out.items() if k != 'problems'}, ensure_ascii=False)[:1500])

if __name__ == '__main__':
    main()
