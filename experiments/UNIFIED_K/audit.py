"""Independent audit of the K1-K6 OOF evaluation (does not import report.py).

Checks: frozen signature and code/artifact hashes, case coverage per fold manifest, query
equality with the source dataset, rows = cases x methods x 5, path digests, repeat
determinism, independent Oracle path check + length recomputation, optimal distance vs
inputs/oracle.json (and Oracle recomputation), bound re-judgement, raw response hashes,
cross-harness equality with SM's recorded rows for shared baseline/K3 methods, and
independently recomputed headline aggregates (stored for report.py to compare).

    python audit.py [--run DIR] [--expect-cases 1000] [--sm-run DIR]
Output: <run>/verification.json
"""
import argparse
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from common import *
from experiments.K3.verification import Oracle

SM_RUN_DEFAULT = r'C:\Users\이규민\AppData\Local\Temp\claude\7325ca17f04be9dd\786c343c-f410-4328-8fe7-8299ad316e4f\scratchpad\sm\sm_part\cv1000_20261112\results\s1_s6_cv1000_20261111\runs\run_20261111_01'
SHARED = ('A_GM', 'N_g2h', 'B_llm', 'K3_wall_euclidean', 'K3_wall_octile', 'A_OCT', 'FOCAL_105')
FIELDS = ('valid', 'length', 'operation', 'successor_expansions', 'storage', 'path_sha256')

def main(R, expect, sm_run):
    frozen = read(R/'frozen.json'); f = frozen['frozen']; methods = f['methods']; bounds = f['bounds']
    assert digest({k: v for k, v in f.items() if k != 'environment'}) == frozen['signature'], 'frozen signature'
    code_ok = all(sha(REPO/rel) == h for rel, h in f['code'].items())
    art_bad = [rel for rel, h in f['artifacts'].items() if not (HERE/rel).exists() or sha(HERE/rel) != h]
    assert sha(INPUTS/'fold_manifest.json') == f['fold_manifest_sha256']
    truth = read(INPUTS/'oracle.json'); source = {q['id']: q for q in cases()}
    seen = {}; groups = defaultdict(list); trials = invalid = violations = 0; problems = []
    for p in sorted(R.glob('folds/fold*/evaluations/*.json')):
        d = read(p); q = d['case']; cid = q['id']; fold = q['fold_id']
        assert cid not in seen, cid; seen[cid] = fold
        assert int(p.parent.parent.name[4:]) == fold and cid in MANIFEST['folds'][fold]['test_case_ids'] and q['split'] == 'test'
        for k in ('start', 'goal', 'range_x', 'range_y', 'horizontal_barriers', 'vertical_barriers'): assert q[k] == source[cid][k], (cid, k)
        assert d['signature'] == frozen['signature'], cid
        assert math.isclose(d['optimal'], truth[cid]['optimal'], abs_tol=1e-9)
        assert len(d['rows']) == len(methods)*5 and {(r['method'], r['repeat']) for r in d['rows']} == {(m, i) for m in methods for i in range(5)}, cid
        ref = {}
        for r in d['rows']:
            assert digest(d['paths'][r['path_sha256']]) == r['path_sha256']
            assert r['total_seconds'] > 0 and r['successor_expansions'] >= 0 and r['operation'] >= 0
            fx = {k: r.get(k) for k in FIELDS+('error', 'api_calls')}
            if r['method'] in ref: assert fx == ref[r['method']], (cid, r['method'])
            ref[r['method']] = fx; trials += 1; invalid += int(not r['valid']); violations += int(r['bound_violation'])
        for v, h in d['raw_hashes'].items():
            rp = INPUTS/'raw'/q['stage']/f'{cid}-first.json' if v == 'first' else R/'raw'/v/f'{cid}.json'
            rec = read(rp); assert rec['response_sha256'] == h == digest(rec['response']) and rec['request_sha256'] == digest(rec['request']), (cid, v)
            assert rec['request']['model'] == MODEL
        groups[q['map_id']].append(d)
    fold_set = sorted(set(seen.values()))
    expected = {c for fo in fold_set for c in MANIFEST['folds'][fo]['test_case_ids']}
    assert set(seen) <= expected
    if expect: assert set(seen) == expected and len(seen) == expect, ('coverage', len(seen), expect)
    # independent geometry
    path_checks = 0
    for mid, ds in groups.items():
        oracle = Oracle(ds[0]['case'])
        for d in ds:
            q = d['case']; opt = oracle.distances(q['goal'])[tuple(q['start'])]
            assert math.isclose(opt, d['optimal'], abs_tol=1e-8), q['id']
            done = set()
            for r in d['rows']:
                path = d['paths'][r['path_sha256']]
                if r['valid']:
                    L = sum(math.dist(a, b) for a, b in zip(path, path[1:])); assert math.isclose(L, r['length'], abs_tol=1e-8)
                    if r['path_sha256'] not in done: oracle.check_path(q, dict(valid=True, path=path, length=L)); done.add(r['path_sha256']); path_checks += 1
                w = bounds.get(r['method']); bad = bool(r['valid'] and w and r['length'] > w*opt+1e-8)
                if bad != r['bound_violation']: problems.append(('bound_flag', q['id'], r['method']))
    # SM cross-harness (deterministic fields, repeat 0)
    matched = mismatched = missing = 0; diffs = []
    sm = Path(sm_run) if sm_run else None
    if sm and sm.exists():
        for cid, fold in seen.items():
            sp = sm/'folds'/f'fold{fold}'/'evaluations'/f'{cid}.json'
            if not sp.exists(): missing += 1; continue
            smd = read(sp); ours = read(R/'folds'/f'fold{fold}'/'evaluations'/f'{cid}.json')
            a = {r['method']: r for r in ours['rows'] if r['repeat'] == 0}; b = {r['method']: r for r in smd['rows'] if r['repeat'] == 0}
            for m in SHARED:
                if m not in a or m not in b: continue
                same = all((math.isclose(a[m][k], b[m][k], abs_tol=1e-9) if isinstance(a[m][k], float) and isinstance(b[m][k], float) else a[m][k] == b[m][k]) for k in FIELDS)
                if same: matched += 1
                else: mismatched += 1; diffs.append(dict(case=cid, method=m, ours={k: a[m][k] for k in FIELDS}, sm={k: b[m][k] for k in FIELDS})) if len(diffs) < 20 else None
    # independent aggregates: GM of per-case median planner time vs B_llm, valid counts, mean ms
    med = defaultdict(dict); val = defaultdict(int)
    for ds in groups.values():
        for d in ds:
            by = defaultdict(list)
            for r in d['rows']: by[r['method']].append(r)
            for m, rr in by.items():
                med[m][d['case']['id']] = (statistics.median(x['total_seconds'] for x in rr), rr[0]['valid'])
                val[m] += int(rr[0]['valid'])
    agg = {}
    for m in methods:
        logs = [math.log(t/med['B_llm'][c][0]) for c, (t, _) in med[m].items() if t > 0 and med['B_llm'][c][0] > 0]
        agg[m] = dict(time_ratio=math.exp(sum(logs)/len(logs)), valid_cases=val[m], mean_planner_ms=1000*statistics.mean(t for t, _ in med[m].values()))
    cmp = None
    if (R/'summary.json').exists():
        s = read(R/'summary.json')['OOF']
        cmp = all(math.isclose(agg[m]['time_ratio'], s[m]['total_seconds']['ratio'], rel_tol=1e-9) and agg[m]['valid_cases'] == s[m]['valid_cases'] for m in methods if m in s)
    out = dict(complete=not problems and code_ok and not art_bad and mismatched == 0, cases=len(seen), folds=fold_set, maps=len(groups), methods=len(methods),
               timing_repeats=5, timed_trials=trials, invalid_trials=invalid, bound_violation_trials=violations, oracle_path_checks=path_checks,
               code_hashes_ok=code_ok, artifact_hash_mismatches=art_bad[:20], problems=problems[:50],
               sm_crosscheck=dict(sm_run=str(sm) if sm else None, methods=list(SHARED), fields=list(FIELDS), matched=matched, mismatched=mismatched,
                                  missing_sm_records=missing, examples=diffs),
               aggregates=agg, report_summary_agrees=cmp, updated_at=now())
    save(R/'verification.json', out)
    print(json.dumps({k: out[k] for k in ('complete', 'cases', 'timed_trials', 'invalid_trials', 'bound_violation_trials', 'oracle_path_checks', 'report_summary_agrees')}
                     | dict(sm_matched=matched, sm_mismatched=mismatched)), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run'); p.add_argument('--expect-cases', type=int, default=1000); p.add_argument('--sm-run', default=SM_RUN_DEFAULT)
    a = p.parse_args(); main(Path(a.run or os.environ.get('UNIFIED_RUN') or RUN).resolve(), a.expect_cases, a.sm_run)
