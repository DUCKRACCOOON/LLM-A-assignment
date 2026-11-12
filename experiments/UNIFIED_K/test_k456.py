"""K4/K5/K6 adapter checks: gm parity, synthetic pilot (30), real fold-0 cases (10).

Run from experiments/UNIFIED_K:  ..\\..\\.venv-gm311\\Scripts\\python.exe test_k456.py
"""
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from harness import run, EXTRA, bound
from common import *
from experiments.K3.verification import Oracle
import methods_k456 as M

K456 = list(M.METHODS)

def per_case(path):
    with open(path, encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def parity():
    """Reproduce recorded gm rows (2026-10-25) with the old gm responses through harness.run."""
    texts = {}
    for line in open(REPO/'experiments/K4/results/run_20261025/responses.jsonl', encoding='utf-8'):
        r = json.loads(line); texts[r['case_id']] = r['text']
    public = {c['case_id']: c['query'] for c in read(REPO/'experiments/K3/results/run_20261025/public_cases.json')}
    checks = []
    sources = [('K4_llm', 'K4', 'focal_llm_w105'), ('K4_noLLM', 'K4', 'focal_goal_w105'), ('K5_rr', 'K5', 'mha_rr_w200'),
               ('K6_main', 'K6', 'adaptive_wall_w200'), ('K6_w105', 'K6_w105', 'adaptive_wall_w105'),
               ('K6_w105_noguide', 'K6_w105', 'no_guide_wall_w105')]
    for name, folder, gm_method in sources:
        rows = [r for r in per_case(REPO/f'experiments/{folder}/results/run_20261025/public/per_case.csv') if r['method'] == gm_method][:3]
        assert len(rows) == 3, (folder, gm_method)
        for row in rows:
            q = dict(public[row['case_id']], id=row['case_id'], stage='public')
            r = run(q, name, dict(text=texts[row['case_id']]))
            exp_key = 'expansions' if folder == 'K4' else 'successor_expansions'
            want_exp = int(row[exp_key])-(int(row['valid'] in ('True', '1')) if folder == 'K4' else 0)
            got = (r['extra']['engine_operation'], r['successor_expansions'], r['storage'], round(r['length'], 9))
            want = (int(row['operation']), want_exp, int(row['storage']), round(float(row['length']), 9))
            assert got == want, (name, row['case_id'], got, want)
            checks.append((name, row['case_id']))
    # K5 DTS: stochastic, reproduce recorded per-seed rows with the recorded seed.
    seeds = read(REPO/'experiments/K5/results/run_20261025/public/per_seed.json')
    rows = [s for s in seeds if s['method'] == 'mha_dts_w200'][:3]
    for row in rows:
        q = dict(public[row['case_id']], id=row['case_id'], stage='public')
        from experiments.K5.search import run_once
        res = run_once(q, 'mha_dts_w200', texts[row['case_id']], seed=row['seed'])
        assert (res['operation'], res['successor_expansions'], res['storage']) == (row['operation'], row['successor_expansions'], row['storage']), row
        checks.append(('K5_dts(engine,seed)', row['case_id']))
    return checks

def exercise(qs, ctx_of, optimal_of, label):
    times = defaultdict(list); n = 0
    for q in qs:
        oracle = Oracle(q); ctx = ctx_of(q); opt = optimal_of(q)
        for m in K456:
            r = run(q, m, ctx)
            assert r['valid'], (label, q['id'], m, r['error'])
            oracle.check_path(q, r)
            w = bound(m)
            if w: assert r['length'] <= w*opt+1e-8, (label, q['id'], m, r['length'], opt)
            assert r['operation'] > 0 and r['successor_expansions'] >= 0 and r['storage'] > 0
            r2 = run(q, m, ctx)
            assert (r2['path'], r2['operation'], r2['successor_expansions']) == (r['path'], r['operation'], r['successor_expansions']), (m, 'nondeterministic')
            times[m].append(r['total_seconds']*1000); n += 1
    return n, {m: round(statistics.median(v), 3) for m, v in times.items()}

if __name__ == '__main__':
    checks = parity(); print('parity rows reproduced:', len(checks), checks)
    refs = pilot_reference(); pilot_opt = {x['query']['id']: x['optimal'] for x in refs}
    n, t = exercise([x['query'] for x in refs], lambda q: dict(text='Generated Path: '+json.dumps(next(x['waypoints'] for x in refs if x['query']['id'] == q['id']))),
                    lambda q: pilot_opt[q['id']], 'pilot')
    print('pilot checks:', n, 'median ms', t)
    oracle = read(INPUTS/'oracle.json'); qs = cases('test', fold=0)[:10]
    n2, t2 = exercise(qs, lambda q: dict(text=text_of(raw(q))), lambda q: oracle[q['id']]['optimal'], 'real')
    print('real fold0 checks:', n2, 'median ms', t2)
    print('ALL PASSED')
