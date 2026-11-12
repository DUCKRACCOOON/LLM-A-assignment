"""Aggregate the single-block 18-method run: LLM-A* scoreboard + 18 methods vs LLM-A* (no normalisation).

Seven metric groups (decided with the team): 1 speed, 2 computation, 3 path quality, 4 LLM cost,
5 memory, 6 correctness/reproducibility, 7 LLM contribution (same-engine control).
Statistics follow SM's report (per-case 5-repeat medians, paired per-case ratios, geometric mean,
map-level bootstrap within folds, 2000 draws) and add standard deviations as the TA notice asks:
geometric SD of per-case ratios, arithmetic mean ± SD of raw values, and repeat-timing CV.
Table-1 style scores use A* (A_GM) as denominator like the paper (geometric mean x100).

    python report18.py
Outputs in results/run_20261112: per_case.csv, summary18.json, report18.md, viz_data.json
"""
import csv
import math
import statistics as st
from collections import Counter, defaultdict
import numpy as np
from u18 import *
from u18 import RUN18, UK_RUN, U18, UK

R = RUN18; BOOT = 2000; SEED = 20261112
# ---- the 18 methods (main condition each) and their same-engine no-LLM control (group 7)
MAIN = [  # (label, condition or seed-group, owner, control, online LLM use)
    ('S1', 'S1', 'SM', 'V_GATE_NORMALIZED', True), ('S2', 'S2_feedback', 'SM', 'S2_control', True),
    ('S3', 'S3', 'SM', 'N_g2h', True), ('S4', 'S4@', 'SM', 'S4_untrained', False),
    ('S5', 'S5', 'SM', 'A_OCT', False), ('S6', 'S6_program', 'SM', 'S6_rule', False),
    ('J1', 'J1_main', 'JY', 'J1_straight', True), ('J2', 'J2_main', 'JY', 'J2_uniform', True),
    ('J3', 'J3_main', 'JY', 'J3_noLLM', True), ('J4', 'J4_main', 'JY', 'J4_graph_octile', False),
    ('J5', 'J5_main', 'JY', 'J5_bidijkstra', False), ('J6', 'J6_main', 'JY', 'J6_full', True),
    ('K1', 'K1_llm', 'GM', 'K1_rule', True), ('K2', 'K2_conformal', 'GM', 'K2_nocal', True),
    ('K3', 'K3_wall_euclidean', 'GM', 'N_g2h', False), ('K4', 'K4_llm', 'GM', 'K4_noLLM', True),
    ('K5', 'K5@', 'GM', 'K5_noLLM', True), ('K6', 'K6_main', 'GM', 'K3_wall_octile', True)]
SEEDS = {'S4@': ['S4_seed11', 'S4_seed22', 'S4_seed33'], 'K5@': ['K5_dts_s0', 'K5_dts_s1', 'K5_dts_s2']}
REF = ['A_GM', 'B_llm', 'B_llm_paper', 'N_g2h', 'A_OCT', 'FOCAL_105', 'K6_w105']
GM_ENGINE = {'A_GM', 'N_g2h', 'B_llm', 'B_llm_paper', 'K3_wall_euclidean', 'K3_wall_octile', 'S1', 'S2_feedback', 'S2_control', 'S3',
             'V_GATE_GM', 'V_GATE_NORMALIZED'}
PAPER_FILE = REPO/'results/main/generations.jsonl'

def raw_seconds_tokens():
    """case -> variant -> (seconds, prompt_tokens, completion_tokens)."""
    out = defaultdict(dict)
    def add(case, v, sec, usage): out[case][v] = (sec, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
    for p in (UK/'inputs/raw/public').glob('*-first.json'):
        d = read(p); add(d['case_id'], 'first', d['request_seconds'], d['response'].get('usage', {}))
    for v in ('k1_ranking', 'k2_cot', 'k2_repe'):
        for p in (UK_RUN/'raw'/v).glob('*.json'):
            d = read(p); add(d['case_id'], v, d['request_seconds'], d['response'].get('usage', {}))
    for p in (U18/'inputs_s/raw/public').glob('*-*.json'):
        d = read(p)
        if d.get('variant') in ('feedback', 'control'): add(d['case_id'], 's2_'+d['variant'], d['request_seconds'], d['response'].get('usage', {}))
    for line in PAPER_FILE.open(encoding='utf-8'):
        d = json.loads(line)
        if d['model'] == 'gpt-4o-mini-2024-07-18' and d['prompt_style'] == 'standard':
            resp = d['response'] if isinstance(d['response'], dict) else json.loads(d['response'])
            add(d['case_id'], 'paper_standard', float(d['seconds']), resp.get('usage', {}))
    return out

def used(method, row, hashes):
    if not row['api_calls']: return []
    if method.startswith('K1'): return ['k1_ranking']
    if method.startswith('K2') or method == 'J2_main': return ['first', 'k2_cot', 'k2_repe']
    if method == 'S2_feedback': return ['first']+(['s2_feedback'] if 's2_feedback' in hashes else [])
    if method == 'S2_control': return ['first']+(['s2_control'] if 's2_control' in hashes else [])
    if method == 'B_llm_paper': return ['paper_standard']
    return ['first']

def paper_op(m, r):
    ex = r.get('extra') or {}
    if 'paper_operation' in ex: return ex['paper_operation']
    if m in GM_ENGINE or m.startswith(('K3', 'K4', 'K5', 'K6')): return r['operation']
    return None   # K1/K2/FOCAL/A_OCT: engine-specific; not Table-1 comparable

def paper_storage(m, r):
    ex = r.get('extra') or {}
    if 'paper_storage' in ex: return ex['paper_storage']
    if m in GM_ENGINE or m.startswith(('K3', 'K4', 'K5', 'K6')): return r['storage']
    return None

def per_case():
    frozen = read(R/'frozen.json')['frozen']; methods = frozen['methods']; bounds = frozen['bounds']
    api = raw_seconds_tokens(); rows = []
    for p in sorted(R.glob('folds/fold*/evaluations/*.json')):
        d = read(p); q = d['case']; g = defaultdict(list)
        for r in d['rows']: g[r['method']].append(r)
        for m in methods:
            rr = sorted(g[m], key=lambda x: x['repeat']); r = rr[0]; ts = [x['total_seconds'] for x in rr]
            us = [api[q['id']][v] for v in used(m, r, d['raw_hashes']) if v in api[q['id']]]
            ex = r.get('extra') or {}
            exp = ex.get('successor_expansions_corrected', r['successor_expansions'])
            t = st.median(ts)
            rows.append(dict(case_id=q['id'], map_id=q['map_id'], fold=q['fold_id'], method=m, valid=r['valid'], optimal=d['optimal'],
                             length=r['length'], heap_pops=r['operation'], paper_operation=paper_op(m, r), expansions=exp,
                             storage=r['storage'], paper_storage=paper_storage(m, r), peak_open=r.get('measured_peak_open'),
                             planner_s=t, planner_cv=(st.pstdev(ts)/st.mean(ts) if st.mean(ts) > 0 else 0.0),
                             api_s=sum(u[0] for u in us), e2e_s=t+sum(u[0] for u in us), responses=len(us),
                             prompt_tokens=sum(u[1] for u in us), completion_tokens=sum(u[2] for u in us),
                             api_calls=r['api_calls'], bound=bounds.get(m), bound_violation=r['bound_violation']))
    # seed groups: per-case mean of numeric fields; keep seed rows for quality tails
    by = defaultdict(dict)
    for r in rows: by[r['method']][r['case_id']] = r
    for grp, ms in SEEDS.items():
        for cid in by[ms[0]]:
            parts = [by[m][cid] for m in ms]; base = dict(parts[0]); base['method'] = grp
            base['valid'] = all(x['valid'] for x in parts)
            for k in ('length', 'heap_pops', 'paper_operation', 'expansions', 'storage', 'paper_storage', 'peak_open', 'planner_s', 'planner_cv',
                      'api_s', 'e2e_s', 'responses', 'prompt_tokens', 'completion_tokens', 'api_calls'):
                vals = [x[k] for x in parts]
                base[k] = None if any(v is None for v in vals) else st.mean(vals)
            base['bound_violation'] = any(x['bound_violation'] for x in parts); rows.append(base)
    return rows, methods, bounds

def ratio(rr, bb, key):
    pairs = []; dirs = Counter(improved=0, equal=0, worsened=0)
    for r in rr:
        b = bb.get(r['case_id']); x = r.get(key); y = b.get(key) if b else None
        if x is None or y is None or (key == 'length' and not (r['valid'] and b['valid'])): continue
        eq = math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-8 if key == 'length' else 1e-12)
        dirs['equal' if eq else 'improved' if x < y else 'worsened'] += 1
        if x > 0 and y > 0: pairs.append((r['fold'], r['map_id'], math.log(x/y)))
    if not pairs: return None
    logs = [x[2] for x in pairs]; grp = defaultdict(list)
    for f, m, x in pairs: grp[(f, m)].append(x)
    rng = np.random.default_rng(SEED); sums = np.zeros(BOOT); counts = np.zeros(BOOT)
    for fold in sorted({f for f, _ in grp}):
        vals = [v for (f, _), v in grp.items() if f == fold]; ss = np.array([sum(v) for v in vals]); nn = np.array([len(v) for v in vals])
        ch = rng.integers(0, len(vals), size=(BOOT, len(vals))); sums += ss[ch].sum(1); counts += nn[ch].sum(1)
    boots = np.exp(sums/counts)
    return dict(ratio=math.exp(st.mean(logs)), ci95=np.quantile(boots, [.025, .975]).tolist(), gsd=math.exp(st.pstdev(logs)),
                n=len(pairs), directions=dict(dirs))

def msd(vals):
    vals = [v for v in vals if v is not None]
    return (st.mean(vals), st.pstdev(vals)) if vals else (None, None)

def stats(m, rr, bb, aa, bound, quality_rows):
    valid = [r for r in rr if r['valid']]
    q = [r['length']/r['optimal'] for r in quality_rows if r['valid']]
    out = dict(method=m, cases=len(rr), valid=len(valid), bound=bound, bound_violations=sum(bool(r['bound_violation']) for r in rr))
    for key in ('planner_s', 'e2e_s', 'expansions', 'heap_pops', 'paper_operation', 'storage', 'paper_storage', 'peak_open', 'length'):
        out[key+'_vs_llm'] = ratio(rr, bb, key); out[key+'_vs_astar'] = ratio(rr, aa, key)
        out[key+'_mean_sd'] = msd([r[key] for r in rr])
    out['planner_cv_median'] = st.median(r['planner_cv'] for r in rr)
    out.update(lstar_gm=math.exp(st.mean(math.log(x) for x in q)) if q else None, lstar_sd=st.pstdev(q) if q else None,
               optimal=sum(abs(x-1) <= 1e-8 for x in q), quality_n=len(q),
               p95_excess=100*(float(np.quantile(q, .95))-1) if q else None, max_excess=100*(max(q)-1) if q else None,
               over5=sum(x > 1.05+1e-8 for x in q), over10=sum(x > 1.1+1e-8 for x in q),
               responses=sum(r['responses'] for r in rr), prompt_tokens=sum(r['prompt_tokens'] for r in rr),
               completion_tokens=sum(r['completion_tokens'] for r in rr), api_s_mean_sd=msd([r['api_s'] for r in rr]))
    out['usd_list'] = out['prompt_tokens']*0.15/1e6+out['completion_tokens']*0.60/1e6
    return out

def build():
    rows, methods, bounds = per_case()
    with (R/'per_case.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = defaultdict(list)
    for r in rows: by[r['method']].append(r)
    idx = lambda m: {r['case_id']: r for r in by[m]}
    bb, aa = idx('B_llm'), idx('A_GM')
    def S(m):
        qrows = [r for s in SEEDS[m] for r in by[s]] if m in SEEDS else by[m]
        bnd = bounds.get(SEEDS[m][0]) if m in SEEDS else bounds.get(m)
        return stats(m, by[m], bb, aa, bnd, qrows)
    summary = dict(reference={m: S(m) for m in REF if m in by}, main={}, controls={}, all={m: S(m) for m in methods})
    for label, cond, owner, ctrl, online in MAIN:
        s = S(cond); s.update(label=label, owner=owner, condition=cond, control=ctrl, online_llm=online)
        c = idx(ctrl)
        s['vs_control'] = {k: ratio(by[cond], c, k) for k in ('planner_s', 'expansions', 'paper_operation', 'length', 'e2e_s')}
        summary['main'][label] = s
    summary['repeat_determinism'] = 'asserted per case in evaluate18 (all 5 repeats identical deterministic fields)'
    save(R/'summary18.json', summary)
    print(json.dumps(dict(rows=len(rows), main=len(summary['main']))), flush=True)
    return summary

if __name__ == '__main__':
    build()
