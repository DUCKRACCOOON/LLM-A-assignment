"""J1-J6 port checks for methods_j.py through UNIFIED_K harness.run.

    ..\\..\\.venv-gm311\\Scripts\\python.exe test_j.py all          # equivalence (1000) + smoke (50) + pilot (30)
    ..\\..\\.venv-gm311\\Scripts\\python.exe test_j.py equivalence [--limit N]

equivalence: replays JY's own frozen responses (J*_full/results/run_20261111/responses.jsonl.gz on
origin/jy d72632c3 = results/main standard/CoT/RePE) on all 1000 public cases and compares every
deterministic field with JY's recorded per-case rows (J*_full/results/run_20261111/evaluations.jsonl.gz):
valid, path, length, successor_expansions, storage, heap-pop operation (J3-J6; JY J1/J2 recorded None),
and for J6 the effective selection-trace sha256 (engine re-run with trace=True outside the harness).
Every valid path is checked by the independent integer Oracle (experiments.K3.verification) and against
the declared bound using UNIFIED_K/inputs/oracle.json optimal lengths.
smoke: unified inputs (SM 'first' standard + UNIFIED_K k2_cot/k2_repe) on 50 cases (every 20th public case),
two runs each for determinism. pilot: 30 synthetic SM pilot cases with fake waypoint text.
No network/API use: all responses are read from disk / git objects.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

U18 = Path(__file__).resolve().parent
UK = U18.parent/'UNIFIED_K'
if str(UK) not in sys.path: sys.path.insert(0, str(UK))
import common                                    # noqa: E402  (REPO, sm_src, UNIFIED_K on sys.path)
from common import REPO, cases, raw, raw_path, text_of, read, save, pilot_reference, INPUTS   # noqa: E402
import harness                                   # noqa: E402
if str(U18) in sys.path: sys.path.remove(str(U18))
sys.path.insert(0, str(U18))
import methods_j as M                            # noqa: E402
from experiments.K3.verification import Oracle   # noqa: E402

for _name, _spec in M.METHODS.items():
    assert _name not in harness.BASE, _name
    harness.EXTRA.setdefault(_name, _spec)
NAMES = list(M.METHODS)
OUT = U18/'results/j_port'
COMMIT = M.JY_COMMIT
FIELDS = ('valid', 'path', 'length', 'successor_expansions', 'storage', 'operation')


def git_blob(path):
    return subprocess.run(['git', 'show', f'{COMMIT}:{path}'], cwd=REPO, check=True, capture_output=True).stdout


def jsonl_gz(path):
    with gzip.open(io.BytesIO(git_blob(path)), 'rt', encoding='utf-8') as f: return [json.loads(line) for line in f]


def content(record): return record['response']['choices'][0]['message'].get('content') or ''


def finish_reason(record): return record['response']['choices'][0].get('finish_reason')


def jy_inputs():
    """JY replies per family (only J1/J2/J3/J6 use responses) and JY rows keyed (jy_condition, case_id)."""
    replies, rows, response_sha = {}, {}, {}
    for j in range(1, 7):
        base = f'experiments/J{j}_full/results/run_20261111/'
        if j in (1, 2, 3, 6):
            blob = git_blob(base+'responses.jsonl.gz'); response_sha[f'J{j}'] = hashlib.sha256(blob).hexdigest()
            fam = {}
            for r in jsonl_gz(base+'responses.jsonl.gz'): fam.setdefault(r['case_id'], {})[r.get('prompt_style', 'standard')] = r
            replies[f'J{j}'] = fam
        for r in jsonl_gz(base+'evaluations.jsonl.gz'): rows[(f'J{j}', r['method'], r['case_id'])] = r
    jy_cases = json.loads(git_blob('experiments/J1_full/results/run_20261111/cases.json'))
    return replies, rows, jy_cases, response_sha


def jy_ctx(fam_replies, cid):
    rep = fam_replies.get(cid, {})
    std = rep.get('standard')
    ctx = dict(text=content(std) if std else '', replies={}, finish={})
    for style, key in (('cot', 'k2_cot'), ('repe', 'k2_repe')):
        if style in rep: ctx['replies'][key] = content(rep[style]); ctx['finish'][key] = finish_reason(rep[style])
    return ctx


def same_path(a, b): return [list(map(int, p)) for p in (a or [])] == [list(map(int, p)) for p in (b or [])]


def check_paper_counts(r, name, counts):
    """paper_operation/aux_operation must partition the harness heap-pop count (J1/J2: only the stale
    anchor-heap discards of paper_operation are heap pops; OPEN selections are array scans)."""
    e = r['extra']
    assert all(k in e for k in ('paper_operation', 'paper_storage', 'aux_operation')), name
    heap_part = e['anchor_heap_pops'] if M.METHODS[name]['family'] in ('J1', 'J2') else e['paper_operation']
    assert heap_part+e['aux_operation'] == r['operation'], (name, heap_part, e['aux_operation'], r['operation'])
    assert e['paper_storage'] > 0 and e['paper_operation'] >= 0, name   # J1: verified incumbent may end before any selection
    counts['paper_operation_zero'] += e['paper_operation'] == 0
    counts['paper_count_checks'] += 1


def check_quality(q, r, name, oracle, opt, counts):
    check_paper_counts(r, name, counts)
    assert r['valid'], (q['id'], name, r['error'])
    oracle.check_path(q, r); counts['oracle_path_checks'] += 1
    assert r['length'] >= opt-1e-7, (q['id'], name, r['length'], opt)
    w = harness.bound(name)
    if w:
        assert r['length'] <= w*opt+1e-7, (q['id'], name, 'bound violation', r['length'], opt, w); counts['bound_checks'] += 1


def equivalence(limit=None):
    replies, rows, jy_cases, response_sha = jy_inputs()
    optimal = read(INPUTS/'oracle.json'); unified = {q['id']: q for q in cases()}
    jy_cases = jy_cases[:limit] if limit else jy_cases
    match = {n: Counter() for n in NAMES}; mismatches = defaultdict(list); counts = Counter()
    times = defaultdict(list); oracle = None; oracle_map = None; started = time.time()
    for i, c in enumerate(jy_cases):
        cid = c['case_id']; q = dict(c['query'], id=cid, stage='public')
        u = unified[cid]
        assert all(u[k] == q[k] for k in ('start', 'goal', 'range_x', 'range_y', 'horizontal_barriers', 'vertical_barriers')), cid
        if oracle_map != c['map_index']: oracle, oracle_map = Oracle(q), c['map_index']
        opt = optimal[cid]['optimal']
        for name in NAMES:
            spec = M.METHODS[name]; fam = spec['family']
            ctx = jy_ctx(replies[fam], cid) if fam in replies else dict(text='', replies={}, finish={})
            r = harness.run(q, name, ctx)
            row = rows[(fam, spec['jy_condition'], cid)]
            assert math.isclose(row['optimum'], opt, abs_tol=1e-7), (cid, row['optimum'], opt)
            got = dict(valid=r['valid'], path=r['path'], length=r['length'], successor_expansions=r['successor_expansions'],
                       storage=r['storage'], operation=r['operation'])
            for f in FIELDS:
                if f == 'operation' and row.get('operation') is None: continue   # JY J1/J2 engine records no total pop count
                ok = (same_path(got[f], row[f]) if f == 'path' else
                      (got[f] is None and row[f] is None) or (got[f] is not None and row[f] is not None and abs(got[f]-row[f]) <= 1e-9) if f == 'length' else
                      got[f] == row[f])
                match[name][f] += ok; match[name][f+'_n'] += 1
                if not ok and len(mismatches[name]) < 10: mismatches[name].append(dict(case_id=cid, field=f, got=None if f == 'path' else got[f], jy=None if f == 'path' else row[f]))
            if r['valid']: check_quality(q, r, name, oracle, opt, counts)
            else: counts['invalid_'+name] += 1
            if fam == 'J6':
                t = M.jys.run_once(q, r['extra']['jy_alias'], text=ctx['text'], trace=True)
                h = hashlib.sha256(json.dumps(t['pop_trace'], separators=(',', ':')).encode()).hexdigest()
                ok = h == row['effective_trace_sha256']; match[name]['trace_sha256'] += ok; match[name]['trace_sha256_n'] += 1
            times[name].append(r['total_seconds']*1000); counts['runs'] += 1
        if (i+1) % 50 == 0:
            print(f'equivalence {i+1}/{len(jy_cases)} cases, {time.time()-started:.0f}s', flush=True)
    summary = {n: {f: f'{match[n][f]}/{match[n][f+"_n"]}' for f in FIELDS+('trace_sha256',) if match[n][f+'_n']} for n in NAMES}
    all_match = all(match[n][f] == match[n][f+'_n'] for n in NAMES for f in FIELDS+('trace_sha256',) if match[n][f+'_n'])
    return dict(cases=len(jy_cases), jy_commit=COMMIT, jy_response_files_sha256=response_sha, all_match=all_match,
                match=summary, mismatch_examples=dict(mismatches), counts=dict(counts),
                median_ms={n: round(statistics.median(v), 3) for n, v in times.items()},
                mean_ms={n: round(statistics.fmean(v), 3) for n, v in times.items()})


def unified_ctx(q):
    """Same as UNIFIED_K evaluate.context (k1 ranking irrelevant for J)."""
    first = raw(q); ctx = dict(text=text_of(first), replies={}, finish={}, fold=q['fold_id'])
    for v in ('k2_cot', 'k2_repe'):
        if raw_path(q, v).exists():
            rec = raw(q, v); ctx['replies'][v] = text_of(rec); ctx['finish'][v] = rec['response']['choices'][0].get('finish_reason')
    return ctx


def smoke(n=50):
    optimal = read(INPUTS/'oracle.json'); qs = cases()[::20][:n]; counts = Counter(); times = defaultdict(list)
    finish = Counter()
    for q in qs:
        oracle = Oracle(q); ctx = unified_ctx(q); opt = optimal[q['id']]['optimal']
        finish.update(f'{k}:{v}' for k, v in ctx['finish'].items())
        for name in NAMES:
            r = harness.run(q, name, ctx); check_quality(q, r, name, oracle, opt, counts)
            assert r['operation'] >= 0 and r['successor_expansions'] >= 0 and r['storage'] > 0, (q['id'], name)
            assert r['api_calls'] == {True: (3 if M.METHODS[name]['family'] == 'J2' else 1), False: 0}[M.METHODS[name]['uses_llm']]
            r2 = harness.run(q, name, ctx)
            assert (r2['path'], r2['operation'], r2['successor_expansions'], r2['storage']) == (r['path'], r['operation'], r['successor_expansions'], r['storage']), (name, 'nondeterministic')
            json.dumps(r['extra'], allow_nan=False)
            times[name].append(r['total_seconds']*1000); counts['runs'] += 1
    return dict(cases=len(qs), case_ids=[q['id'] for q in qs], counts=dict(counts), finish_reasons=dict(finish),
                median_ms={k: round(statistics.median(v), 3) for k, v in times.items()},
                mean_ms={k: round(statistics.fmean(v), 3) for k, v in times.items()})


def pilot():
    refs = pilot_reference(); counts = Counter()
    for item in refs:
        q = item['query']; oracle = Oracle(q); text = 'Generated Path: '+json.dumps(item['waypoints'])
        ctx = dict(text=text, replies={'k2_cot': text, 'k2_repe': text}, finish={'k2_cot': 'stop', 'k2_repe': 'stop'})
        for name in NAMES:
            r = harness.run(q, name, ctx); check_quality(q, r, name, oracle, item['optimal'], counts)
    return dict(cases=len(refs), counts=dict(counts))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('all', 'equivalence', 'smoke', 'pilot'))
    p.add_argument('--limit', type=int); a = p.parse_args()
    result = dict(conditions={n: dict(family=s['family'], bound=s['bound'], role=s['role'], jy_condition=s['jy_condition'],
                                      uses_llm=s['uses_llm']) for n, s in M.METHODS.items()})
    if a.stage in ('all', 'pilot'): result['pilot'] = pilot(); print('pilot', result['pilot'], flush=True)
    if a.stage in ('all', 'smoke'): result['smoke'] = smoke(); print('smoke', json.dumps(result['smoke']['median_ms']), flush=True)
    if a.stage in ('all', 'equivalence'):
        result['equivalence'] = equivalence(a.limit); print(json.dumps(result['equivalence']['match'], indent=1), flush=True)
    if a.stage == 'all' and not a.limit: save(OUT/'test_j.json', result)
    else: save(OUT/f'test_j_{a.stage}{"_"+str(a.limit) if a.limit else ""}.json', result)
    print('ALL MATCH' if result.get('equivalence', {}).get('all_match', True) else 'MISMATCHES PRESENT', flush=True)
