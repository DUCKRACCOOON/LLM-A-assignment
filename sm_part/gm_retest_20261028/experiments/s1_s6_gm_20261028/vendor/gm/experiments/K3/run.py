"""Reproducible offline verification and cold-cache benchmark for K3.

python -m experiments.K3.run --output experiments/K3/results/run_20261025
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from .heuristic import METHODS, PRIMARY, run_once
from .verification import EPS, Oracle, check_bounds, verify_fixtures

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'source_code/dataset/environment_50_30.json'
OLD_RESULTS = ROOT / 'results/main/evaluations.jsonl'
SEED = 20261025


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_cases():
    cases = []
    for mi, env in enumerate(json.loads(DATA.read_text(encoding='utf-8'))):
        for pi, pair in enumerate(env['start_goal']):
            query = {k: env[k] for k in ('range_x', 'range_y', 'horizontal_barriers', 'vertical_barriers')}
            query.update(start=pair[0], goal=pair[1])
            cases.append(dict(case_id=f'm{mi:03d}_p{pi:02d}', map_index=mi, pair_index=pi, query=query))
    return cases


def synthetic_cases():
    rng, cases, rejections = random.Random(SEED), [], []
    for mi in range(30):
        width, height = ((32, 22), (52, 32), (72, 42))[mi % 3]
        count = (4, 6, 8)[(mi // 3) % 3]
        query = dict(range_x=[0, width+1], range_y=[0, height+1], horizontal_barriers=[], vertical_barriers=[])
        for _ in range(count):
            low = rng.randint(0, width-8)
            query['horizontal_barriers'].append([rng.randint(3, height-3), low,
                                                 min(width, low+rng.randint(5, width//2))])
        for _ in range(count):
            low = rng.randint(0, height-6)
            query['vertical_barriers'].append([rng.randint(3, width-3), low,
                                               min(height, low+rng.randint(3, height//2))])
        oracle = Oracle(query)
        rejected = dict(same_point=0, blocked_endpoint=0, disconnected=0, duplicate_pair=0)
        used = set()
        for pi in range(4):
            for _ in range(100000):
                start = (rng.randint(1, width-1), rng.randint(1, height-1))
                goal = (rng.randint(1, width-1), rng.randint(1, height-1))
                if start == goal:
                    rejected['same_point'] += 1
                elif start not in oracle.graph or goal not in oracle.graph:
                    rejected['blocked_endpoint'] += 1
                elif (start, goal) in used:
                    rejected['duplicate_pair'] += 1
                elif start not in oracle.distances(goal):
                    rejected['disconnected'] += 1
                else:
                    used.add((start, goal))
                    cases.append(dict(case_id=f's{mi:03d}_p{pi:02d}', map_index=mi, pair_index=pi,
                                      query=dict(query, start=list(start), goal=list(goal))))
                    break
            else:
                raise RuntimeError('Sampler exhausted; do not silently replace a map')
        rejections.append(dict(map_index=mi, size=[width, height], walls_per_axis=count, rejected=rejected))
    return cases, rejections


def load_old():
    rows = {}
    with OLD_RESULTS.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['method'] in ('astar', 'goal_only'):
                method = 'astar_euclidean' if row['method'] == 'astar' else 'g2h_euclidean'
                rows[(row['case_id'], method)] = row
    return rows


def verify_cases(cases, output):
    output.mkdir()
    totals = dict(cases=0, maps=0, collision_checks=0, admissibility_checks=0, consistency_checks=0,
                  max_admissibility_excess=0.0, max_consistency_excess=0.0, optimal_astar_cases=0)
    map_index, oracle = None, None
    optima = {}
    with (output/'cases.jsonl').open('x', encoding='utf-8') as stream:
        for case in cases:
            if case['map_index'] != map_index:
                map_index = case['map_index']
                oracle = Oracle(case['query'], crosscheck=True)
                totals['collision_checks'] += oracle.collision_checks
                totals['maps'] += 1
            checks = check_bounds(case['query'], oracle)
            assert checks['optimum'] is not None, case['case_id']
            result = run_once(case['query'], 'astar_wall_octile')
            oracle.check_path(case['query'], result)
            assert math.isclose(result['length'], checks['optimum'], abs_tol=EPS), case['case_id']
            optima[case['case_id']] = checks['optimum']
            for key in ('admissibility_checks', 'consistency_checks'):
                totals[key] += checks[key]
            for key in ('max_admissibility_excess', 'max_consistency_excess'):
                totals[key] = max(totals[key], checks[key])
            totals['cases'] += 1
            totals['optimal_astar_cases'] += 1
            stream.write(json.dumps(dict(case_id=case['case_id'], **checks), allow_nan=False)+'\n')
            if totals['cases'] % 50 == 0:
                print(f'Verify {output.name}: {totals["cases"]}/{len(cases)}', flush=True)
    write_json(output/'summary.json', totals)
    write_json(output/'optima.json', optima)
    return optima


def assert_old(row, old):
    for key in ('valid', 'error', 'operation', 'storage', 'unique_pops', 'duplicate_pops', 'infinite_g'):
        assert row[key] == old[key], (row['case_id'], row['method'], key, row[key], old[key])
    assert [list(p) for p in row['path']] == old['path'], (row['case_id'], 'path')
    assert math.isclose(row['length'], old['length'], abs_tol=EPS)


def benchmark(cases, output, optima, old=None):
    output.mkdir()
    rng = random.Random(SEED)
    old_checks = independent_path_checks = 0
    # Graphs exist only for untimed result verification; they are not given to
    # any search. This avoids counting our expensive oracle as online work.
    map_index, oracle = None, None
    with (output/'evaluations.jsonl').open('x', encoding='utf-8') as stream:
        for i, case in enumerate(cases):
            if case['map_index'] != map_index:
                map_index = case['map_index']
                oracle = Oracle(case['query'])
            reference = {}
            for repeat in range(3):
                order = list(METHODS); rng.shuffle(order)
                for position, method in enumerate(order):
                    row = run_once(case['query'], method)
                    row.update(case_id=case['case_id'], map_index=case['map_index'], pair_index=case['pair_index'],
                               method=method, repeat=repeat, order=position, optimum=optima[case['case_id']])
                    if row['valid']:
                        oracle.check_path(case['query'], row)
                        independent_path_checks += 1
                        assert row['length']+EPS >= row['optimum']
                    # Deterministic output is checked; timing is deliberately excluded.
                    stable = {k: row[k] for k in ('valid', 'error', 'path', 'operation', 'storage', 'length')}
                    if method in reference:
                        assert stable == reference[method], (case['case_id'], method, 'repeat mismatch')
                    else:
                        reference[method] = stable
                    if old is not None and (case['case_id'], method) in old:
                        assert_old(row, old[(case['case_id'], method)])
                        old_checks += 1
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            if (i+1) % 25 == 0 or i+1 == len(cases):
                stream.flush()
                print(f'Benchmark {output.name}: {i+1}/{len(cases)}', flush=True)
    write_json(output/'checks.json', dict(cases=len(cases), runs=len(cases)*len(METHODS)*3,
               baseline_saved_row_checks=old_checks, independent_valid_path_checks=independent_path_checks,
               repeats=3, methods=list(METHODS), seed=SEED))


def summarize(output):
    import numpy as np
    buckets = {}
    with (output/'evaluations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            buckets.setdefault((row['case_id'], row['method']), []).append(row)
    rows, by_method = [], {method: {} for method in METHODS}
    for (cid, method), trials in buckets.items():
        assert len(trials) == 3
        row = {k: trials[0].get(k) for k in ('case_id', 'map_index', 'pair_index', 'method', 'valid', 'error',
                'operation', 'storage', 'length', 'unique_pops', 'duplicate_pops', 'infinite_g', 'optimum',
                'heuristic_calls', 'wall_checks', 'blocking_wall_checks', 'wall_count')}
        row['total_seconds'] = statistics.median(t['total_seconds'] for t in trials)
        row['repeat_seconds'] = [t['total_seconds'] for t in trials]
        rows.append(row); by_method[method][cid] = row
    with (output/'per_case.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    comparisons = {}
    for control in ('astar_euclidean', 'g2h_euclidean', 'g2h_octile'):
        comparisons[control] = {}
        for method in METHODS:
            paired = [(r, by_method[control][cid]) for cid, r in by_method[method].items()
                      if r['valid'] and by_method[control][cid]['valid']]
            metric_results = {}
            for metric in ('operation', 'storage', 'length', 'total_seconds'):
                logs = np.array([math.log(r[metric]/b[metric]) for r, b in paired])
                map_ids = sorted({r['map_index'] for r, _ in paired})
                sums = np.array([sum(v for v, (r, _) in zip(logs, paired) if r['map_index'] == mi) for mi in map_ids])
                counts = np.array([sum(r['map_index'] == mi for r, _ in paired) for mi in map_ids])
                rng = np.random.default_rng(SEED)
                sample = rng.integers(0, len(map_ids), (2000, len(map_ids)))
                boots = np.exp(sums[sample].sum(axis=1)/counts[sample].sum(axis=1))
                differences = [r[metric]-b[metric] for r, b in paired]
                tolerance = EPS if metric == 'length' else 0.0
                better = sum(v < -tolerance for v in differences)
                worse = sum(v > tolerance for v in differences)
                metric_results[metric] = dict(geometric_ratio=float(np.exp(logs.mean())),
                    ci95=[float(v) for v in np.quantile(boots, [.025, .975])], pairs=len(paired),
                    better=better, equal=len(paired)-better-worse, worse=worse)
            metric_results['joint'] = dict(
                strictly_better_both=sum(r['operation'] < b['operation'] and r['length'] < b['length']-EPS for r, b in paired),
                operation_better_length_not_worse=sum(r['operation'] < b['operation'] and r['length'] <= b['length']+EPS for r, b in paired),
                neither_worse_at_least_one_better=sum(r['operation'] <= b['operation'] and r['length'] <= b['length']+EPS
                    and (r['operation'] < b['operation'] or r['length'] < b['length']-EPS) for r, b in paired))
            comparisons[control][method] = metric_results
    absolute = {}
    for method, entries in by_method.items():
        valid = [r for r in entries.values() if r['valid']]
        absolute[method] = dict(cases=len(entries), valid=len(valid),
            means={k: statistics.mean(r[k] for r in valid) for k in ('operation', 'storage', 'length', 'total_seconds', 'wall_checks')},
            optimal_paths=sum(math.isclose(r['length'], r['optimum'], abs_tol=EPS) for r in valid))
    write_json(output/'summary.json', dict(primary=PRIMARY, confidence='map-cluster percentile bootstrap, 2000 draws, seed 20261025',
                                          absolute=absolute, comparisons=comparisons))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new folder to preserve raw measurements')
    assert git('branch', '--show-current') == 'gm', 'This experiment is assigned to gm'
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    tracked_inputs = [DATA, OLD_RESULTS, ROOT/'reproduction/core.py', *sorted((ROOT/'experiments/K3').glob('*.py')),
                      ROOT/'experiments/K3/PROTOCOL.md']
    manifest = dict(status='started', started_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
        primary=PRIMARY, methods=METHODS, repeats=3, bootstrap_repeats=2000, command=sys.argv,
        python=sys.version, platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
        git_branch=git('branch', '--show-current'), git_base_commit=git('rev-parse', 'HEAD'),
        git_status=git('status', '--short'), input_sha256={str(p.relative_to(ROOT)): sha(p) for p in tracked_inputs},
        api_calls=0, timing='Cold Geometry + heuristic setup + search + returned-path validation; no I/O or oracle')
    write_json(args.output/'manifest.json', manifest)
    print('Checking independent edge-case fixtures', flush=True)
    write_json(args.output/'fixtures.json', verify_fixtures())
    public = public_cases()
    holdout, rejections = synthetic_cases()
    write_json(args.output/'public_cases.json', public)
    write_json(args.output/'synthetic_cases.json', holdout)
    write_json(args.output/'synthetic_sampling.json', rejections)
    manifest['case_sha256'] = {name: sha(args.output/name) for name in ('public_cases.json', 'synthetic_cases.json')}
    write_json(args.output/'manifest.json', manifest)
    public_optima = verify_cases(public, args.output/'verify_public')
    holdout_optima = verify_cases(holdout, args.output/'verify_synthetic')
    old = load_old()
    assert len(old) == 2000
    pilot = [c for c in public if c['map_index'] % 10 == 0 and c['pair_index'] < 3]
    for name, cases, optima, reference in (('pilot', pilot, public_optima, old),
            ('public', public, public_optima, old), ('synthetic', holdout, holdout_optima, None)):
        benchmark(cases, args.output/name, optima, reference)
        summarize(args.output/name)
    for p in tracked_inputs:
        assert sha(p) == manifest['input_sha256'][str(p.relative_to(ROOT))], f'Input changed during experiment: {p}'
    manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start)
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps(dict(status='completed', seconds=manifest['elapsed_seconds'], output=str(args.output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
