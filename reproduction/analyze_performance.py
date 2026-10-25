"""Offline paired analysis of saved GPT results; no API calls or search reruns.

Run from any directory with Python 3: python3 reproduction/analyze_performance.py.
The map-cluster bootstrap conditions on the one saved generation per prompt.
"""
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/report'
METRICS = ('operation', 'storage', 'length')
PROMPTS = ('standard', 'cot', 'repe')


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def ratios(rows, baseline):
    return {m: 100 * gm([r[m] / baseline[r['case_id']][m] for r in rows])
            for m in METRICS}


def bootstrap(rows, baseline, metric):
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[r['case_id'].split('-')[0]].append(math.log(r[metric] / baseline[r['case_id']][metric]))
    assert len(clusters) == 100 and all(len(v) == 10 for v in clusters.values())
    means = [statistics.mean(v) for _, v in sorted(clusters.items())]
    rng = random.Random(20261014)
    samples = sorted(100 * math.exp(statistics.mean(rng.choices(means, k=100))) for _ in range(2000))
    return [samples[49], samples[1949]]


def main():
    rows = read('results/main/rows.json')
    no_wp = read('results/report/no_waypoint_ablation.json')
    base = {r['case_id']: r for r in rows if r['method'] == 'astar'}
    ablation = {r['case_id']: r for r in no_wp}
    assert len(rows) == 8000 and len(base) == len(no_wp) == len(ablation) == 1000
    assert set(base) == set(ablation)
    assert len({(r['case_id'], r['method'], r['prompt']) for r in rows}) == 8000
    assert all(r['valid'] for r in no_wp)

    # Load just the parser and target filter from core without its vendor import.
    tree = ast.parse((ROOT / 'reproduction/core.py').read_text())
    subset = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                             and n.name in ('parse_path', 'World')], type_ignores=[])
    import re
    namespace = {'ast': ast, 'math': math, 're': re}
    exec(compile(subset, 'core.py parser and filter', 'exec'), namespace)
    parse_path, World = namespace['parse_path'], namespace['World']

    stored_summary = read('results/report/summary.json')['main']
    diagnostics = read('results/report/diagnostics.json')['no_waypoint_baseline']
    calculated_baseline = ratios(no_wp, base)
    for m in METRICS:
        assert math.isclose(calculated_baseline[m], diagnostics[m], rel_tol=1e-12)
    results = {'scope': {'cases': 1000, 'maps': 100, 'cached_responses': 3000,
                         'new_api_calls': 0, 'search_rerun': False,
                         'bootstrap': '2000 paired map-cluster resamples; seed 20261014; saved outputs fixed'},
               'no_waypoint_vs_astar': calculated_baseline, 'prompts': {}}
    paired_rows = []
    for prompt in PROMPTS:
        group = [r for r in rows if r['method'] == 'llm_astar' and r['prompt'] == prompt]
        solo = {r['case_id']: r for r in rows if r['method'] == 'llm_only' and r['prompt'] == prompt}
        assert len(group) == len(solo) == 1000 and all(r['valid'] for r in group)
        stats = {'vs_astar': ratios(group, base), 'vs_no_waypoint': ratios(group, ablation),
                 'paired_counts': {}, 'subgroups': {}, 'no_waypoint_reasons': {},
                 'initial_target_is_goal': 0, 'raw_intermediate_mean': 0,
                 'filtered_intermediate_mean': statistics.mean(len(r['targets'])-2 for r in group)}
        summary = next(s for s in stored_summary if s['method'] == 'llm_astar' and s['prompt'] == prompt)
        for m in METRICS:
            assert math.isclose(stats['vs_astar'][m], summary[m + '_ratio'], rel_tol=1e-12)
            signs = collections.Counter('equal' if math.isclose(r[m], ablation[r['case_id']][m], abs_tol=1e-9)
                                        else 'better' if r[m] < ablation[r['case_id']][m] else 'worse' for r in group)
            stats['paired_counts'][m] = dict(signs)
        stats['operation_vs_no_waypoint_95ci'] = bootstrap(group, ablation, 'operation')
        stats['operation_vs_astar_95ci'] = bootstrap(group, base, 'operation')
        reasons = collections.Counter()
        raw_counts = []
        examples = {}
        for r in group:
            cid = r['case_id']
            evaluation = read(f'results/main/evaluation/{cid}.json')
            assert r == next(v for v in evaluation['rows'] if v['method'] == 'llm_astar' and v['prompt'] == prompt)
            raw = read(f'results/main/raw/{cid}-{prompt}.json')
            payload = raw['payload']
            assert raw['payload_sha256'] == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            assert payload['model'] == 'gpt-3.5-turbo-0125' and payload['temperature'] == 0
            assert raw['response']['model'] == 'gpt-3.5-turbo-0125'
            assert payload['max_tokens'] == 1000 and 'stop' not in payload
            text = raw['response']['choices'][0]['message']['content'] or ''
            path, error = parse_path(text)
            assert error == r['parse_error']
            assert [list(p) for p in path] == solo[cid]['path']
            q = evaluation['case']
            targets = World(q).targets(path, tuple(q['start']), tuple(q['goal']))
            assert [list(p) for p in targets] == r['targets']
            raw_counts.append(sum(p not in (tuple(q['start']), tuple(q['goal'])) for p in path))
            stats['initial_target_is_goal'] += r['targets'][1] == q['goal']
            if len(r['targets']) == 2:
                assert all(math.isclose(r[m], ablation[cid][m], abs_tol=1e-9) for m in METRICS)
                reason = 'parse_failure' if error else 'only_endpoints_generated' if raw_counts[-1] == 0 else 'all_intermediates_filtered'
                reasons[reason] += 1
                examples.setdefault(reason, {'case_id': cid, 'response': text, 'targets': r['targets'],
                                               'operation': r['operation'], 'astar_operation': base[cid]['operation']})
            paired_rows.append({'case_id': cid, 'prompt': prompt, 'target_count': len(r['targets']),
                                'parse_error': error, **{f'{m}_vs_no_waypoint': r[m]/ablation[cid][m] for m in METRICS}})
        stats['raw_intermediate_mean'] = statistics.mean(raw_counts)
        stats['no_waypoint_reasons'] = dict(reasons)
        stats['examples'] = examples
        for name, predicate in [('only_start_goal', lambda r: len(r['targets']) == 2),
                                ('has_intermediates', lambda r: len(r['targets']) > 2),
                                ('parse_failed', lambda r: bool(r['parse_error']))]:
            subset = [r for r in group if predicate(r)]
            stats['subgroups'][name] = {'n': len(subset), 'vs_astar': ratios(subset, base),
                                       'vs_no_waypoint': ratios(subset, ablation)}
        stats['operation_sum_vs_astar_pct'] = 100 * sum(r['operation'] for r in group)/sum(r['operation'] for r in base.values())
        results['prompts'][prompt] = stats
    verification = read('results/verification.json')
    results['source_hashes_match_original_verification'] = all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
        for p, digest in verification['source_sha256'].items())
    results['input_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                              for p in ['results/main/rows.json', 'results/report/no_waypoint_ablation.json']}
    (OUT / 'performance_analysis.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    with (OUT / 'performance_paired.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps({**results, 'prompts': {k: {a:b for a,b in v.items() if a != 'examples'}
                                           for k,v in results['prompts'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
