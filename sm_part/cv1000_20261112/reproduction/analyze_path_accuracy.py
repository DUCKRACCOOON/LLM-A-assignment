"""Post-hoc path optimality analysis; no API calls or changes to frozen trials."""
import collections
import csv
import hashlib
import heapq
import json
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import ROOT, World, MOTIONS, path_length
from run import cases_for

SOURCE = ROOT / 'results/improvements'
OUT = ROOT / 'results/path_accuracy'
SEED = 240702512
BOOTSTRAPS = 10000
ATOL = 1e-8
RTOL = 1e-10
METHODS = ['A', 'D', 'N', 'B', 'E1', 'E2C', 'E2F', 'E3']
KINDS = ['standard', 'cot', 'repe']
PAIRS = [('E1', 'B'), ('E2F', 'E2C'), ('E3', 'B'),
         ('E1', 'N'), ('E2F', 'B'), ('E3', 'N')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(**values):
    values['updated_at'] = datetime.now(timezone.utc).isoformat()
    save(OUT / 'progress.json', values)
    print(json.dumps(values), flush=True)


def cross(ax, ay, bx, by, cx, cy):
    return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)


def between(ax, ay, bx, by, cx, cy):
    return ((np.minimum(ax, bx) <= cx) & (cx <= np.maximum(ax, bx)) &
            (np.minimum(ay, by) <= cy) & (cy <= np.maximum(ay, by)))


def graph(q):
    """Encode the original 8-neighbor collision graph as one byte per node."""
    world = World(q)
    nx, ny = world.x1-world.x0-1, world.y1-world.y0-1
    ids = np.arange(nx*ny, dtype=np.int64)
    xs, ys = world.x0+1+ids//ny, world.y0+1+ids % ny
    flags = np.zeros(nx*ny, dtype=np.uint8)
    checks = 0
    for direction, (dx, dy) in enumerate(MOTIONS):
        bx, by = xs+dx, ys+dy
        inside = (bx > world.x0) & (bx < world.x1) & (by > world.y0) & (by < world.y1)
        blocked = np.zeros(nx*ny, dtype=bool)
        for (cx, cy), (ex, ey) in world.barriers:
            if (cx, cy) == (ex, ey):
                # Preserve GEOS behavior for degenerate barriers exactly.
                import shapely
                lines = shapely.linestrings(np.stack((np.column_stack((xs, ys)), np.column_stack((bx, by))), axis=1))
                barrier = shapely.LineString([(cx, cy), (ex, ey)])
                blocked |= shapely.intersects(lines, barrier)
                continue
            u = cross(xs, ys, bx, by, cx, cy)
            v = cross(xs, ys, bx, by, ex, ey)
            w = cross(cx, cy, ex, ey, xs, ys)
            z = cross(cx, cy, ex, ey, bx, by)
            blocked |= (((u*v < 0) & (w*z < 0)) |
                        ((u == 0) & between(xs, ys, bx, by, cx, cy)) |
                        ((v == 0) & between(xs, ys, bx, by, ex, ey)) |
                        ((w == 0) & between(cx, cy, ex, ey, xs, ys)) |
                        ((z == 0) & between(cx, cy, ex, ey, bx, by)))
        allowed = inside & ~blocked
        flags[allowed] |= np.uint8(1 << direction)
        candidates = ids[inside]
        # Exhaustive original-rule parity on main-sized maps; deterministic
        # coverage across larger maps, including both legal and blocked edges.
        sample = candidates if len(ids) <= 2000 else candidates[np.unique(np.linspace(0, len(candidates)-1, 600, dtype=int))]
        for i in sample:
            a, b = (int(xs[i]), int(ys[i])), (int(bx[i]), int(by[i]))
            assert bool(allowed[i]) == (not world.collision(a, b)), ('edge parity', q['id'], a, b)
        checks += len(sample)
    offsets = [dx*ny+dy for dx, dy in MOTIONS]
    for j, delta in enumerate(offsets):
        allowed_ids = ids[(flags & (1 << j)) != 0]
        assert np.all((flags[allowed_ids+delta] & (1 << ((j+4) % 8))) != 0), 'Graph is not symmetric'
    return {'flags': flags.tobytes(), 'offsets': offsets, 'ny': ny,
            'x0': world.x0, 'y0': world.y0, 'edge_checks': checks}


def node_id(graph_data, p):
    return (int(p[0])-graph_data['x0']-1)*graph_data['ny']+int(p[1])-graph_data['y0']-1


def dijkstra(g, start, goal):
    """Independent shortest-path solver with no heuristic or waypoint logic."""
    source, target = node_id(g, start), node_id(g, goal)
    distances = [math.inf] * len(g['flags'])
    distances[source] = 0.0
    queue = [(0.0, source)]
    transitions = {1 << j: (g['offsets'][j], math.hypot(*motion)) for j, motion in enumerate(MOTIONS)}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            return distance
        remaining = g['flags'][node]
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            offset, cost = transitions[bit]
            neighbor, candidate = node+offset, distance+cost
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    raise RuntimeError('Reference graph has no path')


def audit_path(q, row, g):
    path = row['path']
    assert path and path[0] == q['start'] and path[-1] == q['goal']
    direction_bits = {motion: 1 << j for j, motion in enumerate(MOTIONS)}
    for point in path:
        assert all(isinstance(x, (int, float)) and float(x).is_integer() for x in point)
    for a, b in zip(path, path[1:]):
        motion = b[0]-a[0], b[1]-a[1]
        assert motion in direction_bits, ('Non-grid path', row['case_id'], row['method'])
        i = node_id(g, a)
        assert 0 <= i < len(g['flags']) and g['flags'][i] & direction_bits[motion], 'Illegal path edge'
    length = path_length(path)
    assert math.isclose(length, row['length'], rel_tol=1e-12, abs_tol=1e-9)
    return length


def aggregate(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['method'], r['prompt']].append(r)
    output = []
    for (method, prompt), group in sorted(groups.items(), key=lambda item: (METHODS.index(item[0][0]), str(item[0][1]))):
        good = [r for r in group if r['valid']]
        absolute = np.array([r['excess_length'] for r in good])
        relative = np.array([r['excess_percent'] for r in good])
        output.append({'method': method, 'prompt': prompt, 'n': len(group), 'valid_n': len(good),
                       'optimal_n': sum(r['optimal'] for r in group),
                       'optimal_rate_percent': 100*sum(r['optimal'] for r in group)/len(group),
                       'mean_excess_length': float(np.mean(absolute)),
                       'median_excess_length': float(np.median(absolute)),
                       'mean_excess_percent': float(np.mean(relative)),
                       'median_excess_percent': float(np.median(relative)),
                       'p95_excess_percent': float(np.quantile(relative, .95)),
                       'max_excess_percent': float(np.max(relative)),
                       'within_1percent_rate': 100*sum(r['valid'] and r['excess_percent'] <= 1+1e-9 for r in group)/len(group),
                       'within_5percent_rate': 100*sum(r['valid'] and r['excess_percent'] <= 5+1e-9 for r in group)/len(group)})
    return output


def paired(rows):
    lookup = {(r['case_id'], r['method'], r['prompt']): r for r in rows}
    output = []
    for kind in KINDS:
        for treatment, control in PAIRS:
            matches = [(r, lookup[r['case_id'], control, None if control == 'N' else kind])
                       for r in rows if r['method'] == treatment and r['prompt'] == kind]
            clusters = sorted({a['map_id'] for a, _ in matches})
            cluster_totals = []
            sizes = []
            for map_id in clusters:
                ps = [(a, b) for a, b in matches if a['map_id'] == map_id]
                assert all(a['valid'] and b['valid'] for a, b in ps)
                sizes.append(len(ps))
                cluster_totals.append([sum(100*(int(a['optimal'])-int(b['optimal'])) for a, b in ps),
                                       sum(b['excess_percent']-a['excess_percent'] for a, b in ps),
                                       sum(b['excess_length']-a['excess_length'] for a, b in ps)])
            weights = np.random.default_rng(SEED).multinomial(len(clusters), [1/len(clusters)]*len(clusters), size=BOOTSTRAPS)
            replicates = (weights @ np.array(cluster_totals)) / (weights @ np.array(sizes))[:, None]
            point = np.sum(cluster_totals, axis=0)/sum(sizes)
            record = {'prompt': kind, 'treatment': treatment, 'control': control,
                      'paired_n': len(matches), 'map_clusters': len(clusters),
                      'became_optimal_n': sum(a['optimal'] and not b['optimal'] for a, b in matches),
                      'lost_optimal_n': sum(b['optimal'] and not a['optimal'] for a, b in matches),
                      'shorter_n': sum(a['excess_length'] < b['excess_length']-ATOL for a, b in matches),
                      'longer_n': sum(a['excess_length'] > b['excess_length']+ATOL for a, b in matches),
                      'equal_length_n': sum(math.isclose(a['excess_length'], b['excess_length'], abs_tol=ATOL, rel_tol=RTOL) for a, b in matches)}
            for column, name in enumerate(['optimal_delta_pp', 'mean_excess_reduction_pp', 'mean_excess_length_reduction']):
                record[name] = float(point[column])
                record[name+'_ci95_low'], record[name+'_ci95_high'] = [float(x) for x in np.quantile(replicates[:, column], [.025, .975])]
            output.append(record)
    return output


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    verified = read(SOURCE / 'verification.json')
    imported = read(SOURCE / 'local_import_verification.json')
    protocol = read(SOURCE / 'protocol.json')
    assert verified['complete'] and imported['complete']
    archive = ROOT / 'improvement-results-verified.zip'
    assert sha(archive) == imported['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        expected = json.loads(z.read('archive_manifest.json'))
    inputs = {}
    for stage in ['main', 'scaling']:
        path = SOURCE / stage / 'rows.json'
        inputs[str(path.relative_to(ROOT))] = sha(path)
        assert inputs[str(path.relative_to(ROOT))] == expected[str(path.relative_to(ROOT)).replace('\\', '/')]
    for name, value in protocol['source_sha256'].items():
        assert sha(ROOT / 'reproduction' / name) == value
    assert sha(ROOT / 'vendor/llm-astar/dataset/environment_50_30.json') == protocol['data_sha256']
    outputs, references, audits = {}, [], []
    for stage in ['main', 'scaling']:
        saved = [r for r in read(SOURCE / stage / 'rows.json') if r['method'] in METHODS]
        by_case = collections.defaultdict(list)
        for r in saved:
            by_case[r['case_id']].append(r)
        groups = collections.defaultdict(list)
        for q in cases_for(stage):
            groups[q['map_id'], q.get('scale', 1)].append(q)
        analyzed, checks = [], 0
        for index, qs in enumerate(groups.values(), 1):
            g = graph(qs[0])
            checks += g['edge_checks']
            for q in qs:
                optimum = dijkstra(g, q['start'], q['goal'])
                baseline = next(r for r in by_case[q['id']] if r['method'] == 'A')
                assert math.isclose(optimum, baseline['length'], abs_tol=ATOL, rel_tol=RTOL), ('A* not optimal', q['id'])
                references.append({'stage': stage, 'case_id': q['id'], 'map_id': q['map_id'],
                                   'scale': q.get('scale', 1), 'dijkstra_length': optimum,
                                   'saved_astar_length': baseline['length'],
                                   'absolute_difference': abs(optimum-baseline['length'])})
                for r in by_case[q['id']]:
                    assert r['valid'], 'Unexpected invalid hybrid result; handle denominators explicitly'
                    length = audit_path(q, r, g)
                    optimal = math.isclose(length, optimum, abs_tol=ATOL, rel_tol=RTOL)
                    assert length >= optimum-max(ATOL, RTOL*optimum), 'Path beats certified optimum'
                    excess = 0.0 if optimal else length-optimum
                    analyzed.append({'stage': stage, 'case_id': q['id'], 'map_id': q['map_id'], 'scale': q.get('scale', 1),
                                     'method': r['method'], 'prompt': r['prompt'], 'valid': r['valid'],
                                     'length': length, 'optimal_length': optimum, 'optimal': optimal,
                                     'excess_length': excess, 'excess_percent': 100*excess/optimum})
            if index % 10 == 0 or index == len(groups):
                progress(phase='reference_and_path_audit', stage=stage, graphs_done=index, graphs_total=len(groups), rows_done=len(analyzed))
        outputs[stage] = analyzed
        audits.append({'stage': stage, 'reference_cases': len(cases_for(stage)), 'astar_mismatches': 0,
                       'graph_count': len(groups), 'collision_rule_edge_checks': checks,
                       'grid_paths_verified': len(analyzed), 'rows': len(analyzed)})
        write_csv(OUT / f'{stage}_per_case.csv', analyzed)
        write_csv(OUT / f'{stage}_summary.csv', aggregate(analyzed))
    write_csv(OUT / 'optimal_reference.csv', references)
    progress(phase='bootstrap_and_summary', bootstrap_replicates=BOOTSTRAPS)
    comparisons = paired(outputs['main'])
    write_csv(OUT / 'main_paired_comparisons.csv', comparisons)
    excluded = set(verified['demo_overlap_excluded'])
    without_overlap = [r for r in outputs['main'] if r['case_id'] not in excluded]
    write_csv(OUT / 'main_without_demo_overlap.csv', aggregate(without_overlap))
    sensitivity = paired(without_overlap)
    write_csv(OUT / 'paired_without_demo_overlap.csv', sensitivity)
    by_scale = []
    for scale in range(1, 11):
        by_scale.extend([dict(scale=scale, **r) for r in aggregate([r for r in outputs['scaling'] if r['scale'] == scale])])
    write_csv(OUT / 'scaling_by_factor.csv', by_scale)
    result = {'analysis_type': 'post_hoc_secondary', 'main': aggregate(outputs['main']),
              'scaling': aggregate(outputs['scaling']), 'main_paired': comparisons,
              'main_without_demo_overlap': aggregate(without_overlap),
              'paired_without_demo_overlap': sensitivity, 'scaling_by_factor': by_scale,
              'audit': audits, 'maximum_astar_dijkstra_difference': max(r['absolute_difference'] for r in references)}
    save(OUT / 'analysis.json', result)
    manifest = {'complete': True, 'analysis_type': 'post_hoc_secondary', 'new_api_requests': 0,
                'created_at': datetime.now(timezone.utc).isoformat(), 'source_archive_sha256': imported['archive_sha256'],
                'original_protocol_signature': protocol['signature'], 'input_sha256': inputs,
                'analysis_code_sha256': sha(Path(__file__)), 'optimal_reference': 'independent Dijkstra on original 8-neighbor collision graph',
                'equality_tolerance': {'absolute': ATOL, 'relative': RTOL},
                'excess_aggregation': 'arithmetic mean of per-case absolute excess and percentage excess',
                'bootstrap': {'unit': 'map_id', 'paired': True, 'replicates': BOOTSTRAPS, 'seed': SEED,
                              'interval': 'percentile 95%', 'multiple_comparison_adjustment': False,
                              'scope': 'exploratory variation across maps, not repeated API sampling'},
                'excluded_methods': ['GPT', 'GPT_E2C', 'GPT_E2F'],
                'excluded_reason': 'Raw GPT polylines do not share the eight-neighbor movement space',
                'reference_cases': len(references), 'verified_grid_paths': sum(len(r) for r in outputs.values()),
                'audit': audits, 'elapsed_seconds': time.perf_counter()-started}
    save(OUT / 'analysis_manifest.json', manifest)
    progress(phase='complete', complete=True, reference_cases=len(references), grid_paths=manifest['verified_grid_paths'])


if __name__ == '__main__':
    main()
