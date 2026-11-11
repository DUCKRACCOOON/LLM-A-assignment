"""Independent scan-based FOCAL reference and fixed correctness fixtures."""
import json
import math
import random

from reproduction.core import Geometry
from experiments.K3.heuristic import LowerBound
from experiments.K3.verification import Oracle, fixtures, check_bounds
from .search import EPS, PolylineGuide, graph_search, run_once


def scan_reference(start, goal, graph, h, guide, weight):
    # No heaps, stale entries, or eligibility cache: recompute OPEN/FOCAL each step.
    g, parents, opened = {start: 0.0}, {start: start}, {start}
    sequence = []
    while opened:
        fmin = min(g[n]+h(n) for n in opened)
        eligible = [n for n in opened if g[n]+h(n) <= weight*fmin+EPS]
        node = min(eligible, key=lambda n: (guide(n), h(n), g[n]+h(n), n))
        opened.remove(node)
        sequence.append(node)
        if node == goal:
            path = [node]
            while node != start:
                node = parents[node]; path.append(node)
            return list(reversed(path)), sequence
        for neighbor, cost in graph[node]:
            distance = g[node]+cost
            if distance < g.get(neighbor, math.inf):
                g[neighbor], parents[neighbor] = distance, node
                opened.add(neighbor)
    return [], sequence


def compare(query, text, weight, oracle):
    method = {1.0:'focal_llm_w100', 1.05:'focal_llm_w105', 1.1:'focal_llm_w110', 2.0:'focal_llm_w200'}[weight]
    result = run_once(query, method, text, trace=True, audit=True)
    start, goal = tuple(query['start']), tuple(query['goal'])
    h = LowerBound(Geometry(query), goal)
    guide = PolylineGuide(query, text)
    path, sequence = scan_reference(start, goal, oracle.graph, h, h if guide.error else guide, weight)
    assert result['path'] == path, (query, text, weight, 'path mismatch')
    assert [e['node'] for e in result['trace']] == sequence, (query, text, weight, 'selection mismatch')
    optimum = oracle.distances(goal).get(start)
    if optimum is None:
        assert not result['valid'] and result['error'] == 'unreachable'
    else:
        oracle.check_path(query, result)
        assert result['length'] <= weight*optimum+1e-8
        assert result['lower_bound_at_goal'] <= optimum+1e-8
        if weight == 1:
            assert abs(result['length']-optimum) < 1e-8
    return dict(valid=result['valid'], length=result['length'], optimum=optimum,
                operation=result['operation'], expansions=result['expansions'],
                selection_audits=result['selection_audits'], guide_error=result['guide_error'],
                reopenings=result['reopenings'], eligibility_rebuilds=result['eligibility_rebuilds'])


def verify():
    evidence, total_checks = [], 0
    for name, query in fixtures().items():
        oracle = Oracle(query, crosscheck=True)
        guides = {
            'direct': json.dumps([query['start'], query['goal']]),
            'wrong_loop_outside': json.dumps([query['goal'], [-50, 70], [6, 4], query['start']]),
            'missing': 'No coordinate list',
            'malformed': '[[1,2,3],[4,5,6]]',
        }
        for guide_name, text in guides.items():
            for weight in (1.0, 1.05, 1.1, 2.0):
                checks = compare(query, text, weight, oracle)
                evidence.append(dict(case=name, guide=guide_name, weight=weight, **checks))
                total_checks += checks['selection_audits']
    # Force a previously expanded state to be reopened at a lower g.
    graph = {'S': [('A',4.0),('B',1.0)], 'A':[('G',10.0)], 'B':[('A',1.0),('G',100.0)], 'G':[]}
    rank = {'S':0, 'A':0, 'B':1, 'G':2}
    reopened = graph_search('S', 'G', graph.__getitem__, lambda n:0, rank.__getitem__, weight=4, trace=True, audit=True)
    assert reopened['goal_g'] == 12 and reopened['reopenings'] >= 1 and reopened['reexpansions'] >= 1
    # Admissible but inconsistent h explicitly decreases the FOCAL threshold.
    h = {'S':12.0, 'B':11.0, 'A':0.0, 'G':0.0}
    decreased = graph_search('S', 'G', graph.__getitem__, h.__getitem__, rank.__getitem__, weight=1.05, trace=True, audit=True)
    assert decreased['goal_g'] == 12 and decreased['eligibility_rebuilds'] >= 1
    assert PolylineGuide(fixtures()['empty'], '[[1e308,1e308],[-1e308,-1e308]]').error == 'nonfinite_guide_geometry'
    assert PolylineGuide(fixtures()['empty'], '[[2,4],[2,4],[10,4]]').segments
    bad_query = dict(fixtures()['empty'], start=[0,4])
    assert run_once(bad_query, 'focal_llm_w105', '[[0,4],[10,4]]')['error'] == 'invalid_endpoint'
    bad_query = dict(fixtures()['long_wall'], start=[6,4], goal=[6,4])
    assert run_once(bad_query, 'focal_llm_w105')['error'] == 'invalid_endpoint'
    for bad_weight in (0, .5, float('inf'), float('nan')):
        try:
            graph_search('S','G',graph.__getitem__,lambda n:0,rank.__getitem__,weight=bad_weight)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid weight accepted')
    rng = random.Random(20261026)
    random_evidence = []
    bounds_checked = 0
    for mi in range(40):
        width, height = rng.randint(8,16), rng.randint(8,14)
        query = dict(range_x=[0,width+1], range_y=[0,height+1],
                     horizontal_barriers=[[rng.randint(2,height-2),*sorted(rng.sample(range(width+1),2))] for _ in range(2)],
                     vertical_barriers=[[rng.randint(2,width-2),*sorted(rng.sample(range(height+1),2))] for _ in range(2)])
        oracle = Oracle(query, crosscheck=True)
        nodes = sorted(oracle.graph)
        assert nodes
        goal = rng.choice(nodes)
        distances = oracle.distances(goal)
        start = rng.choice(sorted(distances))
        query.update(start=list(start), goal=list(goal))
        bounds = check_bounds(query, oracle)
        bounds_checked += bounds['admissibility_checks']+bounds['consistency_checks']
        guides = [json.dumps([list(start),list(goal)]), 'missing',
                  json.dumps([list(goal), [width+3,height+4], [1,1], list(start)])]
        for gi, text in enumerate(guides):
            for weight in (1.0,1.05,2.0):
                checks = compare(query,text,weight,oracle)
                random_evidence.append(dict(map_index=mi, guide=gi, weight=weight, **checks))
                total_checks += checks['selection_audits']
    return dict(status='passed', deterministic_fixture_runs=len(evidence), random_maps=40,
                random_fixture_runs=len(random_evidence), full_scan_selection_checks=total_checks,
                random_heuristic_checks=bounds_checked, special_reopening=reopened,
                special_decreasing_threshold=decreased, numeric_and_endpoint_checks='passed',
                fixtures=evidence, random_cases=random_evidence)


if __name__ == '__main__':
    result=verify()
    print(json.dumps({k:v for k,v in result.items() if k not in ('fixtures','random_cases','special_reopening','special_decreasing_threshold')}, indent=2))
