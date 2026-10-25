"""Offline tests: independent Dijkstra, Shapely, and untouched upstream parity."""
import contextlib
import copy
import heapq
import io
import json
import math
import random
import sys
from pathlib import Path
from .core import Geometry, MOVES, llm_only, parse_output, search
from .experiment import ROOT, cases, selected_cases


def dijkstra(query, geometry):
    start, goal = tuple(query['start']), tuple(query['goal'])
    frontier, best = [(0.0, start)], {start: 0.0}
    while frontier:
        cost, node = heapq.heappop(frontier)
        if cost != best[node]:
            continue
        if node == goal:
            return cost
        for neighbor, step in geometry.neighbors(node):
            candidate = cost + step
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                heapq.heappush(frontier,(candidate,neighbor))
    return None


def main():
    import matplotlib
    matplotlib.use('Agg')
    # The search code imports model packages eagerly. These offline stand-ins
    # avoid downloading any model/runtime; fixed responses test search only.
    from types import ModuleType
    for name in ['openai','torch','transformers']:
        if name not in sys.modules:
            sys.modules[name]=ModuleType(name)
    sys.path.insert(0,str(ROOT/'source_code'))
    from llmastar.pather import AStar, LLMAStar
    from llmastar.env.search.plotting import Plotting
    from shapely.geometry import LineString
    Plotting.animation = lambda *a, **k: None

    class FixedModel:
        def __init__(self, text): self.text=text
        def ask(self, **kwargs): return self.text

    evidence = dict(parity=[], geometry_segments=0, checks=[])
    rng = random.Random(20261013)
    for case in selected_cases(30):
        query = case['query']
        geometry = Geometry(query)
        start, goal = query['start'], query['goal']
        # Arbitrary targets deliberately cover poor guidance, filtered boundary
        # targets, fractional coordinates, and a missing-list fallback.
        responses = [json.dumps([start,[25,15],goal]),'No path available']
        for method, response in [('astar','')]+[('llm_astar',t) for t in responses]:
            original = AStar() if method == 'astar' else LLMAStar()
            if method == 'llm_astar': original.model = FixedModel(response)
            with contextlib.redirect_stdout(io.StringIO()):
                before = original.searching(copy.deepcopy(query))
            after = search(query,geometry,method,response)
            for metric in ('operation','storage','length'):
                assert math.isclose(before[metric],after[metric],rel_tol=1e-12), (case['case_id'],method,metric,before[metric],after[metric])
            original_path = original.extract_path(original.PARENT)
            if method == 'astar': original_path.reverse()
            assert [tuple(p) for p in after['path']] == original_path
            assert after['valid']
            evidence['parity'].append(dict(case_id=case['case_id'],method=method,
                                          response=response,operation=after['operation'],storage=after['storage'],length=after['length']))
        assert math.isclose(search(query,geometry)['length'],dijkstra(query,geometry),rel_tol=1e-12)
        lines = [LineString([(a,y),(b,y)]) for y,a,b in geometry.horizontal]
        lines += [LineString([(x,a),(x,b)]) for x,a,b in geometry.vertical]
        for _ in range(100):
            a = (rng.uniform(1,49),rng.uniform(1,29))
            b = (rng.uniform(1,49),rng.uniform(1,29))
            independent = any(LineString([a,b]).intersects(line) for line in lines)
            assert geometry.collision(a,b) == independent, (a,b)
            evidence['geometry_segments'] += 1
        print('Verified',case['case_id'],flush=True)
    q=dict(start=[2,2],goal=[2,4],range_x=[0,7],range_y=[0,7],horizontal_barriers=[[3,0,6]],vertical_barriers=[])
    before=copy.deepcopy(q)
    assert search(q)['error']=='unreachable'
    assert q==before
    q['horizontal_barriers']=[]
    assert parse_output('[[1,2],[bad,3]]')[1]=='malformed_list'
    assert parse_output('[[2,2,7],[2,4,9]]',upstream=True)[0]==[(2,2),(2,4)]
    assert parse_output('[[2,2,7],[2,4,9]]')[1]=='malformed_coordinate'
    assert not llm_only(q,Geometry(q),'[[2,2],[4,4]]')['valid']
    fallback=search(q,method='llm_astar',text='No path available')
    control=search(q,method='goal_only')
    assert all(fallback[k]==control[k] for k in ['operation','storage','length','path'])
    q['goal']=q['start'].copy()
    assert search(q)['valid'] and search(q)['length']==0
    from .proposed import proposed_search
    from .planners import Proposed
    q['goal']=[2,4]
    for response in ['no list','[[2,2],[oops,4]]','[[2,2],[99,99],[2,4]]']:
        result=proposed_search(q,text=response)
        assert result['valid'] and not result['guidance_accepted']
    assert proposed_search(q,text='[[2,2],[2,4]]')['guidance_accepted']
    planner=Proposed(provider=lambda model,prompt:'[[2,2],[2,4]]')
    assert planner.searching(q,filepath=None)['valid']
    evidence['checks']=['30 independent Dijkstra optima','90 upstream path/operation/storage/length parity cases',
                        '3000 continuous segments vs Shapely','unreachable handling','input immutability',
                        'malformed output','endpoint validation','missing-list equals g+2h','start equals goal',
                        'proposal malformed/missing/outside fallback','proposal valid guidance','assignment adapter']
    out=ROOT/'results/verification'
    out.mkdir(parents=True,exist_ok=True)
    (out/'offline_checks.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=evidence['checks'])),flush=True)


if __name__=='__main__':
    main()
