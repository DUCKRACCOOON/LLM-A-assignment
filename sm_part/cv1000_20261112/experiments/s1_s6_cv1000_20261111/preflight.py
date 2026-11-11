import csv
import math
from common import *
from algorithms import run, CountGeometry, delayed
from experiments.K3.verification import fixtures, Oracle
from next_search_core import Grid as ExactGrid
from candidate_core import Grid as CandidateGrid

def preflight():
    checked=0;edge_checks=0
    refs={}
    with (GM/'experiments/K3/results/run_20261025/public/per_case.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):refs[(row['case_id'],row['method'])]=row
    methods={'A_GM':'astar_euclidean','N_g2h':'g2h_euclidean','K3_wall_euclidean':'g2h_wall_euclidean','K3_wall_octile':'g2h_wall_octile'}
    for i,q in enumerate(cases('public')):
        for name,refname in methods.items():
            r=run(q,name);ref=refs[(q['id'],refname)]
            assert r['valid'] and r['operation']==int(ref['operation']) and r['storage']==int(ref['storage']),(q['id'],name)
            assert math.isclose(r['length'],float(ref['length']),abs_tol=1e-8)
            checked+=1
        if i%100==0:print(json.dumps(dict(phase='parity',done=i,total=1000)),flush=True)
    for name,q in fixtures().items():
        q=dict(q,id='fixture_'+name);oracle=Oracle(q,crosscheck=True);opt=oracle.distances(q['goal']).get(tuple(q['start']))
        exact_grid=ExactGrid(q);candidate_grid=CandidateGrid(q)
        for p,edges in oracle.graph.items():
            expected={n for n,_ in edges};actual={exact_grid.point(n) for n,_ in exact_grid.neighbors(exact_grid.index(p))}
            cand={tuple(map(int,candidate_grid.xy[n])) for n in candidate_grid.edges[candidate_grid.index(p)] if n>=0}
            assert expected==actual==cand,(name,p);edge_checks+=len(expected)
        for method in ('A_GM','N_g2h','A_OCT','S5','S6_rule','S6_random','S6_historical','FOCAL_105'):
            r=run(q,method)
            assert r['valid']==(opt is not None),(name,method)
            if opt is not None:
                oracle.check_path(q,r)
                if method in ('A_GM','A_OCT','S5','S6_rule','S6_random','S6_historical'):assert math.isclose(r['length'],opt,abs_tol=1e-8),(name,method,r['length'],opt)
                if method=='FOCAL_105':assert r['length']<=1.05*opt+1e-8
            checked+=1
        for text in ('',json.dumps([q['start'],[4,4],[9,7],q['goal']]),'Generated Path: [[1,2,3],[4,5,6]]'):
            for mode,base in [('never','goal_only'),('immediate','llm_astar')]:
                gm.Geometry.neighbors.cache_clear();r=delayed(q,CountGeometry(q),text,mode)
                gm.Geometry.neighbors.cache_clear();b=gm.search(q,method=base,text=text)
                assert r['path']==b['path'] and r['valid']==b['valid'],(name,mode)
                checked+=1
    # The S1 gate-only intervention must make no extra call and share normalization.
    for q in cases('pilot')[:5]:
        for text in ('malformed',json.dumps([q['start'],q['goal']]),'[[1, 2], [1, 2], [3, 4]]'):
            for method in ('S1','V_GATE_NORMALIZED','S3'):
                r=run(q,method,text);assert r['valid'];checked+=1
    save(OUT/'preflight.json',dict(passed=True,checks=checked,edge_checks=edge_checks,python=platform.python_version(),updated_at=now()))
    print(json.dumps(read(OUT/'preflight.json')),flush=True)

if __name__=='__main__':preflight()
