"""Verify the new configurations against independent selection and geometry."""
import json
import random

from reproduction.core import Geometry
from experiments.K3.heuristic import LowerBound
from experiments.K3.verification import Oracle,fixtures,check_bounds
from experiments.K6.verify import reference
from experiments.K6.search import prepare_targets,run_once as old_run
from .search import PRIMARY,TIGHT,CONFIG,run_once,no_guide_reference


def verify():
    maps=list(fixtures().items());rng=random.Random(20261029)
    for i in range(30):
        width,height=rng.randint(8,16),rng.randint(8,14)
        q=dict(range_x=[0,width+1],range_y=[0,height+1],
               horizontal_barriers=[[rng.randint(2,height-2),*sorted(rng.sample(range(width+1),2))] for _ in range(2)],
               vertical_barriers=[[rng.randint(2,width-2),*sorted(rng.sample(range(height+1),2))] for _ in range(2)])
        o=Oracle(q);goal=rng.choice(sorted(o.graph));start=rng.choice(sorted(o.distances(goal)))
        q.update(start=list(start),goal=list(goal));maps.append((f'random_{i}',q))
    count=selections=collisions=heuristic_checks=adaptive_parity=no_guide_parity=0
    for name,q in maps:
        oracle=Oracle(q,crosscheck=True);collisions+=oracle.collision_checks
        bounds=check_bounds(q,oracle);heuristic_checks+=bounds['admissibility_checks']+bounds['consistency_checks']
        start,goal=tuple(q['start']),tuple(q['goal']);geometry=Geometry(q)
        lower=LowerBound(geometry,goal);valid=sorted(oracle.graph)
        texts=['missing',json.dumps([q['start'],valid[len(valid)//3],valid[-1],q['goal']]),
               json.dumps([[-50,70],[3.5,4],[6,4],[6,4],q['start'],[2,2]])]
        for text in texts:
            rows={}
            for method in TIGHT:
                mode,initial=CONFIG[method]
                targets=[goal] if mode=='off' else prepare_targets(geometry,start,goal,text)[0]
                path,seq=reference(start,goal,oracle.graph,lower,targets,mode,initial,8,1.05,False)
                r=run_once(q,method,text,trace=True,audit=True);rows[method]=r
                assert path==r['path'],(name,method,'path')
                assert seq==[(e['node'],e['target_index'],e['alpha']) for e in r['trace']],(name,method,'sequence')
                if bounds['optimum'] is None:assert not r['valid'] and r['error']=='unreachable'
                else:
                    oracle.check_path(q,r);assert bounds['optimum']-1e-8<=r['length']<=1.05*bounds['optimum']+1e-8
                assert r['operation']==r['stale_pops']+r['successor_expansions']+int(r['valid'])
                assert r['rekey_scans']==r['rekey_entries']
                if mode=='fixed':assert all(e['alpha'] in (0,initial) for e in r['trace'])
                if mode=='off':assert r['guide_distance_checks']==r['heap_rebuilds']==r['alpha_updates']==0
                count+=1;selections+=r['selection_audits']
            old=old_run(q,PRIMARY,text,trace=True,audit=True)
            assert {k:v for k,v in rows[PRIMARY].items() if k!='total_seconds'}=={k:v for k,v in old.items() if k!='total_seconds'}
            adaptive_parity+=1
            if text=='missing':
                for method in TIGHT:
                    assert all(rows[method][k]==rows['no_guide_wall_w105'][k] for k in ('path','operation','storage','length'))
            ref=no_guide_reference(q)
            assert all(ref[k]==rows['no_guide_wall_w105'][k] for k in ('path','operation','storage','goal_g'))
            no_guide_parity+=1
    q=fixtures()['long_wall']
    for method in TIGHT:assert run_once(dict(q,start=[0,4]),method)['error']=='invalid_endpoint'
    # Removing the LLM really removes parsing, even for pathological text.
    a=run_once(q,'no_guide_wall_w105','[[1,2,3]]');b=run_once(q,'no_guide_wall_w105','no response')
    assert {k:v for k,v in a.items() if k!='total_seconds'}=={k:v for k,v in b.items() if k!='total_seconds'}
    return dict(status='passed',reference_cases=count,full_scan_selections=selections,
                collision_checks=collisions,heuristic_checks=heuristic_checks,adaptive_exact_parity=adaptive_parity,
                no_guide_reference_parity=no_guide_parity,invalid_endpoint_checks=len(TIGHT),no_parse_check='passed')


if __name__=='__main__':print(json.dumps(verify(),indent=2))
