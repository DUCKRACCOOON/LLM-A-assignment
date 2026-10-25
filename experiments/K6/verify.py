"""Independent full-scan search/control reference and geometry oracle checks."""
import json
import math
import random

from reproduction.core import Geometry
from experiments.K3.heuristic import LowerBound
from experiments.K3.verification import Oracle,fixtures,check_bounds
from .search import Controller,NEW_METHODS,prepare_targets,graph_search,run_once


def reference(start,goal,graph,h,targets,mode,initial,window,weight,literal):
    # No heap, versions, rekey operations or production Controller.
    opened={start};g={start:0.0};parent={start:start};expanded=set();sequence=[]
    index=0;alpha=initial if len(targets)>1 and mode!='off' else 0.0
    best=math.dist(start,targets[index]);local_best=best;count=0
    def priority(n):
        if n==goal:return g[n]
        low=h(n);distance=math.dist(n,targets[index])
        estimate=(low+alpha*distance) if literal else weight*low+alpha*(weight-1)*(min(distance,low)-low)
        return g[n]+estimate
    while opened:
        node=min(opened,key=lambda n:(priority(n),n))
        sequence.append((node,index,alpha));opened.remove(node)
        if node==goal:
            path=[node]
            while node!=start:node=parent[node];path.append(node)
            return list(reversed(path)),sequence
        for nxt,cost in graph[node]:
            value=g[node]+cost
            if value<g.get(nxt,math.inf):
                g[nxt],parent[nxt]=value,node;opened.add(nxt)
        expanded.add(node);old_index=index
        while index<len(targets)-1 and targets[index] in expanded:index+=1
        if index!=old_index:
            alpha=initial if index<len(targets)-1 and mode!='off' else 0.0
            best=math.dist(node,targets[index]);local_best=best;count=0
        elif mode=='adaptive' and alpha>0:
            local_best=min(local_best,math.dist(node,targets[index]));count+=1
            if count==window:
                if best-local_best>=1:alpha=min(1.,alpha+.25)
                else:
                    alpha/=2
                    if alpha<=1/16:alpha=0.
                best=min(best,local_best);local_best=best;count=0
    return [],sequence


def compare(query,text,method,oracle):
    result=run_once(query,method,text,trace=True,audit=True)
    start,goal=tuple(query['start']),tuple(query['goal'])
    geometry=Geometry(query)
    wall='euclidean' not in method
    h=LowerBound(geometry,goal,base='octile' if wall else 'euclidean',walls=wall)
    targets=[goal] if method=='no_guide_wall_w200' else prepare_targets(geometry,start,goal,text)[0]
    mode='fixed' if method.startswith('fixed') else ('off' if method=='no_guide_wall_w200' else 'adaptive')
    initial=1. if method in ('fixed_full_wall_w200','fixed_literal_wall') else .5
    weight=1.05 if method.endswith('w105') else 2.
    path,seq=reference(start,goal,oracle.graph,h,targets,mode,initial,16 if 'window16' in method else 8,weight,'literal' in method)
    assert path==result['path'],(query,method,'path')
    assert seq==[(e['node'],e['target_index'],e['alpha']) for e in result['trace']],(query,method,'selection')
    optimum=oracle.distances(goal).get(start)
    if optimum is None:
        assert not result['valid'] and result['error']=='unreachable'
    else:
        oracle.check_path(query,result)
        assert result['length']+1e-8>=optimum
        if result['bound_factor'] is not None:assert result['length']<=weight*optimum+1e-8
    assert result['operation']==result['stale_pops']+result['successor_expansions']+int(result['valid'])
    assert result['rekey_entries']==result['rekey_scans']
    return result['selection_audits']


def verify():
    runs=selections=collisions=bound_checks=0
    maps=list(fixtures().items());rng=random.Random(20261028)
    for i in range(30):
        width,height=rng.randint(8,16),rng.randint(8,14)
        q=dict(range_x=[0,width+1],range_y=[0,height+1],
               horizontal_barriers=[[rng.randint(2,height-2),*sorted(rng.sample(range(width+1),2))] for _ in range(2)],
               vertical_barriers=[[rng.randint(2,width-2),*sorted(rng.sample(range(height+1),2))] for _ in range(2)])
        o=Oracle(q);goal=rng.choice(sorted(o.graph));start=rng.choice(sorted(o.distances(goal)))
        q.update(start=list(start),goal=list(goal));maps.append((f'random_{i}',q))
    for name,q in maps:
        o=Oracle(q,crosscheck=True);collisions+=o.collision_checks
        b=check_bounds(q,o);bound_checks+=b['admissibility_checks']+b['consistency_checks']
        valid_points=sorted(o.graph)
        texts=['missing',json.dumps([q['start'],valid_points[len(valid_points)//3],valid_points[-1],q['goal']]),
               json.dumps([q['goal'],[-50,70],[3.5,4],[6,4],[6,4],q['start'],[2,2]])]
        for text in texts:
            for method in NEW_METHODS:
                selections+=compare(q,text,method,o);runs+=1
    # Direct controller observations: improvement, saturation, decay, latch.
    c=Controller([(10,1),(20,1)],(1,1),window=2)
    c.observe((1,1));c.observe((3,1));assert c.alpha==.75 and c.good_windows==1
    c.observe((4,1));c.observe((5,1));assert c.alpha==1.
    for _ in range(8):c.observe((5,1))
    assert c.alpha==0 and c.disables==1
    c.observe((9,1));assert c.alpha==0  # No recovery until target changes.
    c.observe((10,1));assert c.index==1 and c.alpha==0
    c=Controller([(3,1),(2,1),(8,1),(20,1)],(1,1),window=8)
    c.observe((2,1));c.observe((3,1));assert c.index==2 and c.alpha==.5
    # Valid points do not imply valid joining segments; no whole-guide gate.
    q=fixtures()['long_wall'];geometry=Geometry(q)
    targets,meta=prepare_targets(geometry,tuple(q['start']),tuple(q['goal']),
                                 '[[0,4],[6,4],[3.5,4],[2,2],[2,2],[8,2]]')
    assert targets[:-1]==[(2,2),(8,2)] and meta['guide_rejected']=={'blocked_or_outside':2,'fractional':1,'consecutive_duplicate':1}
    assert prepare_targets(geometry,tuple(q['start']),tuple(q['goal']),'[[1,2,3]]')[1]['guide_error']
    assert run_once(dict(q,start=[0,4]),'adaptive_wall_w200')['error']=='invalid_endpoint'
    # Directed graph forces a closed node to be reopened in bounded mode.
    S,A,B,G=(0,0),(1,0),(0,1),(2,0)
    graph={S:[(A,4.),(B,1.)],A:[(G,10.)],B:[(A,1.),(G,100.)],G:[]}
    h={S:0.,A:0.,B:11.,G:0.}
    # weight=1 and admissible inconsistent h discovers A at 4, then at 2.
    result=graph_search(S,G,graph.__getitem__,h.__getitem__,Controller([G],S),weight=1,trace=True,audit=True)
    assert result['goal_g']==12 and result['reopenings']==1
    # No-guide engine agrees with the old weighted engine (including tie order).
    parity=0
    for _,q in maps:
        if tuple(q['start'])==tuple(q['goal']):continue
        a=run_once(q,'no_guide_wall_w200');b=run_once(q,'clean_wastar_wall')
        for field in ('path','operation','storage','length','valid'):assert a[field]==b[field],field
        parity+=1
    for weight in (0,.9,math.inf,math.nan):
        try:graph_search(S,G,graph.__getitem__,h.__getitem__,Controller([G],S),weight=weight)
        except ValueError:pass
        else:raise AssertionError('Invalid weight accepted')
    return dict(status='passed',reference_cases=runs,full_scan_selections=selections,
                independent_collision_checks=collisions,heuristic_checks=bound_checks,
                no_guide_engine_parity_cases=parity,controller_filter_numeric_checks='passed',
                reopening_graph=result)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
