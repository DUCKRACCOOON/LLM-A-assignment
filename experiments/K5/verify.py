"""Heap-free SMHA reference, independent geometry/oracle, scheduler fixtures."""
from collections import Counter
import json
import math
import random

from reproduction.core import Geometry
from experiments.K3.heuristic import LowerBound
from experiments.K3.verification import Oracle, fixtures, check_bounds
from experiments.K4.search import PolylineGuide
from .search import EPS, Scheduler, graph_search, run_once, METHODS


def reference(start, goal, graph, hs, weight, mode, seed):
    # Authoritative sets; no heap versions, stale cleanup, or reward cache.
    opened = [{start} for _ in hs]
    g,parents = {start:0.0},{start:start}
    anchor_closed,aux_closed = set(),set()
    alpha,beta,best = [1.0]*(len(hs)-1),[1.0]*(len(hs)-1),[h(start) for h in hs[1:]]
    rng,turn,sequence = random.Random(seed),0,[]
    while opened[0]:
        f0 = min(g[n]+hs[0](n) for n in opened[0])
        if mode == 'rr':
            arm = turn % (len(hs)-1)
        else:
            samples = [rng.betavariate(a,b) for a,b in zip(alpha,beta)]
            arm = max(range(len(samples)),key=samples.__getitem__)
        turn += 1
        i = arm+1
        if not opened[i] or min(g[n]+hs[i](n) for n in opened[i]) > weight*f0+EPS:
            i = 0
        node = min(opened[i],key=lambda n:(g[n]+hs[i](n),n))
        if g.get(goal,math.inf) <= g[node]+hs[i](node)+EPS:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return list(reversed(path)),sequence,alpha,beta
        sequence.append((node,i))
        for group in opened:
            group.discard(node)
        (anchor_closed if i == 0 else aux_closed).add(node)
        for nxt,cost in graph[node]:
            new = g[node]+cost
            if new < g.get(nxt,math.inf):
                g[nxt],parents[nxt] = new,node
                for group in opened:
                    group.discard(nxt)
                if nxt not in anchor_closed:
                    opened[0].add(nxt)
                    if nxt not in aux_closed:
                        for j in range(1,len(hs)):
                            if new+hs[j](nxt) <= weight*(new+hs[0](nxt))+EPS:
                                opened[j].add(nxt)
        if i and mode != 'rr':
            minimum = min((hs[i](n) for n in opened[i]),default=math.inf)
            if minimum < best[arm]:
                best[arm] = minimum
                alpha[arm] += 1
            else:
                beta[arm] += 1
            if mode == 'dts' and alpha[arm]+beta[arm] > 10:
                alpha[arm],beta[arm] = alpha[arm]*10/11,beta[arm]*10/11
    return [],sequence,alpha,beta


def compare(query,text,weight,mode,seed,oracle):
    start,goal = tuple(query['start']),tuple(query['goal'])
    h = LowerBound(Geometry(query),goal)
    guide = PolylineGuide(query,text)
    hs = [h,lambda n:2*h(n),lambda n:2*math.dist(n,goal),
          (lambda n:2*h(n)) if guide.error else (lambda n:0.0 if n==goal else 2*guide(n))]
    result = graph_search(start,goal,oracle.graph.__getitem__,hs,
                          weight=weight,mode=mode,seed=seed,trace=True,audit=True)
    expected,sequence,alpha,beta = reference(start,goal,oracle.graph,hs,weight,mode,seed)
    assert result['path'] == expected
    assert [(e['node'],e['queue']) for e in result['trace']] == sequence
    assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(alpha,result['bandit_alpha']))
    assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(beta,result['bandit_beta']))
    counts = Counter(n for n,i in sequence)
    assert max(counts.values(),default=0) <= 2
    for node in counts:
        order = [i for n,i in sequence if n == node]
        assert len(order) == 1 or (order[0] != 0 and order[1] == 0)
    optimum = oracle.distances(goal).get(start)
    if optimum is None:
        assert not result['path']
    else:
        result.update(valid=True,length=sum(math.dist(a,b) for a,b in zip(expected,expected[1:])))
        oracle.check_path(query,result)
        assert result['length'] <= weight*optimum+1e-8, (query,weight,mode,seed,result,optimum)
        assert result['anchor_key_at_goal'] <= optimum+1e-8
    assert sum(result['queue_expansions']) == result['successor_expansions']
    assert result['operation'] == result['search_pops']+result['reward_pops']
    assert sum(result['bandit_updates']) == (0 if mode=='rr' else sum(result['queue_expansions'][1:]))
    return result['selection_audits']


def verify():
    total,runs,bounds,collisions = 0,0,0,0
    cases = list(fixtures().items())
    rng = random.Random(20261027)
    for mi in range(30):
        width,height = rng.randint(8,16),rng.randint(8,14)
        q = dict(range_x=[0,width+1],range_y=[0,height+1],
                 horizontal_barriers=[[rng.randint(2,height-2),*sorted(rng.sample(range(width+1),2))] for _ in range(2)],
                 vertical_barriers=[[rng.randint(2,width-2),*sorted(rng.sample(range(height+1),2))] for _ in range(2)])
        o = Oracle(q)
        goal = rng.choice(sorted(o.graph)); start = rng.choice(sorted(o.distances(goal)))
        q.update(start=list(start),goal=list(goal))
        cases.append((f'random_{mi}',q))
    for name,query in cases:
        oracle = Oracle(query,crosscheck=True)
        collisions += oracle.collision_checks
        b = check_bounds(query,oracle)
        bounds += b['admissibility_checks']+b['consistency_checks']
        texts = [json.dumps([query['start'],query['goal']]),'missing',
                 json.dumps([query['goal'],[-50,70],[6,4],query['start']])]
        for text in texts:
            for weight in (1.0,1.05,2.0):
                for mode in ('rr','ts','dts'):
                    for seed in (0,20261027):
                        total += compare(query,text,weight,mode,seed,oracle);runs += 1
    graph = {'S':[('A',4.),('B',1.)],'A':[('G',10.)],'B':[('A',1.),('G',100.)],'G':[]}
    aux = dict(S=0,A=0,B=100,G=0)
    result = graph_search('S','G',graph.__getitem__,[lambda n:0,aux.__getitem__],
                          weight=4,mode='rr',trace=True,audit=True)
    assert result['goal_g']==12 and result['reexpansions']==1
    assert [(e['node'],e['queue']) for e in result['trace']] == [('S',1),('A',1),('B',0),('A',0)]
    # Closed-anchor states may get lower g but must never be requeued.
    ref = reference('S','G',graph,[lambda n:0,aux.__getitem__],4,'rr',0)
    assert result['path']==ref[0]
    bandit = Scheduler([10],seed=0)
    for _ in range(40): bandit.update(0,10)
    assert bandit.rescales==32 and math.isclose(bandit.alpha[0]+bandit.beta[0],10)
    before=bandit.alpha[0]/sum([bandit.alpha[0],bandit.beta[0]])
    bandit.update(0,9)
    assert bandit.rewards == [1] and bandit.alpha[0]/(bandit.alpha[0]+bandit.beta[0])>before
    q = fixtures()['long_wall']
    wrapped = []
    for method in METHODS:
        if method.startswith('mha'):
            a = run_once(q,method,'[[2,4],[20,10],[10,4]]',seed=11,trace=True,audit=True)
            b = run_once(q,method,'[[2,4],[20,10],[10,4]]',seed=11,trace=True,audit=True)
            assert {k:v for k,v in a.items() if k!='total_seconds'} == {k:v for k,v in b.items() if k!='total_seconds'}
            Oracle(q).check_path(q,a);wrapped.append(method)
    bad = dict(q,start=[0,4])
    assert run_once(bad,'mha_dts_w200')['error']=='invalid_endpoint'
    for text in ('[[1,2,3],[4,5,6]]','[[1e308,1e308],[-1e308,-1e308]]'):
        r = run_once(q,'mha_dts_w200',text)
        assert r['valid'] and r['guide_error']
    for weight in (0,.9,math.inf,math.nan):
        try: graph_search('S','G',graph.__getitem__,[lambda n:0,lambda n:0],weight=weight)
        except ValueError: pass
        else: raise AssertionError('Invalid weight accepted')
    return dict(status='passed',reference_cases=runs,full_scan_selection_checks=total,
                independent_collision_checks=collisions,heuristic_checks=bounds,
                random_maps=30,fixtures=10,wrapped_conditions=wrapped,
                aux_then_anchor_case=result,scheduler_rescale_and_reward='passed',
                seed_reproducibility_and_numeric_endpoints='passed')


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
