"""Wall-level relaxation with full-path counterexamples and exact fallback."""
import math
import random
import time
from collections import Counter
from core import intersects
from next_search_core import Grid, search, walls, subset
from next_geometry import Rectangles


def features(q):
    s=tuple(q['start']); g=tuple(q['goal']); scale=math.hypot(q['range_x'][1],q['range_y'][1])
    out=[]
    for a,b in walls(q):
        out.append([float(intersects(s,g,a,b)), math.dist(a,b)/scale,
                    min(math.dist(s,a)+math.dist(a,g),math.dist(s,b)+math.dist(b,g))/scale,
                    float(any(p[0] in (0,q['range_x'][1]-1) or p[1] in (0,q['range_y'][1]-1) for p in (a,b)))])
    return out


def solve(q,mode='rule',compressed=False,instrument=True,program=None,deadline=None):
    t=time.perf_counter(); bs=walls(q); selected=set(); fs=None
    if mode in ('rule','random','program','incremental'):
        fs=features(q); selected={i for i,f in enumerate(fs) if f[0]}
        if mode=='random': selected=set(random.Random(str(q.get('id',q))).sample(range(len(bs)),len(selected)))
        if mode=='program':
            weights=program or [1.,0.,-0.1,0.1]
            k=len(selected)
            selected=set(sorted(range(len(bs)),key=lambda i:(-sum(a*b for a,b in zip(fs[i],weights)),i))[:k])
    prepare=time.perf_counter()-t; rounds=[]; total=Counter(); validation=0.; all_search=0.; reconstruction=0.; fallback=False;lpa=None
    for turn in range(4):
        if turn==3: selected=set(range(len(bs))); fallback=True
        tq=subset(q,selected); graph=(Rectangles(tq,instrument) if compressed else Grid(tq,instrument))
        prepare+=graph.seconds
        if mode=='incremental' and not compressed and not graph.degenerate:
            from next_incremental import LPA
            update=time.perf_counter()
            if lpa is None:lpa=LPA(graph,tq)
            else:lpa.change(graph)
            prepare+=time.perf_counter()-update
            r=lpa.run(deadline)
        else:r=search(graph,tq,instrument=instrument,deadline=deadline)
        total.update(r['counters']); all_search+=r['search_seconds']; reconstruction+=r['reconstruction_seconds']
        check=time.perf_counter(); hits=set(); tested=0
        if r['reachable']:
            for a,b in zip(r['path'],r['path'][1:]):
                for i,(c,d) in enumerate(bs):
                    tested+=1
                    if intersects(tuple(a),tuple(b),c,d):
                        hits.add(i)
                        if mode=='first': break
                if hits and mode=='first': break
        validation+=time.perf_counter()-check
        if instrument: total['validation_wall_intersections']+=tested; total['validation_edges']+=max(0,len(r['path'])-1)
        rounds.append(dict(active_walls=len(selected),new_hits=sorted(hits-selected),expansions=r['counters']['expansions'],length=r['length']))
        if not r['reachable'] or not hits: break
        assert hits-selected, 'counterexample failed to refine'
        selected.update(hits)
    else: raise AssertionError('full-wall fallback failed')
    r.update(counters=dict(total),search_seconds=all_search,reconstruction_seconds=reconstruction,
             preparation_seconds=prepare,required_validation_seconds=validation,rounds=rounds,
             fallback=fallback,active_walls=len(selected),total_seconds=time.perf_counter()-t)
    return r
