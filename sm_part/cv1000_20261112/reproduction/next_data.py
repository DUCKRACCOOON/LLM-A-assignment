"""Seeded map-disjoint synthetic data without human acceptance filtering."""
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
import numpy as np
from core import intersects,scale_case
from next_search_core import Grid

FAMILIES=('original_style_segments','dense_segments','corridors_dead_ends','multiple_detours')
SEEDS={'development':26101410,'validation':26101411,'final':26101412}


def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def geometry(rng,family,j):
    w,h=(51,31) if family==FAMILIES[0] else (101,61)
    q=dict(range_x=[0,w],range_y=[0,h],horizontal_barriers=[],vertical_barriers=[])
    hb=q['horizontal_barriers']; vb=q['vertical_barriers']
    if family in FAMILIES[:2]:
        nh=round(rng.uniform(1,4)) if family==FAMILIES[0] else (6 if j%2==0 else 12)
        nv=round(rng.uniform(1,4)) if family==FAMILIES[0] else nh
        for _ in range(nh):
            y=rng.randrange(1,h-1); a=rng.randrange(1,w-2); b=rng.randrange(a+1,w)
            hb.append([y,a,b])
        for _ in range(nv):
            x=rng.randrange(1,w-1); a=rng.randrange(1,h-2); b=rng.randrange(a+1,h)
            vb.append([x,a,b])
    elif family==FAMILIES[2]:
        gap=j%3+1
        for i,x in enumerate(range(12,95,12)):
            vb.append([x,0,h-2-gap] if i%2==0 else [x,gap+1,h-1])
            y=rng.randrange(12,h-12); hb.append([y,x,min(w-2,x+8)])
    else:
        for i,x in enumerate((20,40,60,80)):
            a=rng.randrange(8,20); b=rng.randrange(39,53)
            vb.append([x,a,b])
            if i%2==(j%2): hb.append([b,x,min(w-2,x+rng.randrange(7,15))])
        # Some wall groups attach to boundaries, others remain separate.
        if j%3==0: hb.append([rng.randrange(25,36),0,13])
    return q


def components(grid):
    labels=np.full(grid.n,-1,dtype=np.int32); groups=[]
    for s in range(grid.n):
        if grid.flat[s] or labels[s]>=0: continue
        ids=[]; rid=len(groups); labels[s]=rid; queue=deque([s])
        while queue:
            n=queue.popleft(); ids.append(n)
            for v,_ in grid.neighbors(n):
                if labels[v]<0: labels[v]=rid; queue.append(v)
        groups.append(ids)
    return labels,groups


def generate(split,maps_per_family=15):
    rng=random.Random(SEEDS[split]); out=[]; log=[]; t=time.perf_counter()
    for family in FAMILIES:
        for j in range(maps_per_family):
            attempts=0
            while True:
                attempts+=1; q=geometry(rng,family,j); grid=Grid(q,False); labels,groups=components(grid)
                good=[g for g in groups if len(g)>=30]
                if good: break
            geom_hash=digest(q); map_id=f'{split}-{family}-{j:03d}'; pairs=[]; rejected=0
            candidates=[i for g in good for i in g]
            while len(pairs)<10:
                a,b=rng.sample(candidates,2)
                if labels[a]!=labels[b]: rejected+=1; continue
                s,g=list(grid.point(a)),list(grid.point(b))
                if family==FAMILIES[0] and not any(intersects(tuple(s),tuple(g),c,d) for c,d in grid.segments):
                    rejected+=1
                    if rejected>100000: raise RuntimeError('query generator rejection limit')
                    continue
                if (s,g) in pairs: rejected+=1; continue
                pairs.append((s,g))
            for i,(s,g) in enumerate(pairs):
                out.append(dict(q,id=f'{map_id}-q{i:02d}',map_id=map_id,base_map_id=map_id,pair_id=i,
                                family=family,geometry_hash=geom_hash,start=s,goal=g))
            log.append(dict(map_id=map_id,geometry_hash=geom_hash,map_attempts=attempts,query_rejections=rejected,
                            direct_queries=sum(not any(intersects(tuple(s),tuple(g),c,d) for c,d in grid.segments) for s,g in pairs)))
    return out,dict(split=split,seed=SEEDS[split],maps=log,generation_seconds=time.perf_counter()-t)


def selected_maps(cases,n):
    answer=set()
    for f in FAMILIES:
        maps={q['map_id']:q['geometry_hash'] for q in cases if q['family']==f}
        answer.update(sorted(maps,key=lambda m:(maps[m],m))[:n])
    return answer


def r3_subset(cases,stage):
    chosen=selected_maps(cases,20 if stage=='final' else 6)
    return [q for q in cases if q['map_id'] in chosen and q['pair_id']<5]


def auxiliary(final):
    chosen=selected_maps(final,10); scaled=[scale_case(q,f) for q in final if q['map_id'] in chosen and q['pair_id']<2 for f in (1,2,4,8)]
    reusemaps=selected_maps(final,5); varying=[q for q in final if q['map_id'] in reusemaps]
    fixed=[]; rng=random.Random(26101413)
    for m in sorted(reusemaps):
        base=next(q for q in final if q['map_id']==m); grid=Grid(base,False); labels,groups=components(grid)
        goal=grid.index(base['goal']); group=groups[int(labels[goal])]
        for j,s in enumerate(rng.sample([n for n in group if n!=goal],10)):
            fixed.append(dict(base,id=f'{m}-fixed{j:02d}',start=list(grid.point(s)),goal=list(base['goal']),pair_id=j))
    return scaled,varying,fixed
