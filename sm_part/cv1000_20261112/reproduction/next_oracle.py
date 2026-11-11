"""Independent segment-intersection graph and SciPy Dijkstra, grouped by map.

The oracle never uses the candidate's occupied-vertex shortcut or rectangles.
Each map also receives direct scalar collision spot checks.
"""
import math
import time
from collections import defaultdict
import numpy as np
from core import World
from next_search_core import independent_distance


def compute(cases,progress=None):
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError:
        return {q['id']:dict(optimal=(lambda x:x if math.isfinite(x) else None)(independent_distance(q)),oracle='scalar_heap_dijkstra') for q in cases}
    groups=defaultdict(list)
    for q in cases:groups[(q['map_id'],q.get('scale',1))].append(q)
    out={}
    for ii,qs in enumerate(groups.values()):
        t=time.perf_counter();q=qs[0];world=World(q);w=q['range_x'][1];h=q['range_y'][1]
        yy,xx=np.mgrid[1:h-1,1:w-1];a=np.stack((xx.ravel(),yy.ravel()),axis=1)
        rr=[];cc=[];cost=[];audit_count=0
        for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
            b=a+(dx,dy);good=(b[:,0]>0)&(b[:,0]<w-1)&(b[:,1]>0)&(b[:,1]<h-1)
            for c0,d0 in world.barriers:
                c=np.array(c0);d=np.array(d0)
                if c0==d0:
                    for j in np.flatnonzero(good):
                        if world.collision(tuple(a[j]),tuple(b[j])):good[j]=False
                    continue
                def cross(u,v):return u[...,0]*v[...,1]-u[...,1]*v[...,0]
                u=cross(b-a,c-a);v=cross(b-a,d-a);z=cross(d-c,a-c);r=cross(d-c,b-c)
                overlap=(np.maximum(a[:,0],b[:,0])>=min(c[0],d[0]))&(np.minimum(a[:,0],b[:,0])<=max(c[0],d[0]))
                overlap&=(np.maximum(a[:,1],b[:,1])>=min(c[1],d[1]))&(np.minimum(a[:,1],b[:,1])<=max(c[1],d[1]))
                good&=~((u*v<=0)&(z*r<=0)&overlap)
            for j in np.linspace(0,len(a)-1,min(128,len(a)),dtype=int):
                aa=tuple(map(int,a[j]));bb=tuple(map(int,b[j]))
                expected=0<bb[0]<w-1 and 0<bb[1]<h-1 and not world.collision(aa,bb)
                assert bool(good[j])==expected,'oracle vector/scalar geometry mismatch';audit_count+=1
            sa=a[good,1]*w+a[good,0];sb=b[good,1]*w+b[good,0]
            rr.extend((sa,sb));cc.extend((sb,sa));cost.extend((np.full(len(sa),math.hypot(dx,dy)),)*2)
        matrix=csr_matrix((np.concatenate(cost),(np.concatenate(rr),np.concatenate(cc))),shape=(w*h,w*h))
        goals=sorted(set(q['goal'][1]*w+q['goal'][0] for q in qs))
        distances=dijkstra(matrix,directed=False,indices=goals)
        for q in qs:
            g=q['goal'][1]*w+q['goal'][0];s=q['start'][1]*w+q['start'][0];v=float(distances[goals.index(g),s])
            out[q['id']]=dict(optimal=v if math.isfinite(v) else None,oracle='independent_segment_graph_scipy_dijkstra',scalar_edge_audits_per_map=audit_count)
        if progress:progress(ii+1,len(groups),time.perf_counter()-t)
    return out
