"""Costed map and goal reuse, including an exact reverse-Dijkstra control."""
import heapq
import math
import random
import statistics
import time
from collections import defaultdict,Counter
import numpy as np
from core import path_length
from next_search_core import Grid,search,audit_path
from next_geometry import Rectangles
from next_refinement import solve as refine


class FieldGuide:
    def __init__(self,field):self.field=field
    def initial(self,s):return ()
    def extend(self,p,n):return ()
    def key(self,n,p):return (float(self.field[n]),)


def goal_field(grid,goal,fmm=False):
    n=grid.n;s=grid.index(goal);dist=np.full(n,np.inf);dist[s]=0.;queue=[(0.,s)];parent={s:s};pops=0
    if not fmm:
        while queue:
            d,s=heapq.heappop(queue);pops+=1
            if d!=dist[s]:continue
            for v,c in grid.neighbors(s):
                z=d+c
                if z<dist[v]-1e-10:dist[v]=z;parent[v]=s;heapq.heappush(queue,(z,v))
    else:
        accepted=np.zeros(n,dtype=bool)
        def cardinal(s):
            x,y=grid.point(s)
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                xx,yy=x+dx,y+dy
                if 0<xx<grid.w-1 and 0<yy<grid.h-1 and not grid.blocked[yy,xx]:yield yy*grid.w+xx,dx==0
        def update(s):
            ax=[];ay=[]
            for v,vert in cardinal(s):
                if accepted[v]:(ay if vert else ax).append(dist[v])
            a,b=sorted((min(ax,default=math.inf),min(ay,default=math.inf)))
            if not math.isfinite(a):return math.inf
            return a+1 if b-a>=1 else .5*(a+b+math.sqrt(max(0.,2-(a-b)**2)))
        while queue:
            d,s=heapq.heappop(queue);pops+=1
            if accepted[s] or d!=dist[s]:continue
            accepted[s]=True
            for v,_ in cardinal(s):
                if accepted[v]:continue
                z=update(v)
                if z<dist[v]:dist[v]=z;heapq.heappush(queue,(z,v))
    return dist,parent,pops


def one_batch(qs,method,selection):
    t=time.perf_counter();g=None;fields={};out=[];preparation=0.;field_seconds=0.;field_pops=0
    if method in ('A_OCT','R1','FOCAL_102','FMM_UNIFORM','REVERSE_DIJKSTRA'):
        g=Rectangles(qs[0],False,selection['r1']) if method=='R1' else Grid(qs[0],False)
        preparation=g.seconds
    for q in qs:
        if method in ('R2','R1_R2'):
            mode=selection.get('combination_r2',selection['r2']) if method=='R1_R2' else selection['r2']
            r=refine(q,mode,method=='R1_R2',False,selection.get('program'))
        elif method in ('FMM_UNIFORM','REVERSE_DIJKSTRA'):
            key=tuple(q['goal'])
            if key not in fields:
                tt=time.perf_counter();fields[key]=goal_field(g,q['goal'],method=='FMM_UNIFORM');field_seconds+=time.perf_counter()-tt
                field_pops+=fields[key][2]
            d,parent,_=fields[key]
            if method=='FMM_UNIFORM':r=search(g,q,1.02,True,guide=FieldGuide(d),instrument=False)
            else:
                node=g.index(q['start']);goal=g.index(q['goal']);ids=[node]
                while ids[-1]!=goal:ids.append(parent[ids[-1]])
                p=g.expand_path(ids);r=dict(path=p,length=path_length(p),reachable=True,counters={'expansions':0})
        else:
            if method=='R1':g.set_query(q)
            r=search(g,q,1.02 if method=='FOCAL_102' else 1.,method=='FOCAL_102',instrument=False)
        out.append(r)
    return dict(total_seconds=time.perf_counter()-t,map_preparation_seconds=preparation,
                goal_preparation_seconds=field_seconds,field_heap_pops=field_pops,distinct_goals=len(fields),results=out)


def run(out,varying,fixed,selection):
    from next_run import save,read,get_oracles,status,time_limit
    methods=['A_OCT','R1','R2','R1_R2','FOCAL_102','FMM_UNIFORM','REVERSE_DIJKSTRA'];rng=random.Random(26101424)
    for workload,qs in [('varying',varying),('fixed',fixed)]:
        oracle=get_oracles('reuse_'+workload,qs);maps=defaultdict(list)
        for q in qs:maps[q['map_id']].append(q)
        for ii,(m,allq) in enumerate(sorted(maps.items())):
            for num in (1,5,10):
                selected=sorted(allq,key=lambda q:q['pair_id'])[:num];order=methods[:];rng.shuffle(order)
                pending={}
                for method in order:
                    p=out/'reuse'/workload/m/f'{method}_{num}.json'
                    if not p.exists():pending[method]=(p,[])
                for repeat in range(5):
                    order=list(pending);rng.shuffle(order)
                    for method in order:
                        p,values=pending[method]
                        with time_limit(60*num):r=one_batch(selected,method,selection)
                        for q,z in zip(selected,r.pop('results')):audit_path(q,z,oracle[q['id']]['optimal'],1.02 if method in ('FOCAL_102','FMM_UNIFORM') else 1.)
                        values.append(r)
                for method,(p,values) in pending.items():
                    save(p,dict(map_id=m,family=selected[0]['family'],workload=workload,queries=num,method=method,timings=values,
                           total_seconds=statistics.median(v['total_seconds'] for v in values),mean_query_seconds=statistics.median(v['total_seconds'] for v in values)/num))
            status('reuse_'+workload,maps_done=ii+1,maps_total=len(maps))
