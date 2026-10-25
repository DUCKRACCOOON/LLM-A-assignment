"""Lifelong Planning A* for monotone wall insertion on the original grid."""
import heapq
import math
import time
from collections import Counter
import numpy as np
from core import path_length
from next_search_core import octile


class LPA:
    def __init__(self,grid,q):
        self.grid=grid;self.q=q;self.s=grid.index(q['start']);self.t=grid.index(q['goal'])
        self.g={};self.rhs={self.s:0.};self.queue=[];self.version={};self.counts=Counter();self.peak=0
        self.push(self.s)
    def key(self,s):
        v=min(self.g.get(s,math.inf),self.rhs.get(s,math.inf))
        return v+octile(self.grid.point(s),self.q['goal']),v,s
    def push(self,s):
        v=self.version.get(s,0)+1;self.version[s]=v
        if self.g.get(s,math.inf)!=self.rhs.get(s,math.inf):
            heapq.heappush(self.queue,(self.key(s),s,v));self.counts['heap_pushes']+=1
        self.peak=max(self.peak,len(self.queue))
    def update(self,s):
        if s!=self.s:
            self.rhs[s]=min((self.g.get(p,math.inf)+c for p,c in self.grid.neighbors(s)),default=math.inf)
        self.push(s)
    def change(self,grid):
        assert not grid.degenerate
        changed=np.flatnonzero(grid.flat!=self.grid.flat);self.grid=grid;affected=set(map(int,changed))
        for s in changed:
            x,y=grid.point(s)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    if 0<x+dx<grid.w-1 and 0<y+dy<grid.h-1:affected.add((y+dy)*grid.w+x+dx)
        for s in sorted(affected):self.update(s)
        self.counts['incremental_affected_vertices']+=len(affected)
    def run(self,deadline=None):
        t=time.perf_counter()
        while self.queue:
            if deadline and self.counts['expansions']%128==0 and time.perf_counter()>deadline:raise TimeoutError('LPA deadline')
            key,s,v=self.queue[0]
            if self.version.get(s)!=v:
                heapq.heappop(self.queue);self.counts['heap_pops']+=1;self.counts['stale_pops']+=1;continue
            # Numerically equal octile f-values can differ by a few ulps after
            # different addition orders. Drain the entire goal-f plateau before
            # stopping, so an underconsistent predecessor cannot remain behind
            # the goal because its f rounded infinitesimally upward.
            if key[0]>self.key(self.t)[0]+1e-8 and self.rhs.get(self.t,math.inf)==self.g.get(self.t,math.inf):break
            heapq.heappop(self.queue);self.counts['heap_pops']+=1;self.counts['expansions']+=1
            if self.g.get(s,math.inf)>self.rhs.get(s,math.inf):
                self.g[s]=self.rhs[s]
                for n,_ in self.grid.neighbors(s):self.update(n)
            else:
                self.g[s]=math.inf;self.update(s)
                for n,_ in self.grid.neighbors(s):self.update(n)
        elapsed=time.perf_counter()-t;t=time.perf_counter();path=[];reached=math.isfinite(self.g.get(self.t,math.inf))
        if reached:
            ids=[self.t]
            while ids[-1]!=self.s:
                n=ids[-1];p,c=min(self.grid.neighbors(n),key=lambda pc:(self.g.get(pc[0],math.inf)+pc[1],pc[0]))
                assert self.g.get(p,math.inf)<self.g[n]-1e-10
                assert abs(self.g[p]+c-self.g[n])<1e-7
                ids.append(p)
            path=self.grid.expand_path(ids[::-1])
        counts=self.counts+self.grid.counts;self.counts=Counter();self.grid.counts=Counter()
        return dict(path=path,length=path_length(path) if path else None,reachable=reached,search_seconds=elapsed,
                    reconstruction_seconds=time.perf_counter()-t,counters=dict(counts),peak_open=self.peak,discovered=len(self.g))
