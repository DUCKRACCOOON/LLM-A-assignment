"""Fresh search backends. Frozen reproduction modules are not modified.

For integer axis-aligned walls a unit 8-grid edge can intersect a wall only
at a grid endpoint. Degenerate walls retain the original GEOS semantics.
No extra 'no corner cutting' rule is introduced.
"""
import heapq
import math
import time
from collections import Counter
import numpy as np
from core import World, MOTIONS, intersects, path_length

SQRT2 = math.sqrt(2)
EPS = 1e-10


def octile(a, b):
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dx, dy)+(SQRT2-1)*min(dx, dy)


def walls(q):
    return [((a,y),(b,y)) for y,a,b in q['horizontal_barriers']] + [((x,a),(x,b)) for x,a,b in q['vertical_barriers']]


def subset(q, ids):
    nh = len(q['horizontal_barriers']); ids=set(ids)
    return dict(q, horizontal_barriers=[b for i,b in enumerate(q['horizontal_barriers']) if i in ids],
                vertical_barriers=[b for i,b in enumerate(q['vertical_barriers']) if i+nh in ids])


class Grid:
    def __init__(self, q, instrument=True):
        t=time.perf_counter(); self.q=q; self.instrument=instrument; self.counts=Counter()
        assert q['range_x'][0] == q['range_y'][0] == 0
        self.w=q['range_x'][1]; self.h=q['range_y'][1]; self.n=self.w*self.h
        self.blocked=np.zeros((self.h,self.w), dtype=np.bool_)
        self.blocked[[0,-1],:]=True; self.blocked[:,[0,-1]]=True
        self.segments=walls(q); self.degenerate=[]
        for a,b in self.segments:
            assert all(isinstance(v,int) for v in (*a,*b)), 'integer walls required'
            if a==b:
                self.degenerate.append((a,b)); continue
            x0,x1=sorted((a[0],b[0])); y0,y1=sorted((a[1],b[1]))
            x0=max(0,x0); x1=min(self.w-1,x1); y0=max(0,y0); y1=min(self.h-1,y1)
            if x0<=x1 and y0<=y1:
                self.blocked[y0:y1+1,x0:x1+1]=True
                self.counts['preprocess_cell_writes']+=(x1-x0+1)*(y1-y0+1)
        self.flat=self.blocked.ravel(); self.seconds=time.perf_counter()-t
        self.counts['preprocess_cells_allocated']=self.n

    def index(self,p): return int(p[1])*self.w+int(p[0])
    def point(self,i): return (int(i%self.w), int(i//self.w))

    def neighbors(self,s):
        x,y=self.point(s)
        if self.flat[s]: return
        for dx,dy in MOTIONS:
            xx,yy=x+dx,y+dy
            if self.instrument: self.counts['neighbor_candidates']+=1; self.counts['collision_tests']+=1
            if not(0<xx<self.w-1 and 0<yy<self.h-1): continue
            n=yy*self.w+xx
            if self.flat[n]: continue
            hit=False
            for a,b in self.degenerate:
                if self.instrument: self.counts['primitive_wall_intersections']+=1
                if intersects((x,y),(xx,yy),a,b): hit=True; break
            if not hit: yield n, SQRT2 if dx and dy else 1.

    def expand_path(self,ids): return [list(self.point(i)) for i in ids]


def search(grid,q,weight=1.,focal=False,euclidean=False,guide=None,instrument=True,deadline=None):
    t=time.perf_counter(); s=grid.index(q['start']); target=grid.index(q['goal'])
    gp=tuple(q['goal']); counts=Counter(); g={s:0.}; parent={s:s}; versions={s:0}
    opened={s}; expanded_before=set(); peak=1; sequence={s:guide.initial(s)} if guide else {}
    def heuristic(n):
        p=grid.point(n)
        return math.hypot(p[0]-gp[0],p[1]-gp[1]) if euclidean else octile(p,gp)
    def sec(n): return (*guide.key(n,sequence[n]),heuristic(n),n) if guide else (heuristic(n),n)
    hs=heuristic(s); queue=[(hs if focal else weight*hs,hs,s,0)]
    waiting=[(hs,s,0)] if focal else []; eligible=[]; result=[]; reached=False
    while opened:
        if deadline is not None and counts['expansions']%128==0 and time.perf_counter()>deadline:
            raise TimeoutError('search deadline exceeded')
        while queue and (queue[0][2] not in opened or versions[queue[0][2]]!=queue[0][3]):
            heapq.heappop(queue); counts['heap_pops']+=1; counts['stale_pops']+=1
        if not queue: break
        if focal:
            limit=weight*queue[0][0]+1e-9
            while waiting and waiting[0][0]<=limit:
                _,n,v=heapq.heappop(waiting); counts['heap_pops']+=1
                if n in opened and versions[n]==v:
                    heapq.heappush(eligible,(sec(n),n,v)); counts['heap_pushes']+=1
                else: counts['stale_pops']+=1
            while eligible:
                _,n,v=heapq.heappop(eligible); counts['heap_pops']+=1
                if n in opened and versions[n]==v: break
                counts['stale_pops']+=1
            else: raise AssertionError('FOCAL lost eligible anchor node')
            assert g[n]+heuristic(n)<=limit+1e-8
        else:
            _,_,n,_=heapq.heappop(queue); counts['heap_pops']+=1
        opened.remove(n); counts['expansions']+=1
        if n in expanded_before: counts['reopens']+=1
        expanded_before.add(n)
        if n==target: reached=True; break
        for v,cost in grid.neighbors(n):
            if instrument: counts['relax_attempts']+=1
            ng=g[n]+cost
            if ng<g.get(v,math.inf)-EPS:
                if instrument: counts['relax_successes']+=1
                g[v]=ng; parent[v]=n; versions[v]=versions.get(v,-1)+1; ver=versions[v]
                if guide: sequence[v]=guide.extend(sequence[n],v)
                hv=heuristic(v); opened.add(v)
                heapq.heappush(queue,(ng+(hv if focal else weight*hv),hv,v,ver)); counts['heap_pushes']+=1
                if focal: heapq.heappush(waiting,(ng+hv,v,ver)); counts['heap_pushes']+=1
        peak=max(peak,len(opened))
    search_seconds=time.perf_counter()-t; tr=time.perf_counter()
    if reached:
        ids=[target]
        while ids[-1]!=s:
            ids.append(parent[ids[-1]])
            assert len(ids)<=grid.n+2, 'parent cycle'
        ids.reverse(); result=grid.expand_path(ids)
    return dict(path=result,length=path_length(result) if result else None,reachable=reached,
                search_seconds=search_seconds,reconstruction_seconds=time.perf_counter()-tr,
                counters=dict(counts+grid.counts),peak_open=peak,discovered=len(g))


def independent_distance(q,all_distances=False):
    """Scalar original geometry and separate heap Dijkstra; never used as guidance."""
    world=World(q); start=tuple(q['start']); goal=tuple(q['goal'])
    dist={start:0.}; queue=[(0.,start)]
    while queue:
        d,s=heapq.heappop(queue)
        if d!=dist[s]: continue
        if s==goal and not all_distances: return d
        for dx,dy in MOTIONS:
            n=(s[0]+dx,s[1]+dy)
            if not(world.x0<n[0]<world.x1 and world.y0<n[1]<world.y1): continue
            if world.collision(s,n): continue
            dn=d+math.hypot(dx,dy)
            if dn<dist.get(n,math.inf)-EPS:
                dist[n]=dn; heapq.heappush(queue,(dn,n))
    return dist if all_distances else math.inf


def audit_path(q,r,optimal=None,bound=1.):
    from core import validate
    if not r['reachable']:
        assert not r['path']
        if optimal is not None: assert not math.isfinite(optimal), 'missed reachable goal'
        return
    p=r['path']; assert all(len(v)==2 and all(type(c)==int for c in v) for v in p)
    assert validate(q,p)[0], 'invalid geometry'
    assert all(max(abs(a[0]-b[0]),abs(a[1]-b[1]))==1 for a,b in zip(p,p[1:])), 'non-unit edge'
    assert math.isclose(path_length(p),r['length'],abs_tol=1e-8,rel_tol=1e-10)
    if optimal is not None:
        assert r['length']<=bound*optimal+1e-8+1e-10*optimal, 'bound violation'
        assert r['length']>=optimal-1e-8-1e-10*optimal, 'shorter than oracle'
