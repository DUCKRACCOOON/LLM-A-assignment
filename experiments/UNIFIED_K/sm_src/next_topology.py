"""Finite rectangle-portal routes; guidance only, not a homotopy certificate."""
import heapq
import math
import random
import time
from collections import defaultdict
from next_geometry import Rectangles
from next_search_core import octile


def routes(q,k=3,instrument=True):
    t=time.perf_counter(); grid=Rectangles(q,instrument); adjacency=defaultdict(set); contacts=defaultdict(list)
    # Construct interfaces from original valid neighbor edges. All work is charged.
    for y in range(1,grid.h-1):
        for x in range(1,grid.w-1):
            s=y*grid.w+x; a=int(grid.label_flat[s])
            if a<0 or not grid.border_flat[s]: continue
            for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
                xx,yy=x+dx,y+dy
                if not(0<xx<grid.w-1 and 0<yy<grid.h-1): continue
                n=yy*grid.w+xx; b=int(grid.label_flat[n])
                if b<0 or a==b: continue
                adjacency[a].add(b); adjacency[b].add(a)
                contacts[tuple(sorted((a,b)))].append(((x+xx)/2,(y+yy)/2))
    portals={key:tuple(sorted(points)[len(points)//2]) for key,points in contacts.items()}
    sr=int(grid.label_flat[grid.index(q['start'])]); gr=int(grid.label_flat[grid.index(q['goal'])])
    def shortest(banned=frozenset()):
        queue=[(0,sr,(sr,))]; seen=set()
        while queue:
            d,s,path=heapq.heappop(queue)
            if s in seen: continue
            seen.add(s)
            if s==gr: return list(path)
            for n in sorted(adjacency[s]):
                if tuple(sorted((s,n))) in banned or n in seen: continue
                heapq.heappush(queue,(d+1,n,path+(n,)))
        return None
    first=shortest(); choices=[]
    if first:
        choices=[first]
        # Bounded alternative generation: remove interfaces, not grid cells.
        edges=list(zip(first,first[1:])); stride=max(1,len(edges)//12)
        for a,b in edges[::stride][:12]:
            p=shortest(frozenset([tuple(sorted((a,b)))]))
            if p and p not in choices: choices.append(p)
    candidates=[]
    for p in choices:
        pts=[list(q['start'])]+[list(portals[tuple(sorted((a,b)))]) for a,b in zip(p,p[1:])]+[list(q['goal'])]
        approx=sum(math.dist(a,b) for a,b in zip(pts,pts[1:]))
        candidates.append(dict(regions=p,points=pts,estimated_length=approx,portal_count=len(p)-1))
    candidates.sort(key=lambda c:(c['estimated_length'],c['portal_count'],c['regions']))
    candidates=candidates[:k]
    for i,c in enumerate(candidates): c['id']=i
    result=dict(candidates=candidates,preparation_seconds=time.perf_counter()-t,
                rectangle_count=len(grid.rects),portal_edges=len(portals),grid_cells_scanned=grid.n,
                generator='bounded_rectangle_portal_routes_not_exact_homotopy_classes')
    return result,grid


class Guide:
    def __init__(self,graph,proposal,ranking):
        self.graph=graph; self.candidates=proposal['candidates']; self.ranking=ranking
    def initial(self,s): return (int(self.graph.label_flat[s]),)
    def extend(self,path,n):
        r=int(self.graph.label_flat[n]); return path if path[-1]==r else path+(r,)
    def key(self,n,path):
        p=self.graph.point(n); best=(1,999,1e9)
        for rank,i in enumerate(self.ranking):
            c=self.candidates[i]; rs=c['regions']; common=0
            for a,b in zip(path,rs):
                if a!=b: break
                common+=1
            matched=common==len(path)
            target=c['points'][min(common,len(c['points'])-1)]
            key=(0 if matched else 1,rank,octile(p,target))
            best=min(best,key)
        return best


def rank(proposal,mode='rule',seed=0):
    ids=list(range(len(proposal['candidates'])))
    if mode=='random': random.Random(seed).shuffle(ids)
    return ids


def prompt(q,proposal):
    import json
    public={k:q[k] for k in ('range_x','range_y','start','goal','horizontal_barriers','vertical_barriers')}
    choices=[{k:c[k] for k in ('id','points','estimated_length','portal_count')} for c in proposal['candidates']]
    return ('Rank all candidate route IDs for efficient 8-neighbor grid search. Walls are line segments including endpoints. '
            'Consider detours, narrow passages and misleading branches. Candidate routes are coarse suggestions, not certified optimal paths. '
            'Return only JSON {"ranking":[all candidate IDs, best first]}. Do not output coordinates or explanations.\n'
            +json.dumps(dict(problem=public,candidates=choices),separators=(',',':')))
