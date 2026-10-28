"""Four-candidate study: independent, bounded grid-search implementations.

CHOMP-inspired covariant descent and a small TransPath-inspired network are
adaptations, not reproductions of either paper's numerical results.
"""
import heapq
import math
import time
import numpy as np
from core import World, MOTIONS, validate, path_length


def octile(points, goal):
    d = np.abs(np.asarray(points) - np.asarray(goal))
    return d.max(axis=-1) + (math.sqrt(2)-1)*d.min(axis=-1)


class Grid:
    """Exact segment-obstacle graph, including wall endpoints and diagonals."""
    def __init__(self, q):
        t = time.perf_counter()
        self.q = q
        self.world = World(q)
        self.w, self.h = self.world.x1+1, self.world.y1+1
        yy, xx = np.mgrid[:self.h, :self.w]
        self.xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        self.n = len(self.xy)
        self.edges = np.full((self.n, 8), -1, dtype=np.int32)
        self.costs = np.array([math.hypot(*m) for m in MOTIONS])
        a = self.xy
        for k, motion in enumerate(MOTIONS):
            b = a + motion
            good = ((a[:,0]>0)&(a[:,0]<self.w-1)&(a[:,1]>0)&(a[:,1]<self.h-1)&
                    (b[:,0]>0)&(b[:,0]<self.w-1)&(b[:,1]>0)&(b[:,1]<self.h-1))
            for c0, d0 in self.world.barriers:
                c, d = np.array(c0), np.array(d0)
                if np.array_equal(c, d):
                    # Preserve the original GEOS degenerate-line behavior.
                    ids = np.flatnonzero(good)
                    for i in ids:
                        if self.world.collision(tuple(a[i]), tuple(b[i])):
                            good[i] = False
                    continue
                def cross(u,v): return u[...,0]*v[...,1]-u[...,1]*v[...,0]
                u,v = cross(b-a,c-a),cross(b-a,d-a)
                z,r = cross(d-c,a-c),cross(d-c,b-c)
                bbox = ((np.maximum(a[:,0],b[:,0])>=min(c[0],d[0])) &
                        (np.minimum(a[:,0],b[:,0])<=max(c[0],d[0])) &
                        (np.maximum(a[:,1],b[:,1])>=min(c[1],d[1])) &
                        (np.minimum(a[:,1],b[:,1])<=max(c[1],d[1])))
                good &= ~((u*v<=0)&(z*r<=0)&bbox)
            self.edges[good,k] = b[good,1]*self.w+b[good,0]
        self.free = np.any(self.edges>=0,axis=1)
        self.seconds = time.perf_counter()-t

    def index(self, point): return int(point[1])*self.w+int(point[0])

    def distances(self, point):
        start = self.index(point)
        dist = np.full(self.n,np.inf); dist[start]=0
        queue=[(0.,start)]; parent=np.full(self.n,-1,dtype=np.int32)
        pops=0
        while queue:
            value,s=heapq.heappop(queue)
            if value!=dist[s]: continue
            pops+=1
            for k,n in enumerate(self.edges[s]):
                if n<0: continue
                value2=value+self.costs[k]
                if value2<dist[n]-1e-10:
                    dist[n]=value2; parent[n]=s; heapq.heappush(queue,(value2,int(n)))
        return dist,parent,pops


def reconstruct(grid, parent, start, goal):
    s,g=grid.index(start),grid.index(goal)
    if s!=g and parent[g]<0: return []
    out=[g]
    while out[-1]!=s:
        out.append(int(parent[out[-1]]))
        if len(out)>grid.n: raise AssertionError('parent cycle')
    return grid.xy[out[::-1]].tolist()


def grid_valid(q,path):
    valid,reason=validate(q,path)
    if valid and any(max(abs(a[0]-b[0]),abs(a[1]-b[1]))!=1 for a,b in zip(path,path[1:])):
        return False,'non_grid_move'
    return valid,reason


def focal(grid,q,secondary=None,weight=1.1,mode='focal',incumbent=None):
    """Admissible octile anchor, full OPEN membership, reopen on improvement.

    Goal may be returned only when selected from FOCAL or a verified incumbent
    is <= weight * minimum anchor f. Secondary priorities cannot remove nodes.
    """
    t=time.perf_counter(); start,goal=grid.index(q['start']),grid.index(q['goal'])
    h=octile(grid.xy,q['goal'])
    secondary=h if secondary is None else np.nan_to_num(secondary,nan=1e9,posinf=1e9,neginf=-1e9)
    g=np.full(grid.n,np.inf); g[start]=0
    parent=np.full(grid.n,-1,dtype=np.int32)
    opened={start}; queue=[(float(h[start]),start,0.)]
    expanded=0; heap_pops=0; peak=1; reached=False; used_incumbent=False
    upper=path_length(incumbent) if incumbent else math.inf
    if incumbent: assert grid_valid(q,incumbent)[0]
    while opened:
        while queue and (queue[0][1] not in opened or queue[0][2]!=g[queue[0][1]]):
            heapq.heappop(queue); heap_pops+=1
        assert queue
        lower=queue[0][0]
        if upper<=weight*lower+1e-9:
            used_incumbent=True; reached=True; break
        ids=np.fromiter(opened,dtype=np.int32)
        if mode=='weighted':
            keys=g[ids]+weight*h[ids]
            s=int(ids[np.lexsort((ids,h[ids],keys))[0]])
        elif mode=='astar':
            s=int(queue[0][1])
        else:
            eligible=ids[(g[ids]+h[ids])<=weight*lower+1e-9]
            s=int(eligible[np.lexsort((eligible,h[eligible],secondary[eligible]))[0]])
        opened.remove(s); expanded+=1
        if s==goal: reached=True; break
        for k,n0 in enumerate(grid.edges[s]):
            n=int(n0)
            if n<0: continue
            ng=g[s]+grid.costs[k]
            if ng<g[n]-1e-10 and ng+h[n]<upper-1e-10:
                g[n]=ng; parent[n]=s; opened.add(n)
                heapq.heappush(queue,(float(ng+h[n]),n,float(ng)))
        peak=max(peak,len(opened))
    path=incumbent if used_incumbent else reconstruct(grid,parent,q['start'],q['goal']) if reached else []
    elapsed=time.perf_counter()-t
    valid,reason=grid_valid(q,path)
    return dict(path=path,valid=valid,failure=reason,length=path_length(path) if path else None,
                expansions=expanded,anchor_heap_pops=heap_pops,discovered=int(np.isfinite(g).sum()),
                peak_open=peak,search_seconds=elapsed,incumbent_returned=used_incumbent)


def normalize_proposal(q,path):
    # Deliberately no collision repair or use of ground truth here.
    p=[list(map(float,x)) for x in path]
    if not p or p[0]!=q['start']: p.insert(0,list(q['start']))
    if p[-1]!=q['goal']: p.append(list(q['goal']))
    p=[x for i,x in enumerate(p) if i==0 or x!=p[i-1]]
    return p


def point_segment_distance(points,path):
    p=np.asarray(points,dtype=float); line=np.asarray(path,dtype=float)
    result=np.full(len(p),np.inf)
    if len(line)<2: return np.zeros(len(p))
    for a,b in zip(line,line[1:]):
        v=b-a; norm=np.dot(v,v)
        proj=a+np.clip(((p-a)@v)/max(norm,1e-12),0,1)[:,None]*v
        result=np.minimum(result,np.linalg.norm(p-proj,axis=1))
    return result


def llm_field(grid,q,paths):
    fields=np.array([point_segment_distance(grid.xy,normalize_proposal(q,p)) for p in paths])
    # Dispersion is computed from proposals only, without labels.
    uncertainty=1.+float(np.mean(np.std(fields[:,grid.free],axis=0)))
    return fields.min(axis=0),uncertainty


def conformal_quantile(scores,alpha):
    values=sorted(float(v) for v in scores)
    k=math.ceil((len(values)+1)*(1-alpha))
    return math.inf if k>len(values) else values[k-1]


def fast_marching(grid,goal,speed=None):
    """First-order isotropic Eikonal fast marching, not graph Dijkstra.

    Accepted cardinal neighbours use the quadratic upwind update. Exact blocked
    edges isolate both sides of zero-width walls. Values are secondary only.
    """
    t=time.perf_counter(); speed=np.ones(grid.n) if speed is None else speed
    arrival=np.full(grid.n,np.inf); accepted=np.zeros(grid.n,dtype=bool)
    s=grid.index(goal); arrival[s]=0; queue=[(0.,s)]; updates=0; pops=0
    def update(n):
        x=[]; y=[]
        for k in (0,2,4,6):
            nb=int(grid.edges[n,k])
            if nb>=0 and accepted[nb]: (x if k in (0,4) else y).append(arrival[nb])
        a=min(x,default=math.inf); b=min(y,default=math.inf)
        a,b=sorted((a,b)); step=1./speed[n]
        if not math.isfinite(a): return math.inf
        if not math.isfinite(b) or b-a>=step: return a+step
        return .5*(a+b+math.sqrt(max(0.,2*step*step-(a-b)**2)))
    while queue:
        value,s=heapq.heappop(queue); pops+=1
        if accepted[s] or value!=arrival[s]: continue
        accepted[s]=True
        for k in (0,2,4,6):
            n=int(grid.edges[s,k])
            if n<0 or accepted[n]: continue
            val=update(n); updates+=1
            if val<arrival[n]: arrival[n]=val; heapq.heappush(queue,(val,n))
    return arrival,dict(field_seconds=time.perf_counter()-t,field_updates=updates,
                        field_heap_pops=pops,field_accepted=int(accepted.sum()))


def rasterize(path):
    result=[]
    for a,b in zip(path,path[1:]):
        a=np.rint(a).astype(int); b=np.rint(b).astype(int)
        count=int(np.abs(b-a).max())
        for x in np.rint(np.linspace(a,b,count+1)).astype(int).tolist():
            if not result or x!=result[-1]: result.append(x)
    return result


def chomp_repair(q,path,iterations=160,points=48):
    """Covariant preconditioned elastic-band descent with obstacle gradients.

    Analytic segment distance penalty, smoothness metric preconditioning,
    fixed endpoints, no homotopy search and no hidden A* bridging.
    """
    t=time.perf_counter(); world=World(q); original=normalize_proposal(q,path)
    line=np.array(original,float); lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    if lengths[-1]==0: return [],dict(repaired=False,iterations=0,repair_seconds=time.perf_counter()-t)
    pos=np.column_stack([np.interp(np.linspace(0,lengths[-1],points),lengths,line[:,i]) for i in (0,1)])
    n=points-2
    metric=2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
    precondition=np.linalg.inv(metric+.1*np.eye(n))
    barriers=np.asarray(world.barriers,float); a,b=barriers[:,0],barriers[:,1]; v=b-a
    vnorm=(v*v).sum(axis=1)
    def cost_grad(p):
        delta=p[1:]-p[:-1]
        cost=.03*(delta*delta).sum()
        grad=np.zeros_like(p); grad[:-1]-=.06*delta; grad[1:]+=.06*delta
        # Obstacle gradients sampled at nodes and segment midpoints.
        samples=np.concatenate([p,(p[:-1]+p[1:])*.5])
        offset=samples[:,None,:]-a[None,:,:]
        proj=a[None,:,:]+np.clip((offset*v).sum(axis=2)/np.maximum(vnorm,1e-12),0,1)[...,None]*v
        diff=samples[:,None,:]-proj; distances=np.linalg.norm(diff,axis=2)
        nearest=distances.argmin(axis=1); ix=np.arange(len(samples)); d=distances[ix,nearest]
        vector=diff[ix,nearest]
        zero=d<1e-8
        if zero.any():
            vv=v[nearest[zero]]; normals=np.stack([-vv[:,1],vv[:,0]],axis=1)
            normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-9)
            vector[zero]=normals*1e-6; d[zero]=1e-6
        penalty=np.maximum(1.15-d,0)
        cost+=4.*(penalty**2).sum()
        sg=-8*penalty[:,None]*vector/np.maximum(d[:,None],1e-9)
        grad+=sg[:points]; grad[:-1]+=.5*sg[points:]; grad[1:]+=.5*sg[points:]
        return float(cost),grad
    best=None; best_length=math.inf; used=0
    for it in range(iterations+1):
        used=it
        if it%8==0 or it==iterations:
            discrete=rasterize(pos)
            if grid_valid(q,discrete)[0]:
                length=path_length(discrete)
                if length<best_length: best,best_length=discrete,length
        if it==iterations: break
        cost,grad=cost_grad(pos); direction=precondition@grad[1:-1]
        maxnorm=np.linalg.norm(direction,axis=1).max()
        direction/=max(1.,maxnorm/2.)
        accepted=False
        for step in (.3,.1,.03,.01):
            trial=pos.copy(); trial[1:-1]-=step*direction
            trial[1:-1,0]=np.clip(trial[1:-1,0],1,world.x1-1)
            trial[1:-1,1]=np.clip(trial[1:-1,1],1,world.y1-1)
            if cost_grad(trial)[0]<cost-1e-9: pos=trial; accepted=True; break
        if not accepted: break
    continuous_valid=validate(q,pos.tolist())[0]
    return best or [],dict(repaired=best is not None,iterations=used,
                           input_valid=validate(q,original)[0],continuous_valid=continuous_valid,
                           grid_valid=best is not None,repair_seconds=time.perf_counter()-t)


def network_inputs(grid,q):
    xy=grid.xy.astype(np.float32); scale=max(grid.w,grid.h)
    start=np.zeros(grid.n,dtype=np.float32); start[grid.index(q['start'])]=1
    goal=np.zeros(grid.n,dtype=np.float32); goal[grid.index(q['goal'])]=1
    return np.stack([(~grid.free).astype(np.float32),start,goal,
                     xy[:,0]/scale,xy[:,1]/scale,
                     octile(xy,q['start'])/scale,octile(xy,q['goal'])/scale]).reshape(7,grid.h,grid.w).astype(np.float32)


def make_network():
    import torch
    from torch import nn
    import torch.nn.functional as F
    class PathTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1=nn.Sequential(nn.Conv2d(7,24,3,padding=1),nn.GELU(),nn.Conv2d(24,24,3,padding=1),nn.GELU())
            self.enc2=nn.Sequential(nn.Conv2d(24,48,3,stride=2,padding=1),nn.GELU())
            self.enc3=nn.Sequential(nn.Conv2d(48,64,3,stride=2,padding=1),nn.GELU())
            self.attn=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True),2,enable_nested_tensor=False)
            self.dec=nn.Sequential(nn.Conv2d(64+24,48,3,padding=1),nn.GELU(),nn.Conv2d(48,24,3,padding=1),nn.GELU(),nn.Conv2d(24,1,1))
        def forward(self,x):
            skip=self.enc1(x); z=self.enc3(self.enc2(skip)); shape=z.shape
            z=self.attn(z.flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
            z=F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            return self.dec(torch.cat([z,skip],dim=1))[:,0]
    return PathTransformer()
