"""Certified empty rectangles and exact perimeter macro edges."""
import time
import numpy as np
from .next_search_core import Grid,octile


class Rectangles(Grid):
    def __init__(self,q,instrument=True,variant='sparse'):
        t=time.perf_counter(); super().__init__(q,instrument); self.variant=variant
        self.labels=np.full((self.h,self.w),-1,dtype=np.int32); self.rects=[]; active={}
        # Equal horizontal free runs on adjacent rows form disjoint rectangles.
        for y in range(1,self.h-1):
            row=~self.blocked[y]; starts=np.flatnonzero(row & ~np.roll(row,1)); ends=np.flatnonzero(row & ~np.roll(row,-1))
            now={}
            for x0,x1 in zip(starts.tolist(),ends.tolist()):
                key=(x0,x1)
                if key in active:
                    rid=active[key]; self.rects[rid][3]=y
                else:
                    rid=len(self.rects); self.rects.append([x0,x1,y,y])
                self.labels[y,x0:x1+1]=rid; now[key]=rid
            active=now
        self.border=np.zeros_like(self.blocked)
        for x0,x1,y0,y1 in self.rects:
            self.border[y0,x0:x1+1]=True; self.border[y1,x0:x1+1]=True
            self.border[y0:y1+1,x0]=True; self.border[y0:y1+1,x1]=True
        self.label_flat=self.labels.ravel(); self.border_flat=self.border.ravel()
        self.inserted={}; self.set_query(q)
        self.counts['rectangles']=len(self.rects); self.counts['retained_vertices']=int(self.border.sum())
        self.disabled=bool(self.degenerate)
        self.seconds=time.perf_counter()-t

    def set_query(self,q):
        self.q=q; self.inserted={}
        for p in (q['start'],q['goal']):
            s=self.index(p); r=int(self.label_flat[s])
            if r>=0 and not self.border_flat[s]: self.inserted.setdefault(r,[]).append(s)

    def perimeter(self,r):
        x0,x1,y0,y1=self.rects[r]
        for x in range(x0,x1+1):
            yield y0*self.w+x
            if y1!=y0: yield y1*self.w+x
        for y in range(y0+1,y1):
            yield y*self.w+x0
            if x1!=x0: yield y*self.w+x1

    def neighbors(self,s):
        if self.disabled:
            yield from super().neighbors(s); return
        r=int(self.label_flat[s])
        if r<0: return
        x,y=self.point(s); x0,x1,y0,y1=self.rects[r]; dest=set()
        for n,c in super().neighbors(s):
            if self.border_flat[n]: dest.add(n)
        if not self.border_flat[s] or self.variant=='full':
            dest.update(self.perimeter(r))
        else:
            # RSR opposite-side fans. Adjacent-side diagonal links below.
            if x==x0 or x==x1:
                xx=x1 if x==x0 else x0; d=abs(xx-x)
                for yy in range(max(y0,y-d),min(y1,y+d)+1): dest.add(yy*self.w+xx)
            if y==y0 or y==y1:
                yy=y1 if y==y0 else y0; d=abs(yy-y)
                for xx in range(max(x0,x-d),min(x1,x+d)+1): dest.add(yy*self.w+xx)
            for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
                d=min(x1-x if dx>0 else x-x0,y1-y if dy>0 else y-y0)
                if d>0: dest.add((y+dy*d)*self.w+x+dx*d)
        dest.update(self.inserted.get(r,[])); dest.discard(s)
        for n in sorted(dest):
            if self.instrument:
                self.counts['macro_edge_checks']+=1; self.counts['fan_scans']+=1
            yield n,octile((x,y),self.point(n))

    def expand_path(self,ids):
        if self.disabled: return super().expand_path(ids)
        out=[list(self.point(ids[0]))]
        for node in ids[1:]:
            bx,by=self.point(node); x,y=out[-1]
            while (x,y)!=(bx,by):
                x+=(bx>x)-(bx<x); y+=(by>y)-(by<y); out.append([x,y])
        return out
