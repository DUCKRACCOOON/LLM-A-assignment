"""Differential tests, independent shortest distances and adversarial guidance."""
import argparse
import json
import math
import random
import time
from core import World,MOTIONS,load_cases
from next_search_core import Grid,search,independent_distance,audit_path
from next_geometry import Rectangles
from next_refinement import solve
from next_topology import routes,Guide,rank


def run(n=2000):
    t=time.perf_counter(); rng=random.Random(26101499); edges=0; results=0
    for i in range(n):
        q=dict(id=f'test{i}',range_x=[0,9],range_y=[0,8],horizontal_barriers=[],vertical_barriers=[],start=[1,1],goal=[7,6])
        for _ in range(i%5):
            if rng.randrange(2):
                a=rng.randrange(0,8); b=rng.randrange(a,9); q['horizontal_barriers'].append([rng.randrange(1,7),a,b])
            else:
                a=rng.randrange(0,7); b=rng.randrange(a,8); q['vertical_barriers'].append([rng.randrange(1,8),a,b])
        grid=Grid(q); points=[grid.point(j) for j in range(grid.n) if not grid.flat[j]]
        if len(points)<2: continue
        q['start'],q['goal']=map(list,rng.sample(points,2))
        if i%19==0:q['goal']=q['start'][:]
        world=World(q)
        if i<100:
            for p in points:
                actual={grid.point(v) for v,_ in grid.neighbors(grid.index(p))}
                expected=set()
                for dx,dy in MOTIONS:
                    v=(p[0]+dx,p[1]+dy)
                    if 0<v[0]<8 and 0<v[1]<7 and not world.collision(p,v):expected.add(v)
                assert actual==expected,(q,p,actual,expected)
                edges+=8
        opt=independent_distance(q)
        for variant in (None,'full','sparse'):
            g=Grid(q) if variant is None else Rectangles(q,variant=variant)
            r=search(g,q); audit_path(q,r,opt);results+=1
        if i<100:
            for mode in ('first','batch','rule','random','program','incremental'):
                r=solve(q,mode,compressed=i%2==0);audit_path(q,r,opt);results+=1
            prop,g=routes(q)
            for w in (1.,1.01,1.02):
                guide=Guide(g,prop,list(reversed(rank(prop)))) if prop['candidates'] else None
                r=search(Grid(q),q,weight=w,focal=True,guide=guide);audit_path(q,r,opt,w);results+=1
    # Dedicated disconnected and endpoint fixtures.
    for i in range(100):
        m=i//5
        q=dict(id=f'unreachable{i}',range_x=[0,14+m],range_y=[0,12],horizontal_barriers=[],vertical_barriers=[[6+m,0,11]],start=[1+i%4,1+i%9],goal=[7+m+i%3,1+(i*3)%9])
        for r in (search(Grid(q),q),search(Rectangles(q),q),solve(q,'rule',True)):
            audit_path(q,r,math.inf);results+=1
    # The prior data has edge cases absent from simple synthetic fixtures.
    for q in load_cases()[::100]:
        opt=independent_distance(q)
        for g in (Grid(q),Rectangles(q)):
            audit_path(q,search(g,q),opt);results+=1
    # Regression for underconsistent predecessors hidden by ulp differences in
    # equal octile f-values after adding a wall. These are development maps.
    from next_data import generate
    medium,_=generate('development',15);repairs=0
    for q in medium:
        reference=search(Grid(q,False),q,instrument=False)
        r=solve(q,'incremental',instrument=False)
        audit_path(q,r,reference['length']);results+=1;repairs+=len(r['rounds'])>1
    return dict(verified=True,random_queries=n,edge_differential_checks=edges,solver_checks=results,
                unreachable_maps=20,unreachable_queries=100,medium_incremental_queries=len(medium),
                medium_queries_requiring_repair=repairs,seconds=time.perf_counter()-t)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=int,default=2000);p.add_argument('--out');a=p.parse_args()
    r=run(a.cases)
    if a.out:
        from pathlib import Path
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2))
    print(json.dumps(r))
