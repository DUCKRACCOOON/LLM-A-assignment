"""Headless port of the author's search semantics (MIT; see vendor LICENSE).

Preserves duplicate heap pops, len(g) storage, target transitions before collision
checks, and the author's waypoint filtering. No graphics or network dependency.
"""
import ast
import copy
import heapq
import json
import math
import re
import runpy
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'llm-astar'
PROMPTS = runpy.run_path(str(VENDOR / 'llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']
MOTIONS = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

def load_cases():
    maps = json.loads((VENDOR / 'dataset/environment_50_30.json').read_text())
    cases = []
    for m in maps:
        for j, sg in enumerate(m['start_goal']):
            q = {k: copy.deepcopy(m[k]) for k in ('range_x','range_y','horizontal_barriers','vertical_barriers')}
            q.update(id=f'm{m["id"]:03d}-p{j:02d}', map_id=m['id'], pair_id=j, start=sg[0], goal=sg[1])
            cases.append(q)
    return cases

def prompt_for(q, kind):
    return PROMPTS[kind].format(**q)

def parse_path(text):
    lists = re.findall(r'\[\[.*?\]\]', text, re.DOTALL)
    if not lists:
        return [], 'missing_list'
    try:
        value = ast.literal_eval(lists[-1])
        if not isinstance(value, list) or len(value) < 2:
            return [], 'too_few_points'
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 and
                   all(type(v) in (int, float) and math.isfinite(v) for v in p) for p in value):
            return [], 'invalid_coordinates'
        return [tuple(p) for p in value], None
    except (ValueError, SyntaxError, TypeError):
        return [], 'parse_error'

def intersects(a,b,c,d):
    # GEOS treats zero-length LineStrings differently from points. Preserve the
    # upstream geometry library's behavior for these dataset edge cases.
    if a == b or c == d:
        from shapely.geometry import LineString
        return bool(LineString([a,b]).intersects(LineString([c,d])))
    def cross(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def within(p,q,r):
        return min(p[0],q[0]) <= r[0] <= max(p[0],q[0]) and min(p[1],q[1]) <= r[1] <= max(p[1],q[1])
    u,v,w,z = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    return ((u*v < 0 and w*z < 0) or
            (u == 0 and within(a,b,c)) or (v == 0 and within(a,b,d)) or
            (w == 0 and within(c,d,a)) or (z == 0 and within(c,d,b)))

class World:
    def __init__(self, q):
        self.x0, self.x1 = q['range_x'][0], q['range_x'][1]-1
        self.y0, self.y1 = q['range_y'][0], q['range_y'][1]-1
        self.h = q['horizontal_barriers']; self.v = q['vertical_barriers']
        self.barriers = [((h[1],h[0]),(h[2],h[0])) for h in self.h]
        self.barriers += [((v[0],v[1]),(v[0],v[2])) for v in self.v]
        self.barriers += [((x,self.y0),(x,self.y1)) for x in (self.x0,self.x1)]
        self.barriers += [((self.x0,y),(self.x1,y)) for y in (self.y0,self.y1)]
        self.obs = {(x,y) for x in range(self.x1+1) for y in (0,self.y1)}
        self.obs |= {(x,y) for y in range(self.y1+1) for x in (0,self.x1)}
        self.obs |= {(x,h[0]) for h in self.h for x in range(h[1],h[2])}
        self.obs |= {(v[0],y) for v in self.v for y in range(v[1],v[2])}
        self.edge_cache = {}

    def collision(self,a,b):
        return any(intersects(a,b,c,d) for c,d in self.barriers)

    def edge_cost(self,a,b):
        key = (a,b) if a < b else (b,a)
        if key not in self.edge_cache:
            self.edge_cache[key] = math.inf if self.collision(a,b) else math.hypot(b[0]-a[0],b[1]-a[1])
        return self.edge_cache[key]

    def targets(self, raw, start, goal):
        nodes = [p for p in raw if p not in self.obs and
                 self.x0+1 < p[0] < self.x1-1 and self.y0+1 < p[1] < self.y1-1]
        if not nodes or nodes[0] != start: nodes.insert(0,start)
        if not nodes or nodes[-1] != goal: nodes.append(goal)
        return nodes

def validate(q, path, world=None):
    world = world or World(q)
    if not path: return False, 'empty_path'
    if tuple(path[0]) != tuple(q['start']) or tuple(path[-1]) != tuple(q['goal']):
        return False, 'wrong_endpoints'
    if any(not (world.x0 < p[0] < world.x1 and world.y0 < p[1] < world.y1) for p in path):
        return False, 'out_of_bounds'
    if any(world.collision(tuple(a),tuple(b)) for a,b in zip(path,path[1:])):
        return False, 'collision'
    return True, None

def path_length(path):
    return sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:]))

def search(q, method='astar', waypoints=None, world=None, keep_visited=False):
    world = world or World(q)
    start, goal = tuple(q['start']), tuple(q['goal'])
    target_list = world.targets(waypoints or [],start,goal) if method == 'llm_astar' else []
    ti = 1; target = target_list[1] if target_list else goal
    g = {start:0.0}; parent = {start:start}; closed=set(); popped=[]; operations=0
    def priority(s):
        h = math.hypot(goal[0]-s[0],goal[1]-s[1])
        if method == 'dijkstra': return g[s]
        if method == 'llm_astar': return g[s]+h+math.hypot(target[0]-s[0],target[1]-s[1])
        if method == 'dynamic_weighted': return g[s]+max(1.0, 2.0*0.99**operations)*h
        return g[s]+h
    queue=[(priority(start),start)]
    t0=time.perf_counter(); reached=False
    while queue:
        _,s=heapq.heappop(queue); operations+=1; closed.add(s)
        if keep_visited: popped.append(s)
        if s==goal: reached=True; break
        for dx,dy in MOTIONS:
            n=(s[0]+dx,s[1]+dy)
            if method=='llm_astar' and n==target and target!=goal:
                ti+=1
                if ti < len(target_list): target=target_list[ti]
                rebuilt=[]
                for _,old in queue: heapq.heappush(rebuilt,(priority(old),old))
                queue=rebuilt
            if n in closed: continue
            cost=g[s]+world.edge_cost(s,n)
            if n not in g: g[n]=math.inf
            if cost < g[n]:
                g[n]=cost; parent[n]=s; heapq.heappush(queue,(priority(n),n))
        if method=='dynamic_weighted':
            # Reconstruction: decay per expansion and re-key OPEN; paper omits details.
            rebuilt=[]
            for _,old in queue: heapq.heappush(rebuilt,(priority(old),old))
            queue=rebuilt
    path=[]
    if reached:
        path=[goal]
        while path[-1]!=start: path.append(parent[path[-1]])
        path.reverse()
    valid,reason=validate(q,path,world)
    return {'operation':operations,'storage':len(g),'length':path_length(path) if path else None,
            'valid':valid,'failure':reason,'path':path,'targets':target_list,
            'seconds':time.perf_counter()-t0, 'visited':popped if keep_visited else None}

def scale_case(q, factor):
    r=copy.deepcopy(q)
    r['id']=q['id']+f'-s{factor:02d}'
    for k in ('start','goal'): r[k]=[v*factor for v in q[k]]
    for k in ('range_x','range_y'): r[k]=[q[k][0]*factor,(q[k][1]-1)*factor+1]
    for k in ('horizontal_barriers','vertical_barriers'): r[k]=[[v*factor for v in row] for row in q[k]]
    r['scale']=factor; r['base_id']=q['id']
    return r
