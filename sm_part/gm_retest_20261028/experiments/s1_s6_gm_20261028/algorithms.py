"""Measured adapters; untouched GM and SM sources remain provenance references."""
import heapq
import math
import time
from collections import Counter
from common import *
import candidate_core as candidate
import next_search_core as exact
import next_refinement as refinement
from next_geometry import Rectangles

class CountGeometry(gm.Geometry):
    def __init__(self,q):
        super().__init__(q);self.collision_calls=0;self.expansions=0;self.expanded=set()
    def collision(self,a,b):
        self.collision_calls+=1;return super().collision(a,b)
    def neighbors(self,s):
        self.expansions+=1;self.expanded.add(s);return super().neighbors(s)

class HeapCounts:
    def __enter__(self):
        self.counts=Counter(); self.original={k:getattr(heapq,k) for k in ('heappop','heappush','heapify')}
        def pop(h): self.counts['heap_pops']+=1;return self.original['heappop'](h)
        def push(h,x):self.counts['heap_pushes']+=1;return self.original['heappush'](h,x)
        def heapify(h):self.counts['heap_rebuilds']+=1;self.counts['heapify_entries']+=len(h);return self.original['heapify'](h)
        heapq.heappop=pop;heapq.heappush=push;heapq.heapify=heapify;return self
    def __exit__(self,*args):
        for k,v in self.original.items():setattr(heapq,k,v)

def normalized(q,geo,text):
    nodes,error=gm.parse_output(text)
    integer=all(all(float(v).is_integer() for v in p) for p in nodes)
    selected=geo.filter_upstream(nodes)
    if not selected or selected[0]!=tuple(q['start']):selected.insert(0,tuple(q['start']))
    if selected[-1]!=tuple(q['goal']):selected.append(tuple(q['goal']))
    return selected,error,integer

def s1(q,geo,text,refine=True):
    targets,error,integer=normalized(q,geo,text); original=len(targets)
    if refine:
        unique=[]
        for p in targets:
            if p in unique:unique=unique[:unique.index(p)+1]
            else:unique.append(p)
        kept=[unique[0]];i=0
        while i<len(unique)-1:
            reach=[j for j in range(i+1,len(unique)) if not geo.collision(unique[i],unique[j])]
            i=max(reach) if reach else i+1;kept.append(unique[i])
        targets=kept
    accepted=error is None and integer and gm.validate_path(geo,targets,q['start'],q['goal']) is None
    result=gm.search(q,geo,'llm_astar' if accepted else 'goal_only',text=json.dumps(targets))
    result.update(guidance_accepted=accepted,input_targets=original,processed_targets=len(targets))
    return result

def delayed(q,geo,text,mode='adaptive'):
    """GM state transitions with the original S3 fixed trigger inserted."""
    start,goal=tuple(q['start']),tuple(q['goal']);g={start:0.};parent={start:start};closed=[];closed_set=set()
    targets=[start,goal];ti=1;target=goal;called=False;trigger=None;rekeys=0;rebuilds=0;distances={}
    def h(s):
        if s not in distances:distances[s]=math.dist(s,goal)
        return distances[s]
    def priority(s):return g[s]+h(s)+math.dist(s,target)
    def inject():
        nonlocal targets,ti,target,called
        nodes,error=gm.parse_output(text,upstream=True);called=True
        if error not in (None,'missing_list'):return error
        targets=geo.filter_upstream(nodes)
        if not targets or targets[0]!=start:targets.insert(0,start)
        if targets[-1]!=goal:targets.append(goal)
        ti=1;target=targets[1];return None
    if start==goal:
        r=gm.search(q,geo,'goal_only');r.update(api_calls=0,trigger=None);return r
    if mode=='immediate':
        err=inject()
        if err:return dict(valid=False,error=err,path=[],length=None,storage=1,api_calls=1)
    queue=[];heapq.heappush(queue,(priority(start),start))
    def rebuild():
        nonlocal queue,rekeys,rebuilds
        rekeys+=len(queue);rebuilds+=1;fresh=[]
        for _,old in queue:heapq.heappush(fresh,(priority(old),old))
        queue=fresh
    d0=math.dist(start,goal);window=max(10,math.ceil(d0));best=d0;previous=d0;error=None
    while queue:
        _,s=heapq.heappop(queue);closed.append(s);closed_set.add(s);best=min(best,h(s))
        if s==goal:break
        for n,cost in geo.neighbors(s):
            if n==target and target!=goal:
                ti+=1
                if ti<len(targets):target=targets[ti]
                rebuild()
            if n in closed_set:continue
            new=g[s]+cost
            if n not in g:g[n]=math.inf
            if new<g[n]:g[n]=new;parent[n]=s;heapq.heappush(queue,(priority(n),n))
        if mode=='adaptive' and not called and queue and len(closed)%window==0:
            if previous-best<.05*d0:
                trigger=dict(pop=len(closed),window=window,progress=previous-best,threshold=.05*d0)
                error=inject()
                if error:break
                rebuild()
            previous=best
    path=[]
    if error is None and goal in parent:
        path=[goal]
        while path[-1]!=start:path.append(parent[path[-1]])
        path.reverse()
    error=error or ('unreachable' if not path else gm.validate_path(geo,path,start,goal,grid=True))
    return dict(valid=error is None,error=error,path=path,length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None,
                storage=len(g),api_calls=int(called),trigger=trigger,queue_rebuilds=rebuilds,rekeyed_entries=rekeys,
                infinite_g=sum(math.isinf(v) for v in g.values()),targets=targets)

def models_load():
    import torch
    torch.set_num_threads(1);models={};t=time.perf_counter()
    for seed in (11,22,33):
        model=candidate.make_network();model.load_state_dict(torch.load(OUT/'models'/f'seed{seed}.pt',map_location='cpu',weights_only=True));model.eval();models[f'S4_seed{seed}']=model
    torch.manual_seed(11);models['S4_untrained']=candidate.make_network().eval()
    return models,time.perf_counter()-t

BASE=['A_GM','N_g2h','B_llm','V_GATE_GM','V_GATE_NORMALIZED','S1','S2_feedback','S2_control','S3','K3_wall_euclidean','K3_wall_octile','A_OCT','FOCAL_105','S5','S6_rule','S6_random','S6_program','S6_historical']
MODEL_METHODS=['S4_seed11','S4_seed22','S4_seed33','S4_untrained']

def run(q,method,text='',revisions=None,models=None,program=None):
    gm.Geometry.neighbors.cache_clear();meta={};revisions=revisions or {};calls=0
    with HeapCounts() as hc:
        started=time.perf_counter()
        if method in BASE[:11]:
            geo=CountGeometry(q);heuristic=None
            if method in ('A_GM','N_g2h','K3_wall_euclidean','K3_wall_octile'):
                if method.startswith('K3_'):heuristic=LowerBound(geo,q['goal'],base='euclidean' if method.endswith('euclidean') else 'octile')
                r=gm.search(q,geo,'astar' if method=='A_GM' else 'goal_only',goal_heuristic=heuristic)
            elif method=='B_llm':r=gm.search(q,geo,'llm_astar',text);calls=1
            elif method=='V_GATE_GM':r=proposed_search(q,geo,text);calls=1
            elif method in ('S1','V_GATE_NORMALIZED'):r=s1(q,geo,text,method=='S1');calls=1
            elif method=='S3':r=delayed(q,geo,text);calls=r['api_calls']
            else:
                diag=diagnosis(q,text);valid=diag['valid'];used=text if valid else revisions[method.split('_')[1]]
                r=gm.search(q,geo,'llm_astar',used);calls=1+int(not valid);r.update(initial_proposal_valid=valid,final_proposal_valid=diagnosis(q,used)['valid'])
            expansions=geo.expansions;meta.update(collision_calls=geo.collision_calls,unique_successor_expansions=len(geo.expanded))
            if heuristic:meta.update(heuristic.stats())
            storage=r.get('storage');storage_definition='g_dictionary_including_infinite'
        elif method=='A_OCT' or method=='S5':
            graph=exact.Grid(q) if method=='A_OCT' else Rectangles(q,variant='sparse')
            r=exact.search(graph,q);expansions=r['counters'].get('expansions',0)-int(r['reachable']);storage=r['discovered'];storage_definition='finite_g_dictionary'
            meta.update(r['counters'],graph_seconds=graph.seconds)
        elif method.startswith('S6_'):
            rounds=[];original=refinement.search
            def traced(*a,**kw):
                result=original(*a,**kw);rounds.append(result);return result
            refinement.search=traced
            try:r=refinement.solve(q,mode='program' if method in ('S6_program','S6_historical') else method[3:],program=[.5,.3,.1,.1] if method=='S6_historical' else program)
            finally:refinement.search=original
            expansions=sum(x['counters'].get('expansions',0)-int(x['reachable']) for x in rounds)
            storage=max(x['discovered'] for x in rounds);storage_definition='max_per_round_finite_g_dictionary'
            meta.update(r['counters'],round_discovered=[x['discovered'] for x in rounds],round_peak_open=[x['peak_open'] for x in rounds])
        else:
            import torch
            graph=candidate.Grid(q);secondary=None;prep=time.perf_counter()
            if method in MODEL_METHODS:
                x=candidate.network_inputs(graph,q)
                with torch.no_grad():secondary=-models[method](torch.from_numpy(x[None])).sigmoid().numpy().ravel()
            meta.update(inference_seconds=time.perf_counter()-prep,graph_seconds=graph.seconds,allocated_g_entries=graph.n)
            r=candidate.focal(graph,q,secondary=secondary,weight=1.05)
            expansions=r['expansions']-int(r['valid']);storage=graph.n;storage_definition='allocated_dense_g_array_entries'
        # Final validation is a common mandatory planner cost, independently of oracle evaluation.
        path=r['path'];geo_check=gm.Geometry(q)
        error=gm.validate_path(geo_check,path,q['start'],q['goal'],grid=True) if path else r.get('error') or r.get('failure') or 'unreachable'
        if path and any(any(not float(c).is_integer() for c in p) for p in path):error='non_integer_grid_point'
        valid=error is None
        elapsed=time.perf_counter()-started
    gm.Geometry.neighbors.cache_clear()
    r.update(valid=valid,error=error,operation=hc.counts['heap_pops'],successor_expansions=expansions,storage=storage,
             storage_definition=storage_definition,total_seconds=elapsed,api_calls=calls,heap_counts=dict(hc.counts),extra=meta)
    return r
