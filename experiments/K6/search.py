"""K6: adaptive waypoint influence, bounded blend and literal-form controls."""
from collections import Counter
import heapq
import math
import time

from reproduction.core import Geometry,parse_output,validate_path
from experiments.K3.heuristic import LowerBound
from experiments.K5.search import run_once as run_previous

PRIMARY='adaptive_wall_w200'
BASELINES=('g2h_euclidean','g2h_wall_octile','g2h_wall_euclidean','clean_wastar_wall',
           'focal_llm_w105','focal_goal_w105','focal_llm_w200',
           'mha_dts_w200','mha_dts_nollm_w200','mha_dts_w105')
STOCHASTIC=('mha_dts_w200','mha_dts_nollm_w200','mha_dts_w105')
NEW_METHODS=(PRIMARY,'fixed_half_wall_w200','fixed_full_wall_w200','adaptive_literal_wall',
             'fixed_literal_wall','no_guide_wall_w200','adaptive_euclidean_w200',
             'adaptive_wall_w105','adaptive_window16_wall_w200')
METHODS=BASELINES+NEW_METHODS


def prepare_targets(geometry,start,goal,text):
    rejected=Counter()
    try:
        points,error=parse_output(text)
    except OverflowError:
        points,error=[],'nonfinite_coordinate'
    targets=[]
    for p in points:
        if any(v!=int(v) for v in p):
            rejected['fractional']+=1;continue
        p=tuple(map(int,p))
        if p in (start,goal):
            rejected['endpoint']+=1;continue
        if geometry.collision(p,p):
            rejected['blocked_or_outside']+=1;continue
        if targets and p==targets[-1]:
            rejected['consecutive_duplicate']+=1;continue
        targets.append(p)
    return targets+[goal],dict(guide_error=error,guide_raw_points=len(points),
                                guide_kept_points=len(targets),guide_rejected=dict(rejected))


class Controller:
    def __init__(self,targets,start,mode='adaptive',initial=.5,window=8):
        if mode not in ('adaptive','fixed','off') or window<1 or not 0<=initial<=1:
            raise ValueError('Invalid controller settings')
        self.targets=list(targets);self.index=0;self.mode=mode;self.initial=initial;self.window=window
        self.alpha=initial if len(targets)>1 and mode!='off' else 0.0
        self.best=math.dist(start,self.target);self.window_best=self.best;self.count=0
        self.alpha_updates=self.target_changes=self.good_windows=self.bad_windows=self.disables=0
        self.observations=self.distance_checks=0
        self.expanded=set();self.alpha_histogram=Counter()

    @property
    def target(self):return self.targets[self.index]

    def observe(self,node):
        self.observations+=1;self.alpha_histogram[str(self.alpha)]+=1
        self.expanded.add(node)
        old=(self.index,self.alpha)
        while self.index<len(self.targets)-1 and self.target in self.expanded:
            self.index+=1;self.target_changes+=1
        if self.index!=old[0]:
            self.alpha=self.initial if self.index<len(self.targets)-1 and self.mode!='off' else 0.0
            self.best=math.dist(node,self.target);self.distance_checks+=1
            self.window_best=self.best;self.count=0
        elif self.mode=='adaptive' and self.alpha>0:
            distance=math.dist(node,self.target);self.distance_checks+=1
            self.window_best=min(self.window_best,distance);self.count+=1
            if self.count==self.window:
                if self.best-self.window_best>=1.0:
                    self.alpha=min(1.0,self.alpha+.25);self.good_windows+=1
                else:
                    self.alpha/=2;self.bad_windows+=1
                    if self.alpha<=1/16:
                        self.alpha=0.0;self.disables+=1
                self.best=min(self.best,self.window_best);self.window_best=self.best;self.count=0
        if self.alpha!=old[1]:self.alpha_updates+=1
        return old!=(self.index,self.alpha)

    def stats(self):
        return dict(alpha_updates=self.alpha_updates,target_changes=self.target_changes,
                    good_windows=self.good_windows,bad_windows=self.bad_windows,guide_disables=self.disables,
                    alpha_histogram=dict(self.alpha_histogram),controller_observations=self.observations,
                    controller_distance_checks=self.distance_checks,final_alpha=self.alpha,
                    final_target_index=self.index,intermediate_targets=len(self.targets)-1)


def graph_search(start,goal,neighbors,heuristic,controller,*,weight=2.0,literal=False,trace=False,audit=False):
    if not math.isfinite(weight) or weight<1:raise ValueError('Weight must be finite and >=1')
    g,parent,active,versions={start:0.0},{start:start},{},{}
    queue,closed,hcache=[],{},{}
    stats=dict(operation=0,stale_pops=0,heap_pushes=0,heap_rebuilds=0,rekey_scans=0,rekey_entries=0,
               rekey_discarded=0,peak_heap_entries=0,successor_expansions=0,reopenings=0,
               selection_audits=0,guide_distance_checks=0)
    events=[];goal_key=None

    def h(node):
        if node not in hcache:
            v=heuristic(node)
            if not math.isfinite(v) or v<0:raise ValueError('Invalid lower bound')
            hcache[node]=v
        return hcache[node]

    def key(node,version):
        lower=h(node)
        if node==goal:
            estimate=0.0
        elif literal:
            distance=math.dist(node,controller.target) if controller.alpha else 0.0
            stats['guide_distance_checks']+=int(controller.alpha>0)
            estimate=lower+controller.alpha*distance
        elif controller.alpha:
            stats['guide_distance_checks']+=1
            distance=math.dist(node,controller.target)
            estimate=weight*lower+controller.alpha*(weight-1)*(min(distance,lower)-lower)
        else:
            estimate=weight*lower
        return g[node]+estimate,node,version

    if abs(h(goal))>1e-10:raise ValueError('h(goal) must be zero')

    def put(node):
        version=versions.get(node,0)+1;versions[node]=active[node]=version
        heapq.heappush(queue,key(node,version));stats['heap_pushes']+=1
        stats['peak_heap_entries']=max(stats['peak_heap_entries'],len(queue))

    put(start)
    while active:
        entry=heapq.heappop(queue);stats['operation']+=1
        _,node,version=entry
        if active.get(node)!=version:
            stats['stale_pops']+=1;continue
        if audit:
            expected=min(key(n,v) for n,v in active.items())
            assert entry==expected,(entry,expected)
            stats['selection_audits']+=1
        del active[node];closed[node]=g[node]
        if trace:
            events.append(dict(node=node,g=g[node],f=entry[0],alpha=controller.alpha,
                               target=controller.target,target_index=controller.index))
        if node==goal:
            goal_key=entry[0];break
        stats['successor_expansions']+=1
        for nxt,cost in neighbors(node):
            if cost<=0 or math.isnan(cost):raise ValueError('Nonpositive edge cost')
            g.setdefault(nxt,math.inf)
            proposal=g[node]+cost
            if proposal<g[nxt]:
                if nxt in closed and nxt not in active:stats['reopenings']+=1
                g[nxt],parent[nxt]=proposal,node;put(nxt)
        if controller.observe(node):
            stats['heap_rebuilds']+=1;stats['rekey_discarded']+=len(queue)
            stats['rekey_scans']+=len(active)
            queue=[key(n,v) for n,v in active.items()]
            heapq.heapify(queue);stats['rekey_entries']+=len(queue)
            stats['peak_heap_entries']=max(stats['peak_heap_entries'],len(queue))
    path=[]
    if goal_key is not None:
        node=goal;path=[node]
        while node!=start:
            node=parent[node]
            if node in path:raise AssertionError('Parent cycle')
            path.append(node)
        path.reverse()
    result=dict(path=path,goal_g=g.get(goal) if path else None,goal_key=goal_key,storage=len(g),
                unique_pops=len(closed),reexpansions=stats['successor_expansions']+int(bool(path))-len(closed),
                infinite_g=sum(math.isinf(v) for v in g.values()),bound_factor=None if literal else weight,
                **stats,**controller.stats())
    if trace:result['trace']=events
    return result


def run_once(query,method,text='',*,seed=0,trace=False,audit=False):
    if method not in METHODS:raise ValueError(method)
    if method in BASELINES:return run_previous(query,method,text,seed=seed,trace=trace,audit=audit)
    Geometry.neighbors.cache_clear();started=time.perf_counter()
    geometry=Geometry(query);start,goal=tuple(query['start']),tuple(query['goal'])
    wall='euclidean' not in method
    lower=LowerBound(geometry,goal,base='octile' if wall else 'euclidean',walls=wall)
    if method=='no_guide_wall_w200':
        targets=[goal];metadata=dict(guide_error=None,guide_raw_points=0,guide_kept_points=0,guide_rejected={})
    else:targets,metadata=prepare_targets(geometry,start,goal,text)
    mode='fixed' if method.startswith('fixed') else ('off' if method=='no_guide_wall_w200' else 'adaptive')
    initial=1.0 if method in ('fixed_full_wall_w200','fixed_literal_wall') else .5
    controller=Controller(targets,start,mode,initial,window=16 if 'window16' in method else 8)
    weight=1.05 if method.endswith('w105') else 2.0
    if geometry.collision(start,start) or geometry.collision(goal,goal):
        result=dict(path=[],valid=False,error='invalid_endpoint',length=None,operation=0,storage=0)
    else:
        result=graph_search(start,goal,geometry.neighbors,lower,controller,weight=weight,
                            literal='literal' in method,trace=trace,audit=audit)
        path=result['path'];error=validate_path(geometry,path,start,goal,grid=True) if path else 'unreachable'
        length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None
        result.update(valid=error is None,error=error,length=length)
        if path:assert length<=result['goal_g']+1e-8
    result['total_seconds']=time.perf_counter()-started
    result.update(metadata,**lower.stats())
    Geometry.neighbors.cache_clear()
    return result
