"""Configuration-only w=1.05 ablation of the frozen K6 search engine."""
import math
import time

from reproduction.core import Geometry,validate_path
from experiments.K3.heuristic import LowerBound
from experiments.K4.search import graph_search as weighted_reference
from experiments.K6.search import Controller,prepare_targets,graph_search,run_once as old_run

PRIMARY='adaptive_wall_w105'
BASELINES=('g2h_euclidean','g2h_wall_octile','g2h_wall_euclidean')
TIGHT=(PRIMARY,'no_guide_wall_w105','fixed_half_wall_w105','fixed_full_wall_w105')
METHODS=BASELINES+TIGHT
CONFIG={PRIMARY:('adaptive',.5),'no_guide_wall_w105':('off',0.),
        'fixed_half_wall_w105':('fixed',.5),'fixed_full_wall_w105':('fixed',1.)}


def run_once(query,method,text='',*,trace=False,audit=False):
    if method not in METHODS:raise ValueError(method)
    if method in BASELINES:return old_run(query,method,trace=trace,audit=audit)
    Geometry.neighbors.cache_clear();started=time.perf_counter()
    geometry=Geometry(query);start,goal=tuple(query['start']),tuple(query['goal'])
    lower=LowerBound(geometry,goal,base='octile',walls=True)
    mode,initial=CONFIG[method]
    if mode=='off':
        targets=[goal];metadata=dict(guide_error=None,guide_raw_points=0,guide_kept_points=0,guide_rejected={})
    else:targets,metadata=prepare_targets(geometry,start,goal,text)
    controller=Controller(targets,start,mode,initial,window=8)
    if geometry.collision(start,start) or geometry.collision(goal,goal):
        result=dict(path=[],valid=False,error='invalid_endpoint',length=None,operation=0,storage=0)
    else:
        result=graph_search(start,goal,geometry.neighbors,lower,controller,weight=1.05,trace=trace,audit=audit)
        path=result['path'];error=validate_path(geometry,path,start,goal,grid=True) if path else 'unreachable'
        length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None
        result.update(valid=error is None,error=error,length=length)
        if path:assert length<=result['goal_g']+1e-8
    result['total_seconds']=time.perf_counter()-started
    result.update(metadata,**lower.stats());Geometry.neighbors.cache_clear()
    return result


def no_guide_reference(query):
    """Separate engine for parity only; never part of benchmark timings."""
    Geometry.neighbors.cache_clear();geometry=Geometry(query)
    start,goal=tuple(query['start']),tuple(query['goal']);lower=LowerBound(geometry,goal)
    result=weighted_reference(start,goal,geometry.neighbors,lower,lower,weight=1.05,focal=False)
    Geometry.neighbors.cache_clear()
    return result
