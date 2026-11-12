"""Adapters preserving each branch's method, on a common input/model/host.

No oracle is accepted by run(). All imports of SM search code are namespaced.
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from reproduction.core import Geometry, search, parse_output, validate_path
from experiments.K3.heuristic import LowerBound
from experiments.K4.search import run_once as run_k4,PolylineGuide
from experiments.K5.search import graph_search as mha_search
from experiments.K6.search import graph_search as adaptive_search,Controller,prepare_targets
from experiments.JY.search import run_once as run_jy,LandmarkIndex
from .vendor_sm import candidate_core as cc,next_search_core as nc,next_topology as topo,next_refinement as refine
from .vendor_sm.next_geometry import Rectangles
from .vendor_sm.core import parse_path,World,search as sm_search
from .interventions import refine as simplify,delayed_search

ROOT=Path(__file__).resolve().parents[2]
PRIMARY=tuple([f'K{i}' for i in range(1,7)]+[f'J{i}' for i in range(1,7)]+[f'S{i}' for i in range(1,7)])
CONTROLS=('Astar','LLMA','G2H','SM_LLMA','Octile','K1_rule','K1_random','FOCAL0','K4_noLLM','K4_exact_noLLM',
          'K4_exact','K5_noLLM','K6_noLLM','J1_straight','J2_uniform','J3_noLLM','J6_full','S4_untrained','S1_gate')
EXACT_VARIANTS=('K1_exact','K1_rule_exact','K2_exact','K5_exact','K6_exact','J1_exact','J2_exact','S4_exact')
METHODS=PRIMARY+CONTROLS+EXACT_VARIANTS
EXACT={'Astar','Octile','J3','J4','J5','S5','S6','K4_exact','K4_exact_noLLM','J3_noLLM'}
EXACT.update(EXACT_VARIANTS)
BOUNDED={'K1','K2','K4','K5','K6','J1','J2','S4','K1_rule','K1_random','FOCAL0',
         'K4_noLLM','K5_noLLM','K6_noLLM','J1_straight','J2_uniform','S4_untrained'}


def load_model(untrained=False):
    import torch
    torch.set_num_threads(1);torch.manual_seed(11)
    model=cc.make_network()
    if not untrained:
        path=Path(__file__).parent/'assets/seed11.pt'
        expected=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.eval()
    return model


def content(r):return r['response']['choices'][0]['message'].get('content') or ''


def clean_paths(texts):
    return [parse_path(t)[0] for t in texts]


def finish(q,r):
    path=[tuple(p) for p in r.get('path',[])];geo=Geometry(q)
    error=validate_path(geo,path,q['start'],q['goal'],grid=True) if path else r.get('error',r.get('failure','unreachable'))
    r.update(valid=error is None,error=error,path=path,
             length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None)
    return r


def run(method,q,replies,context,seed=20261028):
    """Time reading/normalizing responses, ALL online preparation and validation.

    File loading/model loading are outside, as in a warm resident service. A
    method's charged response IDs determine its calls, latency and tokens.
    """
    Geometry.neighbors.cache_clear();started=time.perf_counter();used=[];extra={}
    exact_variant=method in EXACT_VARIANTS
    family=method[:-6] if exact_variant else method
    needs_standard=family in ('LLMA','SM_LLMA','K4','K4_exact','K5','K6','J1','J3','J6','S1','S1_gate','S2','J6_full')
    text=content(replies['standard']) if needs_standard else ''
    raw=parse_path(text)[0] if family in ('SM_LLMA','J1','S1','S1_gate') else []
    if method in ('Astar','LLMA','G2H','K3'):
        geo=Geometry(q)
        kind={'Astar':'astar','LLMA':'llm_astar','G2H':'goal_only','K3':'goal_only'}[method]
        r=search(q,geo,method=kind,text=text,goal_heuristic=LowerBound(geo,q['goal'],base='euclidean') if method=='K3' else None)
        r['successor_expansions']=(r.get('operation') or 0)-int(r['valid'])
        if method=='LLMA':used=['standard']
    elif method=='SM_LLMA':
        r=sm_search(q,method='llm_astar',waypoints=raw);used=['standard']
        r['successor_expansions']=r['operation']-int(r['valid'])
    elif method in ('Octile','J3','J4','J5','J6','J3_noLLM','J6_full'):
        m={'Octile':'astar_octile','J3':'j3_llm_prune','J4':'j4_alt4','J5':'j5_nbs_octile',
           'J6':'j6_lazy','J3_noLLM':'j3_weight_prune','J6_full':'j6_full'}[method]
        idx=LandmarkIndex(q,4) if method=='J4' else None
        r=run_jy(q,m,text,index=idx)
        r['successor_expansions']=r['expansions'];r['operation']=r['cold_operation']
        if method in ('J3','J6','J6_full'):used=['standard']
    elif method in ('K4','K4_exact','K4_noLLM','K4_exact_noLLM'):
        key='focal_goal_w105' if method=='K4_noLLM' else 'focal_llm_w100' if method in ('K4_exact','K4_exact_noLLM') else 'focal_llm_w105'
        r=run_k4(q,key,text)
        r['successor_expansions']=r['expansions']-int(r['valid'])
        if method in ('K4','K4_exact'):used=['standard']
    elif family in ('K5','K5_noLLM'):
        geo=Geometry(q);start,goal=tuple(q['start']),tuple(q['goal']);lower=LowerBound(geo,goal);cache={}
        def h(n):
            if n not in cache:cache[n]=lower(n)
            return cache[n]
        helpers=[h,lambda n:2*h(n),lambda n:2*math.dist(n,goal)]
        if family=='K5':
            guide=PolylineGuide(q,text);helpers.append((lambda n:2*h(n)) if guide.error else (lambda n:0.0 if n==goal else 2*guide(n)))
            used=['standard']
        r=mha_search(start,goal,geo.neighbors,helpers,weight=1.0 if exact_variant else 1.05,mode='dts',seed=seed)
    elif family in ('K6','K6_noLLM'):
        geo=Geometry(q);start,goal=tuple(q['start']),tuple(q['goal']);lower=LowerBound(geo,goal)
        if family=='K6':targets,metadata=prepare_targets(geo,start,goal,text);used=['standard']
        else:targets,metadata=[goal],{}
        ctl=Controller(targets,start,'adaptive' if family=='K6' else 'off',.5,window=8)
        r=adaptive_search(start,goal,geo.neighbors,lower,ctl,weight=1.0 if exact_variant else 1.05);r.update(metadata)
    elif family in ('K1','K1_rule','K1_random'):
        proposal,graph=topo.routes(q,k=2);ids=list(range(len(proposal['candidates'])))
        ranking=topo.rank(proposal,'random' if family=='K1_random' else 'rule',seed=seed)
        if family=='K1' and len(ids)>=2:
            used=['ranking']
            try:
                match=re.search(r'\{[^{}]*\}',content(replies['ranking']),re.S)
                chosen=json.loads(match.group()).get('ranking')
                assert isinstance(chosen,list) and all(type(v)==int for v in chosen) and sorted(chosen)==ids
                assert replies['ranking']['response']['choices'][0].get('finish_reason','stop')=='stop'
                ranking=chosen;extra['ranking_accepted']=True
            except (ValueError,TypeError,AttributeError,AssertionError):extra['ranking_accepted']=False
        # Same original unit-grid FOCAL as SM R3, not compressed route execution.
        grid=nc.Grid(q);guide=topo.Guide(graph,proposal,ranking)
        r=nc.search(grid,q,weight=1.0 if exact_variant else 1.05,focal=True,guide=guide)
        r['operation']=r['counters']['heap_pops'];r['storage']=r['discovered']
        r['successor_expansions']=r['counters']['expansions']-int(r['reachable'])
        extra.update(candidate_count=len(ids),ranking=ranking)
    elif family in ('K2','J1','J2','S4','FOCAL0','J1_straight','J2_uniform','S4_untrained'):
        grid=cc.Grid(q);secondary=None;incumbent=None;meta={};field=None
        if family in ('K2','J2'):
            used=['standard','cot','repe'];paths=clean_paths([content(replies[k]) for k in used]);field,scale=cc.llm_field(grid,q,paths)
        if family=='K2':
            radius=context['quantile']*scale
            secondary=np.maximum(0,field-radius);extra['set_fraction']=float(np.mean(field[grid.free]<=radius))
        elif family in ('J1','J1_straight'):
            if family=='J1':used=['standard']
            repaired,meta=cc.chomp_repair(q,raw if family=='J1' else [q['start'],q['goal']])
            incumbent=repaired or None;secondary=cc.point_segment_distance(grid.xy,repaired) if repaired else None
        elif family in ('J2','J2_uniform'):
            speed=np.where(field<=2.,2.,1.) if field is not None else None
            secondary,meta=cc.fast_marching(grid,q['goal'],speed)
        elif family in ('S4','S4_untrained'):
            import torch
            model=context['untrained' if family=='S4_untrained' else 'model']
            with torch.inference_mode():secondary=-model(torch.from_numpy(cc.network_inputs(grid,q)[None])).sigmoid().numpy().ravel()
        r=cc.focal(grid,q,secondary,weight=1.0 if exact_variant else 1.05,incumbent=incumbent)
        # This original engine selects FOCAL by array scanning; its heap counter
        # omits explicit node selections. Keep it separate, never invent pops.
        r['operation']=None;r['storage']=r['discovered']
        r['successor_expansions']=r['expansions']-int(r['valid'] and not r['incumbent_returned'])
        extra.update(meta)
    elif method in ('S1','S1_gate','S2','S3'):
        if method in ('S1','S1_gate'):
            if method=='S1':targets,meta=simplify(q,raw)
            else:
                from .vendor_sm.core import validate
                world=World(q);targets=world.targets(raw,tuple(q['start']),tuple(q['goal']));accepted,reason=validate(q,targets,world)
                meta=dict(accepted=accepted,rejection_reason=reason,gate_only=True)
                if not accepted:targets=[]
            r=search(q,method='llm_astar',text=json.dumps(targets));extra.update(meta);used=['standard']
        elif method=='S2':
            used=['standard'];selected=text
            parsed,parse_error=parse_output(text);reason=parse_error or validate_path(Geometry(q),parsed,q['start'],q['goal'])
            if reason:
                # Include validation/prompt construction required in deployment.
                from .prepare import correction_prompt
                correction_prompt(q,text,context.get('standard_template',''))
                assert 'correction' in replies,'Required correction response missing'
                used.append('correction');selected=content(replies['correction'])
            r=search(q,method='llm_astar',text=selected)
        else:
            r=delayed_search(q,lambda:parse_path(content(replies['standard']))[0])
            used=['standard'] if r['api_calls'] else []
        r['successor_expansions']=(r.get('operation') or 0)-int(r.get('valid',False))
    elif method in ('S5','S6'):
        if method=='S5':graph=Rectangles(q);r=nc.search(graph,q)
        else:r=refine.solve(q,mode='program',program=[.5,.3,.1,.1])
        r['operation']=r['counters']['heap_pops'];r['storage']=r['discovered']
        r['successor_expansions']=r['counters']['expansions']-sum(1 for _ in r.get('rounds',[1]) if r.get('reachable'))
    else:raise ValueError(method)
    r=finish(q,r);local=time.perf_counter()-started;Geometry.neighbors.cache_clear()
    charged=[replies[k] for k in used]
    api=sum(r['seconds'] for r in charged)
    return dict(r,method=method,local_seconds=local,api_seconds=api,reconstructed_total_seconds=local+api,
                llm_calls=len(used),used_responses=used,prompt_tokens=sum(r['response'].get('usage',{}).get('prompt_tokens',0) for r in charged),
                completion_tokens=sum(r['response'].get('usage',{}).get('completion_tokens',0) for r in charged),
                bound=1.0 if method in EXACT else 1.05 if method in BOUNDED else None,metadata=extra)
