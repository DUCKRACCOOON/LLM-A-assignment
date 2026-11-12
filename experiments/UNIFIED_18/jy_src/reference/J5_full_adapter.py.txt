"""Unchanged JY NBS/bidirectional search plus a same-engine Dijkstra control."""
import time

from experiments.COMMON18.backends import finish
from reproduction.core import Geometry
from experiments.JY.search import run_once,astar

ALIASES=dict(Astar='legacy_astar',LLMA='legacy_llm',G2H='legacy_g2h',Octile='astar_octile',WallAstar='astar_wall',
    Dijkstra=None,J5='j5_nbs_octile',J5_euclidean='j5_nbs_euclidean',J5_bidijkstra='j5_bidijkstra')
METHODS=tuple(ALIASES)
BIDIRECTIONAL=('J5','J5_euclidean','J5_bidijkstra')


def run(method,q,replies,context,seed=20261111):
    Geometry.neighbors.cache_clear();started=time.perf_counter();used=['standard'] if method=='LLMA' else []
    text=(replies['standard']['response']['choices'][0]['message'].get('content') or '') if used else ''
    if method=='Dijkstra':
        geometry=Geometry(q);r=astar(q,geometry.neighbors,lambda n:0.)
    else:
        r=run_once(q,ALIASES[method],text=text,audit=context.get('audit_pair_selection',False))
    r['successor_expansions']=r['expansions'];r=finish(q,r)
    local=time.perf_counter()-started;Geometry.neighbors.cache_clear();charged=[replies[k] for k in used]
    api=sum(v['seconds'] for v in charged)
    return dict(r,method=method,local_seconds=local,api_seconds=api,reconstructed_total_seconds=local+api,
        llm_calls=len(used),used_responses=used,prompt_tokens=sum(v['response']['usage']['prompt_tokens'] for v in charged),
        completion_tokens=sum(v['response']['usage']['completion_tokens'] for v in charged),
        bound=None if method in ('LLMA','G2H') else 1.0,metadata={})
