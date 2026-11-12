"""Fresh ALT preparation per query, retaining existing JY graph/search code."""
import time

from experiments.COMMON18.backends import finish
from reproduction.core import Geometry
from experiments.JY.search import run_once,LandmarkIndex

ALIASES=dict(Astar='legacy_astar',LLMA='legacy_llm',G2H='legacy_g2h',Octile='astar_octile',WallAstar='astar_wall',
    GraphOctile='j4_alt4',J4='j4_alt4',J4_alt8='j4_alt8')
METHODS=tuple(ALIASES)
COUNTS=dict(GraphOctile=0,J4=4,J4_alt8=8)


def run(method,q,replies,context,seed=20261111):
    Geometry.neighbors.cache_clear();started=time.perf_counter();used=['standard'] if method=='LLMA' else []
    text=(replies['standard']['response']['choices'][0]['message'].get('content') or '') if used else ''
    index=None;prep=0.
    if method in COUNTS:
        p=time.perf_counter();index=LandmarkIndex(q,COUNTS[method]);prep=time.perf_counter()-p
    r=run_once(q,ALIASES[method],text=text,index=index)
    r['successor_expansions']=r['expansions'];r=finish(q,r)
    local=time.perf_counter()-started;Geometry.neighbors.cache_clear();context['index_for_audit']=index
    query_pops=r['operation'];charged=[replies[k] for k in used];api=sum(v['seconds'] for v in charged)
    return dict(r,method=method,local_seconds=local,api_seconds=api,reconstructed_total_seconds=local+api,
        preprocessing_seconds=prep,prepared_query_seconds=local-prep,index_internal_seconds=index.seconds if index else 0.,
        query_operation=query_pops,operation=query_pops+r.get('preprocessing_pops',0),
        llm_calls=len(used),used_responses=used,prompt_tokens=sum(v['response']['usage']['prompt_tokens'] for v in charged),
        completion_tokens=sum(v['response']['usage']['completion_tokens'] for v in charged),
        bound=None if method in ('LLMA','G2H') else 1.0,metadata={})
