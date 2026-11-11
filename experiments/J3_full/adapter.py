"""Expose existing JY upper-bound engines without changing their algorithms."""
import time

from experiments.COMMON18.backends import finish
from reproduction.core import Geometry
from experiments.JY.search import run_once

ALIASES=dict(Astar='legacy_astar',LLMA='legacy_llm',G2H='legacy_g2h',Octile='astar_octile',WallAstar='astar_wall',
    J3='j3_llm_prune',J3_no_prune='j3_llm_noprune',J3_noLLM='j3_weight_prune',J3_noLLM_no_prune='j3_weight_noprune')
METHODS=tuple(ALIASES)
LLM={'LLMA','J3','J3_no_prune'}


def run(method,q,replies,context,seed=20261111):
    Geometry.neighbors.cache_clear();started=time.perf_counter();used=['standard'] if method in LLM else []
    text=(replies['standard']['response']['choices'][0]['message'].get('content') or '') if used else ''
    r=run_once(q,ALIASES[method],text=text);r['successor_expansions']=r['expansions'];r=finish(q,r)
    local=time.perf_counter()-started;Geometry.neighbors.cache_clear();charged=[replies[k] for k in used]
    api=sum(v['seconds'] for v in charged)
    return dict(r,method=method,local_seconds=local,api_seconds=api,reconstructed_total_seconds=local+api,
        llm_calls=len(used),used_responses=used,prompt_tokens=sum(v['response']['usage']['prompt_tokens'] for v in charged),
        completion_tokens=sum(v['response']['usage']['completion_tokens'] for v in charged),
        bound=None if method in ('LLMA','G2H') else 1.0,metadata={})
