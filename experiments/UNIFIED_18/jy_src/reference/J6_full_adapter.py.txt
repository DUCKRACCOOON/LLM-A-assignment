"""Same existing guided engine and trace collection for full/lazy variants."""
import time

from experiments.COMMON18.backends import finish
from reproduction.core import Geometry
from experiments.JY.search import run_once

ALIASES=dict(Astar='legacy_astar',LLMA='legacy_llm',G2H='legacy_g2h',Octile='astar_octile',WallAstar='astar_wall',
    J6_full='j6_full',J6='j6_lazy')
METHODS=tuple(ALIASES)
GUIDED=('J6_full','J6')
LLM=('LLMA',)+GUIDED


def run(method,q,replies,context,seed=20261111):
    Geometry.neighbors.cache_clear();started=time.perf_counter();used=['standard'] if method in LLM else []
    text=(replies['standard']['response']['choices'][0]['message'].get('content') or '') if used else ''
    r=run_once(q,ALIASES[method],text=text,trace=method in GUIDED)
    r['successor_expansions']=r['expansions'];r=finish(q,r)
    local=time.perf_counter()-started;Geometry.neighbors.cache_clear();charged=[replies[k] for k in used]
    api=sum(v['seconds'] for v in charged)
    return dict(r,method=method,local_seconds=local,api_seconds=api,reconstructed_total_seconds=local+api,
        llm_calls=len(used),used_responses=used,prompt_tokens=sum(v['response']['usage']['prompt_tokens'] for v in charged),
        completion_tokens=sum(v['response']['usage']['completion_tokens'] for v in charged),
        bound=1.0 if method in ('Astar','Octile','WallAstar') else None,metadata={})
