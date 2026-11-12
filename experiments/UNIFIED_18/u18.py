"""Shared setup for the 18-method single-block evaluation (S1-S6, J1-J6, K1-K6 + baselines).

Reuses the UNIFIED_K adaptation of SM's unified protocol unchanged (common.py paths/data,
harness.run measurement wrapper). UNIFIED_K files are not modified (their hashes are frozen);
this module only extends harness.EXTRA with the S and J adapters and the paper-prompt
LLM-A* condition, and redirects outputs to UNIFIED_18/results.
"""
import os
import sys
from pathlib import Path

U18 = Path(__file__).resolve().parent
UK = U18.parent/'UNIFIED_K'
if str(UK) not in sys.path: sys.path.insert(0, str(UK))
import common                                   # noqa: E402  (sets REPO, sm_src, UNIFIED_K on sys.path)
from common import *                            # noqa: E402,F401,F403
import harness                                  # noqa: E402
if str(U18) in sys.path: sys.path.remove(str(U18))
sys.path.insert(0, str(U18))

RUN18 = U18/'results/run_20261112'
OUT18 = RUN18/'folds'/f'fold{FOLD}'
UK_RUN = common.RUN                             # UNIFIED_K run: k1/k2 responses + K2 fold calibrations

def b_llm_paper(q, ctx):
    """Original LLM-A* (same GM engine as B_llm) fed the ORIGINAL paper-prompt standard response (Table 1 setting)."""
    geo = harness.CountGeometry(q)
    r = gm.search(q, geo, 'llm_astar', ctx['replies'].get('paper_standard', ''))
    r.update(successor_expansions=geo.expansions, storage=r.get('storage'), storage_definition='g_dictionary_including_infinite', api_calls=1,
             extra=dict(collision_calls=geo.collision_calls, unique_successor_expansions=len(geo.expanded)))
    return r

LOCAL = {'B_llm_paper': dict(fn=b_llm_paper, bound=None, family='baseline', note='original LLM-A*, paper prompt response (Table 1 setting)')}

def extend():
    import methods_s, methods_j
    for mod in (LOCAL, methods_s.METHODS, methods_j.METHODS):
        for name, spec in mod.items():
            assert name not in harness.EXTRA and name not in harness.BASE, name
            harness.EXTRA[name] = spec

extend()
ALL = harness.BASE+list(harness.EXTRA)
run, bound, family = harness.run, harness.bound, harness.family
