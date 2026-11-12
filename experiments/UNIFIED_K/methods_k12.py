"""K1 (route-class ranking) and K2 (conformal LLM corridor) adapters for harness.run.

Engines are SM code copied verbatim into sm_src/ (commit 0c0a6082):
  K1: next_topology.routes/rank/Guide + next_search_core.search(focal=True, weight=1.05)
      — same composition as COMMON18 backends.run family 'K1' (jy d72632c3).
  K2: cv_candidate_core.llm_field/focal(weight=1.05) — the same FOCAL engine and
      Grid as the harness FOCAL_105 no-guide control. cv_candidate_core differs from
      candidate_core only by two extra scan counters (focal_scanned_entries,
      focal_array_selections); llm_field/normalize_proposal/conformal_quantile are identical.
      Composition follows COMMON18 backends.run family 'K2'.

ctx keys used: text (SM shared standard 'first' response), replies {'k1_ranking','k2_cot','k2_repe'}
(texts), finish {'k1_ranking': finish_reason}, calibration {'quantile': float} (fold-level,
from calibrate_k2.py), fold (int).
All per-query work (topology construction, response parsing, field construction) is inside fn.
"""
import json
import re
import numpy as np
from common import FOLD
import core as smcore
import cv_candidate_core as cc
import next_search_core as nc
import next_topology as topo


def _k1(mode):
    def fn(q, ctx):
        proposal, graph = topo.routes(q, k=2); ids = list(range(len(proposal['candidates'])))
        seed = f'20261112:{ctx.get("fold", FOLD)}:{q["id"]}'
        ranking = topo.rank(proposal, 'random' if mode == 'random' else 'rule', seed=seed)
        extra = dict(candidate_count=len(ids)); calls = 0
        if mode == 'llm' and len(ids) >= 2:
            replies = ctx.get('replies', {})
            if 'k1_ranking' in replies:
                calls = 1
                try:
                    match = re.search(r'\{[^{}]*\}', replies['k1_ranking'], re.S)
                    chosen = json.loads(match.group()).get('ranking')
                    assert isinstance(chosen, list) and all(type(v) == int for v in chosen) and sorted(chosen) == ids
                    assert (ctx.get('finish', {}).get('k1_ranking') or 'stop') == 'stop'
                    ranking = chosen; extra['ranking_accepted'] = True
                except (ValueError, TypeError, AttributeError, AssertionError):
                    extra['ranking_accepted'] = False
            else:
                extra['ranking_missing'] = True
        grid = nc.Grid(q); guide = topo.Guide(graph, proposal, ranking)
        r = nc.search(grid, q, weight=1.05, focal=True, guide=guide)
        extra.update(ranking=ranking, counters=r['counters'], engine_peak_open=r['peak_open'])
        return dict(path=r['path'], error=None if r['path'] else 'unreachable', storage=r['discovered'],
                    storage_definition='finite_g_dictionary', successor_expansions=r['counters']['expansions']-int(r['reachable']),
                    api_calls=calls, peak_open=r['peak_open'], extra=extra)
    return fn


def proposals(q, texts):
    """COMMON18 clean_paths: SM core.parse_path; a failed parse gives [] -> straight start-goal proposal."""
    parsed = [smcore.parse_path(t) for t in texts]
    return [p for p, _ in parsed], [e for _, e in parsed]


def _k2(conformal):
    def fn(q, ctx):
        replies = ctx.get('replies', {})
        texts = [ctx.get('text', ''), replies.get('k2_cot', ''), replies.get('k2_repe', '')]
        paths, errors = proposals(q, texts)
        grid = cc.Grid(q); field, scale = cc.llm_field(grid, q, paths)
        extra = dict(parse_errors=errors, uncertainty_scale=scale, graph_seconds=grid.seconds)
        if conformal:
            radius = ctx['calibration']['quantile']*scale
            secondary = np.maximum(0, field-radius); extra.update(radius=radius, set_fraction=float(np.mean(field[grid.free] <= radius)))
        else:
            secondary = field
        r = cc.focal(grid, q, secondary, weight=1.05)
        extra.update({k: r[k] for k in ('anchor_heap_pops', 'focal_scanned_entries', 'focal_array_selections', 'incumbent_returned')})
        return dict(path=r['path'], error=None if r['path'] else (r.get('failure') or 'unreachable'), storage=grid.n,
                    storage_definition='allocated_dense_g_array_entries', successor_expansions=r['expansions']-int(r['valid']),
                    api_calls=3, peak_open=r['peak_open'], extra=extra)
    return fn


METHODS = {
    'K1_llm': dict(fn=_k1('llm'), bound=1.05, family='K1', note='LLM ranks bounded rectangle-portal route candidates; FOCAL w=1.05'),
    'K1_rule': dict(fn=_k1('rule'), bound=1.05, family='K1', note='same engine, rule (estimated length) order, no API'),
    'K1_random': dict(fn=_k1('random'), bound=1.05, family='K1', note='same engine, seeded random order, no API'),
    'K2_conformal': dict(fn=_k2(True), bound=1.05, family='K2', note='3-response corridor field minus fold conformal radius; FOCAL w=1.05'),
    'K2_nocal': dict(fn=_k2(False), bound=1.05, family='K2', note='same 3-response field without conformal radius'),
}
