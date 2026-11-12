"""J1-J6 (JY) adapters for the UNIFIED_K harness.run contract (see UNIFIED_K/harness.py docstring).

Engines are byte-exact copies from origin/jy d72632c3 under jy_src/ (PROVENANCE.json):
  jy_src/vendor_sm/candidate_core.py  COMMON18 copy of SM candidate_core (+ JY's documented
                                      verified-incumbent repair)      -> J1 (CHOMP repair + FOCAL), J2 (FMM + FOCAL)
  jy_src/vendor_sm/core.py            COMMON18 copy of SM core (parse_path, World)
  jy_src/JY/search.py                 JY engines                       -> J3 (j3_*), J4 (LandmarkIndex + j4_alt4),
                                                                          J5 (j5_*), J6 (j6_*)
Each fn reproduces the corresponding branch of JY's own runner (COMMON18/backends.run for J1/J2,
J3_full..J6_full/adapter.run for J3-J6; copies in jy_src/reference/) minus JY's bookkeeping
(API latency/token accounting). All per-query work - response parsing, dense Grid build, CHOMP
repair, LLM distance field, Fast Marching, J3 seed search, J4 graph + landmark Dijkstra
(LandmarkIndex is rebuilt for every query, as in JY's cold protocol), NBS queues, J6 rekeying - runs
inside fn, so harness.run times it.

heapq: every JY engine (and reproduction.core / experiments.K3.heuristic it calls) uses
``import heapq`` + ``heapq.heappop(...)``; no ``from heapq import`` anywhere, so harness.HeapCounts
sees every pop without patching.

Shims (adapter-side only; no copied file is edited):
  1. vendor_sm/core.py resolves ROOT = Path(__file__).parents[3] to find source_code/ (prompt
     templates, dataset). Under jy_src/vendor_sm it is one level deeper than in COMMON18, so
     _load_vendor_core() executes the unchanged file with __file__ set to its original location
     (experiments/COMMON18/vendor_sm/core.py), which only affects that path computation.
  2. J6 runs with trace=False (JY timed trace=True to audit full/lazy selection-sequence equality).
     trace only appends selected nodes to a list; search decisions are unchanged. test_j.py
     re-checks JY's recorded trace hashes with trace=True outside the harness.
  3. J5 runs with audit=False, as in JY's timed dataset run (audit only asserts pair choice).
  4. J3 only: a read-only wrapper around jy_src.JY.search's module-level `search` name records each
     seed search's (operation, storage) so the refinement and seed stages can be reported separately.
  5. Python 3.12 sum(): JY ran on CPython 3.12.14 (compensated float sum()); we run 3.11.9 (naive).
     Float sums take part in decisions in two places: vendor_sm/core.path_length (CHOMP keeps the
     strictly shorter rasterised path; FOCAL incumbent bound) and JY.search.astar's pruning bound U (J3).
     Equal-length alternatives then tie-break differently in the last bit. _sum_py312 (a port of CPython
     3.12 builtin_sum_impl) is injected as module-global `sum` of vendor_sm.core (permanently; module used
     only here) and of JY.search (during J3 calls only). Without it: J1 997-999/1000 and J3 ~65-75% match.

extra['paper_operation'/'paper_storage'/'aux_operation'(/'aux_storage')] follow the LLM-A* Table 1
definitions for the method's MAIN search (selections from OPEN incl. goal and stale pops; states given a
g value); helper work (J2 FMM field, J3 seed searches, J4 landmark Dijkstras) goes to aux_*. Per-family
definitions are in the code below and in J_PORT.md.

Inputs: ctx['text'] = standard response (J1, J3, J6); J2 also ctx['replies']['k2_cot'/'k2_repe'].
finish_reason is recorded, never used to drop a response (JY J2 PROTOCOL l.9 and K2 adapter: truncated
responses are parsed as-is; a missing/failed parse becomes the straight start-goal proposal).
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path: sys.path.append(str(HERE))
if str(REPO) not in sys.path: sys.path.append(str(REPO))
JY_COMMIT = 'd72632c3e3ade5b9cd7e38a181caff4f51e3546e'


def _load_vendor_core():
    name = 'jy_src.vendor_sm.core'
    if name in sys.modules: return sys.modules[name]
    import jy_src.vendor_sm  # noqa: F401  (empty package __init__)
    spec = importlib.util.spec_from_file_location(name, HERE/'jy_src/vendor_sm/core.py')
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(REPO/'experiments/COMMON18/vendor_sm/core.py')   # shim 1: ROOT = parents[3] -> REPO
    sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None); raise
    assert module.ROOT == REPO and module.VENDOR.is_dir(), module.ROOT
    return module


def _sum_py312(iterable, start=0):
    """CPython 3.12 builtin sum() semantics (bltinmodule.c builtin_sum_impl): ints add exactly; from the
    first float on, floats use Neumaier compensation, ints are added uncompensated, and the compensation
    term is added at the end. Python 3.11 sum() adds floats naively."""
    items, total = iter(iterable), start
    if type(total) is not float:
        for x in items:
            total = total+x
            if type(total) is float: break
        else: return total
        if type(total) is not float: return total
    f, c = total, 0.0
    for x in items:
        if type(x) is float:
            t = f+x
            c += (f-t)+x if abs(f) >= abs(x) else (x-t)+f
            f = t
        elif type(x) in (int, bool): f += x
        else: raise TypeError(type(x).__name__)   # never needed by jy_src.JY.search
    return f+c if c and math.isfinite(c) else f


vcore = _load_vendor_core()
vcore.sum = _sum_py312   # shim 5: path_length() in CHOMP best-path choice and FOCAL incumbent bound
import jy_src.vendor_sm.candidate_core as cc   # noqa: E402
import jy_src.JY.search as jys                 # noqa: E402

DROP = {'path', 'valid', 'error', 'length', 'pop_trace', 'failure'}


def _plain_extra(r, rename_operation=True):
    extra = {k: v for k, v in r.items() if k not in DROP and (v is None or isinstance(v, (bool, int, float, str)))}
    if rename_operation and 'operation' in extra: extra['engine_operation'] = extra.pop('operation')   # JY's own pop count
    for k in ('total_seconds', 'query_seconds', 'search_seconds'):
        if k in extra: extra['engine_'+k] = extra.pop(k)
    return extra


def _finish_flags(ctx, keys):
    finish = ctx.get('finish', {}) or {}
    return {f'finish_{k}': finish.get(k) for k in keys}


# ---------------- J1 / J2: COMMON18 candidate_core FOCAL (w=1.05) ----------------

def _focal_result(grid, r, calls, extra, aux_operation):
    # paper_operation: FOCAL OPEN is a set scanned per selection (no heap pops for selections), so it is
    # selections from OPEN incl. the goal (= engine expansions) + stale anchor-heap discards.
    # paper_storage: finite-g states; the dense g array pre-fills all grid.n cells with inf (allocated_g_entries).
    extra.update(paper_operation=int(r['expansions'])+int(r['anchor_heap_pops']), paper_storage=int(r['discovered']),
                 aux_operation=int(aux_operation),
                 anchor_heap_pops=r['anchor_heap_pops'], incumbent_returned=r['incumbent_returned'],
                 allocated_g_entries=int(grid.n), graph_seconds=grid.seconds, engine_search_seconds=r['search_seconds'])
    path = [list(map(int, p)) for p in r['path']] if r['path'] else []
    return dict(path=path, error=None if path else (r.get('failure') or 'unreachable'), storage=int(r['discovered']),
                storage_definition='finite_g_entries_of_dense_array (JY discovered)',
                successor_expansions=int(r['expansions'])-int(r['valid'] and not r['incumbent_returned']),
                api_calls=calls, peak_open=int(r['peak_open']), extra=extra)


def _j1(llm):
    """backends.run family J1 / J1_straight: CHOMP-inspired repair of the (LLM | straight) polyline,
    repaired path = verified incumbent + point-segment distance secondary, FOCAL w=1.05."""
    def fn(q, ctx):
        raw = vcore.parse_path(ctx.get('text', ''))[0] if llm else []
        grid = cc.Grid(q)
        repaired, meta = cc.chomp_repair(q, raw if llm else [q['start'], q['goal']])
        incumbent = repaired or None
        secondary = cc.point_segment_distance(grid.xy, repaired) if repaired else None
        r = cc.focal(grid, q, secondary, weight=1.05, incumbent=incumbent)
        extra = dict(meta)
        if llm: extra['parsed_points'] = len(raw)
        return _focal_result(grid, r, int(llm), extra, aux_operation=0)   # CHOMP repair uses no queue
    return fn


def _j2(llm):
    """backends.run family J2 / J2_uniform: min distance field of 3 parsed proposals (standard, CoT, RePE),
    speed 2 within distance 2 else 1 (uniform speed 1 for the control), Fast Marching from goal as FOCAL
    secondary, w=1.05."""
    def fn(q, ctx):
        grid = cc.Grid(q); field = None; extra = {}
        if llm:
            replies = ctx.get('replies', {}) or {}
            texts = [ctx.get('text', ''), replies.get('k2_cot', ''), replies.get('k2_repe', '')]
            parsed = [vcore.parse_path(t) for t in texts]
            field, scale = cc.llm_field(grid, q, [p for p, _ in parsed])
            extra.update(parse_errors=[e for _, e in parsed], uncertainty_scale=scale,
                         missing_replies=[k for k in ('k2_cot', 'k2_repe') if k not in replies],
                         **_finish_flags(ctx, ('k2_cot', 'k2_repe')))
        speed = np.where(field <= 2., 2., 1.) if field is not None else None
        secondary, meta = cc.fast_marching(grid, q['goal'], speed)
        extra.update(meta)
        r = cc.focal(grid, q, secondary, weight=1.05, incumbent=None)
        return _focal_result(grid, r, 3 if llm else 0, extra, aux_operation=meta['field_heap_pops'])   # FMM field build
    return fn


# ---------------- J3-J6: experiments/JY/search.py run_once ----------------

def _run_recording_seeds(q, alias, text, index, trace):
    """J3 only.
    Shim 4: read-only wrapper around the module-level `search` name in jy_src.JY.search
    (= reproduction.core.search, used by run_once for the J3 seed and its g+2h fallback) to record each
    seed search's own (operation, storage); run_once only keeps the seed pop sum and the summed storage.
    Shim 5: JY ran on Python 3.12.14, whose sum() is compensated; ours is 3.11.9. astar() sets the pruning
    bound U = sum(dist) of the seed path and prunes on f >= U, so when the seed is already optimal the
    last-bit difference flips the tie (3.12: prune at the first pop; 3.11: full refinement). The 3.12 sum
    is injected as the module-global `sum` of jy_src.JY.search for the J3 call only (other JY engines use
    float sums only for reported lengths; ints are unaffected; J5's per-iteration int sum stays builtin)."""
    original, seeds = jys.search, []
    def recording(*args, **kwargs):
        res = original(*args, **kwargs); seeds.append((res.get('operation') or 0, res.get('storage') or 0)); return res
    jys.search = recording; jys.sum = _sum_py312
    try: r = jys.run_once(q, alias, text=text, index=index, trace=trace)
    finally:
        jys.search = original; del jys.sum
    return r, seeds


def _jy(alias, llm, landmarks=None, storage_definition='', trace=False):
    def fn(q, ctx):
        text = ctx.get('text', '') if llm else ''
        index = jys.LandmarkIndex(q, landmarks) if landmarks is not None else None
        if alias.startswith('j3_'): r, seeds = _run_recording_seeds(q, alias, text, index, trace)
        else: r = jys.run_once(q, alias, text=text, index=index, trace=trace)
        extra = _plain_extra(r); extra['jy_alias'] = alias
        if alias.startswith('j3_'):
            # main search = exact refinement A*; the seed search(es) (LLM-A* / g+2h, + fallback) are aux.
            assert sum(o for o, _ in seeds) == r['seed_operation'] and seeds, (seeds, r['seed_operation'])
            extra.update(paper_operation=int(r['refinement_operation']), paper_storage=int(r['storage'])-int(seeds[-1][1]),
                         aux_operation=int(r['seed_operation']), aux_storage=int(sum(s for _, s in seeds)),
                         seed_searches=len(seeds))
        elif index is not None:
            # main search = query A*; aux = landmark Dijkstras (graph build has no queue).
            extra.update(paper_operation=int(r['operation']), paper_storage=int(r['storage']),
                         aux_operation=int(index.pops), aux_storage=int(index.table_entries),
                         landmarks=[list(p) for p in index.landmarks], index_internal_seconds=index.seconds,
                         query_operation=r['operation'])
        else:
            # J5: all pops of the bidirectional loop (expansion pops + waiting->ready promotion pops + stale);
            #     storage = forward+backward g. J6: all pops incl. stale and lazy-refresh pops; storage = len(g).
            extra.update(paper_operation=int(r['operation']), paper_storage=int(r['storage']), aux_operation=0)
        path = [list(p) for p in r['path']] if r['path'] else []
        peak = r.get('peak_heap_entries')
        return dict(path=path, error=r.get('error') if not path else None, storage=int(r['storage']),
                    storage_definition=storage_definition, successor_expansions=int(r['expansions']),
                    api_calls=int(llm), extra=extra, **({'peak_open': int(peak)} if peak is not None else {}))
    return fn


J3_STORAGE = 'seed_gm_g_dictionary_including_infinite + refinement_finite_g_dictionary (JY two-stage sum)'
J4_STORAGE = 'query_finite_g_dictionary (landmark tables/graph excluded; see extra landmark_entries, graph_nodes, graph_edges)'
J5_STORAGE = 'forward+backward finite_g_dictionaries (JY sum)'
J6_STORAGE = 'finite_g_dictionary'


def spec(fn, family, bound, role, jy_condition, uses_llm, note):
    return dict(fn=fn, bound=bound, family=family, note=note, role=role, jy_condition=jy_condition, uses_llm=uses_llm)


METHODS = {
    # J1 - J1_full/PROTOCOL.md l.11-13: main J1 w=1.05; direct contribution control J1/J1_straight.
    'J1_main': spec(_j1(True), 'J1', 1.05, 'main', 'J1', True, 'CHOMP-inspired repair of LLM polyline + FOCAL w=1.05 (COMMON18 engine)'),
    'J1_straight': spec(_j1(False), 'J1', 1.05, 'control', 'J1_straight', False, 'same repair+FOCAL, straight start-goal initialisation, no API'),
    # J2 - J2_full/PROTOCOL.md l.14-15: main J2 w=1.05; direct control J2/J2_uniform (uniform-speed FMM).
    'J2_main': spec(_j2(True), 'J2', 1.05, 'main', 'J2', True, '3-response (standard/CoT/RePE) speed field -> Fast Marching secondary, FOCAL w=1.05'),
    'J2_uniform': spec(_j2(False), 'J2', 1.05, 'control', 'J2_uniform', False, 'same FMM with uniform speed 1, FOCAL w=1.05, no API'),
    # J3 - J3_full/PROTOCOL.md l.10-13: main = LLM-A* seed (g+2h fallback) + exact A* refinement with U pruning.
    'J3_main': spec(_jy('j3_llm_prune', True, storage_definition=J3_STORAGE), 'J3', 1.0, 'main', 'J3', True, 'LLM-A* seed path as upper bound U, octile A* refinement with g+h>=U pruning'),
    'J3_no_prune': spec(_jy('j3_llm_noprune', True, storage_definition=J3_STORAGE), 'J3', 1.0, 'control', 'J3_no_prune', True, 'same LLM seed + refinement, pruning removed'),
    'J3_noLLM': spec(_jy('j3_weight_prune', False, storage_definition=J3_STORAGE), 'J3', 1.0, 'control', 'J3_noLLM', False, 'g+2h seed + same pruned refinement, no API'),
    # J4 - J4_full/PROTOCOL.md l.7-12: main J4 = 4 landmarks, index rebuilt per query; GraphOctile = same graph/engine, 0 landmarks.
    'J4_main': spec(_jy('j4_alt4', False, landmarks=4, storage_definition=J4_STORAGE), 'J4', 1.0, 'main', 'J4', False, 'ALT, 4 graph-farthest landmarks, per-query cold preprocessing, no LLM'),
    'J4_graph_octile': spec(_jy('j4_alt4', False, landmarks=0, storage_definition=J4_STORAGE), 'J4', 1.0, 'control', 'GraphOctile', False, 'same per-query graph + A* engine, octile only (0 landmarks)'),
    # J5 - J5_full/PROTOCOL.md l.6,9-10: main J5 = octile NBS; J5_bidijkstra = same bidirectional function without heuristic.
    'J5_main': spec(_jy('j5_nbs_octile', False, storage_definition=J5_STORAGE), 'J5', 1.0, 'main', 'J5', False, 'NBS with octile heuristic, no LLM'),
    'J5_bidijkstra': spec(_jy('j5_bidijkstra', False, storage_definition=J5_STORAGE), 'J5', 1.0, 'control', 'J5_bidijkstra', False, 'same bidirectional function, Dijkstra (no heuristic)'),
    # J6 - J6_full/PROTOCOL.md l.7-11: main J6 = lazy certified rekey; J6_full = full rekey+heapify. No optimality guarantee.
    'J6_main': spec(_jy('j6_lazy', True, storage_definition=J6_STORAGE), 'J6', None, 'main', 'J6', True, 'clean guided LLM-A* engine, lazy epoch/offset rekey on target change'),
    'J6_full': spec(_jy('j6_full', True, storage_definition=J6_STORAGE), 'J6', None, 'control', 'J6_full', True, 'same guided engine, full OPEN rekey + heapify on target change'),
}
