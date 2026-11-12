"""SM S1-S6 adapters for the unified harness (UNIFIED_K/harness.py contract).

Each fn reproduces one branch of SM's measured wrapper ``sm_src/algorithms.py::run``
(origin/sm 0c0a6082, sha 7718cf4c...) line for line, calling SM's own helpers
(algorithms.s1 / algorithms.delayed, reproduction.proposed.proposed_search,
next_geometry.Rectangles + next_search_core.search, next_refinement.solve,
cv_candidate_core.Grid/network_inputs/focal). harness.run supplies the parts SM's run
did around the branch (neighbors cache clear, HeapCounts, timer, final grid validation),
so the timed region is the same as in SM.

Timing policy (identical to SM evaluate.py):
  * model loading (torch.load of the fold's S4 checkpoints, untrained init) and file IO are
    outside the timer: ``prepare(fold)`` caches per fold; ``warmup(q, fold)`` runs one
    untimed forward pass per model for each new grid size (SM did this per case before
    timing). ``replies(q)`` -- called by evaluate18.context() before timing -- does both.
  * parsing, diagnosis, revision choice, graph build, inference, search, re-search,
    refinement validation are inside fn (hence inside the harness timer).

ctx keys used: text (SM first standard response), fold, replies['s2_feedback'],
replies['s2_control'] (S2 second responses; only exist when the first proposal is invalid).
All engines call heapq through the module attribute (verified: reproduction/core.py,
next_search_core.py, cv_candidate_core.py, algorithms.delayed), so HeapCounts sees all pops.
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UK = HERE.parent/'UNIFIED_K'
for p in (str(UK), ):
    if p not in sys.path: sys.path.insert(0, p)
import common as _uk_common                     # UNIFIED_K/common.py (puts sm_src + repo on sys.path)
from common import gm, digest, read, proposed_search
import cv_candidate_core as candidate
import next_search_core as exact
import next_refinement as refinement
from next_geometry import Rectangles

INPUTS_S = HERE/'inputs_s'
SMSRC = UK/'sm_src'

# SM's algorithms.py under a unique module name (avoids a top-level 'algorithms' collision).
_spec = importlib.util.spec_from_file_location('sm_algorithms', SMSRC/'algorithms.py')
sm_alg = importlib.util.module_from_spec(_spec); sys.modules['sm_algorithms'] = sm_alg; _spec.loader.exec_module(sm_alg)
CountGeometry = sm_alg.CountGeometry


def diagnosis(q, text):
    """Verbatim copy of sm_src/common.py::diagnosis (that module cannot be imported here: it reads SM run paths)."""
    nodes, error = gm.parse_output(text); geo = gm.Geometry(q)
    error = error or gm.validate_path(geo, nodes, q['start'], q['goal'])
    detail = dict(valid=error is None, error=error)
    if error == 'wrong_endpoints': detail.update(observed_start=nodes[0] if nodes else None, observed_goal=nodes[-1] if nodes else None, required_start=q['start'], required_goal=q['goal'])
    elif error == 'outside_boundary': detail['first_bad_point'] = next(dict(index=i, point=p) for i, p in enumerate(nodes) if not geo.inside(p))
    elif error == 'collision':
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            if geo.collision(a, b): detail['first_bad_segment'] = dict(index=i, segment=[a, b]); break
        if 'first_bad_segment' in detail:
            a, b = detail['first_bad_segment']['segment']
            for kind in ('horizontal_barriers', 'vertical_barriers'):
                for j, wall in enumerate(q[kind]):
                    single = dict(q, horizontal_barriers=[], vertical_barriers=[]); single[kind] = [wall]
                    if gm.Geometry(single).collision(a, b): detail['first_bad_wall'] = dict(kind=kind, index=j, wall=wall); break
                if 'first_bad_wall' in detail: break
    return detail

sm_alg.diagnosis = diagnosis   # algorithms.run references a global 'diagnosis' from SM common


# ---------------------------------------------------------------- fold artefacts (untimed)
_models = {}; _weights = {}; _warmed = set(); LOAD_SECONDS = {}

def prepare(fold):
    """Load the fold's S4 models (3 trained seeds + seeded untrained) and S6 selected weights. Untimed, cached."""
    if fold in _models: return _models[fold]
    import torch
    torch.set_num_threads(1); t = time.perf_counter(); models = {}
    base = INPUTS_S/'folds'/f'fold{fold}'
    for seed in (11, 22, 33):
        model = candidate.make_network()
        model.load_state_dict(torch.load(base/'models'/f'seed{seed}.pt', map_location='cpu', weights_only=True)); model.eval()
        models[f'S4_seed{seed}'] = model
    torch.manual_seed(20261111+fold); models['S4_untrained'] = candidate.make_network().eval()
    _models[fold] = models; _weights[fold] = read(base/'program_selection.json')['weights']
    LOAD_SECONDS[fold] = time.perf_counter()-t
    return models

def warmup(q, fold):
    """SM evaluate.py: one untimed forward pass per model for each new grid size (per fold process)."""
    import torch
    models = prepare(fold); size = (fold,)+tuple(q['range_x']+q['range_y'])
    if size in _warmed: return
    graph = candidate.Grid(q); x = torch.from_numpy(candidate.network_inputs(graph, q)[None])
    for model in models.values():
        with torch.no_grad(): model(x)
    _warmed.add(size)

def _record(q, variant):
    p = INPUTS_S/'raw'/'public'/f'{q["id"]}-{variant}.json'
    if not p.exists(): return None
    rec = read(p)
    assert rec['request_sha256'] == digest(rec['request']) and rec['response_sha256'] == digest(rec['response']), p
    choice = rec['response']['choices'][0]
    return choice['message'].get('content') or '', choice.get('finish_reason'), rec['response_sha256']

def replies(q):
    """Hook for evaluate18.context(): S2 second responses + untimed per-fold preparation.

    Returns {'s2_feedback': (text, finish_reason, response_sha256), 's2_control': (...)} for the
    936 cases whose first proposal failed SM's diagnosis; {} otherwise. finish_reason is NOT
    filtered (SM used the content even when finish_reason == 'length').
    """
    fold = q.get('fold_id', _uk_common.FOLD)
    prepare(fold); warmup(q, fold)
    out = {}
    for v in ('feedback', 'control'):
        r = _record(q, v)
        if r is not None: out[f's2_{v}'] = r
    return out


# ---------------------------------------------------------------- branches (timed by harness.run)
GM_STORAGE = 'g_dictionary_including_infinite'

def _gm_branch(kind):
    def fn(q, ctx):
        text = ctx.get('text', ''); geo = CountGeometry(q)
        if kind == 'V_GATE_GM': r = proposed_search(q, geo, text); calls = 1
        elif kind in ('S1', 'V_GATE_NORMALIZED'): r = sm_alg.s1(q, geo, text, kind == 'S1'); calls = 1
        elif kind == 'S3': r = sm_alg.delayed(q, geo, text); calls = r['api_calls']
        else:   # S2_feedback / S2_control
            diag = diagnosis(q, text); valid = diag['valid']
            used = text if valid else ctx['replies']['s2_'+kind.split('_')[1]]
            r = gm.search(q, geo, 'llm_astar', used); calls = 1+int(not valid)
            r.update(initial_proposal_valid=valid, final_proposal_valid=diagnosis(q, used)['valid'])
        # Paper Table 1: gm.search reports operation=len(closed) (every OPEN pop incl. duplicates and the goal pop)
        # and storage=len(g) (inf-initialised entries included). algorithms.delayed (S3) keeps the same closed
        # list but does not return it: every pop calls neighbors() except the goal pop, so len(closed) =
        # expansions + goal_popped (goal popped <=> path returned). All heap pops of these engines are main-OPEN pops.
        paper_op = r['operation'] if 'operation' in r else geo.expansions+int(bool(r['path']))
        r.update(successor_expansions=geo.expansions, storage=r.get('storage'), storage_definition=GM_STORAGE, api_calls=calls,
                 extra=dict(collision_calls=geo.collision_calls, unique_successor_expansions=len(geo.expanded),
                            paper_operation=paper_op, paper_storage=r.get('storage'), aux_operation=0))
        return r
    return fn

def s5(q, ctx):
    graph = Rectangles(q, variant='sparse')
    r = exact.search(graph, q)
    r.update(successor_expansions=r['counters'].get('expansions', 0)-int(r['reachable']), storage=r['discovered'],
             storage_definition='finite_g_dictionary', api_calls=0, extra=dict(r['counters'], graph_seconds=graph.seconds))
    # Plain A* on the compressed graph: heap_pops = all OPEN pops (stale incl., goal pop incl.); g dict = finite entries only.
    r['extra'].update(paper_operation=r['counters'].get('heap_pops', 0), paper_storage=r['discovered'], aux_operation=0)
    return r

def _s6(kind):
    def fn(q, ctx):
        fold = ctx['fold']; loaded_in_call = fold not in _weights
        if loaded_in_call: prepare(fold)   # only if the evaluator skipped prepare() (flagged below)
        program = _weights[fold]
        rounds = []; original = refinement.search
        def traced(*a, **kw):
            result = original(*a, **kw); rounds.append(result); return result
        refinement.search = traced
        rq = dict(q, id=f'20261111:{fold}:{q["id"]}') if kind == 'S6_random' else q
        try: r = refinement.solve(rq, mode='program' if kind in ('S6_program', 'S6_historical') else kind[3:],
                                  program=[.5, .3, .1, .1] if kind == 'S6_historical' else program)
        finally: refinement.search = original
        r['peak_open'] = max(x['peak_open'] for x in rounds)
        r.update(successor_expansions=sum(x['counters'].get('expansions', 0)-int(x['reachable']) for x in rounds),
                 storage=max(x['discovered'] for x in rounds), storage_definition='max_per_round_finite_g_dictionary', api_calls=0,
                 extra=dict(r['counters'], round_discovered=[x['discovered'] for x in rounds], round_peak_open=[x['peak_open'] for x in rounds]))
        # Every round is a re-search of the main algorithm (no helper search): OPEN pops summed over rounds.
        # Each round has its own fresh g dict; the union of states across rounds is not exposed by the engine,
        # so paper_storage = max per-round len(g) (= SM storage) and the per-round sum is given separately.
        # NB: solve() returns the LAST round's result dict after overwriting its 'counters' with the all-round
        # total, so rounds[-1]['counters'] is already the total. SM's successor_expansions (kept unchanged above,
        # it is a compared det field) therefore double-counts rounds 1..n-1 when n>1 rounds; the corrected
        # value is given as successor_expansions_corrected. paper_operation uses the total (no double count).
        r['extra'].update(paper_operation=r['counters'].get('heap_pops', 0),
                          successor_expansions_corrected=r['counters'].get('expansions', 0)-sum(int(x['reachable']) for x in rounds),
                          paper_storage=max(x['discovered'] for x in rounds),
                          paper_storage_sum_rounds=sum(x['discovered'] for x in rounds), aux_operation=0)
        if loaded_in_call: r['extra']['artefacts_loaded_in_timed_call'] = True
        return r
    return fn

def _s4(kind):
    def fn(q, ctx):
        import torch
        fold = ctx['fold']; loaded_in_call = fold not in _models
        models = prepare(fold)       # cached; loading here only if the evaluator skipped prepare() (flagged below)
        graph = candidate.Grid(q); prep = time.perf_counter()
        x = candidate.network_inputs(graph, q)
        with torch.no_grad(): secondary = -models[kind](torch.from_numpy(x[None])).sigmoid().numpy().ravel()
        meta = dict(inference_seconds=time.perf_counter()-prep, graph_seconds=graph.seconds, allocated_g_entries=graph.n)
        if loaded_in_call: meta['artefacts_loaded_in_timed_call'] = True
        r = candidate.focal(graph, q, secondary=secondary, weight=1.05)
        # FOCAL engine: each expansion (incl. goal) is an array selection from OPEN; the anchor heap is only
        # popped to discard stale entries (incl. entries of already-expanded nodes). Paper operation = both.
        # g is a dense array pre-filled with inf for every cell (= storage); paper_storage = cells given a finite g.
        meta.update(paper_operation=r['focal_array_selections']+r['anchor_heap_pops'],
                    paper_operation_selections=r['focal_array_selections'], paper_operation_stale_pops=r['anchor_heap_pops'],
                    paper_storage=r['discovered'], aux_operation=0)
        r.update(successor_expansions=r['expansions']-int(r['valid']), storage=graph.n,
                 storage_definition='allocated_dense_g_array_entries', api_calls=0, extra=meta)
        return r
    return fn


def spec(fn, family, bound, role, note):
    return dict(fn=fn, bound=bound, family=family, role=role, note=note)

METHODS = {
    # S1: normalisation + collision-free shortcut + validity gate. Controls: gate-only (V_GATE_*), B_llm.
    'V_GATE_GM': spec(_gm_branch('V_GATE_GM'), 'S1', None, 'control:S1', 'GM proposed_search validity gate only (no normalisation)'),
    'V_GATE_NORMALIZED': spec(_gm_branch('V_GATE_NORMALIZED'), 'S1', None, 'control:S1', 'same normalisation+gate as S1 without dedup/shortcut (direct S1 control)'),
    'S1': spec(_gm_branch('S1'), 'S1', None, 'main', 'S1 dedup/loop removal, collision-free shortcut, validity gate'),
    # S2: one revision of an invalid first proposal. Control: same extra call without the diagnosis report.
    'S2_feedback': spec(_gm_branch('S2_feedback'), 'S2', None, 'main', 'S2: invalid first proposal -> revised answer with external diagnosis (replies s2_feedback)'),
    'S2_control': spec(_gm_branch('S2_control'), 'S2', None, 'control:S2_feedback', 'same extra call, generic re-review without diagnosis (replies s2_control)'),
    # S3: g+2h, at most one delayed guidance injection. Controls: N_g2h (no guidance), B_llm (immediate).
    'S3': spec(_gm_branch('S3'), 'S3', None, 'main', 'S3 stall-triggered single guidance injection; api_calls=0 if never triggered'),
    # S4: per-fold trained path-probability net as FOCAL w=1.05 secondary. Controls: FOCAL_105, S4_untrained.
    'S4_seed11': spec(_s4('S4_seed11'), 'S4', 1.05, 'main', 'S4 fold model seed 11'),
    'S4_seed22': spec(_s4('S4_seed22'), 'S4', 1.05, 'main', 'S4 fold model seed 22'),
    'S4_seed33': spec(_s4('S4_seed33'), 'S4', 1.05, 'main', 'S4 fold model seed 33 (S4 = per-case mean of the three seeds)'),
    'S4_untrained': spec(_s4('S4_untrained'), 'S4', 1.05, 'control:S4', 'same net, untrained init torch.manual_seed(20261111+fold)'),
    # S5: sparse rectangle compression + exact search. Control: A_OCT (same backend).
    'S5': spec(s5, 'S5', 1.0, 'main', 'S5 sparse rectangles, exact octile search, grid path expansion'),
    # S6: wall-subset relaxation with counterexample refinement (<=4 rounds, full-wall fallback). Controls: S6_rule, S6_random, A_OCT.
    'S6_program': spec(_s6('S6_program'), 'S6', 1.0, 'main', 'S6 new LLM wall-score coefficients selected per fold (program_selection.json)'),
    'S6_rule': spec(_s6('S6_rule'), 'S6', 1.0, 'control:S6_program', 'initial walls = those crossing the start-goal segment'),
    'S6_random': spec(_s6('S6_random'), 'S6', 1.0, 'control:S6_program', 'same number of random initial walls, seed "20261111:<fold>:<case id>"'),
    'S6_historical': spec(_s6('S6_historical'), 'S6', 1.0, 'reference', 'historical coefficients [0.5,0.3,0.1,0.1] (exposed to public data; excluded from ranking)'),
}
