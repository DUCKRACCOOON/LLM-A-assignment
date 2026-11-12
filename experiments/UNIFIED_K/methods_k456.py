"""K4/K5/K6 adapters for the unified harness.

Each adapter calls gm's original run_once (experiments/K4|K5|K6|K6_w105/search.py, unchanged)
with the method id used as PRIMARY / control in the 2026-10-25 gm experiments. run_once builds
the Geometry, parses the LLM text and prepares guidance inside the call, so all per-query
preparation is timed by harness.run. All three engines use ``import heapq`` and call
``heapq.heappop``/``heappush``/``heapify`` through the module, so harness.HeapCounts sees
every pop (no shim needed). run_once also validates its own path (gm's mandatory check);
harness then repeats the common SM final validation — both inside the timer.

successor_expansions: real neighbor expansions (goal selection excluded), the same
definition gm used in K5/K6 (`successor_expansions`) and K4 (`expansions - valid`).
storage: len(g) dictionary including infinite placeholders, as in gm engines.
"""
import hashlib
from experiments.K4.search import run_once as k4_run
from experiments.K5.search import run_once as k5_run
from experiments.K6.search import run_once as k6_run
from experiments.K6_w105.search import run_once as k6t_run

STORAGE_DEF = 'g_dictionary_including_infinite'
EXTRA_KEYS = ('operation', 'unique_pops', 'reexpansions', 'reopenings', 'stale_pops', 'open_pops', 'waiting_pops',
              'focal_pops', 'search_pops', 'reward_pops', 'queue_expansions', 'heap_pushes', 'heap_rebuilds',
              'rekey_scans', 'rekey_discarded', 'rekey_entries', 'peak_heap_entries', 'eligibility_rebuilds',
              'rebuild_scans', 'guide_error', 'guide_segments', 'guide_segment_checks', 'guide_calls',
              'guide_raw_points', 'guide_kept_points', 'guide_distance_checks', 'wall_checks', 'alpha_updates',
              'target_changes', 'guide_disables', 'goal_g', 'bound_factor', 'bound_limit', 'goal_queue',
              'infinite_g', 'queue_labels', 'arm_pulls', 'final_alpha')

def k5_seed(case_id, index):
    """Fixed per case and seed index (fold-independent: each case is evaluated in exactly one fold)."""
    return int.from_bytes(hashlib.sha256(f'UNIFIED_K|K5|20261112|{case_id}|{index}'.encode()).digest()[:8], 'big')

def adapt(result, uses_llm, expansions_key):
    path = [list(p) for p in result.get('path') or []]
    if expansions_key == 'expansions':
        expansions = result.get('expansions', 0)-int(bool(result.get('valid')))
    else:
        expansions = result.get('successor_expansions', 0)
    extra = {k: result[k] for k in EXTRA_KEYS if k in result}
    extra['engine_operation'] = extra.pop('operation', None)   # gm's own pop definition, kept for parity
    extra['engine_seconds'] = result.get('total_seconds')
    return dict(path=path, error=result.get('error'), storage=result.get('storage', 0), storage_definition=STORAGE_DEF,
                successor_expansions=expansions, api_calls=int(uses_llm), extra=extra,
                **({'peak_open': result['peak_heap_entries']} if 'peak_heap_entries' in result else {}))

def make(runner, gm_method, uses_llm, expansions_key, seed_index=None):
    def fn(q, ctx):
        text = ctx.get('text', '') if uses_llm else ''
        if seed_index is None: result = runner(q, gm_method, text)
        else:
            seed = k5_seed(q['id'], seed_index)
            result = runner(q, gm_method, text, seed=seed)
            result['seed'] = seed
        out = adapt(result, uses_llm, expansions_key)
        if seed_index is not None: out['extra'].update(seed=result['seed'], seed_index=seed_index)
        out['extra']['gm_method'] = gm_method
        return out
    return fn

def spec(runner, gm_method, family, bound, uses_llm, expansions_key, note, seed_index=None):
    return dict(fn=make(runner, gm_method, uses_llm, expansions_key, seed_index), bound=bound, family=family,
                note=note, gm_method=gm_method, uses_llm=uses_llm)

METHODS = {
    # K4: FOCAL with LLM polyline secondary, 5% bound (gm PRIMARY 'focal_llm_w105').
    'K4_llm': spec(k4_run, 'focal_llm_w105', 'K4', 1.05, True, 'expansions', 'gm K4 PRIMARY'),
    'K4_noLLM': spec(k4_run, 'focal_goal_w105', 'K4', 1.05, False, 'expansions', 'same FOCAL engine, secondary = wall lower bound (no LLM)'),
    # K5: shared MHA* (anchor + wall + goal + LLM queues), DTS scheduler, w=2 (gm PRIMARY 'mha_dts_w200').
    'K5_dts_s0': spec(k5_run, 'mha_dts_w200', 'K5', 2.0, True, 'successor_expansions', 'gm K5 PRIMARY, seed index 0', 0),
    'K5_dts_s1': spec(k5_run, 'mha_dts_w200', 'K5', 2.0, True, 'successor_expansions', 'gm K5 PRIMARY, seed index 1', 1),
    'K5_dts_s2': spec(k5_run, 'mha_dts_w200', 'K5', 2.0, True, 'successor_expansions', 'gm K5 PRIMARY, seed index 2', 2),
    'K5_rr': spec(k5_run, 'mha_rr_w200', 'K5', 2.0, True, 'successor_expansions', 'same queues, fixed round-robin instead of DTS (deterministic)'),
    'K5_noLLM': spec(k5_run, 'mha_dts_nollm_w200', 'K5', 2.0, False, 'successor_expansions', 'DTS without the LLM queue, seed index 0', 0),
    # K6: adaptive waypoint weight α, w=2 (gm PRIMARY 'adaptive_wall_w200'); 5% follow-up engine (K6_w105).
    'K6_main': spec(k6_run, 'adaptive_wall_w200', 'K6', 2.0, True, 'successor_expansions', 'gm K6 PRIMARY'),
    'K6_w105': spec(k6t_run, 'adaptive_wall_w105', 'K6', 1.05, True, 'successor_expansions', 'gm K6_w105 PRIMARY (5% bound)'),
    'K6_w105_noguide': spec(k6t_run, 'no_guide_wall_w105', 'K6', 1.05, False, 'successor_expansions', 'same 5% engine, targets=[goal], α off'),
}
