"""Measured run() wrapper — SM unified protocol (sm_src/algorithms.py) with a method registry.

The measurement policy is unchanged from SM: heapq is patched to count pops/pushes, the
whole planner call plus the mandatory final grid validation is inside the timer, and
paths are re-validated with the GM geometry. Baseline branches are copied verbatim.

Adapter contract (methods_k12.py / methods_k456.py):
    METHODS = {name: dict(fn=callable(q, ctx) -> dict, bound=float|None, family='K1'..'K6', note=str)}
    ctx = dict(text=<standard first response text>, replies={variant: text}, fold=int,
               calibration=<fold calibration dict or None>, seed=int)
    The returned dict must contain:
        path (list of [x,y] integer grid points, [] on failure), error (None or str, optional),
        storage (int), storage_definition (str), successor_expansions (int), api_calls (int)
    Optional: length, peak_open, extra (dict, JSON-serialisable), operation_note (str).
    Engines must call heapq.heappop/heappush through the module attribute (``import heapq``),
    never ``from heapq import heappop``, so the shared HeapCounts counter sees every pop.
    All per-query preparation (parsing, field construction, ranking parse ...) must happen
    inside fn so it is timed. Fold-level calibration/selection is done beforehand and passed in ctx.
"""
import heapq
import math
import time
from collections import Counter
from common import *
import cv_candidate_core as candidate
import next_search_core as exact

class CountGeometry(gm.Geometry):
    def __init__(self, q):
        super().__init__(q); self.collision_calls = 0; self.expansions = 0; self.expanded = set()
    def collision(self, a, b):
        self.collision_calls += 1; return super().collision(a, b)
    def neighbors(self, s):
        self.expansions += 1; self.expanded.add(s); return super().neighbors(s)

class HeapCounts:
    def __enter__(self):
        self.counts = Counter(); self.original = {k: getattr(heapq, k) for k in ('heappop', 'heappush', 'heapify')}
        def pop(h):
            self.counts['heap_pops'] += 1; self.counts['max_single_heap_entries'] = max(self.counts['max_single_heap_entries'], len(h)); return self.original['heappop'](h)
        def push(h, x):
            self.counts['heap_pushes'] += 1; value = self.original['heappush'](h, x)
            self.counts['max_single_heap_entries'] = max(self.counts['max_single_heap_entries'], len(h)); return value
        def heapify(h): self.counts['heap_rebuilds'] += 1; self.counts['heapify_entries'] += len(h); return self.original['heapify'](h)
        heapq.heappop = pop; heapq.heappush = push; heapq.heapify = heapify; return self
    def __exit__(self, *args):
        for k, v in self.original.items(): setattr(heapq, k, v)

BASE = ['A_GM', 'N_g2h', 'B_llm', 'K3_wall_euclidean', 'K3_wall_octile', 'A_OCT', 'FOCAL_105']
BASE_BOUND = {'A_GM': 1., 'A_OCT': 1., 'FOCAL_105': 1.05}
BASE_FAMILY = {'K3_wall_euclidean': 'K3', 'K3_wall_octile': 'K3'}

def registry():
    reg = {}
    for module in ('methods_k12', 'methods_k456'):
        try: mod = __import__(module)
        except ModuleNotFoundError as exc:
            if exc.name == module: continue
            raise
        for name, spec in mod.METHODS.items():
            assert name not in reg and name not in BASE, name
            reg[name] = spec
    return reg

EXTRA = registry()
METHODS = BASE+list(EXTRA)

def bound(method):
    return BASE_BOUND.get(method) if method in BASE else EXTRA[method].get('bound')

def family(method):
    return BASE_FAMILY.get(method, 'baseline') if method in BASE else EXTRA[method]['family']

def run(q, method, ctx):
    gm.Geometry.neighbors.cache_clear(); meta = {}; calls = 0; text = ctx.get('text', '')
    with HeapCounts() as hc:
        started = time.perf_counter()
        if method in ('A_GM', 'N_g2h', 'B_llm', 'K3_wall_euclidean', 'K3_wall_octile'):
            geo = CountGeometry(q); heuristic = None
            if method == 'B_llm': r = gm.search(q, geo, 'llm_astar', text); calls = 1
            else:
                if method.startswith('K3_'): heuristic = LowerBound(geo, q['goal'], base='euclidean' if method.endswith('euclidean') else 'octile')
                r = gm.search(q, geo, 'astar' if method == 'A_GM' else 'goal_only', goal_heuristic=heuristic)
            expansions = geo.expansions; meta.update(collision_calls=geo.collision_calls, unique_successor_expansions=len(geo.expanded))
            if heuristic: meta.update(heuristic.stats())
            storage = r.get('storage'); storage_definition = 'g_dictionary_including_infinite'
        elif method == 'A_OCT':
            graph = exact.Grid(q); r = exact.search(graph, q); expansions = r['counters'].get('expansions', 0)-int(r['reachable'])
            storage = r['discovered']; storage_definition = 'finite_g_dictionary'; meta.update(r['counters'], graph_seconds=graph.seconds)
        elif method == 'FOCAL_105':
            graph = candidate.Grid(q); meta.update(graph_seconds=graph.seconds, allocated_g_entries=graph.n)
            r = candidate.focal(graph, q, secondary=None, weight=1.05)
            expansions = r['expansions']-int(r['valid']); storage = graph.n; storage_definition = 'allocated_dense_g_array_entries'
        else:
            r = EXTRA[method]['fn'](q, ctx)
            expansions = r['successor_expansions']; storage = r['storage']; storage_definition = r['storage_definition']
            calls = r.get('api_calls', 0); meta.update(r.pop('extra', {}) or {})
        path = r['path']; geo_check = gm.Geometry(q)
        error = gm.validate_path(geo_check, path, q['start'], q['goal'], grid=True) if path else r.get('error') or r.get('failure') or 'unreachable'
        if path and any(any(not float(c).is_integer() for c in p) for p in path): error = 'non_integer_grid_point'
        valid = error is None
        elapsed = time.perf_counter()-started
    gm.Geometry.neighbors.cache_clear()
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:])) if valid and path else None
    r.update(valid=valid, error=error, length=length, operation=hc.counts['heap_pops'], successor_expansions=expansions, storage=storage,
             storage_definition=storage_definition, total_seconds=elapsed, api_calls=calls, heap_counts=dict(hc.counts), extra=meta,
             measured_peak_open=r.get('peak_open', hc.counts['max_single_heap_entries']),
             peak_open_definition='engine_logical_open' if 'peak_open' in r else 'max_single_heap_entries_including_duplicates')
    return r
