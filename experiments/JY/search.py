"""Fresh search engines for J3--J6. Upstream reproduction is never modified.

All online edges come from reproduction.core.Geometry. Independent verification
uses the integer orientation implementation in experiments.K3.verification.
"""
import heapq
import math
import time

from reproduction.core import Geometry, MOVES, parse_output, search, validate_path
from experiments.K3.heuristic import LowerBound, octile, run_once as run_k3

QUANTUM = 10**9
METHODS = ('legacy_astar', 'legacy_llm', 'legacy_g2h', 'k3_wall',
           'astar_octile', 'astar_wall', 'clean_g2h',
           'j3_weight_noprune', 'j3_weight_prune', 'j3_llm_noprune', 'j3_llm_prune',
           'j4_alt4', 'j4_alt8', 'j5_bidijkstra', 'j5_nbs_euclidean', 'j5_nbs_octile',
           'j6_full', 'j6_lazy')
EXACT = set(METHODS) - {'legacy_llm', 'legacy_g2h', 'k3_wall', 'clean_g2h', 'j6_full', 'j6_lazy'}


def fresh_geometry(query):
    Geometry.neighbors.cache_clear()
    return Geometry(query)


def path_to(parent, node):
    path = [node]
    while parent[path[-1]] != path[-1]:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def finish(query, path, stats):
    geo = Geometry(query)
    error = validate_path(geo, path, query['start'], query['goal'], grid=True) if path else 'unreachable'
    return dict(stats, valid=error is None, error=error, path=path,
                length=sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else None)


def astar(query, neighbors, heuristic, weight=1.0, incumbent=None, prune=False):
    """Version-free stale-g checking, lower-g reopening, optional feasible bound.

    No discovered vertex is permanently banned by pruning: a later smaller g
    can still insert it. The incumbent path is independent of this search tree.
    """
    start, goal = tuple(query['start']), tuple(query['goal'])
    g, parent, expanded, hs = {start: 0.0}, {start: start}, {}, {}
    queue, pops, stale, expansions, relax, pruned, reopens = [], 0, 0, 0, 0, 0, 0
    best_path = [tuple(p) for p in incumbent] if incumbent else []
    upper = sum(math.dist(a, b) for a, b in zip(best_path, best_path[1:])) if best_path else math.inf

    def h(n):
        if n not in hs:
            hs[n] = heuristic(n)
        return hs[n]

    heapq.heappush(queue, (weight*h(start), start, 0.0))
    pushes, peak = 1, 1
    while queue:
        priority, node, cost = heapq.heappop(queue); pops += 1
        if cost != g[node] or expanded.get(node, math.inf) <= cost:
            stale += 1; continue
        if prune and priority >= upper:
            pruned += 1; break  # weight=1 for bound refinement only.
        if node == goal:
            best_path = path_to(parent, goal); break
        if node in expanded:
            reopens += 1
        expanded[node] = cost; expansions += 1
        for other, step in neighbors(node):
            relax += 1
            if not math.isfinite(step):
                continue
            candidate = cost + step
            if candidate >= g.get(other, math.inf):
                continue
            if prune and candidate+h(other) >= upper:
                pruned += 1; continue
            g[other], parent[other] = candidate, node
            heapq.heappush(queue, (candidate+weight*h(other), other, candidate)); pushes += 1
        peak = max(peak, len(queue))
    return finish(query, best_path, dict(operation=pops, expansions=expansions, storage=len(g),
                  stale_pops=stale, relax_attempts=relax, heap_pushes=pushes,
                  peak_heap_entries=peak, reopenings=reopens, pruned=pruned,
                  heuristic_evaluations=len(hs)))


class LandmarkIndex:
    """Undirected ALT, graph-farthest landmarks, including disconnected components.

    The first landmark is lexicographically smallest. Uncovered components have
    infinite distance and are selected first. No start, goal, or oracle is used
    to choose landmarks. Graph generation and ALL landmark Dijkstra costs count.
    """
    def __init__(self, query, count):
        started = time.perf_counter(); geo = fresh_geometry(query)
        self.graph = {}
        for x in range(geo.x0+1, geo.x1):
            for y in range(geo.y0+1, geo.y1):
                node = (x, y)
                if not geo.collision(node, node):
                    self.graph[node] = tuple((n, c) for n, c in geo.neighbors(node) if math.isfinite(c))
        self.landmarks, self.tables = [], []
        self.pops = self.expansions = self.pushes = self.relax = 0
        minimum = {n: math.inf for n in self.graph}
        for _ in range(min(count, len(self.graph))):
            remaining = [n for n in self.graph if n not in self.landmarks]
            landmark = min(remaining, key=lambda n: (-minimum[n], n))
            self.landmarks.append(landmark)
            dist = {landmark: 0.0}; queue = [(0.0, landmark)]; self.pushes += 1
            while queue:
                d, node = heapq.heappop(queue); self.pops += 1
                if d != dist[node]:
                    continue
                self.expansions += 1
                for other, step in self.graph[node]:
                    self.relax += 1; candidate = d+step
                    if candidate < dist.get(other, math.inf):
                        dist[other] = candidate; heapq.heappush(queue, (candidate, other)); self.pushes += 1
            self.tables.append(dist)
            for n, d in dist.items():
                minimum[n] = min(minimum[n], d)
        self.seconds = time.perf_counter()-started
        self.table_entries = sum(map(len, self.tables))
        self.edge_entries = sum(map(len, self.graph.values()))
        Geometry.neighbors.cache_clear()

    def heuristic(self, goal):
        pairs = [(t, t[goal]) for t in self.tables if goal in t]

        def h(n):
            return max([octile(n, goal), *[abs(t[n]-dg) for t, dg in pairs if n in t]])
        return h


def bidirectional(query, geometry, kind='nbs', metric=octile, audit=False):
    """NBS Algorithms 1--3 (Chen et al. 2017), or bidirectional Dijkstra.

    NBS uses waiting(f) and ready(g) queues, active versions and reopening.
    Four queue removals/promotions/stale pops are counted, not just expansions.
    Both directions use the original symmetric, undirected grid edges.
    """
    start, goal = tuple(query['start']), tuple(query['goal'])
    if start == goal:
        return finish(query, [start], dict(operation=0, expansions=0, storage=2, stale_pops=0,
                    relax_attempts=0, heap_pushes=0, queue_promotions=0, pairs=0, pair_audits=0))
    gs, parents = [{start: 0.0}, {goal: 0.0}], [{start: start}, {goal: goal}]
    active, versions, closed = [{}, {}], [{}, {}], [{}, {}]
    waiting, ready = [[], []], [[], []]
    hs = [{}, {}]; targets = [goal, start]
    stats = dict(operation=0, expansions=0, stale_pops=0, relax_attempts=0,
                 heap_pushes=0, queue_promotions=0, pairs=0, pair_audits=0,
                 reopenings=0, peak_heap_entries=0)
    upper, best_path, lower = math.inf, [], 0.0

    def h(d, n):
        if n not in hs[d]:
            hs[d][n] = metric(n, targets[d])
        return hs[d][n]

    def put(d, n):
        v = versions[d].get(n, 0)+1; versions[d][n] = v; active[d][n] = (v, 'w')
        c = gs[d][n]
        heapq.heappush(waiting[d], ((c+h(d, n)) if kind == 'nbs' else c, -c, n, v))
        stats['heap_pushes'] += 1

    def pop(heap):
        stats['operation'] += 1
        return heapq.heappop(heap)

    def clean(d, place):
        heap = waiting[d] if place == 'w' else ready[d]
        while heap and active[d].get(heap[0][-2]) != (heap[0][-1], place):
            pop(heap); stats['stale_pops'] += 1

    def update(n):
        nonlocal upper, best_path
        if n in gs[0] and n in gs[1] and gs[0][n]+gs[1][n] < upper:
            upper = gs[0][n]+gs[1][n]
            best_path = path_to(parents[0], n)+list(reversed(path_to(parents[1], n)))[1:]

    def expand(d, n):
        active[d].pop(n)
        c = gs[d][n]
        if n in closed[d]: stats['reopenings'] += 1
        closed[d][n] = c; stats['expansions'] += 1; update(n)
        for other, step in geometry.neighbors(n):
            stats['relax_attempts'] += 1
            if not math.isfinite(step): continue
            candidate = c+step
            if candidate < gs[d].get(other, math.inf):
                gs[d][other] = candidate; parents[d][other] = n; update(other); put(d, other)

    def promote(d):
        _, _, n, v = pop(waiting[d]); active[d][n] = (v, 'r')
        heapq.heappush(ready[d], (gs[d][n], n, v)); stats['heap_pushes'] += 1
        stats['queue_promotions'] += 1

    put(0, start); put(1, goal)
    while active[0] and active[1]:
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], sum(map(len, waiting+ready)))
        if kind == 'dijkstra':
            for d in (0, 1): clean(d, 'w')
            if not all(waiting): break
            if waiting[0][0][0]+waiting[1][0][0] >= upper: break
            d = min((0, 1), key=lambda k: (waiting[k][0][0], k))
            _, _, n, _ = pop(waiting[d]); expand(d, n)
            continue
        # Promote ALL equal-bound entries as well so the minimum-g tie order
        # agrees with Algorithm 1's Cartesian reference (rather than the
        # optional high-g equal-f tie strategy in Algorithm 3's discussion).
        for d in (0, 1):
            clean(d, 'w'); clean(d, 'r')
            while waiting[d] and waiting[d][0][0] <= lower:
                promote(d); clean(d, 'w')
        while True:
            for d in (0, 1): clean(d, 'w'); clean(d, 'r')
            if not active[0] or not active[1]: break
            eligible = [d for d in (0, 1) if waiting[d] and waiting[d][0][0] <= lower]
            if eligible:
                promote(eligible[0]); continue
            if all(ready) and ready[0][0][0]+ready[1][0][0] <= lower: break
            values = [q[0][0] for q in waiting if q]
            if all(ready): values.append(ready[0][0][0]+ready[1][0][0])
            if not values: break
            lower = min(values)
        if not all(ready) or lower >= upper: break
        nodes = [ready[d][0][1] for d in (0, 1)]
        if audit:
            pairs = [(max(gs[0][u]+h(0,u), gs[1][v]+h(1,v), gs[0][u]+gs[1][v]),
                      gs[0][u], gs[1][v], u, v) for u in active[0] for v in active[1]]
            best_lb = min(p[0] for p in pairs)
            # Float sums can differ by 2e-15 for mathematically identical f.
            # Audit minimum bound to tolerance, then EXACT feasible-threshold
            # g tie ordering used by the queue implementation.
            best = min((p[1],p[2],p[3],p[4]) for p in pairs if p[0] <= lower)
            chosen = (max(gs[0][nodes[0]]+h(0,nodes[0]), gs[1][nodes[1]]+h(1,nodes[1]),
                          gs[0][nodes[0]]+gs[1][nodes[1]]), gs[0][nodes[0]], gs[1][nodes[1]])
            assert math.isclose(chosen[0],best_lb,abs_tol=1e-8), (chosen,best_lb)
            assert chosen[1:] == best[:2], (chosen,best)
            stats['pair_audits'] += 1
        # Select the pair BEFORE expanding either side.
        for d in (0, 1): pop(ready[d])
        for d in (0, 1): expand(d, nodes[d])
        stats['pairs'] += 1
    stats['storage'] = sum(map(len, gs))
    stats['forward_storage'], stats['backward_storage'] = map(len, gs)
    return finish(query, best_path, stats)


def guided(query, geometry, text='', lazy=False, trace=False, force_targets=None):
    """J6 same clean engine, full rekey versus certified lazy metric bounds.

    Static cached waypoint list ONLY; there is no online waypoint generation.
    Active versions discard stale-g entries identically in both queue strategies.
    Original neighbor-triggered target changes (even before collision) are kept.
    Integer 1e-9 priority units make full/lazy ties exactly reproducible.

    A target change t->t' shifts EVERY old queue lower bound by at least
    -(ceil(1e9*dist(t,t'))+1), in O(1). Triangle inequality makes each bound safe.
    A stale top is recomputed and reinserted until a current-epoch top wins.
    """
    start, goal = tuple(query['start']), tuple(query['goal'])
    nodes, error = parse_output(text, upstream=True)
    if force_targets is not None:
        nodes, error = list(force_targets), None
    targets = geometry.filter_upstream(nodes) if error is None else []
    if not targets or targets[0] != start: targets.insert(0, start)
    if targets[-1] != goal: targets.append(goal)
    if start == goal:
        return finish(query, [start], dict(operation=1, expansions=0, storage=1,
            rekey_evaluations=0, target_changes=0, stale_pops=0, lazy_refreshes=0,
            heap_pushes=1, relax_attempts=0, peak_heap_entries=1, pop_trace=[]))
    ti, target, epoch, offset = 1, targets[1], 0, 0
    g, parent, closed, versions, active = {start: 0.0}, {start: start}, set(), {}, {}
    queue, hg, events = [], {}, []
    stats = dict(operation=0, expansions=0, storage=0, rekey_evaluations=0,
                 target_changes=0, stale_pops=0, lazy_refreshes=0, heap_pushes=0,
                 relax_attempts=0, peak_heap_entries=0, bound_shift_units=0)

    def score(n):
        if n not in hg: hg[n] = math.dist(n, goal)
        stats['rekey_evaluations'] += 1
        return round((g[n]+hg[n]+math.dist(n, target))*QUANTUM)

    def put(n):
        v = versions.get(n, 0)+1; versions[n] = v; active[n] = v
        heapq.heappush(queue, (score(n)-offset, n, v, epoch)); stats['heap_pushes'] += 1

    def change(new):
        nonlocal target, epoch, offset, queue
        old = target; target = new; epoch += 1; stats['target_changes'] += 1
        if lazy:
            shift = math.ceil(math.dist(old, new)*QUANTUM)+1
            offset -= shift; stats['bound_shift_units'] += shift
        else:
            queue = [(score(n), n, v, epoch) for n, v in active.items()]
            heapq.heapify(queue)

    put(start); final = []
    while queue:
        _, n, v, saved_epoch = heapq.heappop(queue); stats['operation'] += 1
        if active.get(n) != v:
            stats['stale_pops'] += 1; continue
        if lazy and saved_epoch != epoch:
            heapq.heappush(queue, (score(n)-offset, n, v, epoch))
            stats['heap_pushes'] += 1; stats['lazy_refreshes'] += 1; continue
        active.pop(n); closed.add(n)
        if trace: events.append(n)
        if n == goal:
            final = path_to(parent, goal); break
        stats['expansions'] += 1
        for other, step in geometry.neighbors(n):
            stats['relax_attempts'] += 1
            if other == target and target != goal:
                ti += 1
                if ti < len(targets): change(targets[ti])
            if other in closed or not math.isfinite(step): continue
            candidate = g[n]+step
            if candidate < g.get(other, math.inf):
                g[other], parent[other] = candidate, n; put(other)
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], len(queue))
    stats['storage'] = len(g); stats['guide_parse_error'] = error
    stats['pop_trace'] = events
    return finish(query, final, stats)


def run_once(query, method, text='', index=None, audit=False, trace=False):
    if method not in METHODS: raise ValueError(method)
    Geometry.neighbors.cache_clear()
    started = time.perf_counter()
    geometry = Geometry(query)
    if method in ('legacy_astar', 'legacy_g2h', 'legacy_llm'):
        old = {'legacy_astar':'astar','legacy_g2h':'goal_only','legacy_llm':'llm_astar'}[method]
        r = search(query, geometry, method=old, text=text)
        r['expansions'] = r['operation']-int(r['valid'])
        r['stale_pops'] = 0  # Original duplicates actually expand again.
    elif method == 'k3_wall':
        r = search(query, geometry, method='goal_only', goal_heuristic=LowerBound(geometry, query['goal'],base='euclidean'))
        r['expansions'] = r['operation']-int(r['valid']); r['stale_pops'] = 0
    elif method in ('astar_octile','astar_wall','clean_g2h'):
        h = LowerBound(geometry, query['goal']) if method == 'astar_wall' else lambda n: octile(n, query['goal'])
        r = astar(query, geometry.neighbors, h, weight=2 if method=='clean_g2h' else 1)
    elif method.startswith('j3_'):
        seed_start = time.perf_counter()
        seed = search(query, geometry, method='llm_astar' if 'llm' in method else 'goal_only', text=text)
        fallback = False
        if not seed['valid']:
            fallback = True
            failed = seed
            seed = search(query, geometry, method='goal_only')
            seed['operation'] += failed.get('operation') or 0
        seed_seconds = time.perf_counter()-seed_start
        h = lambda n: octile(n, query['goal'])
        refine = astar(query, geometry.neighbors, h, incumbent=seed['path'], prune=method.endswith('_prune'))
        r = dict(refine, seed_operation=seed['operation'], refinement_operation=refine['operation'],
                 seed_seconds=seed_seconds, seed_length=seed['length'], seed_fallback=fallback,
                 operation=seed['operation']+refine['operation'],
                 expansions=seed['operation']-int(seed['valid'])+refine['expansions'],
                 storage=seed['storage']+refine['storage'])
    elif method.startswith('j4_'):
        if index is None: raise ValueError('ALT index required; preprocessing is charged separately')
        goal = tuple(query['goal'])
        r = astar(query, lambda n:index.graph.get(n,()), index.heuristic(goal))
        r.update(preprocessing_seconds=index.seconds, preprocessing_pops=index.pops,
                 preprocessing_expansions=index.expansions, landmark_entries=index.table_entries,
                 graph_nodes=len(index.graph), graph_edges=index.edge_entries,
                 landmarks=index.landmarks)
    elif method.startswith('j5_'):
        kind = 'dijkstra' if method=='j5_bidijkstra' else 'nbs'
        metric = math.dist if method.endswith('euclidean') else octile
        r = bidirectional(query, geometry, kind=kind, metric=metric, audit=audit)
    else:
        r = guided(query, geometry, text, lazy=method=='j6_lazy', trace=trace)
    r.setdefault('preprocessing_seconds',0.0)
    r['query_seconds'] = time.perf_counter()-started
    r['total_seconds'] = r['query_seconds']+r['preprocessing_seconds']
    r['cold_operation'] = r['operation']+r.get('preprocessing_pops',0)
    r['cold_expansions'] = r['expansions']+r.get('preprocessing_expansions',0)
    r['new_api_calls'] = 0
    Geometry.neighbors.cache_clear()
    return r
