"""FOCAL with reopening and exact eligibility; no model/network dependency."""
import heapq
import math
import time

from reproduction.core import Geometry, parse_output, validate_path
from experiments.K3.heuristic import LowerBound, run_once as run_k3

EPS = 1e-10  # Numeric membership slack; final oracle verification uses 1e-8.
PRIMARY = 'focal_llm_w105'
METHODS = ('astar_euclidean', 'g2h_euclidean', 'g2h_wall_octile', 'g2h_wall_euclidean',
           'clean_wastar_wall', 'focal_goal_w105', 'focal_straight_w105', PRIMARY,
           'focal_llm_w100', 'focal_llm_w110', 'focal_llm_w200', 'focal_llm_euclidean_w105')


class PolylineGuide:
    """Nearest projection + remaining arclength. Never used in the lower bound."""
    def __init__(self, query, text='', straight=False):
        try:
            points, self.error = ([], None) if straight else parse_output(text)
        except OverflowError:
            points, self.error = [], 'nonfinite_guide_geometry'
        self.raw_points = len(points)
        self.calls = self.segment_checks = 0
        self.segments = []
        self.cache = {}
        if self.error:
            return
        points = [tuple(query['start']), *points, tuple(query['goal'])]
        lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
        total = sum(lengths)
        if not math.isfinite(total) or any(not math.isfinite(d*d) for d in lengths):
            self.error = 'nonfinite_guide_geometry'
            return
        suffix = total
        for a, b, length in zip(points, points[1:], lengths):
            suffix -= length
            if length:
                self.segments.append((a, (b[0]-a[0], b[1]-a[1]), length, max(0.0, suffix)))

    def __call__(self, node):
        if node in self.cache:
            return self.cache[node]
        self.calls += 1
        best = (math.inf, math.inf)
        for a, delta, length, suffix in self.segments:
            self.segment_checks += 1
            t = max(0.0, min(1.0, ((node[0]-a[0])*delta[0]+(node[1]-a[1])*delta[1])/(length*length)))
            distance = math.hypot(node[0]-a[0]-t*delta[0], node[1]-a[1]-t*delta[1])
            remaining = (1-t)*length+suffix
            best = min(best, (distance, remaining))
        value = sum(best) if self.segments else 0.0
        self.cache[node] = value
        return value


def graph_search(start, goal, neighbors, heuristic, secondary, *, weight=1.05, focal=True,
                 trace=False, audit=False):
    """Generic engine. audit=True checks selection against a full-scan reference.

    Active versions are authoritative. OPEN tracks min(g+h); waiting contains
    ineligible entries; FOCAL orders eligible entries by a static guide. A lower
    threshold rebuilds eligibility, so admissible inconsistent h is supported.
    """
    if not math.isfinite(weight) or weight < 1:
        raise ValueError('weight must be finite and >= 1')
    g, parent, active, versions, closed = {start: 0.0}, {start: start}, {}, {}, {}
    hcache, scache, queues = {}, {}, {'open': [], 'waiting': [], 'focal': []}
    stats = dict(open_pops=0, waiting_pops=0, focal_pops=0, heap_pushes=0, stale_pops=0,
                 expansions=0, reopenings=0, eligibility_rebuilds=0, rebuild_scans=0,
                 peak_heap_entries=0, promotions=0, selection_audits=0)
    events = []
    threshold, certificate = -math.inf, None

    def h(node):
        if node not in hcache:
            hcache[node] = heuristic(node)
            assert math.isfinite(hcache[node]) and hcache[node] >= 0
        return hcache[node]

    def key(node, version):
        if node not in scache:
            scache[node] = h(node) if secondary is heuristic else secondary(node)
            assert math.isfinite(scache[node])
        return (scache[node], h(node), g[node]+h(node), node, version)

    def push(which, entry):
        heapq.heappush(queues[which], entry)
        stats['heap_pushes'] += 1
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'], sum(map(len, queues.values())))

    def pop(which):
        stats[which+'_pops'] += 1
        return heapq.heappop(queues[which])

    def current(entry):
        return active.get(entry[-2]) == entry[-1]

    def put(node):
        version = versions.get(node, 0)+1
        versions[node] = active[node] = version
        f = g[node]+h(node)
        push('open', ((f if focal else g[node]+weight*h(node)), node, version))
        if focal:
            if f <= threshold+EPS:
                push('focal', key(node, version))
            else:
                push('waiting', (f, node, version))

    put(start)
    while active:
        if focal:
            while queues['open'] and not current(queues['open'][0]):
                pop('open'); stats['stale_pops'] += 1
            assert queues['open']
            fmin = queues['open'][0][0]
            new_threshold = weight*fmin
            if new_threshold < threshold:
                # Reclassification is counted; no stale priorities can retain
                # illegal membership when a lower-g path reduces the threshold.
                queues['waiting'].clear(); queues['focal'].clear()
                stats['eligibility_rebuilds'] += 1
                for node, version in active.items():
                    stats['rebuild_scans'] += 1
                    f = g[node]+h(node)
                    push('focal', key(node, version)) if f <= new_threshold+EPS else push('waiting', (f, node, version))
            threshold = new_threshold
            while queues['waiting']:
                entry = queues['waiting'][0]
                if not current(entry):
                    pop('waiting'); stats['stale_pops'] += 1
                elif entry[0] <= threshold+EPS:
                    _, node, version = pop('waiting')
                    stats['promotions'] += 1
                    push('focal', key(node, version))
                else:
                    break
            while queues['focal'] and not current(queues['focal'][0]):
                pop('focal'); stats['stale_pops'] += 1
            assert queues['focal'], 'Nonempty OPEN must have a FOCAL candidate'
            entry = pop('focal')
            node, version = entry[-2:]
            assert g[node]+h(node) <= threshold+EPS
            if audit:
                expected = min(key(n, v) for n, v in active.items() if g[n]+h(n) <= threshold+EPS)
                assert entry == expected, (entry, expected)
                assert math.isclose(fmin, min(g[n]+h(n) for n in active), abs_tol=EPS)
                stats['selection_audits'] += 1
        else:
            entry = pop('open')
            if not current(entry):
                stats['stale_pops'] += 1
                continue
            node, version = entry[-2:]
            fmin = None
        del active[node]
        stats['expansions'] += 1
        closed[node] = g[node]
        if trace:
            events.append(dict(node=node, g=g[node], f=g[node]+h(node), fmin=fmin,
                               limit=threshold if focal else None))
        if node == goal:
            certificate = fmin
            break
        for neighbor, cost in neighbors(node):
            assert cost > 0
            if neighbor not in g:
                g[neighbor] = math.inf
            candidate = g[node]+cost
            if candidate < g[neighbor]:
                if neighbor in closed and neighbor not in active:
                    stats['reopenings'] += 1
                g[neighbor], parent[neighbor] = candidate, node
                put(neighbor)
    path = []
    if goal in closed:
        node = goal
        path = [node]
        while node != start:
            node = parent[node]; path.append(node)
        path.reverse()
    result = dict(path=path, goal_g=g.get(goal) if path else None,
                  operation=sum(stats[k] for k in ('open_pops', 'waiting_pops', 'focal_pops')),
                  storage=len(g), unique_pops=len(closed), reexpansions=stats['expansions']-len(closed),
                  infinite_g=sum(math.isinf(v) for v in g.values()),
                  lower_bound_at_goal=certificate, bound_factor=weight if focal else None,
                  bound_limit=weight*certificate if certificate is not None else None, **stats)
    if trace:
        result['trace'] = events
    return result


def run_once(query, method, text='', trace=False, audit=False):
    if method not in METHODS:
        raise ValueError(method)
    if method in METHODS[:4]:
        result = run_k3(query, method, trace=trace)
        result.update(expansions=result['operation'], open_pops=result['operation'], waiting_pops=0,
                      focal_pops=0, guide_segment_checks=0, guide_error=None,
                      bound_factor=None, lower_bound_at_goal=None, bound_limit=None)
        return result
    Geometry.neighbors.cache_clear()
    started = time.perf_counter()
    geometry = Geometry(query)
    goal, start = tuple(query['goal']), tuple(query['start'])
    walls = 'euclidean' not in method
    lower = LowerBound(geometry, goal, base='octile' if walls else 'euclidean', walls=walls)
    guide = None
    secondary = lower
    if 'llm' in method or 'straight' in method:
        guide = PolylineGuide(query, text, straight='straight' in method)
        if not guide.error:
            secondary = guide
    is_focal = method != 'clean_wastar_wall'
    weight = int(method.rsplit('w', 1)[1])/100 if is_focal else 2.0
    if geometry.collision(start, start) or geometry.collision(goal, goal):
        result = dict(path=[], operation=0, storage=0, expansions=0, valid=False,
                      error='invalid_endpoint', length=None)
    else:
        result = graph_search(start, goal, geometry.neighbors, lower, secondary,
                              weight=weight, focal=is_focal, trace=trace, audit=audit)
        path = result['path']
        error = validate_path(geometry, path, start, goal, grid=True) if path else 'unreachable'
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else None
        result.update(valid=error is None, error=error, length=length)
        if path:
            assert length <= result['goal_g']+1e-8
            if is_focal:
                assert length <= result['bound_limit']+1e-8
    result['total_seconds'] = time.perf_counter()-started
    result.update(lower.stats(), guide_error=guide.error if guide else None,
                  guide_raw_points=guide.raw_points if guide else 0,
                  guide_segments=len(guide.segments) if guide else 0,
                  guide_calls=guide.calls if guide else 0,
                  guide_segment_checks=guide.segment_checks if guide else 0)
    Geometry.neighbors.cache_clear()
    return result
