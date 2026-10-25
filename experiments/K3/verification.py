"""Independent integer geometry and Dijkstra oracle; never used as a heuristic."""
import heapq
import math

from reproduction.core import Geometry
from .heuristic import LowerBound, METHODS, run_once

EPS = 1e-8
STEPS = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y)


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, c):
    return cross(a, b, c) == 0 and all(min(a[i], b[i]) <= c[i] <= max(a[i], b[i]) for i in (0, 1))


def intersects(a, b, c, d):
    u, v, w, z = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return (u*v < 0 and w*z < 0) or any((
        u == 0 and on_segment(a, b, c), v == 0 and on_segment(a, b, d),
        w == 0 and on_segment(c, d, a), z == 0 and on_segment(c, d, b)))


class Oracle:
    def __init__(self, query, crosscheck=False):
        x0, x1 = query['range_x']; y0, y1 = query['range_y']
        self.walls = [((a, y), (b, y)) for y, a, b in query['horizontal_barriers'] if a != b]
        self.walls += [((x, a), (x, b)) for x, a, b in query['vertical_barriers'] if a != b]
        self.bounds = x0, x1-1, y0, y1-1
        self.graph = {}
        for x in range(x0+1, x1-1):
            for y in range(y0+1, y1-1):
                p = (x, y)
                if not self.blocked(p, p):
                    self.graph[p] = []
        self.collision_checks = 0
        geometry = Geometry(query) if crosscheck else None
        # Include blocked interior states in the cross-check, including attempts
        # to touch wall endpoints or step outside the grid.
        for x in range(x0+1, x1-1):
            for y in range(y0+1, y1-1):
                p = (x, y)
                for dx, dy in STEPS:
                    q = (x+dx, y+dy)
                    blocked = self.blocked(p, q)
                    if geometry:
                        assert blocked == geometry.collision(p, q), (p, q, query)
                        self.collision_checks += 1
                    if not blocked:
                        self.graph[p].append((q, math.sqrt(dx*dx+dy*dy)))

    def blocked(self, a, b):
        x0, x1, y0, y1 = self.bounds
        if any(not (x0 < p[0] < x1 and y0 < p[1] < y1) for p in (a, b)):
            return True
        return any(intersects(a, b, c, d) for c, d in self.walls)

    def distances(self, goal):
        goal = tuple(goal)
        if goal not in self.graph:
            return {}
        best, queue = {goal: 0.0}, [(0.0, goal)]
        while queue:
            dist, p = heapq.heappop(queue)
            if dist != best[p]:
                continue
            for q, cost in self.graph[p]:
                candidate = dist + cost
                if candidate < best.get(q, math.inf):
                    best[q] = candidate
                    heapq.heappush(queue, (candidate, q))
        return best

    def check_path(self, query, result):
        path = [tuple(p) for p in result['path']]
        assert result['valid'] and path[0] == tuple(query['start']) and path[-1] == tuple(query['goal'])
        assert all(p in self.graph for p in path)
        assert all(any(p == b for p, _ in self.graph[a]) for a, b in zip(path, path[1:]))
        cost = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        assert math.isclose(cost, result['length'], abs_tol=EPS)


def check_bounds(query, oracle):
    actual = oracle.distances(query['goal'])
    geometry = Geometry(query)
    counts = dict(admissibility_checks=0, consistency_checks=0, max_admissibility_excess=0.0,
                  max_consistency_excess=0.0, reachable_nodes=len(actual))
    for base, walls in (('octile', False), ('euclidean', True), ('octile', True)):
        h = LowerBound(geometry, query['goal'], base=base, walls=walls)
        values = {p: h(p) for p in oracle.graph}
        assert abs(h(tuple(query['goal']))) < EPS
        for p, distance in actual.items():
            excess = values[p] - distance
            assert values[p] >= 0 and excess <= EPS, ('inadmissible', base, walls, p, excess, query)
            counts['admissibility_checks'] += 1
            counts['max_admissibility_excess'] = max(counts['max_admissibility_excess'], excess)
        for p, edges in oracle.graph.items():
            for q, cost in edges:
                excess = values[p] - cost - values[q]
                assert excess <= EPS, ('inconsistent', base, walls, p, q, excess, query)
                counts['consistency_checks'] += 1
                counts['max_consistency_excess'] = max(counts['max_consistency_excess'], excess)
    counts['optimum'] = actual.get(tuple(query['start']))
    return counts


def fixtures():
    def make(start=(2, 4), goal=(10, 4), h=(), v=()):
        return dict(range_x=[0, 13], range_y=[0, 10], start=list(start), goal=list(goal),
                    horizontal_barriers=list(h), vertical_barriers=list(v))
    return {
        'empty': make(),
        'long_wall': make(v=[(6, 1, 7)]),
        'reversed_wall': make(v=[(6, 7, 1)]),
        'crossing_walls': make(h=[(5, 3, 9)], v=[(6, 2, 7)]),
        'boundary_attached_wall': make(v=[(6, 0, 6)]),
        'degenerate_walls': make(v=[(6, 4, 4)], h=[(4, 6, 6)]),
        'start_equals_goal': make(goal=(2, 4)),
        'unreachable': make(v=[(6, 0, 9)]),
        'endpoint_contact': make(start=(5, 2), goal=(7, 2), v=[(6, 2, 6)]),
        'same_line_detour': make(start=(6, 1), goal=(6, 7), v=[(6, 2, 6)]),
    }


def verify_fixtures():
    results = {}
    for name, query in fixtures().items():
        oracle = Oracle(query, crosscheck=True)
        checks = check_bounds(query, oracle)
        rows = {}
        for method in METHODS + ('astar_wall_octile',):
            row = run_once(query, method)
            if checks['optimum'] is None:
                assert not row['valid'] and row['error'] == 'unreachable'
            else:
                oracle.check_path(query, row)
                if method.startswith('astar'):
                    assert math.isclose(row['length'], checks['optimum'], abs_tol=EPS)
            rows[method] = {k: row[k] for k in ('valid', 'error', 'length', 'operation')}
        results[name] = dict(checks=checks, methods=rows, collision_checks=oracle.collision_checks)
    assert results['long_wall']['methods'] == results['reversed_wall']['methods']
    assert results['empty']['methods'] == results['degenerate_walls']['methods']
    # Analytical one-wall lower bound: equal-length detours around y=1 and y=7.
    q = fixtures()['long_wall']
    assert math.isclose(LowerBound(Geometry(q), q['goal'])(tuple(q['start'])), 10.0)
    return results
