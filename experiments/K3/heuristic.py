"""K3 lower bound on the original eight-neighbor, line-barrier graph.

Removing all but one wall, the outer boundary, and the grid-direction restriction
relaxes the problem. If the straight segment intersects that wall, the infimum
continuous detour is via either endpoint. Endpoint contact is allowed only in
this lower-bound problem, NEVER in the actual search. Maxima (not sums) of these
distances and the obstacle-free octile metric remain lower bounds.
"""
import math
import time

from reproduction.core import Geometry, search

METHODS = (
    'astar_euclidean', 'g2h_euclidean', 'g2h_octile',
    'g2h_wall_euclidean', 'g2h_wall_octile',
)
PRIMARY = 'g2h_wall_octile'


def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


class LowerBound:
    def __init__(self, geometry, goal, base='octile', walls=True):
        if base not in ('euclidean', 'octile'):
            raise ValueError(base)
        self.goal, self.base = tuple(goal), base
        self.walls = []
        self.calls = self.wall_checks = self.intersections = 0
        if walls:
            # axis=1: horizontal wall, axis=0: vertical wall.
            for axis, barriers in ((1, geometry.horizontal), (0, geometry.vertical)):
                for coordinate, low, high in barriers:
                    if low == high:
                        continue  # Same degenerate-wall convention as upstream.
                    low, high = sorted((low, high))
                    a = (low, coordinate) if axis == 1 else (coordinate, low)
                    b = (high, coordinate) if axis == 1 else (coordinate, high)
                    self.walls.append((axis, coordinate, low, high, a, b,
                                       math.dist(a, self.goal), math.dist(b, self.goal)))

    def __call__(self, node):
        self.calls += 1
        goal = self.goal
        best = octile(node, goal) if self.base == 'octile' else math.dist(node, goal)
        for axis, coordinate, low, high, a, b, ga, gb in self.walls:
            self.wall_checks += 1
            side_n, side_g = node[axis] - coordinate, goal[axis] - coordinate
            if side_n * side_g > 0:
                continue
            other = 1 - axis
            delta = goal[axis] - node[axis]
            if delta == 0:
                # Same line as wall: closed interval overlap, including endpoints.
                hit = max(min(node[other], goal[other]), low) <= min(max(node[other], goal[other]), high)
            else:
                # Integer cross-multiplication avoids a floating intersection
                # position near an endpoint. The input graph has integer nodes.
                crossing = node[other] * delta + (coordinate - node[axis]) * (goal[other] - node[other])
                lower, upper = low * delta, high * delta
                if delta < 0:
                    lower, upper = upper, lower
                hit = lower <= crossing <= upper
            if hit:
                self.intersections += 1
                best = max(best, min(math.dist(node, a) + ga, math.dist(node, b) + gb))
        return best

    def stats(self):
        return dict(heuristic_calls=self.calls, wall_checks=self.wall_checks,
                    blocking_wall_checks=self.intersections, wall_count=len(self.walls),
                    precomputed_endpoint_distances=2 * len(self.walls))


def run_once(query, method, trace=False):
    if method not in METHODS + ('astar_wall_octile',):
        raise ValueError(method)
    # The original decorator caches across instances and holds self references.
    # Clear outside the timer so every trial starts cold without unbounded cache
    # growth. No graph/heuristic precomputation is shared across queries/methods.
    Geometry.neighbors.cache_clear()
    started = time.perf_counter()
    geometry = Geometry(query)
    heuristic = None
    if method not in ('astar_euclidean', 'g2h_euclidean'):
        heuristic = LowerBound(geometry, query['goal'],
                               base='euclidean' if method.endswith('_euclidean') else 'octile',
                               walls='wall' in method)
    result = search(query, geometry, method='astar' if method.startswith('astar') else 'goal_only',
                    goal_heuristic=heuristic, trace=trace)
    elapsed = time.perf_counter() - started
    result['total_seconds'] = elapsed
    result.update(heuristic.stats() if heuristic else dict(
        heuristic_calls=None, wall_checks=0, blocking_wall_checks=0,
        wall_count=0, precomputed_endpoint_distances=0))
    Geometry.neighbors.cache_clear()
    return result
