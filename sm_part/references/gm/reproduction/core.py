"""Search behavior adapted from the MIT-licensed source_code/llmastar.

The baseline intentionally retains duplicate queue pops, infinite g entries,
neighbor-triggered targets, and the upstream waypoint filter. Robust input/error
reporting and cached collision checks are infrastructure, not claimed innovations.
"""
import ast
import heapq
import math
import re
import time
from functools import lru_cache

MOVES = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


class Geometry:
    def __init__(self, query):
        self.x0, upper_x = query['range_x']
        self.y0, upper_y = query['range_y']
        self.x1, self.y1 = upper_x - 1, upper_y - 1
        self.horizontal = tuple(map(tuple, query['horizontal_barriers']))
        self.vertical = tuple(map(tuple, query['vertical_barriers']))
        if not self.x0 < self.x1 or not self.y0 < self.y1:
            raise ValueError('Invalid exclusive-upper-bound ranges')
        self.obs = set()
        for y, a, b in self.horizontal:
            self.obs.update((x, y) for x in range(a, b))
        for x, a, b in self.vertical:
            self.obs.update((x, y) for y in range(a, b))

    def inside(self, p):
        return self.x0 < p[0] < self.x1 and self.y0 < p[1] < self.y1

    def collision(self, a, b):
        # The rectangular interior is convex: interior endpoints keep a segment
        # inside. Closed barrier segments include both endpoints, as in Shapely.
        if not self.inside(a) or not self.inside(b):
            return True
        dx, dy = b[0] - a[0], b[1] - a[1]
        for y, left, right in self.horizontal:
            if left == right:
                continue  # Upstream Shapely degenerate LineString has no intersection.
            left, right = min(left, right), max(left, right)
            if dy == 0:
                if a[1] == y and max(min(a[0], b[0]), left) <= min(max(a[0], b[0]), right):
                    return True
            else:
                t = (y - a[1]) / dy
                if 0 <= t <= 1 and left <= a[0] + t * dx <= right:
                    return True
        for x, low, high in self.vertical:
            if low == high:
                continue
            low, high = min(low, high), max(low, high)
            if dx == 0:
                if a[0] == x and max(min(a[1], b[1]), low) <= min(max(a[1], b[1]), high):
                    return True
            else:
                t = (x - a[0]) / dx
                if 0 <= t <= 1 and low <= a[1] + t * dy <= high:
                    return True
        return False

    @lru_cache(maxsize=None)
    def neighbors(self, s):
        return tuple(((s[0] + dx, s[1] + dy),
                      math.inf if self.collision(s, (s[0] + dx, s[1] + dy)) else math.hypot(dx, dy))
                     for dx, dy in MOVES)

    def filter_upstream(self, nodes):
        return [p for p in nodes if p not in self.obs
                and self.x0 + 1 < p[0] < self.x1 - 1
                and self.y0 + 1 < p[1] < self.y1 - 1]


def parse_output(text, upstream=False):
    matches = re.findall(r'\[\[.*?\]\]', text, re.DOTALL)
    if not matches:
        return [], 'missing_list'
    # Same limited subtraction normalization as upstream, without eval().
    raw = re.sub(r'(\d+) - (\d+\.\d+)',
                 lambda m: str(int(m[1]) - float(m[2])), matches[-1])
    try:
        nodes = ast.literal_eval(raw)
        if not isinstance(nodes, list) or not nodes:
            return [], 'malformed_list'
        for p in nodes:
            if (not isinstance(p, (list, tuple)) or len(p) < 2 or (not upstream and len(p) != 2)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in p[:2])):
                return [], 'malformed_coordinate'
        return [tuple(p[:2]) for p in nodes], None
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return [], 'malformed_list'


def validate_path(geometry, path, start, goal, grid=False):
    if not path or tuple(path[0]) != tuple(start) or tuple(path[-1]) != tuple(goal):
        return 'wrong_endpoints'
    if any(not geometry.inside(p) for p in path):
        return 'outside_boundary'
    if any(geometry.collision(a, b) for a, b in zip(path, path[1:])):
        return 'collision'
    if len(path) == 1 and geometry.collision(path[0], path[0]):
        return 'collision'
    if grid and any((b[0] - a[0], b[1] - a[1]) not in MOVES for a, b in zip(path, path[1:])):
        return 'invalid_grid_step'
    return None


def llm_only(query, geometry, text):
    nodes, error = parse_output(text)
    error = error or validate_path(geometry, nodes, query['start'], query['goal'])
    # Invalid polylines still retain their geometric length for diagnostic use;
    # main length aggregates condition on valid paths and state their denominator.
    length = sum(math.dist(a, b) for a, b in zip(nodes, nodes[1:])) if nodes else None
    return dict(valid=error is None, error=error, length=length, path=nodes,
                operation=None, storage=None, raw_waypoints=len(nodes))


def search(query, geometry=None, method='astar', text='', trace=False):
    started = time.perf_counter()
    geometry = geometry or Geometry(query)
    start, goal = tuple(query['start']), tuple(query['goal'])
    nodes, parse_error = (parse_output(text, upstream=True) if method == 'llm_astar' else ([], None))
    schema_error = parse_output(text)[1] if method == 'llm_astar' else None
    if method not in ('astar', 'dwa', 'llm_astar', 'goal_only'):
        raise ValueError('Unknown search method: ' + method)
    # Missing lists use the upstream start/goal fallback. Malformed lists would
    # crash upstream; record that failure rather than quietly changing baseline.
    if method == 'llm_astar' and parse_error not in (None, 'missing_list'):
        return dict(valid=False, error=parse_error, operation=None, storage=None,
                    length=None, path=[], parse_error=parse_error, seconds=time.perf_counter()-started)
    targets = geometry.filter_upstream(nodes)
    retained = len(targets)
    if not targets or targets[0] != start:
        targets.insert(0, start)
    if targets[-1] != goal:
        targets.append(goal)
    if start == goal:
        error = validate_path(geometry, [start], start, goal)
        return dict(valid=error is None, error=error, path=[start], length=0.0,
                    operation=1, storage=1, seconds=time.perf_counter()-started,
                    upstream_exception_fixed='start_equals_goal')
    if not geometry.inside(start) or not geometry.inside(goal):
        return dict(valid=False, error='invalid_endpoint', path=[], length=None,
                    operation=0, storage=0, seconds=time.perf_counter()-started)
    target_index, target, weight = 1, targets[1], 2.0
    g, parent, closed, closed_set = {start: 0.0}, {start: start}, [], set()
    queue, events, rebuilds, rebuilt_entries = [], [], 0, 0
    distances = {}

    def h_goal(s):
        if s not in distances:
            distances[s] = math.hypot(goal[0]-s[0], goal[1]-s[1])
        return distances[s]

    def priority(s):
        h = h_goal(s)
        if method == 'llm_astar':
            h += math.hypot(target[0]-s[0], target[1]-s[1])
        elif method == 'goal_only':
            h *= 2
        elif method == 'dwa':
            h *= weight
        return g[s] + h

    def rebuild():
        nonlocal queue, rebuilds, rebuilt_entries
        rebuilt_entries += len(queue)
        rebuilds += 1
        fresh = []
        for _, node in queue:
            heapq.heappush(fresh, (priority(node), node))
        queue = fresh

    heapq.heappush(queue, (priority(start), start))
    while queue:
        popped_priority, s = heapq.heappop(queue)
        closed.append(s)  # Duplicate pops intentionally count as operations.
        closed_set.add(s)
        if trace:
            events.append(dict(event='pop', node=s, g=g[s], f=popped_priority, target=target))
        if s == goal:
            break
        if method == 'dwa':
            # Explicit implementation assumption: decay after each nonterminal
            # pop, lower bound 1, rekey every remaining entry at that event.
            next_weight = max(1.0, weight * .99)
            if next_weight != weight:
                weight = next_weight
                rebuild()
        for neighbor, cost in geometry.neighbors(s):
            if method == 'llm_astar' and neighbor == target and goal != target:
                target_index += 1
                if target_index < len(targets):
                    target = targets[target_index]
                rebuild()
                if trace:
                    events.append(dict(event='target', observed=neighbor, target=target,
                                       collision=math.isinf(cost)))
            if neighbor in closed_set:
                continue
            new_cost = g[s] + cost
            if neighbor not in g:
                g[neighbor] = math.inf
            if new_cost < g[neighbor]:
                g[neighbor], parent[neighbor] = new_cost, s
                heapq.heappush(queue, (priority(neighbor), neighbor))
    path = []
    if goal in parent:
        s = goal
        path = [s]
        while s != start:
            s = parent[s]
            path.append(s)
        path.reverse()
    error = 'unreachable' if not path else validate_path(geometry, path, start, goal, grid=True)
    result = dict(valid=error is None, error=error, path=path,
                  operation=len(closed), storage=len(g),
                  length=sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else None,
                  unique_pops=len(closed_set), duplicate_pops=len(closed)-len(closed_set),
                  infinite_g=sum(math.isinf(v) for v in g.values()),
                  queue_rebuilds=rebuilds, rekeyed_entries=rebuilt_entries,
                  targets=targets, raw_waypoints=len(nodes), retained_waypoints=retained,
                  target_index=target_index, final_target=target, parse_error=parse_error,
                  response_schema_error=schema_error,
                  seconds=time.perf_counter()-started)
    if trace:
        result['trace'] = events
    return result
