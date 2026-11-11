"""Preregistered independent interventions; original core.py remains unchanged."""
import heapq
import math
import time

from core import World, MOTIONS, validate, parse_path, path_length, prompt_for
from run import payload


def refine(q, raw):
    world = World(q)
    original = world.targets(raw, tuple(q['start']), tuple(q['goal']))
    unique = []
    for p in original:
        if p in unique:
            unique = unique[:unique.index(p)+1]
        else:
            unique.append(p)
    kept = [unique[0]]
    i = 0
    while i < len(unique)-1:
        reachable = [j for j in range(i+1, len(unique)) if not world.collision(unique[i], unique[j])]
        j = max(reachable) if reachable else i+1
        kept.append(unique[j])
        i = j
    accepted, reason = validate(q, kept, world)
    selected = kept if accepted else []
    return selected, {'input_targets': len(original), 'simplified_targets': len(kept),
                      'accepted': accepted, 'rejection_reason': reason,
                      'retained_targets': len(world.targets(selected, tuple(q['start']), tuple(q['goal'])))}


def diagnosis(q, text):
    path, error = parse_path(text)
    world = World(q)
    valid, reason = validate(q, path, world)
    reason = error or reason
    details = {'valid': valid and error is None, 'error': reason}
    if error:
        details['detail'] = 'No coordinate list could be parsed by the fixed parser: '+error
    elif reason == 'wrong_endpoints':
        details['detail'] = {'observed_start': path[0], 'observed_goal': path[-1],
                             'required_start': q['start'], 'required_goal': q['goal']}
    elif reason == 'out_of_bounds':
        details['detail'] = next({'point_index': i, 'point': p} for i,p in enumerate(path)
                               if not(world.x0 < p[0] < world.x1 and world.y0 < p[1] < world.y1))
    elif reason == 'collision':
        from core import intersects
        details['detail'] = next({'segment_index': i, 'segment': [a,b], 'barrier': [c,d]}
            for i,(a,b) in enumerate(zip(path,path[1:])) for c,d in world.barriers if intersects(a,b,c,d))
    return details


def correction_payload(q, kind, text, feedback):
    import json
    body = payload(q, kind)
    report = json.dumps(diagnosis(q,text), ensure_ascii=False, sort_keys=True) if feedback else 'Not provided.'
    body['messages'][0]['content'] = (prompt_for(q,kind)
        + '\n\nPrevious response:\n'+text
        + '\n\nExternal validation report:\n'+report
        + '\n\nReview the previous response and provide one revised answer for the same original problem. '
          'Follow the original output instructions and include the final coordinate list. '
          'Do not change the start, goal, map boundaries, or barriers.')
    return body


def delayed_search(q, get_waypoints, mode='adaptive', world=None):
    """Same search semantics, except when the target list is injected.

    Callback is called once at a checkpoint, before continuing the next pop.
    immediate/never are parity-test controls, not tuned experimental arms.
    """
    world = world or World(q)
    start, goal = tuple(q['start']), tuple(q['goal'])
    target_list = world.targets([], start, goal)
    ti, target = 1, goal
    g = {start: 0.0}; parent = {start: start}; closed = set()
    operations = 0; called = False; callback_seconds = 0.0
    def priority(s):
        return g[s]+math.hypot(goal[0]-s[0],goal[1]-s[1])+math.hypot(target[0]-s[0],target[1]-s[1])
    t0 = time.perf_counter()
    trigger = None
    if mode == 'immediate':
        t = time.perf_counter(); raw = get_waypoints(); callback_seconds += time.perf_counter()-t
        target_list = world.targets(raw, start, goal); target = target_list[1]; called = True
        trigger = {'pop': 0}
    queue = [(priority(start),start)]
    d0 = math.hypot(goal[0]-start[0],goal[1]-start[1])
    window = max(10, math.ceil(d0)); best = d0; previous_best = d0
    reached = False
    while queue:
        _,s = heapq.heappop(queue); operations += 1; closed.add(s)
        best = min(best, math.hypot(goal[0]-s[0],goal[1]-s[1]))
        if s == goal:
            reached = True; break
        for dx,dy in MOTIONS:
            n = (s[0]+dx,s[1]+dy)
            if n == target and target != goal:
                ti += 1
                if ti < len(target_list): target = target_list[ti]
                rebuilt = []
                for _,old in queue: heapq.heappush(rebuilt,(priority(old),old))
                queue = rebuilt
            if n in closed: continue
            cost = g[s]+world.edge_cost(s,n)
            if n not in g: g[n] = math.inf
            if cost < g[n]:
                g[n] = cost; parent[n] = s; heapq.heappush(queue,(priority(n),n))
        if mode == 'adaptive' and not called and queue and operations % window == 0:
            progress = previous_best-best
            if progress < 0.05*d0:
                trigger = {'pop': operations, 'window': window, 'best_distance': best,
                           'interval_progress': progress, 'threshold': 0.05*d0}
                t = time.perf_counter(); raw = get_waypoints(); callback_seconds += time.perf_counter()-t
                called = True
                target_list = world.targets(raw, start, goal); ti = 1; target = target_list[1]
                rebuilt = []
                for _,old in queue: heapq.heappush(rebuilt,(priority(old),old))
                queue = rebuilt
            previous_best = best
    path = []
    if reached:
        path = [goal]
        while path[-1] != start: path.append(parent[path[-1]])
        path.reverse()
    valid, reason = validate(q,path,world)
    elapsed = time.perf_counter()-t0
    return {'operation':operations, 'storage':len(g), 'length':path_length(path) if path else None,
            'valid':valid, 'failure':reason, 'path':path, 'targets':target_list,
            'seconds':elapsed-callback_seconds, 'elapsed_with_callback_seconds':elapsed,
            'callback_seconds':callback_seconds, 'api_calls':int(called), 'trigger':trigger}
