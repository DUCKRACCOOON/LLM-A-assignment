"""Shared MHA* (RSS 2014 Fig. 3) with RR/TS/DTS scheduling.

The consistent anchor is mandatory; auxiliary heuristics can be inadmissible.
This implementation fixes w1=1; w2 controls the returned cost bound.
"""
import heapq
import math
import random
import time

from reproduction.core import Geometry, validate_path
from experiments.K3.heuristic import LowerBound
from experiments.K4.search import PolylineGuide, run_once as run_k4

EPS = 1e-10
PRIMARY = 'mha_dts_w200'
BASELINES = ('g2h_euclidean', 'g2h_wall_octile', 'g2h_wall_euclidean',
             'clean_wastar_wall', 'focal_llm_w105', 'focal_goal_w105', 'focal_llm_w200')
STOCHASTIC = (PRIMARY, 'mha_ts_w200', 'mha_dts_nollm_w200', 'mha_dts_w105')
METHODS = BASELINES + (PRIMARY, 'mha_rr_w200', 'mha_ts_w200',
                      'mha_dts_nollm_w200', 'mha_wall_w200', 'mha_dts_w105')


class Scheduler:
    def __init__(self, initial_h, mode='dts', seed=0, cap=10):
        if mode not in ('rr', 'ts', 'dts') or cap < 2 or not math.isfinite(cap):
            raise ValueError('Invalid scheduler or history cap')
        self.mode, self.cap, self.rng = mode, cap, random.Random(seed)
        self.best = list(initial_h)
        self.alpha, self.beta = [1.0]*len(initial_h), [1.0]*len(initial_h)
        self.choices, self.updates, self.rewards = ([0]*len(initial_h) for _ in range(3))
        self.samples = self.rescales = self.turn = 0

    def choose(self):
        if self.mode == 'rr':
            arm = self.turn % len(self.best)
        else:
            draws = [self.rng.betavariate(a,b) for a,b in zip(self.alpha,self.beta)]
            self.samples += len(draws)
            arm = max(range(len(draws)), key=draws.__getitem__)
        self.turn += 1
        self.choices[arm] += 1
        return arm

    def update(self, arm, minimum_h):
        if self.mode == 'rr':
            return
        reward = minimum_h < self.best[arm]
        self.updates[arm] += 1
        self.rewards[arm] += int(reward)
        if reward:
            self.best[arm] = minimum_h
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        if self.mode == 'dts' and self.alpha[arm]+self.beta[arm] > self.cap:
            scale = self.cap/(self.cap+1)
            self.alpha[arm] *= scale
            self.beta[arm] *= scale
            self.rescales += 1

    def stats(self):
        return dict(bandit_choices=self.choices, bandit_updates=self.updates, bandit_rewards=self.rewards,
                    bandit_alpha=self.alpha, bandit_beta=self.beta, beta_samples=self.samples,
                    bandit_rescales=self.rescales)


def graph_search(start, goal, neighbors, heuristics, *, weight=2.0, mode='dts', seed=0,
                 cap=10, trace=False, audit=False):
    if not math.isfinite(weight) or weight < 1 or len(heuristics) < 2:
        raise ValueError('Need weight >= 1 and an anchor plus at least one auxiliary')
    size = len(heuristics)
    g, parent, versions = {start:0.0}, {start:start}, {}
    closed_anchor, closed_aux = set(), set()
    active = [dict() for _ in range(size)]
    queues, rewards = ([[] for _ in range(size)] for _ in range(2))
    cache = [dict() for _ in range(size)]
    stats = dict(search_pops=0, reward_pops=0, stale_pops=0, heap_pushes=0,
                 peak_heap_entries=0, successor_expansions=0, anchor_fallbacks=0,
                 queue_expansions=[0]*size, selection_audits=0)
    events, certificate, chosen_queue, heap_size = [], None, None, 0

    def h(i, node):
        if node not in cache[i]:
            value = heuristics[i](node)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Heuristics must be finite and nonnegative')
            cache[i][node] = value
        return cache[i][node]

    def key(i, node):
        return g[node]+h(i,node)

    for i in range(size):
        if abs(h(i,goal)) > EPS:
            raise ValueError('Heuristics must be zero at the goal')
    scheduler = Scheduler([h(i,start) for i in range(1,size)], mode, seed, cap)

    def push(heap, entry):
        nonlocal heap_size
        heapq.heappush(heap,entry)
        stats['heap_pushes'] += 1
        heap_size += 1
        stats['peak_heap_entries'] = max(stats['peak_heap_entries'],heap_size)

    def pop(heap, kind):
        nonlocal heap_size
        stats[kind] += 1
        heap_size -= 1
        return heapq.heappop(heap)

    def put(i,node,version):
        active[i][node] = version
        push(queues[i],(key(i,node),node,version))
        if i and mode != 'rr':
            push(rewards[i],(h(i,node),node,version))

    def update(node, initial=False):
        version = versions.get(node,0)+1
        versions[node] = version
        for a in active:
            a.pop(node,None)
        if node not in closed_anchor:
            put(0,node,version)
            if node not in closed_aux:
                for i in range(1,size):
                    if initial or key(i,node) <= weight*key(0,node)+EPS:
                        put(i,node,version)

    def top(i, reward=False):
        heap = rewards[i] if reward else queues[i]
        while heap and active[i].get(heap[0][1]) != heap[0][2]:
            pop(heap,'reward_pops' if reward else 'search_pops')
            stats['stale_pops'] += 1
        return heap[0] if heap else (math.inf,None,None)

    update(start, initial=True)
    while active[0]:
        anchor_min = top(0)[0]
        arm = scheduler.choose()
        selected = arm+1
        candidate = top(selected)
        if candidate[0] > weight*anchor_min+EPS:
            selected = 0
            candidate = top(0)
            stats['anchor_fallbacks'] += 1
        if audit:
            expected = min((key(selected,n),n,v) for n,v in active[selected].items())
            assert candidate == expected
            assert abs(anchor_min-min(key(0,n) for n in active[0])) < EPS
            stats['selection_audits'] += 1
        if g.get(goal,math.inf) <= candidate[0]+EPS:
            certificate, chosen_queue = anchor_min, selected
            break
        _, node, _ = pop(queues[selected],'search_pops')
        for a in active:
            a.pop(node,None)
        (closed_anchor if selected == 0 else closed_aux).add(node)
        stats['successor_expansions'] += 1
        stats['queue_expansions'][selected] += 1
        if trace:
            events.append(dict(node=node,g=g[node],queue=selected,requested=arm+1,
                               anchor_min=anchor_min,selected_key=candidate[0]))
        for neighbor,cost in neighbors(node):
            if cost <= 0 or math.isnan(cost):
                raise ValueError('Edge costs must be positive')
            g.setdefault(neighbor,math.inf)
            proposal = g[node]+cost
            if proposal < g[neighbor]:
                g[neighbor],parent[neighbor] = proposal,node
                update(neighbor)
        if selected and mode != 'rr':
            min_h = top(selected,reward=True)[0]
            if audit:
                assert min_h == min((h(selected,n) for n in active[selected]),default=math.inf)
            scheduler.update(arm,min_h)
    path = []
    if certificate is not None:
        node = goal
        path = [node]
        while node != start:
            node = parent[node]
            if node in path:
                raise AssertionError('Parent cycle')
            path.append(node)
        path.reverse()
    result = dict(path=path,goal_g=g.get(goal) if path else None,
                  operation=stats['search_pops']+stats['reward_pops'], storage=len(g),
                  unique_pops=len(closed_anchor|closed_aux), reexpansions=len(closed_anchor&closed_aux),
                  infinite_g=sum(math.isinf(v) for v in g.values()),
                  bound_factor=weight, anchor_key_at_goal=certificate, goal_queue=chosen_queue,
                  bound_limit=weight*certificate if certificate is not None else None,
                  **stats, **scheduler.stats())
    if trace:
        result['trace'] = events
    return result


def run_once(query, method, text='', *, seed=0, trace=False, audit=False):
    if method not in METHODS:
        raise ValueError(method)
    if method in BASELINES:
        result = run_k4(query,method,text,trace=trace,audit=audit)
        # Existing engines count the terminal goal as a selected node.
        result['successor_expansions'] = result['expansions']-int(result['valid'])
        return result
    Geometry.neighbors.cache_clear()
    started = time.perf_counter()
    geometry = Geometry(query)
    start,goal = tuple(query['start']),tuple(query['goal'])
    lower = LowerBound(geometry,goal)
    lower_cache = {}

    def h0(node):
        if node not in lower_cache:
            lower_cache[node] = lower(node)
        return lower_cache[node]

    helpers = [h0,lambda n:2*h0(n)]
    labels = ['anchor','wall']
    guide = None
    if method != 'mha_wall_w200':
        helpers.append(lambda n:2*math.dist(n,goal)); labels.append('goal')
        if 'nollm' not in method:
            guide = PolylineGuide(query,text)
            # A polyline may revisit the goal: roundoff can favor an earlier
            # projection with nonzero remaining length. Enforce MHA's h(goal)=0.
            helpers.append((lambda n:2*h0(n)) if guide.error else (lambda n:0.0 if n==goal else 2*guide(n)))
            labels.append('llm')
    mode = 'rr' if method in ('mha_rr_w200','mha_wall_w200') else ('ts' if method=='mha_ts_w200' else 'dts')
    weight = 1.05 if method.endswith('w105') else 2.0
    if geometry.collision(start,start) or geometry.collision(goal,goal):
        result = dict(path=[],valid=False,error='invalid_endpoint',length=None,operation=0,storage=0)
    else:
        result = graph_search(start,goal,geometry.neighbors,helpers,weight=weight,mode=mode,
                              seed=seed,trace=trace,audit=audit)
        path = result['path']
        error = validate_path(geometry,path,start,goal,grid=True) if path else 'unreachable'
        length = sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None
        result.update(valid=error is None,error=error,length=length)
        if path:
            assert length <= result['goal_g']+1e-8
            assert length <= result['bound_limit']+1e-8
    result['total_seconds'] = time.perf_counter()-started
    result.update(lower.stats(), queue_labels=labels, guide_error=guide.error if guide else None,
                  guide_segments=len(guide.segments) if guide else 0,
                  guide_segment_checks=guide.segment_checks if guide else 0,
                  guide_calls=guide.calls if guide else 0)
    Geometry.neighbors.cache_clear()
    return result
