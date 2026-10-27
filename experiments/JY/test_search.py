"""Correctness tests for graph algorithms and certified OPEN updates."""
import math
import random
import unittest

from reproduction.core import Geometry
from experiments.K3.verification import Oracle, fixtures
from .search import (EXACT, METHODS, LandmarkIndex, astar, bidirectional,
                     guided, run_once)


class SearchTests(unittest.TestCase):
    def test_fixtures_against_independent_geometry_and_distances(self):
        for name, q in fixtures().items():
            oracle = Oracle(q, crosscheck=True)
            opt = oracle.distances(q['goal']).get(tuple(q['start']))
            for method in METHODS:
                with self.subTest(fixture=name, method=method):
                    index = LandmarkIndex(q, 4 if method=='j4_alt4' else 8) if method.startswith('j4') else None
                    r = run_once(q, method, '[[2,4],[3,8],[9,8],[10,4]]', index=index,
                                 audit=method.startswith('j5_nbs'))
                    self.assertEqual(r['valid'], opt is not None)
                    if opt is not None:
                        oracle.check_path(q, r)
                        self.assertGreaterEqual(r['length']+1e-8, opt)
                        if method in EXACT:
                            self.assertAlmostEqual(r['length'], opt, places=7)

    def test_lower_g_reopening_and_pruning(self):
        # Admissible inconsistent h forces reopening A after B's improvement.
        edges = {'S':[('A',3),('B',1)],'A':[('G',3)],'B':[('A',1),('G',9)],'G':[]}
        h = {'S':0,'A':0,'B':4,'G':0}
        # Generic astar uses coordinate path geometry; use an independent tiny
        # grid with a deliberately weakened, inconsistent admissible heuristic.
        q=fixtures()['long_wall']; oracle=Oracle(q); distances=oracle.distances(q['goal'])
        fun=lambda n: distances.get(n,0) if (n[0]+n[1])%2 else 0
        r=astar(q,lambda n:oracle.graph[n],fun)
        self.assertAlmostEqual(r['length'],distances[tuple(q['start'])])
        p=astar(q,lambda n:oracle.graph[n],fun,incumbent=r['path'],prune=True)
        self.assertAlmostEqual(p['length'],r['length'])

    def test_alt_disconnected_and_all_node_bounds(self):
        for q in (fixtures()['long_wall'],fixtures()['unreachable']):
            index=LandmarkIndex(q,4); oracle=Oracle(q,crosscheck=True)
            self.assertEqual(set(index.graph),set(oracle.graph))
            h=index.heuristic(tuple(q['goal'])); actual=oracle.distances(q['goal'])
            for n in index.graph:
                self.assertTrue(math.isfinite(h(n)))
                if n in actual:self.assertLessEqual(h(n),actual[n]+1e-8)
                for other,c in index.graph[n]:self.assertLessEqual(h(n),c+h(other)+1e-8)

    def test_random_nbs_pair_selection_and_optimality(self):
        rng=random.Random(20261027)
        for i in range(35):
            q=dict(range_x=[0,13],range_y=[0,11],start=[1,1],goal=[11,9],
                   horizontal_barriers=[[rng.randrange(2,9),rng.randrange(1,5),rng.randrange(6,11)] for _ in range(2)],
                   vertical_barriers=[[rng.randrange(2,10),rng.randrange(1,4),rng.randrange(5,9)] for _ in range(2)])
            oracle=Oracle(q);opt=oracle.distances(q['goal']).get(tuple(q['start']))
            for method in ('j5_nbs_euclidean','j5_nbs_octile','j5_bidijkstra'):
                r=run_once(q,method,audit=method!='j5_bidijkstra')
                self.assertEqual(r['valid'],opt is not None)
                if opt is not None:self.assertAlmostEqual(r['length'],opt,places=7)

    def test_lazy_exact_same_pop_sequence_on_target_changes(self):
        rng=random.Random(270610)
        for q in fixtures().values():
            for _ in range(10):
                targets=[(rng.randrange(2,11),rng.randrange(2,8)) for _ in range(7)]
                a=guided(q,Geometry(q),force_targets=targets,lazy=False,trace=True)
                b=guided(q,Geometry(q),force_targets=targets,lazy=True,trace=True)
                for k in ('path','length','valid','expansions','storage','pop_trace','target_changes'):
                    self.assertEqual(a[k],b[k],k)


if __name__=='__main__':unittest.main()
