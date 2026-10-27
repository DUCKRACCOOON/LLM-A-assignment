"""Post-run targeted check; no benchmark settings or code are changed."""
import random
import unittest

from experiments.K3.verification import Oracle,fixtures
from .search import astar


class ReopeningRegression(unittest.TestCase):
    def test_admissible_inconsistent_h_actually_reopens(self):
        q=fixtures()['long_wall'];oracle=Oracle(q);actual=oracle.distances(q['goal'])
        rng=random.Random(2710)
        # The second fixed draw forces lower-g reopening. Each h is zero or
        # the true distance: admissible but deliberately inconsistent.
        # The oracle is used in this test ONLY, never in benchmark search.
        for _ in range(2):
            h={n:(d if rng.random()<.5 else 0.0) for n,d in actual.items()}
        r=astar(q,lambda n:oracle.graph[n],lambda n:h[n])
        self.assertGreater(r['reopenings'],0)
        self.assertAlmostEqual(r['length'],actual[tuple(q['start'])],places=8)
        oracle.check_path(q,r)


if __name__=='__main__':unittest.main()
