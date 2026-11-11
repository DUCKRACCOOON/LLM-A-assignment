"""Independent invariants for the new search, geometry, and calibration."""
import math
import unittest
import numpy as np
from core import load_cases, World, search, validate
from candidate_core import Grid, focal, fast_marching, conformal_quantile, chomp_repair, grid_valid


def simple(**kw):
    q=dict(id='unit',map_id=-1,pair_id=0,range_x=[0,16],range_y=[0,13],
           start=[2,2],goal=[12,9],horizontal_barriers=[],vertical_barriers=[])
    q.update(kw); return q


class CandidateTests(unittest.TestCase):
    def test_exact_geometry(self):
        for q in [simple(vertical_barriers=[[7,0,8]]),load_cases()[0],load_cases()[570]]:
            grid=Grid(q); world=World(q)
            for i in range(grid.n):
                for k,(dx,dy) in enumerate(__import__('core').MOTIONS):
                    a=tuple(grid.xy[i]); b=(a[0]+dx,a[1]+dy)
                    interior=0<a[0]<grid.w-1 and 0<a[1]<grid.h-1 and 0<b[0]<grid.w-1 and 0<b[1]<grid.h-1
                    expected=interior and not world.collision(a,b)
                    self.assertEqual(grid.edges[i,k]>=0,expected,(q['id'],a,b))

    def test_focal_bounds_with_adversarial_guidance(self):
        rng=np.random.default_rng(20261014)
        for q in [simple(),simple(vertical_barriers=[[7,0,8]])]+load_cases()[230:240]:
            grid=Grid(q); dist,_,_=grid.distances(q['goal']); optimum=dist[grid.index(q['start'])]
            legacy=search(q,'dijkstra')
            self.assertAlmostEqual(legacy['length'],optimum)
            for mode in ['astar','weighted','focal']:
                r=focal(grid,q,rng.normal(size=grid.n),mode=mode)
                self.assertTrue(r['valid']); self.assertLessEqual(r['length'],1.1*optimum+1e-8)
            r=focal(grid,q,weight=1.)
            self.assertAlmostEqual(r['length'],optimum)
            r2=focal(grid,q,incumbent=r['path'])
            self.assertTrue(r2['valid']); self.assertLessEqual(r2['length'],1.1*optimum+1e-8)

    def test_fmm_is_eikonal_not_dijkstra(self):
        q=simple(start=[5,5],goal=[7,7]); grid=Grid(q)
        field,_=fast_marching(grid,q['goal'])
        self.assertEqual(field[grid.index(q['goal'])],0.)
        self.assertAlmostEqual(field[grid.index([7,6])],1.)
        self.assertAlmostEqual(field[grid.index([6,6])],1+1/math.sqrt(2))
        q=simple(vertical_barriers=[[7,0,12]]); grid=Grid(q)
        field,_=fast_marching(grid,q['goal'])
        self.assertTrue(math.isinf(field[grid.index(q['start'])]))

    def test_finite_sample_quantile(self):
        self.assertEqual(conformal_quantile(range(20),.1),18)
        self.assertTrue(math.isinf(conformal_quantile([1,2,3],.1)))

    def test_repair_never_accepts_invalid_grid_path(self):
        for q in [simple(),simple(vertical_barriers=[[7,0,8]])]:
            p,meta=chomp_repair(q,[q['start'],q['goal']],iterations=32)
            if p: self.assertTrue(grid_valid(q,p)[0])
            self.assertEqual(bool(p),meta['repaired'])
        self.assertFalse(grid_valid(simple(),[[2,2],[12,9]])[0])

if __name__=='__main__': unittest.main()
