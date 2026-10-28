"""Backend regression checks using saved responses only; no network calls."""
import json
import math
from pathlib import Path
import unittest

from reproduction.core import Geometry
from experiments.K3.verification import Oracle,fixtures
from .backends import METHODS,run,load_model


class CommonBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (Path(__file__).resolve().parents[2]/'results/main/generations.jsonl').open(encoding='utf-8') as stream:
            r=json.loads(stream.readline())
        cls.replies={k:r for k in ('standard','cot','repe','correction')}
        cls.replies['ranking']=dict(r,response={'choices':[{'message':{'content':'{"ranking":[0,1]}'}}]})
        cls.ctx=dict(quantile=6.5,model=load_model(),untrained=load_model(True))

    def test_reachable_edge_cases_every_condition(self):
        for name in ('empty','long_wall','reversed_wall','crossing_walls','degenerate_walls','endpoint_contact'):
            q=fixtures()[name];oracle=Oracle(q,crosscheck=True);opt=oracle.distances(q['goal'])[tuple(q['start'])]
            for method in METHODS:
                with self.subTest(fixture=name,method=method):
                    r=run(method,q,self.replies,self.ctx)
                    self.assertTrue(r['valid']);oracle.check_path(q,r)
                    if r['bound']:self.assertLessEqual(r['length'],opt*r['bound']+1e-7)

    def test_s3_never_mode_matches_original_g2h(self):
        from .interventions import delayed_search
        from reproduction.core import search
        q=fixtures()['long_wall'];a=delayed_search(q,lambda:[],mode='never');b=search(q,method='goal_only')
        for key in ('path','length','operation','storage'):self.assertEqual(a[key],b[key])

    def test_s3_immediate_mode_matches_native_llma(self):
        from .interventions import delayed_search
        from .vendor_sm.core import search,parse_path
        q=fixtures()['long_wall'];raw=[(2,4),(3,8),(9,8),(10,4)]
        a=delayed_search(q,lambda:raw,mode='immediate');b=search(q,method='llm_astar',waypoints=raw)
        for key in ('path','length','operation','storage'):self.assertEqual(a[key],b[key])

    def test_exact_incumbent_survives_open_exhaustion(self):
        from .backends import cc
        for name in ('long_wall','crossing_walls','endpoint_contact','same_line_detour'):
            q=fixtures()[name];o=Oracle(q);dist=o.distances(q['goal']);goal=tuple(q['goal']);path=[tuple(q['start'])]
            while path[-1]!=goal:
                p=path[-1];path.append(min(n for n,c in o.graph[p] if abs(c+dist[n]-dist[p])<1e-8))
            r=cc.focal(cc.Grid(q),q,weight=1.0,incumbent=path)
            self.assertTrue(r['valid']);self.assertTrue(r['incumbent_returned'])
            self.assertAlmostEqual(r['length'],dist[path[0]],places=8)


if __name__=='__main__':unittest.main()
