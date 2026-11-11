import ast
import contextlib
import copy
import heapq
import io
import math
import random
import runpy
import types
import unittest

from shapely.geometry import LineString
from core import VENDOR, World, intersects, load_cases, search, parse_path, validate, PROMPTS

def upstream_class(relative, name):
    tree=ast.parse((VENDOR/relative).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    env=types.SimpleNamespace(Env=runpy.run_path(str(VENDOR/'llmastar/env/search/env.py'))['Env'])
    class Plot:
        def __init__(self,*args): pass
        def animation(self,*args): pass
    class FakeGPT:
        def __init__(self,*args,**kwargs): pass
        def ask(self,*args,**kwargs): return '[[5,5],[26,9],[25,23],[20,20]]'
    namespace={'math':math,'heapq':heapq,'env':env,'plotting':types.SimpleNamespace(Plotting=Plot),
               'is_lines_collision':lambda a,b:LineString(a).intersects(LineString(b)),
               'ChatGPT':FakeGPT,'Llama3':FakeGPT,'sysprompt_parse':'','example_parse':None,
               'gpt_prompt':PROMPTS,'list_parse':lambda text:parse_path(text)[0]}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(VENDOR/relative),'exec'),namespace)
    return namespace[name]

class ReproductionTests(unittest.TestCase):
    def test_collision_against_shapely(self):
        rng=random.Random(128)
        for _ in range(3000):
            a,b,c,d=[tuple(rng.randrange(0,12) for _ in range(2)) for _ in range(4)]
            if a==b or c==d: continue
            self.assertEqual(intersects(a,b,c,d),bool(LineString([a,b]).intersects(LineString([c,d]))),(a,b,c,d))

    def test_degenerate_barriers(self):
        for a,b,c,d in [((0,0),(3,3),(1,1),(1,1)),((0,1),(3,1),(1,1),(1,1)),
                         ((1,1),(1,1),(0,0),(3,3))]:
            self.assertEqual(intersects(a,b,c,d),bool(LineString([a,b]).intersects(LineString([c,d]))),(a,b,c,d))

    def test_upstream_search_parity(self):
        AStar=upstream_class('llmastar/pather/a_star/a_star.py','AStar')
        LLMAStar=upstream_class('llmastar/pather/llm_a_star/llm_a_star.py','LLMAStar')
        cases=load_cases()
        chosen=[cases[0]]+random.Random(123).sample(cases,14)
        for q in chosen:
            for kind,cls in [('astar',AStar),('llm_astar',LLMAStar)]:
                original=copy.deepcopy(q)
                with contextlib.redirect_stdout(io.StringIO()): ref=cls().searching(copy.deepcopy(q))
                got=search(q,kind,[(5,5),(26,9),(25,23),(20,20)])
                for metric in ('operation','storage'): self.assertEqual(got[metric],ref[metric],(q['id'],kind,metric))
                self.assertAlmostEqual(got['length'],ref['length'],places=8)
                self.assertEqual(q,original)

    def test_astar_optimality(self):
        for q in random.Random(71).sample(load_cases(),20):
            w=World(q)
            a=search(q,world=w); d=search(q,'dijkstra',world=w)
            self.assertTrue(a['valid'],q['id'])
            self.assertAlmostEqual(a['length'],d['length'],places=8)

    def test_segment_validation_and_parser(self):
        q={'start':[2,2],'goal':[8,8],'range_x':[0,11],'range_y':[0,11],
           'horizontal_barriers':[[5,0,10]],'vertical_barriers':[]}
        self.assertFalse(validate(q,[(2,2),(8,8)])[0])
        self.assertEqual(parse_path('No path')[1],'missing_list')
        self.assertEqual(parse_path('[[2,2],[8,8]]')[0],[(2,2),(8,8)])
        self.assertEqual(parse_path('[[2,2],[True,8]]')[1],'invalid_coordinates')

if __name__=='__main__': unittest.main(verbosity=2)
