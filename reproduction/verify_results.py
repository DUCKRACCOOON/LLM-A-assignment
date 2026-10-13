"""End-to-end record integrity, full-data optimum check, and live-output parity."""
import contextlib
import copy
import io
import json
import math
import sys
from collections import Counter
from types import ModuleType
from .core import Geometry, parse_output, search, validate_path
from .experiment import ROOT, cases, records
from .verify import dijkstra


def main():
    out=ROOT/'results/main'
    generations=records(out/'generations.jsonl')
    evaluations=records(out/'evaluations.jsonl')
    assert len(generations)==6000
    assert len({r['request_sha256'] for r in generations})==6000
    assert all(r['status']=='ok' for r in generations)
    assert set(Counter((r['model'],r['prompt_style']) for r in generations).values())=={1000}
    assert len(evaluations)==15000
    assert len({r['evaluation_id'] for r in evaluations})==15000
    lookup={c['case_id']:c for c in cases()}
    geometries={}
    astar={r['case_id']:r for r in evaluations if r['method']=='astar'}
    for case in lookup.values():
        geo=geometries.setdefault(case['map_index'],Geometry(case['query']))
        optimal=dijkstra(case['query'],geo)
        assert optimal is not None and abs(optimal-astar[case['case_id']]['length'])<1e-9
    for r in evaluations:
        if not r['valid']: continue
        case=lookup[r['case_id']]
        assert validate_path(geometries[case['map_index']],r['path'],case['query']['start'],case['query']['goal'],grid=r['method']!='llm_only') is None
    import matplotlib
    matplotlib.use('Agg')
    for name in ['openai','torch','transformers']:
        if name not in sys.modules: sys.modules[name]=ModuleType(name)
    sys.path.insert(0,str(ROOT/'source_code'))
    from llmastar.pather import LLMAStar
    from llmastar.env.search.plotting import Plotting
    Plotting.animation=lambda *a,**k:None
    class Fixed:
        def __init__(self,text): self.text=text
        def ask(self,**k): return self.text
    # All structurally malformed responses, plus fixed spaced successful cases.
    selected=[]
    for i,g in enumerate(sorted(generations,key=lambda r:(r['case_id'],r['model'],r['prompt_style']))):
        text=g['response']['choices'][0]['message'].get('content') or ''
        error=parse_output(text)[1]
        if error not in (None,'missing_list') or i%200==0:
            selected.append((g,text))
    parity=[]
    for g,text in selected:
        q=lookup[g['case_id']]['query']
        original=LLMAStar()
        original.model=Fixed(text)
        after=search(q,method='llm_astar',text=text)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                before=original.searching(copy.deepcopy(q))
        except (IndexError,TypeError,KeyError,ValueError) as exc:
            assert not after['valid'],(g['case_id'],type(exc).__name__)
            status='matching_failure:'+type(exc).__name__
        else:
            assert after['valid'],(g['case_id'],'upstream succeeded but reproduction failed')
            for metric in ['operation','storage','length']:
                assert math.isclose(before[metric],after[metric],rel_tol=1e-12),(g['case_id'],metric,before,after)
            assert original.extract_path(original.PARENT)==after['path']
            status='matching_success'
        parity.append(dict(case_id=g['case_id'],model=g['model'],prompt=g['prompt_style'],status=status))
    result=dict(generations=6000,evaluations=15000,unique_requests=True,all_conditions_n=1000,
                independent_dijkstra_checks=1000,all_successful_paths_revalidated=True,
                real_response_upstream_parity=parity,passed=True)
    (ROOT/'results/verification/final_integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,dijkstra=1000,real_response_parity=len(parity))))


if __name__=='__main__':main()
