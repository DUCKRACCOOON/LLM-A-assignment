import argparse
import math
import statistics
import re
import time
from common import *
from algorithms import run
from collect import obtain

def generate():
    for index in range(12):
        prompt=(
          'Propose four finite numeric coefficients, each between -2 and 2, for prioritizing walls in an exact grid path planner. '
          'The score is a linear weighted sum of these features, in order: '
          '(1) start-goal straight segment intersects wall (0 or 1), (2) wall length divided by map diagonal, '
          '(3) min(distance(start,wall endpoint)+distance(endpoint,goal)) divided by map diagonal, '
          '(4) any wall endpoint touches map boundary (0 or 1). '
          'The highest scoring k walls are initially activated, where k is the number intersecting the start-goal segment. '
          'After exact relaxed search, all walls are checked and collision walls are added; after 3 failures all walls are enabled. '
          'Aim to reduce total preparation, repeated search and validation cost. '
          'Maps have horizontal and vertical closed line-segment barriers and 8-neighbor movement. '
          'No evaluation-map outcomes are available. This is proposal '+str(index+1)+' of 12. '
          'Return only a JSON array of exactly four coefficients. Do not output code.')
        q=dict(id=f'proposal{index:02d}',stage='s6_programs')
        body=dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=1000,store=False)
        record,_=obtain(q,'program',body);text=text_of(record)
        try:
            # v2 accepts one surrounding JSON code fence, identically for all
            # proposals. Original strict-v1 outcomes and response hashes survive.
            clean=text.strip()
            match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',clean)
            if match:clean=match.group(1)
            weights=json.loads(clean)
            assert isinstance(weights,list) and len(weights)==4
            assert all(type(v) in (float,int) and math.isfinite(v) and -2<=v<=2 for v in weights)
            valid=True
        except Exception:weights=None;valid=False
        save(OUT/'programs'/f'proposal{index:02d}.json',dict(index=index,weights=weights,valid=valid,parser='v2_single_json_fence',raw_sha256=record['response_sha256']))
        print(json.dumps(dict(proposal=index,valid=valid)),flush=True)

def select(refresh=False):
    target=OUT/'program_selection.json'
    if target.exists() and not refresh:print(json.dumps(read(target)),flush=True);return
    started=time.perf_counter()
    if target.exists() and refresh:save(OUT/'program_selection_initial.json',read(target))
    candidates=[read(p) for p in sorted((OUT/'programs').glob('proposal*.json'))]
    valid=[p for p in candidates if p['valid']]
    if not valid:raise RuntimeError('No valid S6 proposals; historical fallback must remain separately labelled')
    # Deduplicate identical coefficients computationally, while retaining all generated records and costs.
    unique={}
    for candidate in valid:unique.setdefault(tuple(candidate['weights']),candidate)
    results=[]
    for weights,candidate in unique.items():
        path=OUT/('programs_selection_v2' if refresh else 'programs')/f'selection{candidate["index"]:02d}.json'
        if path.exists():result=read(path)
        else:
            result=dict(index=candidate['index'],weights=list(weights),splits={})
            for split in ('train','validation'):
                qs=[q for q in cases('public') if q['split']==split];rows=[]
                for q in qs:
                    r=run(q,'S6_program',program=weights);a=run(q,'A_OCT')
                    assert r['valid'] and math.isclose(r['length'],a['length'],abs_tol=1e-8)
                    rows.append(dict(case_id=q['id'],operation=r['operation'],reference_operation=a['operation'],seconds=r['total_seconds'],reference_seconds=a['total_seconds']))
                # Deterministic selection avoids choosing a coefficient due to timing noise.
                result['splits'][split]=dict(heap_pop_ratio=math.exp(statistics.mean(math.log(r['operation']/r['reference_operation']) for r in rows)),rows=rows)
            save(path,result)
        results.append(result);print(json.dumps(dict(phase='s6_select',proposal=result['index'],train=result['splits']['train']['heap_pop_ratio'],validation=result['splits']['validation']['heap_pop_ratio'])),flush=True)
    shortlist=sorted(results,key=lambda x:(x['splits']['train']['heap_pop_ratio'],x['index']))[:3]
    chosen=min(shortlist,key=lambda x:(x['splits']['validation']['heap_pop_ratio'],x['index']))
    cost_rows=[r for result in results for part in result['splits'].values() for r in part['rows']]
    result=dict(weights=chosen['weights'],proposal_index=chosen['index'],criterion='top 3 train geometric mean heap-pop ratio, then minimum validation ratio, index breaks ties',
                valid_proposals=len(valid),unique_proposals=len(unique),selection_uses_test=False,selected_at=now(),
                selection_seconds=time.perf_counter()-started,candidate_planner_seconds=sum(r['seconds'] for r in cost_rows),
                reference_planner_seconds=sum(r.get('reference_seconds',0) for r in cost_rows),selection_queries=len(cost_rows),
                cost_measurement='complete repeated selection with reference cost' if refresh else 'initial selection')
    if refresh:
        previous=read(OUT/'program_selection_initial.json');assert result['weights']==previous['weights'] and result['proposal_index']==previous['proposal_index']
    save(target,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','select']);p.add_argument('--refresh',action='store_true');a=p.parse_args()
    generate() if a.action=='generate' else select(a.refresh)
