"""Evaluate a prespecified proposal and generate paired uncertainty analyses."""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from .core import Geometry
from .experiment import cases, records, append, manifest
from .proposed import proposed_search
from .summarize import aggregate, geometric_mean


def bootstrap_map_ratio(rows, comparator, metric, repetitions=2000):
    by_map=defaultdict(list)
    for r in rows:
        other=comparator.get(r['case_id'])
        if not r['valid'] or not other or not other['valid']:
            continue
        a,b=r.get(metric),other.get(metric)
        if a is not None and b and a>0:
            by_map[r['map_index']].append(math.log(a/b))
    keys=sorted(by_map)
    if not keys:
        return dict(n=0,maps=0,ratio_percent=None,ci95=None)
    sums=[sum(by_map[k]) for k in keys]
    counts=[len(by_map[k]) for k in keys]
    rng=random.Random(20261013)
    boot=[]
    for _ in range(repetitions):
        chosen=rng.choices(range(len(keys)),k=len(keys))
        boot.append(100*math.exp(sum(sums[i] for i in chosen)/sum(counts[i] for i in chosen)))
    boot.sort()
    return dict(n=sum(counts),maps=len(keys),ratio_percent=100*math.exp(sum(sums)/sum(counts)),
                ci95=[boot[int(.025*repetitions)],boot[int(.975*repetitions)]],repetitions=repetitions)


def run(directory):
    out=Path(directory)
    case_lookup={c['case_id']:c for c in cases()}
    geometries={}
    generations=records(out/'generations.jsonl')
    base=records(out/'evaluations.jsonl')
    prior=records(out/'proposed.jsonl')
    known={r['evaluation_id'] for r in prior}
    manifest_name=manifest(out,dict(action='diagnostics',gate='valid_polyline_and_integer_targets',fallback='goal_only'))
    with (out/'proposed.jsonl').open('a',encoding='utf-8') as f:
        for g in generations:
            if g['status']!='ok': continue
            case=case_lookup[g['case_id']]
            geo=geometries.setdefault(case['map_index'],Geometry(case['query']))
            content=g['response']['choices'][0]['message'].get('content') or ''
            for variant in ['validated','parse_only','astar_fallback']:
                eid=g['request_sha256']+':'+variant
                if eid in known: continue
                r=dict(case_id=case['case_id'],map_index=case['map_index'],model=g['model'],prompt_style=g['prompt_style'],
                       method=variant,evaluation_id=eid,request_sha256=g['request_sha256'],manifest=manifest_name,
                       **proposed_search(case['query'],geo,content,variant))
                append(f,r)
                known.add(eid)
    proposed=records(out/'proposed.jsonl')
    astar={r['case_id']:r for r in base if r['method']=='astar'}
    goal_only={r['case_id']:r for r in base if r['method']=='goal_only'}
    groups=defaultdict(list)
    for r in proposed:
        groups[(r['method'],r['model'],r['prompt_style'])].append(r)
    summaries=[]
    for (variant,model,prompt),rows in sorted(groups.items()):
        upstream={r['case_id']:r for r in base if r['method']=='llm_astar' and r['model']==model and r['prompt_style']==prompt}
        summaries.append(dict(variant=variant,model=model,prompt=prompt,**aggregate(rows,astar),
                              gate_counts=dict(Counter(str(r['gate_reason']) for r in rows)),
                              accepted=sum(r['guidance_accepted'] for r in rows),
                              vs_llm_astar={k:bootstrap_map_ratio(rows,upstream,k) for k in ['operation','storage','length']},
                              vs_goal_only={k:bootstrap_map_ratio(rows,goal_only,k) for k in ['operation','length']}))
    result=dict(proposal=summaries,bootstrap_unit='map; same-case pairs; joint successes only',
                random_seed=20261013,baseline_success={k:sum(r['valid'] for r in base if r['method']==k) for k in ['astar','dwa','goal_only']})
    (out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(evaluations=len(proposed),conditions=len(summaries))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/main')
    run(parser.parse_args().output)
