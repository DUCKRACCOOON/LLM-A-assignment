"""Create tables from recorded observations; never invent missing rows."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from .experiment import records, MODELS, PROMPTS


def geometric_mean(values):
    values = list(values)
    if not values:
        return None
    if any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError('Geometric mean needs strictly positive finite ratios')
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def aggregate(group, baseline):
    valid = [r for r in group if r['valid']]
    row = dict(n=len(group), valid_n=len(valid), valid_percent=100*len(valid)/len(group) if group else None,
               failures=dict(Counter(r.get('error') for r in group if not r['valid'])))
    for metric in ('operation','storage','length'):
        ratios = [r[metric]/baseline[r['case_id']][metric] for r in valid
                  if r.get(metric) is not None and baseline[r['case_id']].get(metric,0)>0]
        ratio = geometric_mean(ratios)
        row[metric+'_percent'] = 100*ratio if ratio is not None else None
        row[metric+'_n'] = len(ratios)
    return row


def summarize(directory):
    out = Path(directory)
    evaluations = records(out/'evaluations.jsonl')
    generations = records(out/'generations.jsonl')
    baseline = {r['case_id']:r for r in evaluations if r['method']=='astar'}
    groups = defaultdict(list)
    for r in evaluations:
        groups[(r['method'],r['model'],r['prompt_style'])].append(r)
    expected = [('astar',None,None),('dwa',None,None)]
    expected += [(method,model,style) for method in ('llm_only','llm_astar') for model in MODELS for style in PROMPTS]
    expected += [('goal_only',None,None)]
    rows = []
    for method,model,style in expected:
        row = dict(method=method,model=model,prompt_style=style,
                   **aggregate(groups[(method,model,style)],baseline))
        rows.append(row)
    usage = defaultdict(lambda:dict(requests=0,api_failures=0,prompt_tokens=0,completion_tokens=0,
                                    cached_tokens=0,transport_retries=0,finish_reasons=Counter(),latencies=[]))
    for r in generations:
        u=usage[r['model']]
        u['requests']+=1
        u['transport_retries']+=max(0,len(r['attempts'])-(r['status']!='ok'))
        u['latencies'].append(r['seconds'])
        if r['status']!='ok':
            u['api_failures']+=1
            continue
        actual=r['response']['usage']
        for k in ['prompt_tokens','completion_tokens']:
            u[k]+=actual[k]
        u['cached_tokens']+=actual.get('prompt_tokens_details',{}).get('cached_tokens',0)
        u['finish_reasons'][r['response']['choices'][0]['finish_reason']]+=1
    for u in usage.values():
        times=sorted(u.pop('latencies'))
        u['mean_request_seconds']=sum(times)/len(times)
        u['p95_request_seconds']=times[min(len(times)-1,math.ceil(len(times)*.95)-1)]
    result=dict(rows=rows,usage=dict(usage),baseline_cases=len(baseline),generations=len(generations),
                length_denominator='Only collision-free paths with correct endpoints; see length_n. LLM-only operates on polylines.',
                status='complete' if len(baseline)==1000 and all(row['n']==1000 for row in rows[:14]) and all(r['status']=='ok' for r in generations) else 'partial')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['method','model','prompt_style','n','valid_n','valid_percent','operation_percent','operation_n','storage_percent','storage_n','length_percent','length_n']
    with (out/'table1.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows[:14])
    text=['# Recorded reproduction results','',f"Status: **{result['status']}**. Generations: {len(generations)}.",'',
          '| Method | Model | Prompt | N | Valid % | Operation % | Storage % | Length % (valid n) |',
          '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(v): return '—' if v is None else f'{v:.2f}'
    for r in rows:
        text.append(f"| {r['method']} | {r['model'] or '—'} | {r['prompt_style'] or '—'} | {r['n']} | {fmt(r['valid_percent'])} | {fmt(r['operation_percent'])} | {fmt(r['storage_percent'])} | {fmt(r['length_percent'])} ({r['length_n']}) |")
    text.extend(['','All percentages are ratios to the same-case A* baseline, except valid-path percentage.',
                 'Geometric means use valid paths only; each metric has its own recorded denominator.',
                 'goal_only is an additional g+2h diagnostic, excluded from the 14-row Table 1.',
                 'DWA is our documented interpretation of the incompletely specified baseline.'])
    (out/'table1.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/main')
    summarize(parser.parse_args().output)
