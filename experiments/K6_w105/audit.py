"""Independent checks of saved repeats, parity counts, medians and summaries."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from experiments.K3.run import sha,write_json
from .run import ROOT,REPEATS,METRICS,read_rows
from .search import METHODS,TIGHT,PRIMARY


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed' and manifest['git_branch']=='gm' and manifest['new_api_calls']==0
    for name,value in manifest['input_sha256'].items():assert sha(ROOT/name)==value,name
    for name,value in manifest['fixed_data_sha256'].items():assert sha(root/name)==value,name
    totals={}
    for group,n in [('pilot',30),('public',1000),('transfer',30)]:
        times=defaultdict(list);first={};signatures={};bounds=0
        for r in read_rows(root/group/'evaluations.jsonl.gz'):
            key=r['case_id'],r['method'];assert r['valid'] and r['method'] in METHODS
            times[key].append(r['total_seconds'])
            sig={k:v for k,v in r.items() if k not in ('total_seconds','seconds','order','repeat')}
            if key in signatures:assert sig==signatures[key]
            else:signatures[key]=sig;first[key]=r
            if r['method'] in TIGHT:
                bounds+=1;assert r['length']<=1.05*r['optimum']+1e-8
                assert r['operation']==r['stale_pops']+r['successor_expansions']+1
                assert r['rekey_entries']==r['rekey_scans']
                assert sum(r['alpha_histogram'].values())==r['successor_expansions']
        assert len(times)==n*len(METHODS) and all(len(v)==REPEATS for v in times.values())
        checks=json.loads((root/group/'checks.json').read_text(encoding='utf-8'))
        assert checks['runs']==checks['independent_paths']==n*35
        assert checks['bound_checks']==bounds==n*20 and checks['bound_violations']==0
        assert checks['previous_parity']==n*20 and checks['repeat_checks']==n*28
        assert checks['no_guide_reference_runs']==n and checks['no_guide_reference_checks']==n*5
        compact={ (r['case_id'],r['method']):r for r in read_rows(root/group/'per_case.jsonl.gz') }
        assert set(compact)==set(first)
        for key,values in times.items():
            assert compact[key]['total_seconds']==statistics.median(values)
            for metric in METRICS:
                if metric!='total_seconds':assert compact[key][metric]==first[key][metric]
        fallbacks=0
        for (cid,m),r in first.items():
            if m==PRIMARY and r['guide_kept_points']==0:
                for method in TIGHT:
                    for field in ('path','operation','storage','length','successor_expansions'):
                        assert first[cid,method][field]==first[cid,'no_guide_wall_w105'][field]
                    fallbacks+=1
        summary=json.loads((root/group/'summary.json').read_text(encoding='utf-8'));ratios=0
        for control in METHODS:
            for method in METHODS:
                for metric in METRICS:
                    values=[]
                    for (cid,m),r in first.items():
                        if m!=method:continue
                        v=statistics.median(times[cid,m]) if metric=='total_seconds' else r[metric]
                        b=statistics.median(times[cid,control]) if metric=='total_seconds' else first[cid,control][metric]
                        values.append(math.log(v/b))
                    expected=math.exp(math.fsum(values)/len(values));actual=summary['comparisons'][control][method][metric]['geometric_ratio']
                    assert math.isclose(expected,actual,rel_tol=1e-12);ratios+=1
        totals[group]=dict(runs=checks['runs'],bound_checks=bounds,previous_parity=checks['previous_parity'],
            no_guide_reference_checks=checks['no_guide_reference_checks'],fallback_path_checks=fallbacks,
            median_checks=len(times),independent_ratios=ratios)
    analysis=json.loads((root/'analysis/manifest.json').read_text(encoding='utf-8'))
    for name,value in analysis['input_sha256'].items():assert sha(Path(name))==value,name
    result=dict(status='passed',groups=totals,input_hashes=len(manifest['input_sha256']),fixed_data_hashes=len(manifest['fixed_data_sha256']),analysis_hashes=len(analysis['input_sha256']))
    write_json(root/'final_audit.json',result)
    write_json(root/'artifacts_sha256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
