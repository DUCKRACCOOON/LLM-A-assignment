"""Audit saved bytes, repeat signatures, counters and independent aggregation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from experiments.K3.run import sha,write_json
from .run import ROOT,read_rows,case_seed,METRICS
from .search import BASELINES,METHODS,NEW_METHODS,STOCHASTIC


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    root=args.input;manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed' and manifest['git_branch']=='gm' and manifest['new_api_calls']==0
    for name,value in manifest['input_sha256'].items():assert sha(ROOT/name)==value,name
    for name,value in manifest['fixed_data_sha256'].items():assert sha(root/name)==value,name
    totals={}
    for group,ncases in [('pilot',30),('public',1000),('transfer',30)]:
        counts=defaultdict(int);buckets=defaultdict(list);signatures={};bounds=0
        for r in read_rows(root/group/'evaluations.jsonl.gz'):
            assert r['valid'] and r['method'] in METHODS
            key=(r['case_id'],r['method'],r['seed_index']);counts[key]+=1
            if r['method'] in STOCHASTIC:assert 0<=r['seed_index']<5 and r['seed']==case_seed(r['case_id'],r['seed_index'])
            else:assert r['seed_index']==-1
            sig={k:r.get(k) for k in ('path','operation','storage','length','successor_expansions','alpha_histogram','rekey_scans','final_alpha','queue_expansions','bandit_rewards')}
            if key in signatures:assert signatures[key]==sig
            signatures[key]=sig;buckets[key].append({k:r[k] for k in METRICS})
            if r.get('bound_factor') is not None:
                bounds+=1;assert r['length']<=r['bound_factor']*r['optimum']+1e-8
            if r['method'] in NEW_METHODS:
                assert r['operation']==r['stale_pops']+r['successor_expansions']+1
                assert r['rekey_scans']==r['rekey_entries']
                assert sum(r['alpha_histogram'].values())==r['successor_expansions']
        assert len(counts)==ncases*31 and set(counts.values())=={3}
        checks=json.loads((root/group/'checks.json').read_text(encoding='utf-8'))
        assert checks['runs']==ncases*93 and checks['independent_path_checks']==checks['runs']
        assert checks['baseline_parity_checks']==ncases*66 and checks['no_guide_parity_checks']==ncases*3
        assert checks['bound_checks']==bounds and checks['bound_violations']==0
        per_case=defaultdict(list)
        for (cid,m,si),rows in buckets.items():
            value=dict(rows[0]);value['total_seconds']=statistics.median(r['total_seconds'] for r in rows)
            per_case[cid,m].append(value)
        means={key:{metric:statistics.mean(r[metric] for r in rows) for metric in METRICS} for key,rows in per_case.items()}
        summary=json.loads((root/group/'summary.json').read_text(encoding='utf-8'))
        aggregates=0
        for method in METHODS:
            for metric in METRICS:
                vals=[math.log(v[metric]/means[cid,'g2h_euclidean'][metric]) for (cid,m),v in means.items() if m==method]
                expected=math.exp(statistics.mean(vals));actual=summary['comparisons']['g2h_euclidean'][method][metric]['geometric_ratio']
                assert math.isclose(expected,actual,rel_tol=1e-12);aggregates+=1
        totals[group]=dict(runs=checks['runs'],seed_trials=len(counts),baseline_parity=checks['baseline_parity_checks'],
                           no_guide_parity=checks['no_guide_parity_checks'],independent_paths=checks['independent_path_checks'],
                           bound_checks=bounds,aggregate_ratios=aggregates)
    analysis=json.loads((root/'analysis/manifest.json').read_text(encoding='utf-8'))
    for name,value in analysis['input_sha256'].items():assert sha(Path(name))==value,name
    result=dict(status='passed',groups=totals,input_hashes=len(manifest['input_sha256']),
                fixed_data_hashes=len(manifest['fixed_data_sha256']),analysis_hashes=len(analysis['input_sha256']))
    write_json(root/'final_audit.json',result)
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'}
    write_json(root/'artifacts_sha256.json',files);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
