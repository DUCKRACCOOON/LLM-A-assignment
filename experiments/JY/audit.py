"""Audit persisted experiment: independent graph paths, counters and summaries."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.K3.run import write_json,sha
from experiments.K3.verification import Oracle
from .run import ROOT,read_rows
from .search import EXACT


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed'
    assert manifest['git_branch']=='jy' and manifest['new_api_calls']==0
    for k,v in manifest['source_sha256'].items():assert sha(ROOT/k)==v,('source changed',k)
    for k,v in manifest['input_sha256'].items():assert sha(ROOT/k)==v,('input changed',k)
    totals={}
    for group,expected_n in [('public',1000),('transfer',30),('synthetic',120)]:
        cases={c['case_id']:c for c in json.loads((out/(group+'_cases.json')).read_text(encoding='utf-8'))}
        assert len(cases)==expected_n
        optima=json.loads((out/group/'optima.json').read_text());checks=json.loads((out/group/'checks.json').read_text())
        summary=json.loads((out/group/'summary.json').read_text());counts=Counter();paired=defaultdict(dict);last=None;oracle=None
        metrics=defaultdict(list);seen={};pathchecks=optimalchecks=parity=0
        for r in read_rows(out/group/'evaluations.jsonl.gz'):
            assert r['new_api_calls']==0 and r['valid'];c=cases[r['case_id']]
            if c['map_index']!=last:oracle=Oracle(c['query']);last=c['map_index']
            oracle.check_path(c['query'],r);pathchecks+=1
            assert abs(r['optimum']-optima[r['case_id']])<1e-8
            if r['method'] in EXACT:
                assert abs(r['length']-r['optimum'])<1e-8;optimalchecks+=1
            counts[r['case_id'],r['method']]+=1
            sig={k:r.get(k) for k in ('path','length','operation','storage','expansions','target_changes','pop_trace_sha256')}
            key=(r['case_id'],r['method'])
            if key in seen:assert seen[key]==sig
            seen[key]=sig;metrics[r['method']].append(r)
            if r['method'].startswith('j6_'):
                pair=(r['case_id'],r['repeat']);paired[pair][r['method']]=r
                if len(paired[pair])==2:
                    a,b=paired.pop(pair).values()
                    for k in ('path','length','storage','expansions','target_changes','pop_trace_sha256'):assert a[k]==b[k]
                    parity+=1
            if r['method'].startswith('j3_'):assert r['operation']==r['seed_operation']+r['refinement_operation']
            if r['method'].startswith('j4_'):
                assert r['cold_operation']==r['operation']+r['preprocessing_pops']
                assert math.isclose(r['total_seconds'],r['query_seconds']+r['preprocessing_seconds'],abs_tol=1e-12)
        assert not paired and set(counts.values())=={3}
        assert pathchecks==checks['runs']==expected_n*len(checks['methods'])*3
        assert optimalchecks==checks['optimality_checks'];assert parity==checks.get('j6_parity_checks',0)
        aggregationchecks=0
        for m,rs in metrics.items():
            bycase=defaultdict(list)
            for r in rs:bycase[r['case_id']].append(r)
            s=summary['methods'][m];assert s['n']==expected_n and s['valid']==expected_n
            for metric in ('operation','expansions','storage','length','total_seconds','query_seconds','cold_operation','cold_expansions'):
                vals=[statistics.median(r[metric] for r in trials) for trials in bycase.values()]
                assert math.isclose(statistics.mean(vals),s['means'][metric],rel_tol=1e-12)
                assert math.isclose(sum(vals),s['totals'][metric],rel_tol=1e-12)
                aggregationchecks+=2
        totals[group]=dict(cases=expected_n,runs=pathchecks,independent_path_checks=pathchecks,
            optimality_checks=optimalchecks,j6_pop_sequence_parity=parity,aggregate_checks=aggregationchecks,passed=True)
        print(f'Audit {group}: {pathchecks} paths passed',flush=True)
    # Compare source-independent oracle optimum values to the run's recorded
    # oracle: all goals are recomputed once, not inferred from A* results.
    for group in totals:
        last=None;oracle=None
        for c in json.loads((out/(group+'_cases.json')).read_text(encoding='utf-8')):
            if c['map_index']!=last:oracle=Oracle(c['query']);last=c['map_index']
            actual=oracle.distances(c['query']['goal'])[tuple(c['query']['start'])]
            saved=json.loads((out/group/'optima.json').read_text())[c['case_id']]
            assert abs(actual-saved)<1e-8
    result=dict(passed=True,groups=totals,new_api_calls=0,input_and_frozen_source_hashes_verified=True,
        independent_optimum_rechecks=1150,audit_code_sha256=sha(Path(__file__)))
    write_json(out/'final_audit.json',result)
    hashes={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifacts_sha256.json'}
    write_json(out/'artifacts_sha256.json',hashes)
    print('All stored records, independent paths, optima, aggregates and hashes passed.')


if __name__=='__main__':main()
