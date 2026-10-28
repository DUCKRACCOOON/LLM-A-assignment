"""Independent persisted-data audit: requests, map separation, paths, bounds."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experiments.K3.verification import Oracle
from experiments.K3.run import sha,public_cases
from .prepare import save,rows,digest,split,MODEL
from .backends import ROOT,METHODS,PRIMARY


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();out=a.input.resolve()
    completed=json.loads((out/'benchmark_complete.json').read_text(encoding='utf-8'));assert completed['passed']
    manifest=json.loads((out/'benchmark_manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,('timed source changed',name)
    assert sha(out/'generations.jsonl')==manifest['collection_sha256'],'Responses changed during replay'
    assert sha(out/'calibration.json')==manifest['calibration_sha256'],'Calibration changed during replay'
    original_protocol=json.loads((Path(__file__).parent/'assets/original_training_protocol.json').read_text(encoding='utf-8'))
    division=json.loads((out/'split.json').read_text(encoding='utf-8'));assert division==split()==original_protocol['splits']
    groups=[set(division[k]) for k in ('train','validation','calibration','test')]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i))
    assert original_protocol['data_sha256']==sha(ROOT/'source_code/dataset/environment_50_30.json')
    allcases={c['case_id']:c for c in public_cases()};casegroups={}
    for stage in ('test','calibration'):
        cases=json.loads((out/(stage+'_cases.json')).read_text(encoding='utf-8'));assert len(cases)==200
        assert {c['map_index'] for c in cases}==set(division[stage])
        for c in cases:assert c==allcases[c['case_id']]
        casegroups[stage]=cases
    generations=rows(out/'generations.jsonl');assert len(generations)<=1600
    cache={r['request_sha256']:r for r in generations};assert len(cache)==len(generations)
    prompt_tokens=completion_tokens=0;kindcounts=Counter()
    for r in generations:
        assert r['status']=='ok' and digest(r['request'])==r['request_sha256']
        request=r['request'];assert request['model']==MODEL and request['temperature']==0 and request['store'] is False
        assert request['max_tokens']==(256 if r['kind']=='ranking' else 1000)
        assert r['response']['model']==MODEL
        prompt_tokens+=r['response']['usage']['prompt_tokens'];completion_tokens+=r['response']['usage']['completion_tokens']
        kindcounts[r['kind']]+=1
    assert prompt_tokens<=5000000 and completion_tokens<=1600000
    index=json.loads((out/'response_index.json').read_text(encoding='utf-8'))
    for c in casegroups['test']+casegroups['calibration']:
        for k in ('standard','cot','repe'):
            r=cache[index[c['case_id']][k]];assert r['case_id']==c['case_id'] and r['kind']==k
    optima=json.loads((out/'optima.json').read_text(encoding='utf-8'))
    cases={c['case_id']:c for c in casegroups['test']};oracle_cache={};distance_checks=0
    for c in cases.values():
        if c['map_index'] not in oracle_cache:oracle_cache[c['map_index']]=Oracle(c['query'])
        opt=oracle_cache[c['map_index']].distances(c['query']['goal'])[tuple(c['query']['start'])]
        assert abs(opt-optima[c['case_id']])<1e-7;distance_checks+=1
    counts=Counter();signatures={};pathchecks=boundchecks=0;percase=defaultdict(list);actualfailures=0
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);c=cases[r['case_id']];q=c['query'];m=r['method'];assert m in METHODS
            key=(r['case_id'],m,r['seed_index']);counts[key]+=1;percase[key].append(r)
            oracle=oracle_cache[c['map_index']]
            if r['valid']:
                oracle.check_path(q,r);pathchecks+=1
                assert r['length']>=r['optimum']-1e-7
                if r['bound'] is not None:
                    assert r['length']<=r['optimum']*r['bound']+1e-7;boundchecks+=1
            else:actualfailures+=1
            assert r['map_index']==c['map_index'] and abs(r['optimum']-optima[r['case_id']])<1e-7
            sig={k:r.get(k) for k in ('valid','path','length','operation','storage','successor_expansions','llm_calls')}
            if key in signatures:assert signatures[key]==sig
            signatures[key]=sig
            charged=[cache[index[r['case_id']][kind]] for kind in r['used_responses']]
            assert r['llm_calls']==len(charged)
            assert abs(r['api_seconds']-sum(x['seconds'] for x in charged))<1e-10
            assert r['prompt_tokens']==sum(x['response']['usage']['prompt_tokens'] for x in charged)
            assert r['completion_tokens']==sum(x['response']['usage']['completion_tokens'] for x in charged)
            assert abs(r['reconstructed_total_seconds']-r['local_seconds']-r['api_seconds'])<1e-10
    assert set(counts.values())=={3}
    expected_runs=200*3*(len(METHODS)+6)  # three K5 variants have two extra seeds
    assert sum(counts.values())==completed['runs']==expected_runs
    assert pathchecks==completed['path_checks'] and boundchecks==completed['bound_checks']
    assert actualfailures==completed['failures']
    # Independent arithmetic recheck, including seed aggregation order.
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));seeds=defaultdict(list)
    diagnostic_rechecks=0
    for m,b in [('K1','K1_rule'),('K1_exact','K1_rule_exact')]:
        pairs=[(percase[cid,m,-1][0],percase[cid,b,-1][0]) for cid in cases]
        expected=dict(cases=len(pairs),same_path=sum(r['path']==v['path'] for r,v in pairs),
            same_expansions=sum(r['successor_expansions']==v['successor_expansions'] for r,v in pairs),
            same_ranking=sum(r['metadata'].get('ranking')==v['metadata'].get('ranking') for r,v in pairs))
        assert summary['diagnostics'][m]==expected
        diagnostic_rechecks+=4
    assert summary['diagnostics']['S1_gate_accepted']==sum(percase[cid,'S1_gate',-1][0]['metadata'].get('accepted',False) for cid in cases)
    assert summary['diagnostics']['K2_full_map_sets']==sum(percase[cid,'K2',-1][0]['metadata'].get('set_fraction')==1.0 for cid in cases)
    diagnostic_rechecks+=2
    metrics=('local_seconds','api_seconds','reconstructed_total_seconds','llm_calls','prompt_tokens','completion_tokens')
    for (cid,m,si),rs in percase.items():
        seeds[m,cid].append({k:statistics.median(r[k] for r in rs) for k in metrics})
    collapsed=defaultdict(dict)
    for (m,cid),rs in seeds.items():collapsed[m][cid]={k:statistics.mean(r[k] for r in rs) for k in metrics}
    rechecks=0
    for m,cs in collapsed.items():
        for k in metrics:
            actual=statistics.mean(v[k] for v in cs.values());stored=summary['absolute'][m]['means'][k]
            assert math.isclose(actual,stored,rel_tol=1e-12,abs_tol=1e-12),(m,k)
            rechecks+=1
        for k in ('local_seconds','reconstructed_total_seconds','llm_calls','prompt_tokens','completion_tokens'):
            floor=1e-12 if 'seconds' in k else 1
            actual=math.exp(statistics.mean(math.log(max(v[k],floor)/max(collapsed['LLMA'][cid][k],floor)) for cid,v in cs.items()))
            stored=summary['comparisons']['LLMA'][m][k]['geometric_ratio']
            assert math.isclose(actual,stored,rel_tol=1e-12,abs_tol=1e-12),(m,k,'paired')
            rechecks+=1
    memory=json.loads((out/'memory_summary.json').read_text(encoding='utf-8'));assert {r['method'] for r in memory}==set(METHODS)
    memory_paths=0
    for r in memory:
        assert r['sampled_peak_rss_bytes']>=r['baseline_rss_bytes'] and r['samples']>0
        ps=json.loads((out/'memory'/(r['method']+'_paths.json')).read_text(encoding='utf-8'));assert len(ps)==10
        for v in ps:
            c=cases[v['case_id']]
            # The worker stores result identity/validity; full paths are audited
            # in timed records. This is not a duplicate independent path check.
            assert c['pair_index']==0;memory_paths+=1
    # No new credential patterns in the deliverables, including raw requests.
    patterns=[re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}')]
    scanned=0
    for path in Path(__file__).parent.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.txt','.json','.jsonl','.csv'):
            data=path.read_bytes();assert not any(p.search(data) for p in patterns),('credential pattern',str(path));scanned+=1
    result=dict(passed=True,requests=len(generations),request_kinds=dict(kindcounts),prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,model=MODEL,logical_cases=200,methods=len(METHODS),primary=len(PRIMARY),
        runs=expected_runs,independent_path_checks=pathchecks,bound_checks=boundchecks,failed_runs=actualfailures,
        independent_dijkstra_checks=distance_checks,aggregate_rechecks=rechecks,diagnostic_rechecks=diagnostic_rechecks,memory_workers=len(memory),
        memory_case_records=memory_paths,map_split_disjoint=True,checkpoint_training_dataset_and_split_match=True,
        current_timed_source_hashes_match=True,responses_unchanged_during_replay=True,
        credential_scanned_text_files=scanned,audit_code_sha256=sha(Path(__file__)))
    save(out/'audit.json',result)
    hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'}
    save(out/'artifacts_sha256.json',hashes)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
