"""Independently audit all persisted queries, responses, paths and aggregates."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics as st

from .run import ROOT,METHODS,load,replies
from experiments.K3.run import public_cases,sha
from experiments.K3.verification import Oracle
from experiments.COMMON18.prepare import MODEL,save,digest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    protocol=load(out/'protocol.json');complete=load(out/'complete.json');assert complete['complete']
    for name,value in protocol['source_sha256'].items():assert sha(ROOT/name)==value,('source changed',name)
    assert sha(out/'responses.jsonl.gz')==protocol['frozen_response_sha256']
    assert sha(ROOT/protocol['source_response_file'])==protocol['source_response_sha256']
    assert sha(ROOT/'source_code/dataset/environment_50_30.json')==protocol['data_sha256']
    assert sha(Path(__file__).with_name('PROTOCOL.md'))==protocol['protocol_sha256']
    cases=load(out/'cases.json');assert cases==public_cases();assert len(cases)==1000
    cache=replies(out);assert set(cache)=={c['case_id'] for c in cases}
    template=runpy.run_path(str(ROOT/'source_code/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']['standard']
    finishes=Counter()
    for c in cases:
        q=c['query'];r=cache[c['case_id']]['standard']
        text=template.format(start=q['start'],goal=q['goal'],horizontal_barriers=q['horizontal_barriers'],vertical_barriers=q['vertical_barriers'])
        expected=dict(model=MODEL,messages=[dict(role='user',content=text)],temperature=0,max_tokens=1000,store=False)
        assert r['request']==expected and r['request_sha256']==digest(expected) and r['status']=='ok'
        assert r['model']==MODEL and r['response']['model']==MODEL
        finishes[r['response']['choices'][0]['finish_reason']]+=1
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    counts=Counter((r['case_id'],r['method']) for r in records)
    assert len(records)==7000 and set(counts.values())=={1}
    assert set(counts)=={(c['case_id'],m) for c in cases for m in METHODS}
    bycase=defaultdict(list)
    for r in records:bycase[r['case_id']].append(r)
    checked=boundchecks=exactchecks=failures=0;optima=load(out/'optima.json')
    for c in cases:
        q=c['query']
        if c['pair_index']==0:oracle=Oracle(q)
        optimum=oracle.distances(q['goal'])[tuple(q['start'])];assert math.isclose(optimum,optima[c['case_id']],abs_tol=1e-8)
        for r in bycase[c['case_id']]:
            assert r['map_index']==c['map_index'] and r['pair_index']==c['pair_index'] and r['repeat']==0
            assert abs(r['optimum']-optimum)<1e-8
            if r['valid']:
                oracle.check_path(q,r);checked+=1;assert r['length']>=optimum-1e-7
                if r['bound']:
                    assert r['length']<=optimum*r['bound']+1e-7;boundchecks+=1
                if r['bound']==1:exactchecks+=1
            else:failures+=1
            called=r['method'] in ('LLMA','J1','J1_exact');response=cache[c['case_id']]['standard']
            assert r['llm_calls']==int(called)
            assert r['used_responses']==(['standard'] if called else [])
            assert r['response_sha256']==(response['request_sha256'] if called else None)
            assert r['api_seconds']==(response['seconds'] if called else 0)
            for k in ('prompt_tokens','completion_tokens'):
                assert r[k]==(response['response']['usage'][k] if called else 0)
            assert abs(r['reconstructed_total_seconds']-r['api_seconds']-r['local_seconds'])<1e-10
    assert checked==complete['path_checks'] and boundchecks==complete['bound_checks'] and exactchecks==complete['exact_checks']
    assert failures==complete['failed']==len(load(out/'failures.json'))
    geometry=load(out/'geometry.json');assert {r['map_index'] for r in geometry}==set(range(100))
    summary=load(out/'summary.json');rechecks=0
    for group,maps in [('all1000',set(range(100))),('common18_test200',set(protocol['common18_test_maps']))]:
        group_records=[r for r in records if r['map_index'] in maps]
        base={r['case_id']:r for r in group_records if r['method']=='LLMA'}
        for m in METHODS:
            rs=[r for r in group_records if r['method']==m];v=summary['groups'][group]['absolute'][m]
            assert v['cases']==len(rs) and v['valid']==sum(r['valid'] for r in rs)
            for key,value in v['means'].items():
                assert math.isclose(value,st.mean(r[key] for r in rs),rel_tol=1e-12,abs_tol=1e-12);rechecks+=1
            valid=[r for r in rs if r['valid']]
            expected=math.exp(st.mean(math.log(r['length']/r['optimum']) for r in valid))
            assert math.isclose(v['stretch_gm'],expected,rel_tol=1e-12)
            exact=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs)
            assert math.isclose(v['exact_percent'],exact,abs_tol=1e-10)
            repair=v['repair']
            assert repair['repaired_valid']==sum(r['metadata'].get('repaired',False) for r in rs)
            assert repair['normalized_input_valid']==sum(r['metadata'].get('input_valid',False) for r in rs)
            assert repair['invalid_to_valid']==sum(not r['metadata'].get('input_valid',False) and r['metadata'].get('repaired',False) for r in rs)
            assert repair['valid_to_invalid']==sum(r['metadata'].get('input_valid',False) and not r['metadata'].get('repaired',False) for r in rs)
            assert repair['incumbent_returned']==sum(r.get('incumbent_returned',False) for r in rs)
            assert math.isclose(repair['mean_seconds'],st.mean(r['metadata'].get('repair_seconds',0) for r in rs),abs_tol=1e-12)
            for key in ('local_seconds','reconstructed_total_seconds','successor_expansions','length'):
                vals=[math.log(max(r[key],1e-12)/max(base[r['case_id']][key],1e-12)) for r in rs if r[key] is not None and base[r['case_id']][key] is not None]
                expected=math.exp(st.mean(vals));actual=summary['groups'][group]['comparisons']['LLMA'][m][key]['geometric_ratio']
                assert math.isclose(actual,expected,rel_tol=1e-12);rechecks+=1
    patterns=[re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}')]
    text_files=0
    for path in Path(__file__).parent.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.json','.csv'):
            assert not any(p.search(path.read_bytes()) for p in patterns),('credential pattern',str(path));text_files+=1
    with gzip.open(out/'responses.jsonl.gz','rb') as f:response_bytes=f.read()
    assert not any(p.search(response_bytes) for p in patterns)
    result=dict(passed=True,cases=1000,maps=100,conditions=len(METHODS),repeats=1,runs=len(records),new_api_calls=0,
        independently_checked_paths=checked,bound_checks=boundchecks,exact_checks=exactchecks,failed=failures,
        independent_dijkstra_queries=1000,geometry_maps=100,directed_edges=sum(r['directed_edges'] for r in geometry),
        request_checks=1000,finish_reasons=dict(finishes),aggregate_rechecks=rechecks,credential_scanned_text_files=text_files,
        source_hashes_match=True,frozen_responses_unchanged=True,audit_source_sha256=sha(Path(__file__)))
    save(out/'audit.json',result)
    save(out/'artifacts_sha256.json',{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts_sha256.json'})
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
