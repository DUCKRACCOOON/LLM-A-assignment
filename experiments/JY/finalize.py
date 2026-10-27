"""Post-report audit: paired arithmetic, saved K3 parity, links and final hashes."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import unquote

from experiments.K3.run import sha,write_json
from .run import ROOT,read_rows
from .report import METRICS


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    audit=json.loads((out/'final_audit.json').read_text());assert audit['passed']
    manifest=json.loads((out/'manifest.json').read_text())
    for k,v in manifest['source_sha256'].items():assert sha(ROOT/k)==v,('frozen source',k)
    for k,v in manifest['input_sha256'].items():assert sha(ROOT/k)==v,('input',k)
    k3={r['case_id']:r for r in read_rows(ROOT/'experiments/K3/results/run_20261025/public/evaluations.jsonl')
        if r['repeat']==0 and r['method']=='g2h_wall_euclidean'}
    groupchecks={};geometries={}
    for group in ('public','transfer','synthetic'):
        maps={}
        for c in json.loads((out/(group+'_cases.json')).read_text(encoding='utf-8')):
            q=c['query']
            signature=json.dumps([q['range_x'],q['range_y'],
                sorted((y,min(a,b),max(a,b)) for y,a,b in q['horizontal_barriers'] if a!=b),
                sorted((x,min(a,b),max(a,b)) for x,a,b in q['vertical_barriers'] if a!=b)])
            maps[c['map_index']]=signature
        assert len(set(maps.values()))==len(maps)
        geometries[group]=set(maps.values())
        buckets=defaultdict(list);parity=0
        for r in read_rows(out/group/'evaluations.jsonl.gz'):
            buckets[r['method'],r['case_id']].append(r)
            if group=='public' and r['method']=='k3_wall':
                old=k3[r['case_id']]
                for metric in ('path','length','operation','storage','valid'):assert r[metric]==old[metric],(r['case_id'],metric)
                parity+=1
        by=defaultdict(dict)
        for (m,cid),trials in buckets.items():
            by[m][cid]={k:statistics.median(r[k] for r in trials) for k in METRICS}
        summary=json.loads((out/group/'summary.json').read_text());comparisons=0
        for base,methods in summary['comparisons'].items():
            for m,measurements in methods.items():
                ids=sorted(set(by[m])&set(by[base]))
                for metric,stored in measurements.items():
                    ratios=[]
                    for cid in ids:
                        a,b=by[m][cid][metric],by[base][cid][metric]
                        ratios.append(a/b if metric in ('total_seconds','query_seconds','length') else max(a,1)/max(b,1))
                    gm=math.exp(sum(map(math.log,ratios))/len(ratios))
                    assert math.isclose(gm,stored['ratio'],rel_tol=1e-12)
                    assert stored['n']==len(ids) and 0<stored['ci95'][0]<=stored['ci95'][1]
                    comparisons+=1
        groupchecks[group]=dict(paired_aggregate_rechecks=comparisons,saved_k3_baseline_parity=parity,passed=True)
    assert geometries['synthetic'].isdisjoint(geometries['public']|geometries['transfer'])
    files=list((ROOT/'experiments/JY').glob('*.md'))
    for method in ('J3','J4','J5','J6'):files+=list((ROOT/f'experiments/{method}').glob('*.md'))
    linkchecks=0
    for f in files:
        text=f.read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in link or link.startswith('#'):continue
            target=unquote(link.split('#')[0].strip('<>'))
            assert (f.parent/target).exists(),(str(f),target)
            linkchecks+=1
    # No credential values are printed. Check only newly authored deliverables.
    secret_patterns=[re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}')]
    inspected=0
    for directory in ('experiments/JY','experiments/J3','experiments/J4','experiments/J5','experiments/J6','docs'):
        for f in (ROOT/directory).rglob('*'):
            if f.is_file() and f.suffix in ('.py','.md','.json','.csv','.jsonl'):
                data=f.read_bytes();assert not any(p.search(data) for p in secret_patterns),('credential pattern',str(f))
                inspected+=1
    result=dict(passed=True,groups=groupchecks,markdown_links_checked=linkchecks,
        deliverable_text_files_secret_scanned=inspected,new_api_calls=0,finalize_code_sha256=sha(Path(__file__)))
    result['new_synthetic_map_geometries_unique_and_disjoint']=True
    write_json(out/'supplemental_audit.json',result)
    hashes={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifacts_sha256.json'}
    write_json(out/'artifacts_sha256.json',hashes)
    # Hash authored code/docs as well as raw artifacts. This excludes itself.
    authored={}
    for directory in ('experiments/JY','experiments/J3','experiments/J4','experiments/J5','experiments/J6','docs'):
        for f in sorted((ROOT/directory).rglob('*')):
            if f.is_file() and 'results' not in f.parts and '__pycache__' not in f.parts and f.name!='deliverables_sha256.json':
                authored[str(f.relative_to(ROOT))]=sha(f)
    (ROOT/'experiments/JY/deliverables_sha256.json').write_bytes(
        (json.dumps(authored,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
