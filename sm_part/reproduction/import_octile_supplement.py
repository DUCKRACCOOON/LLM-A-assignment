"""Locally verify the isolated development-only heuristic supplement."""
import argparse
import hashlib
import json
import math
import zipfile
from functools import lru_cache
from pathlib import Path,PurePosixPath
from core import ROOT,World,path_length
from next_data import digest

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('--sha256',required=True);a=ap.parse_args()
    arc=Path(a.archive);assert hashlib.sha256(arc.read_bytes()).hexdigest()==a.sha256
    out=ROOT/'results/next_octile_supplement'
    with zipfile.ZipFile(arc) as z:
        names=z.namelist();assert len(names)==len(set(names)) and z.testzip() is None
        manifest=json.loads(z.read('archive_manifest.json'));assert set(names)==set(manifest)|{'archive_manifest.json'}
        for name,h in manifest.items():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            assert hashlib.sha256(z.read(name)).hexdigest()==h
            assert name=='supplement_octile.py' or name.startswith('results/next_octile_supplement/')
            if name=='supplement_octile.py':assert hashlib.sha256((ROOT/'reproduction'/name).read_bytes()).hexdigest()==h
        for name in manifest:
            if name.startswith('results/'):
                p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    mainout=ROOT/'results/next_search_v1';assert read(mainout/'local_verification.json')['verified']
    protocol=read(out/'protocol.json');frozen=read(mainout/'protocol_frozen.json')
    assert protocol['source_hashes']==frozen['source_hashes'] and protocol['main_signature']==frozen['signature']
    assert read(out/'status.json')['complete']
    cases={q['id']:q for q in read(mainout/'data/development.json')};oracle=read(mainout/'oracle/r1_development.json')
    maps={q['map_id']:q for q in cases.values()}
    @lru_cache(maxsize=4)
    def world(m):return World(maps[m])
    rows=[]
    for p in sorted((out/'evaluation').glob('*/*.json')):
        r=read(p);q=cases[r['id']];rows.append(r)
        assert r['input_hash']==digest(q) and r['optimal']==oracle[q['id']]['optimal']
        assert r['status']=='ok' and len(r['timings'])==5
        path=r['path'];assert path[0]==q['start'] and path[-1]==q['goal'];w=world(q['map_id'])
        assert all(all(type(c)==int for c in v) and 0<v[0]<w.x1 and 0<v[1]<w.y1 for v in path)
        for a0,b0 in zip(path,path[1:]):
            assert max(abs(a0[0]-b0[0]),abs(a0[1]-b0[1]))==1 and math.isfinite(w.edge_cost(tuple(a0),tuple(b0)))
        assert math.isclose(path_length(path),r['length'],abs_tol=1e-8,rel_tol=1e-10)
        assert math.isclose(r['length'],r['optimal'],abs_tol=1e-8,rel_tol=1e-10)
    assert len(rows)==1200
    for m in ('A_OCT','A_EUC'):assert {r['id'] for r in rows if r['method']==m}==set(cases)
    # Bootstrap RNG consumption follows the server's development-map order.
    # Alphabetical filesystem order otherwise changes finite bootstrap draws.
    case_order={key:i for i,key in enumerate(cases)}
    rows.sort(key=lambda r:case_order[r['id']])
    from next_report import comparison
    expected=read(out/'summary.json')
    for label,metric in [('time','total_seconds'),('expansions','expansions')]:
        c=comparison(rows,'A_EUC','A_OCT',metric)
        assert all(math.isclose(c[k],expected[label][k],rel_tol=1e-12) for k in ('ratio','low','high'))
    v=dict(verified=True,rows=len(rows),exact_path_checks=len(rows),archive_sha256=a.sha256,
           script_and_frozen_sources_match=True,summary_recomputed=True)
    (out/'local_verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

if __name__=='__main__':main()
