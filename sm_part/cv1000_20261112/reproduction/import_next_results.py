"""Verify a server result archive before copying only new result files locally."""
import argparse
import hashlib
import json
import math
import statistics
import time
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path,PurePosixPath
from core import ROOT,World,path_length
from next_data import digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('--sha256',required=True);a=ap.parse_args()
    archive=Path(a.archive);assert hashlib.sha256(archive.read_bytes()).hexdigest()==a.sha256
    out=ROOT/'results/next_search_v1';t=time.perf_counter()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names));assert z.testzip() is None
        manifest=json.loads(z.read('archive_manifest.json'));assert set(names)==set(manifest)|{'archive_manifest.json'}
        for name,h in manifest.items():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            assert name.startswith(('results/next_search_v1/','reproduction/'))
            assert hashlib.sha256(z.read(name)).hexdigest()==h,name
            if name.startswith('reproduction/'):
                assert (ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,('source differs',name)
        for name in manifest:
            if name.startswith('results/next_search_v1/'):
                p=(ROOT/name).resolve();assert p.is_relative_to(out.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    frozen=json.loads((out/'protocol_frozen.json').read_text())
    status=json.loads((out/'status.json').read_text())
    assert status['phase']=='complete' and status['complete'] is True
    signature=frozen['signature'];assert digest(dict(frozen,signature=None))==signature
    expected_hash=digest(frozen['source_hashes'])
    cases={}
    for p in (out/'data').glob('*.json'):
        if p.stem.endswith('_manifest'):continue
        data=json.loads(p.read_text())
        if isinstance(data,list):cases.update({q['id']:q for q in data})
    geoms={}
    for q in cases.values():geoms[(q['base_map_id'],q.get('scale',1))]=q
    @lru_cache(maxsize=8)
    def world(key):return World(geoms[key])
    checked=set();rows=0;failures=0;counts=Counter()
    for p in sorted((out/'evaluation').glob('*/*/*.json')):
        r=json.loads(p.read_text());rows+=1;q=cases[r['id']];counts[(r['stage'],r['method'])]+=1
        if rows%5000==0:print(json.dumps(dict(progress_rows=rows,unique_paths_checked=len(checked))),flush=True)
        assert r['input_hash']==digest(q) and r['source_hash']==expected_hash
        if r['status']!='ok':failures+=1;continue
        path=r['path'];assert tuple(path[0])==tuple(q['start']) and tuple(path[-1])==tuple(q['goal'])
        key=(q['base_map_id'],q.get('scale',1));w=world(key)
        pathkey=(key,tuple(map(tuple,path)))
        if pathkey not in checked:
            for v in path:assert all(type(c)==int for c in v) and 0<v[0]<w.x1 and 0<v[1]<w.y1
            for v,u in zip(path,path[1:]):
                assert max(abs(v[0]-u[0]),abs(v[1]-u[1]))==1
                assert math.isfinite(w.edge_cost(tuple(v),tuple(u)))
            checked.add(pathkey)
        length=path_length(path);assert math.isclose(length,r['length'],abs_tol=1e-8,rel_tol=1e-10)
        if r['optimal'] is not None:
            assert length>=r['optimal']-1e-8-1e-10*r['optimal']
            if r['bound'] is not None:assert length<=r['bound']*r['optimal']+1e-8+1e-10*r['optimal']
        assert len(r['timings'])==5
    server=json.loads((out/'verification.json').read_text());assert rows==server['total_rows'] and server['verified']
    expected_final=frozen['final_maps']*10
    for method in ['A_OCT','N_LEGACY','N_CLEAN','WA_101','WA_102','WA_200','FOCAL_101','FOCAL_102','R1','R2','R1_R2']:
        assert counts[('final',method)]==expected_final,(method,counts[('final',method)])
    for method in ['A_OCT','N_CLEAN','R1','R2','R1_R2']:
        assert counts[('scaling',method)]==320
    modes=['rule','random']+(['llm'] if frozen['selected']['r3_gate'] else [])
    for w in (1.,1.01,1.02):
        for mode in modes:
            for seed in ((26101431,26101432,26101433) if mode=='random' else (0,)):
                assert counts[('r3_final',f'TOPO:{mode}:{w}:{seed}')]==400
    splits=[json.loads((out/'data'/f'{s}.json').read_text()) for s in ('development','validation','final')]
    hashes=[{q['geometry_hash'] for q in qs} for qs in splits]
    assert all(not hashes[i].intersection(hashes[j]) for i in range(3) for j in range(i))
    reuse=[json.loads(p.read_text()) for p in (out/'reuse').glob('*/*/*.json')]
    reuse_counts=Counter((r['workload'],r['method'],r['queries']) for r in reuse)
    for workload in ('varying','fixed'):
        for method in ('A_OCT','R1','R2','R1_R2','FOCAL_102','FMM_UNIFORM','REVERSE_DIJKSTRA'):
            for n in (1,5,10):assert reuse_counts[(workload,method,n)]==20
    for r in reuse:
        assert len(r['timings'])==5
        assert math.isclose(statistics.median(t['total_seconds'] for t in r['timings']),r['total_seconds'],rel_tol=1e-12)
        assert math.isclose(r['total_seconds']/r['queries'],r['mean_query_seconds'],rel_tol=1e-12)
    result=dict(verified=True,archive_sha256=a.sha256,archive_files=len(manifest),rows=rows,failures=failures,
                unique_paths_checked=len(checked),source_files_match=True,protocol_signature_verified=True,
                expected_final_scaling_r3_counts_verified=True,map_disjoint_splits_verified=True,
                reuse_batches_verified=len(reuse),reuse_paths='server audited on each repeat; not stored for local path re-audit',seconds=time.perf_counter()-t)
    (out/'local_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
