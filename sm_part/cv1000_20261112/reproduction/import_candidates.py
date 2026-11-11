"""Import only a hash-verified candidate experiment archive; validate locally."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile
from core import ROOT

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('archive');parser.add_argument('--sha256',required=True)
    args=parser.parse_args(); archive=Path(args.archive); assert sha(archive.read_bytes())==args.sha256
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); assert len(names)==len(set(names))
        manifest=json.loads(z.read('MANIFEST.json')); assert set(names)==set(manifest)|{'MANIFEST.json'}
        for name,wanted in manifest.items():
            p=PurePosixPath(name); assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            assert name.startswith(('results/candidates_v1/','reproduction/'))
            b=z.read(name); assert sha(b)==wanted,name
            if name.startswith('reproduction/'):
                assert (ROOT/name).is_file() and sha((ROOT/name).read_bytes())==wanted,('Local source mismatch',name)
        for name in manifest:
            if name.startswith('results/candidates_v1/'):
                path=ROOT/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    from candidate_run import freeze,verify,OUT,save
    signature=freeze(); result=verify(signature)
    # Separate scalar geometry / legacy Dijkstra checks the optimum used by the
    # vectorized graph. One query per main/OOD map, every scaling query.
    from core import search
    checked=[]
    for stage in ['main','ood','scaling']:
        for p in sorted((OUT/'evaluation'/stage).glob('*.json')):
            d=json.loads(p.read_text()); q=d['case']
            if stage!='scaling' and q['pair_id']!=0: continue
            legacy=search(q,'dijkstra')
            assert legacy['valid']
            assert abs(legacy['length']-d['rows'][0]['optimal'])<1e-7,(stage,q['id'])
            checked.append({'stage':stage,'case_id':q['id'],'length':legacy['length']})
    result['independent_legacy_dijkstra_checks']=checked
    result.update(archive_sha256=args.sha256,archive_files=len(manifest),archive_bytes=archive.stat().st_size,local_source_match=True)
    save(OUT/'local_verification.json',result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
