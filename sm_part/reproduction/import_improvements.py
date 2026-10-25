"""Import the downloaded verified archive without overwriting unrelated files."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path,PurePosixPath
from core import ROOT
from run import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('archive');ap.add_argument('--sha256',required=True);args=ap.parse_args()
    archive=Path(args.archive)
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual!=args.sha256:raise RuntimeError('Archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive path')
        manifest=json.loads(z.read('archive_manifest.json'))
        if set(names)!=set(manifest)|{'archive_manifest.json'}:raise RuntimeError('Unlisted archive entries')
        for name,expected in manifest.items():
            rel=PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts or tuple(rel.parts[:2])!=('results','improvements'):
                raise RuntimeError('Archive path outside expected destination')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('File hash mismatch: '+name)
            p=ROOT.joinpath(*rel.parts)
            if p.exists() and p.read_bytes()!=data:raise RuntimeError('Conflicting existing file: '+name)
        for name in manifest:
            p=ROOT.joinpath(*PurePosixPath(name).parts);p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():p.write_bytes(z.read(name))
    verification=json.loads((ROOT/'results/improvements/verification.json').read_text(encoding='utf-8'))
    if not verification.get('complete') or verification['total_rows']!=28270:raise RuntimeError('Completion contract failed')
    from improve_report import verify_stage
    stages={s:verify_stage(s)[1] for s in ('pilot','main','scaling')}
    save(ROOT/'results/improvements/local_import_verification.json',{'complete':True,'archive_sha256':actual,
        'files_verified':len(manifest),'stages':stages})
    print(json.dumps({'complete':True,'files':len(manifest),'rows':sum(s['rows'] for s in stages.values())}))

if __name__=='__main__':main()
