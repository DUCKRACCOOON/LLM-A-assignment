"""Build a credential-free submission archive from an explicit allowlist."""
import hashlib
import json
import zipfile
from pathlib import Path
from .experiment import ROOT


def main():
    destination=ROOT/'output/LLM_Astar_CodeFile.zip'
    destination.parent.mkdir(exist_ok=True)
    directories=['reproduction','source_code','report','results/main','results/generalization','results/verification','최신','review','materials']
    files=[ROOT/n for n in ['README.md','requirements.txt','.env.example','UPSTREAM.md','LLM-Astar_Project_Plan.md',
                           'output/LLM_Astar_Report_KO.txt','output/ai_check/copyleaks_result.png',
                           'output/ai_check/copyleaks_result.txt']]
    for directory in directories:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    # Do not include development snapshots or compiled/cache/private files.
    files=sorted(set(p for p in files if '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo']
                     and 'proposed_development' not in p.name and 'diagnostics_development' not in p.name))
    secret=(ROOT/'.env').read_text(encoding='utf-8').strip().split('=',1)[1] if (ROOT/'.env').exists() else None
    hashes={}
    for p in files:
        relative=p.relative_to(ROOT).as_posix()
        if any(part in ('.git','.env','local_private','contribution_reports_private') for part in p.relative_to(ROOT).parts):
            raise RuntimeError('Excluded private path in package')
        content=p.read_bytes()
        if secret and secret.encode() in content: raise RuntimeError('Credential detected in '+relative)
        hashes[relative]=hashlib.sha256(content).hexdigest()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(hashes,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert '.env' not in z.namelist()
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT/'output/package_manifest.json').write_text(json.dumps(dict(file=destination.name,sha256=digest,
          entries=len(files),bytes=destination.stat().st_size,credential_scan='passed'),indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(destination),entries=len(files),mb=round(destination.stat().st_size/1e6,2),sha256=digest)))


if __name__=='__main__':main()
