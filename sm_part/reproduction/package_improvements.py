"""Build allowlisted deployment package; never includes credentials or old raw responses."""
import hashlib
import json
import zipfile
from pathlib import Path
from core import ROOT
from run import save

def main():
    ref={}
    rows=json.loads((ROOT/'results/main/rows.json').read_text())
    for row in rows:
        if row['method'] in ('astar','dynamic_weighted'):
            method='A' if row['method']=='astar' else 'D'
            ref.setdefault(row['case_id'],{})[method]={k:row[k] for k in ('operation','storage','length','valid')}
    for row in json.loads((ROOT/'results/report/no_waypoint_ablation.json').read_text()):
        ref[row['case_id']]['N']={k:row[k] for k in ('operation','storage','length','valid')}
    assert len(ref)==1000 and all(set(v)=={'A','D','N'} for v in ref.values())
    source=['core.py','api_client.py','run.py','test_reproduction.py','improve_core.py','improve_run.py','improve_report.py','test_improvements.py']
    files=[ROOT/'reproduction'/name for name in source]
    files += [ROOT/'requirements-lock.txt']
    for relative in ['dataset/environment_50_30.json','llmastar/pather/llm_a_star/prompt.py',
        'llmastar/pather/llm_a_star/llm_a_star.py','llmastar/pather/a_star/a_star.py','llmastar/env/search/env.py','LICENSE']:
        files.append(ROOT/'vendor/llm-astar'/relative)
    dest=ROOT/'dist/gpt-improvements-v1.zip';dest.parent.mkdir(exist_ok=True)
    entries={str(p.relative_to(ROOT)).replace('\\','/'):p.read_bytes() for p in files}
    entries['improvement_reference.json']=json.dumps(ref,ensure_ascii=False,sort_keys=True).encode('utf-8')
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items():z.writestr(name,data)
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    info={'archive':str(dest),'files':len(entries),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    save(ROOT/'results/improvements_deployment.json',info)
    print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':main()
