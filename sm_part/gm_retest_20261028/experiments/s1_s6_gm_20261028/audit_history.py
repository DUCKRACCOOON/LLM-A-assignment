"""Post-benchmark historical path comparison; no new searches or API calls."""
import math
import subprocess
from datetime import datetime
from common import *

def main():
    assert read(OUT/'pipeline_status.json')['complete'], 'Run after timed evaluation and verification'
    ref='c56eb83b7df4be06d402daba36e0a2b45afa950b'
    repo=ROOT/'tmp/sm-publish'
    source='experiments/K3/results/run_20261025/public/evaluations.jsonl'
    dest=OUT/'historical/gm_k3_public_evaluations.jsonl'
    if not dest.exists():
        content=subprocess.check_output(['git','-c',f'safe.directory={repo.as_posix()}','-C',str(repo),'show',f'{ref}:{source}'])
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    names={'astar_euclidean':'A_GM','g2h_euclidean':'N_g2h','g2h_wall_euclidean':'K3_wall_euclidean','g2h_wall_octile':'K3_wall_octile'}
    historical={}
    with dest.open(encoding='utf-8-sig') as file:
        for line in file:
            r=json.loads(line)
            if r['method'] in names and r['repeat']==0:historical[(r['case_id'],names[r['method']])]=r
    mismatches=[];checks=0
    for path in sorted((OUT/'evaluations/main/public').glob('*.json')):
        record=read(path)
        for r in record['rows']:
            if r['method'] not in names.values() or r['repeat']!=0:continue
            old=historical[(record['case']['id'],r['method'])];path_now=record['paths'][r['path_sha256']]
            diff=[]
            for key in ('valid','operation','storage'):
                if r[key]!=old[key]:diff.append(key)
            if not math.isclose(r['length'],old['length'],abs_tol=1e-8,rel_tol=1e-10):diff.append('length')
            if path_now!=old['path']:diff.append('path')
            if diff:mismatches.append(dict(case_id=record['case']['id'],method=r['method'],fields=diff))
            checks+=1
    assert checks==4000
    result=dict(checked=checks,passed=not mismatches,mismatches=mismatches,source_commit=ref,source_path=source,source_sha256=sha(dest),verified_at=now())
    save(OUT/'historical_path_verification.json',result)
    initial=read(OUT/'program_selection_initial.json');stamp=(OUT/'program_selection.log').stat().st_ctime
    elapsed=datetime.fromisoformat(initial['selected_at']).timestamp()-stamp
    save(OUT/'selection_initial_wall_clock.json',dict(seconds=elapsed,start_utc=datetime.fromtimestamp(stamp,timezone.utc).isoformat(),end_utc=initial['selected_at'],
         measurement='Windows Python 3.11 file creation time of redirected selection log to recorded selection completion; includes process launch, approximate wall-clock rather than perf_counter'))
    print(json.dumps(dict(checked=checks,mismatches=len(mismatches),initial_selection_wall_seconds=elapsed)),flush=True)

if __name__=='__main__':main()
