"""Run this authorized experiment through collection, benchmark, and report."""
import subprocess
import time
from common import *

def status(phase,**extra):
    value=dict(phase=phase,updated_at=now(),complete=False,**extra)
    save(OUT/'pipeline_status.json',value);print(json.dumps(value),flush=True)

def main():
    status('waiting_for_full_collection')
    while True:
        collection=read(OUT/'collection_status.json')
        if collection.get('phase')=='failed':raise RuntimeError('Response collection failed; see collection_status.json')
        if collection.get('complete') and collection.get('stage') is None:break
        time.sleep(5)
    assert read(OUT/'training_status.json')['complete']
    assert read(OUT/'trained_model_checks.json')['passed']
    assert read(OUT/'preflight.json')['passed']
    assert (OUT/'program_selection.json').exists()
    for phase,script,action in [('benchmark','evaluate.py','all'),('verify_and_report','report.py','all')]:
        status(phase)
        with (OUT/f'{phase}.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,str(HERE/script),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError(f'{phase} exited {proc.returncode}; see {phase}.log')
    value=dict(phase='completed',complete=True,completed_at=now(),report=str(OUT/'report.md'))
    save(OUT/'pipeline_status.json',value);print(json.dumps(value),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=str(e));raise
