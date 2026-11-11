"""Resume authorized collection and offline study after all inner selections."""
import subprocess
import time
from common import *

def phase(name,script):
    save(RUN/'pipeline_status.json',dict(phase=name,complete=False,updated_at=now()))
    with (RUN/f'{name}.log').open('a',encoding='utf-8') as log:
        proc=subprocess.run([sys.executable,str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'{name} failed; inspect {name}.log')

if __name__=='__main__':
    try:
        start=time.monotonic()
        while not ((RUN/'preparation_status.json').exists() and read(RUN/'preparation_status.json')['complete']):
            if time.monotonic()-start>1800:raise RuntimeError('Preparation not complete after 30 minutes')
            time.sleep(5)
        phase('collection','collect_all.py')
        phase('offline_finish','finish.py')
        assert read(RUN/'manifest.json')['status']=='completed'
        save(RUN/'pipeline_status.json',dict(phase='completed',complete=True,report=str(RUN/'report.md'),updated_at=now()))
    except Exception as exc:
        save(RUN/'pipeline_status.json',dict(phase='failed',complete=False,error=str(exc),updated_at=now()));raise
