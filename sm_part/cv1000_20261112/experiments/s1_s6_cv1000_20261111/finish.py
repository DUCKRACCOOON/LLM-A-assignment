"""Complete the authorized offline benchmark, memory study and report."""
import subprocess
import time
from common import *
from offline_prepare import execute

if __name__=='__main__':
    assert read(RUN/'preparation_status.json')['complete']
    assert read(RUN/'collection_status.json')['complete']
    if (RUN/'tail_collection_status.json').exists():assert read(RUN/'tail_collection_status.json')['complete']
    if (RUN/'middle_collection_protocol.json').exists():assert read(RUN/'middle_collection_status.json')['complete']
    execute(0,'evaluate.py',['freeze'],'freeze.log')
    for fold in range(5):execute(fold,'evaluate.py',['evaluate'],'benchmark.log')
    save(RUN/'evaluation_status.json',dict(complete=True,cases=1000,trials=110000,updated_at=now()))
    execute(0,'report.py',['verify'],'verify.log')
    execute(0,'memory.py',[],'memory.log')
    execute(0,'report.py',['report'],'report.log')
    print(json.dumps(dict(complete=True,report=str(RUN/'report.md'))),flush=True)
