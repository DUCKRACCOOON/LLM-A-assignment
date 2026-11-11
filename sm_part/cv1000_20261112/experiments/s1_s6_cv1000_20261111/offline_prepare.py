"""After training, select S6 on inner data and check synthetic pilot only."""
import subprocess
import time
from common import *

def execute(fold,script,args,logname):
    env=dict(os.environ,CV_FOLD=str(fold),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    dest=RUN/'folds'/f'fold{fold}';command=[sys.executable,str(HERE/script)]+args;t=time.perf_counter()
    with (dest/logname).open('a',encoding='utf-8') as log:
        proc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    history=read(RUN/'run_log.json') if (RUN/'run_log.json').exists() else []
    history.append(dict(command=command,fold=fold,returncode=proc.returncode,seconds=time.perf_counter()-t,ended_at=now(),log=str(dest/logname)));save(RUN/'run_log.json',history)
    if proc.returncode:raise RuntimeError(f'{script} fold {fold} failed; see {logname}')

def main():
    assert read(RUN/'training_status.json')['complete'],'Wait until all 15 models are trained'
    for fold in range(5):
        execute(fold,'programs.py',['select'],'program_selection.log')
        execute(fold,'evaluate.py',['pilot'],'pilot.log')
        save(RUN/'preparation_status.json',dict(fold=fold,complete=False,updated_at=now()));print(json.dumps(dict(fold=fold,selected_and_pilot_passed=True)),flush=True)
    execute(0,'instrumentation_check.py',[],'instrumentation_check.log')
    save(RUN/'preparation_status.json',dict(complete=True,folds=5,updated_at=now()))

if __name__=='__main__':main()
