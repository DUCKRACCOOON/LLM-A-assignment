import subprocess
from common import *

if __name__=='__main__':
    assert read(RUN/'folds/fold0/preflight.json')['passed']
    for fold in range(5):
        env=dict(os.environ,CV_FOLD=str(fold),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
        dest=RUN/'folds'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
        save(RUN/'training_status.json',dict(fold=fold,phase='training',complete=False,updated_at=now()))
        with (dest/'training.log').open('w',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(HERE/'train.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Training fold {fold} failed; inspect log')
        print(json.dumps(dict(fold=fold,trained=True)),flush=True)
    save(RUN/'training_status.json',dict(complete=True,models=15,updated_at=now()))
