import subprocess
import argparse
from common import *

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--programs-only',action='store_true');args=parser.parse_args()
    assert read(RUN/'folds/fold0/preflight.json')['passed']
    for fold in range(5):
        env=dict(os.environ,CV_FOLD=str(fold))
        dest=RUN/'folds'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
        with (dest/'program_generation.log').open('a',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,str(HERE/'programs.py'),'generate'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'S6 proposal collection fold {fold} failed')
        print(json.dumps(dict(fold=fold,proposals_collected=True)),flush=True)
    if args.programs_only:sys.exit(0)
    from collect import collect
    collect('public',6)
    save(RUN/'collection_status.json',dict(complete=True,cases=1000,s6_proposals=60,updated_at=now()))
