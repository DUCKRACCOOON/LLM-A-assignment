from common import *
from algorithms import models_load, run

models, seconds=models_load()
truth=read(OUT/'oracle.json')
queries=cases('pilot')+cases('transfer')[:3]+cases('synthetic')[:3]
n=0;max_ratio=0.
for q in queries:
    for name in models:
        result=run(q,name,models=models);optimum=truth[q['id']]['optimal']
        assert result['valid'] and result['length']<=1.05*optimum+1e-8
        n+=1;max_ratio=max(max_ratio,result['length']/optimum)
result=dict(passed=True,checks=n,max_length_ratio=max_ratio,model_loading_seconds=seconds)
save(OUT/'trained_model_checks.json',result)
print(json.dumps(result),flush=True)
