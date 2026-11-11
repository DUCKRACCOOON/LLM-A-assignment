"""Check learned models on disjoint development maps, never outer test data."""
from common import *
from algorithms import models_load,run,MODEL_METHODS
from experiments.K3.verification import Oracle

if __name__=='__main__':
    models,seconds=models_load();count=0;worst=1.
    for item in read(RUN/'data/pilot_reference.json'):
        q=item['query'];oracle=Oracle(q)
        for method in MODEL_METHODS:
            r=run(q,method,models=models);assert r['valid'];oracle.check_path(q,r)
            assert r['length']<=1.05*item['optimal']+1e-8
            worst=max(worst,r['length']/item['optimal']);count+=1
    save(OUT/'model_audit.json',dict(passed=True,checks=count,max_cost_ratio=worst,model_loading_seconds=seconds,
                                  source='30 disjoint synthetic development maps',updated_at=now()))
    print(json.dumps(read(OUT/'model_audit.json')),flush=True)
