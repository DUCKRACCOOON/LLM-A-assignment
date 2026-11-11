"""Audit saved training provenance before any outer evaluation."""
import math
import numpy as np
from common import *

if __name__=='__main__':
    assert read(RUN/'training_status.json')['complete']
    source={q['id']:q for q in cases()};checks=[]
    for fold,part in enumerate(MANIFEST['folds']):
        dest=RUN/'folds'/f'fold{fold}';meta=read(dest/'models/training_data.json');data=np.load(dest/'models/training_data.npz')
        assert sha(dest/'models/training_data.npz')==meta['sha256']
        expected={q['id'] for q in source.values() if q['map_index'] in part['train_maps']+part['validation_maps']}
        assert len(expected)==800 and set(meta['case_ids'])==expected and not expected&set(part['test_case_ids'])
        val=np.array([source[cid]['map_index'] in part['validation_maps'] for cid in meta['case_ids']])
        assert np.array_equal(val,data['validation']) and int(val.sum())==160
        ii=np.flatnonzero(~val);positive=float((data['y'][ii]*data['mask'][ii]).sum());weight=min(30.,(float(data['mask'][ii].sum())-positive)/max(positive,1.))
        for seed in (11,22,33):
            sm=read(dest/'models'/f'seed{seed}.json');assert sha(dest/'models'/f'seed{seed}.pt')==sm['sha256']
            assert math.isclose(sm['positive_weight'],weight,rel_tol=1e-6)
            assert sm['best_epoch']==min(sm['history'],key=lambda x:x['validation_loss'])['epoch']
            assert sm['epochs']==60 and len(sm['history'])==60 and sm['parameters']==160297
            checks.append(dict(fold=fold,seed=seed,training_cases=640,validation_cases=160,test_cases_excluded=200,
                               best_epoch=sm['best_epoch'],seconds=sm['seconds'],checkpoint_sha256=sm['sha256']))
    save(RUN/'training_audit.json',dict(passed=True,models=len(checks),maps_disjoint=True,train_only_class_weights=True,
         validation_only_checkpoint_selection=True,models_detail=checks,updated_at=now()))
    print(json.dumps(dict(training_audit_passed=True,models=len(checks))),flush=True)
