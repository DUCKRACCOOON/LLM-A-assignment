"""Development-only sensitivity to heap wrapper overhead, after training."""
import random
import statistics
import math
from collections import Counter
from common import *
import algorithms

class NoHeapWrapper:
    def __enter__(self):self.counts=Counter();return self
    def __exit__(self,*args):pass

if __name__=='__main__':
    assert read(RUN/'training_status.json')['complete']
    models,_=algorithms.models_load();original=algorithms.HeapCounts;rows=[]
    for item in read(RUN/'data/pilot_reference.json')[:10]:
        q=item['query'];text='Generated Path: '+json.dumps(item['waypoints'])
        for method in algorithms.BASE+algorithms.MODEL_METHODS:
            timings={True:[],False:[]};ref=None
            for repeat in range(3):
                modes=[True,False];random.Random(f'{q["id"]}:{method}:{repeat}').shuffle(modes)
                for enabled in modes:
                    algorithms.HeapCounts=original if enabled else NoHeapWrapper
                    try:r=algorithms.run(q,method,text,{'feedback':text,'control':text},models,[.5,.3,.1,.1])
                    finally:algorithms.HeapCounts=original
                    signature=(r['valid'],r['length'],digest(r['path']))
                    if ref is None:ref=signature
                    assert ref==signature;timings[enabled].append(r['total_seconds'])
            rows.append(dict(case_id=q['id'],method=method,wrapped_seconds=statistics.median(timings[True]),unwrapped_seconds=statistics.median(timings[False])))
    summary={}
    for method in algorithms.BASE+algorithms.MODEL_METHODS:
        rr=[r for r in rows if r['method']==method]
        summary[method]=math.exp(statistics.mean(math.log(r['wrapped_seconds']/r['unwrapped_seconds']) for r in rr))
    save(RUN/'instrumentation_check.json',dict(verified_paths_unchanged=True,development_maps=10,repeats=3,
         scope='Only heap wrapper removed; other counters remain. Exploratory, no tuning or selection based on these timings.',ratios=summary,rows=rows))
    print(json.dumps(dict(development_check_passed=True,paired_conditions=len(rows))),flush=True)
