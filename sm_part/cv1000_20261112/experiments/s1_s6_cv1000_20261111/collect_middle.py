"""Collect a still-distant middle segment after the tail has fully drained."""
import concurrent.futures
import time
from common import *
from collect import one

if __name__=='__main__':
    while not ((RUN/'tail_collection_status.json').exists() and read(RUN/'tail_collection_status.json')['complete']):time.sleep(5)
    head=read(RUN/'folds/fold0/collection_status.json');start=min(500,head.get('done',500)+120)
    qs=cases()[start:500]
    assert not any(raw_path(q,'first').exists() for q in qs)
    save(RUN/'middle_collection_protocol.json',dict(start=start,stop=500,workers=32,case_ids=[q['id'] for q in qs],started_at=now()))
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for result in pool.map(one,qs):done.append(result)
    save(RUN/'middle_collection_results.json',done)
    save(RUN/'middle_collection_status.json',dict(complete=True,cases=len(done),new_calls=sum(r['new_calls'] for r in done),updated_at=now()))
