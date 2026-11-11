"""Disjoint tail collection; foreground feeder resumes from saved tail cache."""
import concurrent.futures
from common import *
from collect import one

if __name__=='__main__':
    qs=cases('public')[500:]
    existing=len([p for p in (RUN/'raw/public').glob('*.json') if int(p.name[1:4])>=50])
    save(RUN/'tail_collection_protocol.json',dict(case_ids=[q['id'] for q in qs],workers=32,started_at=now(),existing_responses=existing,
        reason='Disjoint tail acceleration while the six-worker foreground collects the first half; same requests and shared per-case responses'))
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for result in pool.map(one,qs):
            done.append(result)
            if len(done)%10==0:
                save(RUN/'tail_collection_status.json',dict(done=len(done),total=500,complete=False,updated_at=now()))
    save(RUN/'tail_collection_results.json',done)
    save(RUN/'tail_collection_accounting.json',dict(responses_before_restart=existing,new_calls=sum(r['new_calls'] for r in done)))
    save(RUN/'tail_collection_status.json',dict(done=len(done),total=500,complete=True,updated_at=now()))
