import math
import random
import shutil
from common import *
from experiments.K3.verification import Oracle

def geometry_hash(q):
    return digest({k:sorted(q[k]) if 'barriers' in k else q[k] for k in ('range_x','range_y','horizontal_barriers','vertical_barriers')})

def generate(count,seed,excluded):
    rng=random.Random(seed);out=[]
    while len(out)<count:
        q=dict(range_x=[0,51],range_y=[0,31],horizontal_barriers=[],vertical_barriers=[])
        for _ in range(rng.randint(2,7)):
            if rng.random()<.5:
                a,b=sorted(rng.sample(range(0,51),2));q['horizontal_barriers'].append([rng.randrange(2,28),a,b])
            else:
                a,b=sorted(rng.sample(range(0,31),2));q['vertical_barriers'].append([rng.randrange(2,48),a,b])
        key=geometry_hash(q)
        if key in excluded:continue
        oracle=Oracle(q);points=sorted(oracle.graph)
        if len(points)<2:continue
        s,g=rng.sample(points,2);ds=oracle.distances(g)
        if s not in ds or ds[s]<10:continue
        q.update(start=list(s),goal=list(g),id=f'dev{len(out):03d}',map_id=f'dev:{seed}:{len(out)}',stage='pilot',split='development',map_index=len(out),pair_id=0)
        path=[s]
        while path[-1]!=g:
            u=path[-1];v=min(oracle.graph[u],key=lambda e:(e[1]+ds.get(e[0],math.inf),e[0]))[0]
            assert ds[v]<ds[u];path.append(v)
        oracle.check_path(q,dict(valid=True,path=path,length=ds[s]))
        short=[path[0]];i=0
        while i<len(path)-1:
            i=max(j for j in range(i+1,len(path)) if not oracle.blocked(path[i],path[j]));short.append(path[i])
        assert gm.validate_path(gm.Geometry(q),short,s,g) is None
        out.append(dict(query=q,path=path,waypoints=short,optimal=ds[s],geometry_sha256=key));excluded.add(key)
    return out

def main():
    assert sys.version_info[:2]==(3,11)
    if (RUN/'protocol.json').exists():print('Already prepared');return
    allq=cases();excluded={geometry_hash(q) for q in allq};assert len(excluded)==100
    examples=generate(5,20261112,excluded);pilot=generate(30,20261113,excluded)
    source=GM/'source_code/llmastar/pather/llm_a_star/prompt.py'
    original=runpy.run_path(str(source))['gpt_prompt']['standard']
    header=original.split('Start Point:')[0];tail='Start Point: {start}'+original.split('Start Point: {start}',1)[1]
    blocks=[]
    for e in examples:
        q=e['query'];blocks.append('Start Point: '+str(q['start'])+'\nGoal Point: '+str(q['goal'])+'\nHorizontal Barriers: '+str(q['horizontal_barriers'])+'\nVertical Barriers: '+str(q['vertical_barriers'])+'\nGenerated Path: '+json.dumps(e['waypoints']))
    PROMPT_FILE.parent.mkdir(parents=True,exist_ok=True);PROMPT_FILE.write_text(header+'\n\n'.join(blocks)+'\n\n'+tail,encoding='utf-8')
    save(RUN/'prompts/examples.json',examples);save(RUN/'data/pilot_reference.json',pilot);save(RUN/'data/pilot.json',[x['query'] for x in pilot])
    save(RUN/'data/public.json',allq);save(RUN/'fold_manifest.json',MANIFEST)
    testids=[]
    for p in MANIFEST['folds']:
        sets=[set(p[k+'_maps']) for k in ('train','validation','test')]
        assert list(map(len,sets))==[64,16,20] and not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        testids+=p['test_case_ids']
    assert len(testids)==len(set(testids))==1000
    shutil.copyfile(PLAN/'plan.md',RUN/'plan.md')
    save(RUN/'protocol.json',dict(status='prepared',created_at=now(),environment=environment(),model=MODEL,temperature=0,max_tokens=1000,
       folds=5,train_validation_test=[640,160,200],timing_repeats=5,planned_trials=110000,model_seeds=[11,22,33],
       train_threads=4,native_threads=1,new_generations=True,transport_retries=0,primary_baseline='B_llm',
       prompt_sha256=sha(PROMPT_FILE),source_prompt_sha256=sha(source),fold_manifest_sha256=sha(RUN/'fold_manifest.json'),
       source_input_sha256=sha(GM/'experiments/K3/results/run_20261025/public_cases.json'),
       prompt_change='Five disjoint synthetic demonstration maps; original instructions and output format retained',
       timing='Five offline planner repeats; shared freshly measured API latency added only as reconstructed end-to-end estimate',
       scope='OOF public1000, historically exposed; not hidden independent evaluation'))
    print(json.dumps(dict(prepared=True,maps=100,cases=1000,disjoint_examples=5,disjoint_pilot=30,environment=environment())),flush=True)

if __name__=='__main__':main()
