"""Resumable, cached-LLM study on the existing GPU server. No API calls."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from core import ROOT, load_cases, parse_path, scale_case, path_length
from candidate_core import (Grid, focal, reconstruct, llm_field, conformal_quantile,
    fast_marching, chomp_repair, network_inputs, make_network, point_segment_distance,
    normalize_proposal, grid_valid)

OUT=ROOT/'results'/'candidates_v1'
SEEDS=[11,22,33]
PROTOCOL={
 'version':'candidates-20261014-v1','split_seed':261014,'weight':1.1,
 'train_maps':50,'validation_maps':10,'calibration_maps':20,'test_maps':20,
 'split_unit':'map; 10 start-goal pairs kept together',
 'prompt_demo_maps_forced_train':[0,2,3,4,5],
 'llm':'existing verified improvements/main/raw/*-first.json; no regeneration or API calls',
 'prompts':['standard','cot','repe'],'cp_alpha':.1,
 'cp_score':'max over 10 pairs and all canonical shortest-path cells of distance to union of three LLM polylines / (1 + spatial mean prompt dispersion)',
 'cp_set':'free grid cells within qhat * proposal-only dispersion of union of LLM polylines',
 'cp_guarantee_scope':'exchangeable non-demo maps; simultaneous inclusion of the 10 canonical optimal paths per map; no OOD claim; selection biases of original dataset remain',
 'cp_use':'secondary FOCAL priority = distance outside conformal set; never prune by prediction',
 'chomp':'48 samples; 160 covariant descent steps; analytic obstacle penalty; exact grid revalidation; single standard prompt',
 'chomp_scope':'CHOMP-inspired 2D adaptation, not original CHOMP implementation',
 'fmm':'first-order cardinal upwind Eikonal update; full-grid field; uniform speed control and speed 2 within LLM radius 2',
 'network':'7 input channels; conv encoder 24/48/64; 2 transformer layers, 4 heads, 128 FFN; convolutional decoder',
 'transpath_scope':'TransPath-inspired learned path-probability + FOCAL, not official pretrained model or exact paper reproduction',
 'train_labels':'all grid cells with d(start,x)+d(x,goal)<=optimal+.5; exact graph distances',
 'training':{'seeds':SEEDS,'epochs':60,'batch_size':32,'lr':.001,'optimizer':'AdamW','weight_decay':.0001,'checkpoint':'lowest validation BCE; test never used'},
 'primary_control':'FOCAL_EUCLIDEAN (same w=1.1 and same graph/backend)',
 'metrics':['validity','length/optimal','node expansions','discovered nodes','preparation seconds','search seconds','cold local seconds','historical GPT latency separately'],
 'cold_local':'cold graph construction + preparation/inference + search; resident warmed neural model; transfer and CUDA synchronization included; one warmup per geometry logged separately',
 'timing_limits':'single execution per case on shared server; hardware load recorded; descriptive latency only, not repeated performance benchmark',
 'ood':'20 unseen alternating-wall maze maps x 5 queries; only LLM-free methods; no conformal coverage claim',
 'scaling':'first query on first 10 held-out maps at factors 2 and 4; LLM coordinates transformed from cached responses, no new GPT requests',
 'sources':['https://arxiv.org/abs/2107.07511','https://publications.ri.cmu.edu/chomp-covariant-hamiltonian-optimization-for-motion-planning',
            'https://math.berkeley.edu/~sethian/2006/Applications/Robotics/hold_robotics.html','https://ojs.aaai.org/index.php/AAAI/article/view/26465'],
}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)
def status(phase,**kw):
    data=dict(phase=phase,updated_utc=now(),complete=False,**kw)
    save(OUT/'status.json',data); print(json.dumps(data,ensure_ascii=False),flush=True)

def splits():
    demos=PROTOCOL['prompt_demo_maps_forced_train']
    other=[i for i in range(100) if i not in demos]; random.Random(PROTOCOL['split_seed']).shuffle(other)
    return dict(train=demos+other[:45],validation=other[45:55],calibration=other[55:75],test=other[75:])

def freeze():
    sources={p.name:sha(p) for p in [ROOT/'reproduction'/x for x in ['candidate_core.py','candidate_run.py','candidate_report.py','test_candidates.py','core.py']]}
    p=dict(protocol=PROTOCOL,splits=splits(),sources=sources,data_sha256=sha(ROOT/'vendor/llm-astar/dataset/environment_50_30.json'))
    p['signature']=digest(p)
    dest=OUT/'protocol.json'
    if dest.exists(): assert json.loads(dest.read_text())==p,'Frozen protocol changed; use a new output directory'
    else: save(dest,p)
    return p['signature']

def raw_proposals(q):
    paths=[]; latency=[]; hashes=[]
    base=q.get('base_id',q['id']); factor=q.get('scale',1)
    for kind in PROTOCOL['prompts']:
        p=ROOT/'results/improvements/main/raw'/f'{base}-{kind}-first.json'
        record=json.loads(p.read_text(encoding='utf-8'))
        assert digest(record['response'])==record['response_sha256']
        assert digest(record['payload'])==record['payload_sha256']
        assert record['case_id']==base
        path,_=parse_path(record['response']['choices'][0]['message']['content'] or '')
        paths.append([[a*factor,b*factor] for a,b in path])
        latency.append(record['latency_seconds'])
        hashes.append(sha(p))
    return paths,latency,hashes

def prepare_data():
    dest=OUT/'training_data.npz'; meta=OUT/'data_manifest.json'
    if dest.exists() and meta.exists():
        m=json.loads(meta.read_text()); assert sha(dest)==m['sha256']; return
    split=splits(); qs=[q for q in load_cases() if q['map_id'] in split['train']+split['validation']]
    t=time.perf_counter(); inputs=[]; labels=[]; masks=[]; validation=[]; ids=[]; labels_pops=0
    groups=defaultdict(list)
    for q in qs: groups[q['map_id']].append(q)
    for mi,(_,cases) in enumerate(groups.items()):
        grid=Grid(cases[0])
        for q in cases:
            ds,_,p1=grid.distances(q['start']); dg,_,p2=grid.distances(q['goal']); labels_pops+=p1+p2
            optimum=ds[grid.index(q['goal'])]; assert math.isfinite(optimum)
            inputs.append(network_inputs(grid,q)); labels.append(((ds+dg)<=optimum+.5).reshape(grid.h,grid.w))
            masks.append(grid.free.reshape(grid.h,grid.w)); validation.append(q['map_id'] in split['validation']); ids.append(q['id'])
        if mi%10==0: status('prepare_training_data',maps=mi+1,total_maps=60)
    np.savez_compressed(dest,x=np.asarray(inputs),y=np.asarray(labels,dtype=np.float32),mask=np.asarray(masks,dtype=np.float32),validation=np.asarray(validation))
    save(meta,dict(case_ids=ids,sha256=sha(dest),seconds=time.perf_counter()-t,label_dijkstra_expansions=labels_pops))

def train():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU required for this training run')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    data=np.load(OUT/'training_data.npz'); validation=data['validation']
    x=torch.from_numpy(data['x']).cuda(); y=torch.from_numpy(data['y']).cuda(); mask=torch.from_numpy(data['mask']).cuda()
    train_ids=np.flatnonzero(~validation); val_ids=np.flatnonzero(validation)
    positive=float((y[train_ids]*mask[train_ids]).sum().item())
    negative=float(mask[train_ids].sum().item())-positive
    pos_weight=torch.tensor(min(30.,negative/max(positive,1.)),device='cuda')
    criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight)
    for seed in SEEDS:
        checkpoint=OUT/'models'/f'seed{seed}.pt'; metapath=checkpoint.with_suffix('.json')
        if metapath.exists():
            assert sha(checkpoint)==json.loads(metapath.read_text())['sha256']; continue
        torch.manual_seed(seed); np.random.seed(seed); model=make_network().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=math.inf; best_epoch=0; history=[]; t=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(60):
            model.train(); epochloss=0.; order=np.random.permutation(train_ids)
            for i in range(0,len(order),32):
                idx=order[i:i+32]; logits=model(x[idx]); loss=(criterion(logits,y[idx])*mask[idx]).sum()/mask[idx].sum()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); epochloss+=float(loss.item())*len(idx)
            model.eval(); val_loss=0.
            with torch.no_grad():
                for i in range(0,len(val_ids),32):
                    idx=val_ids[i:i+32]; loss=(criterion(model(x[idx]),y[idx])*mask[idx]).sum()/mask[idx].sum()
                    val_loss+=float(loss.item())*len(idx)
            val_loss/=len(val_ids); history.append(dict(epoch=epoch+1,train_loss=epochloss/len(order),validation_loss=val_loss))
            if val_loss<best:
                best=val_loss; best_epoch=epoch+1; checkpoint.parent.mkdir(parents=True,exist_ok=True)
                torch.save(model.state_dict(),checkpoint)
            if (epoch+1)%10==0: status('training',seed=seed,epoch=epoch+1,epochs=60,validation_loss=val_loss,best_epoch=best_epoch)
        torch.cuda.synchronize()
        save(metapath,dict(seed=seed,epochs=60,best_epoch=best_epoch,best_validation_loss=best,seconds=time.perf_counter()-t,
                           peak_cuda_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
                           sha256=sha(checkpoint),history=history,positive_weight=pos_weight.item(),cuda_device=torch.cuda.get_device_name()))
        del model,optimizer

def calibrate():
    dest=OUT/'calibration.json'
    if dest.exists(): return json.loads(dest.read_text())
    t=time.perf_counter(); scores={}; detail=[]; groups=defaultdict(list)
    for q in load_cases():
        if q['map_id'] in splits()['calibration']: groups[q['map_id']].append(q)
    for mi,(mid,cases) in enumerate(groups.items()):
        grid=Grid(cases[0]); group=[]
        for q in cases:
            _,parents,_=grid.distances(q['start']); path=reconstruct(grid,parents,q['start'],q['goal']); assert path
            proposals,_,hashes=raw_proposals(q); field,scale=llm_field(grid,q,proposals)
            score=max(float(field[grid.index(x)]) for x in path)/scale
            group.append(score); detail.append(dict(case_id=q['id'],score=score,scale=scale,raw_hashes=hashes))
        scores[str(mid)]=max(group)
        if mi%5==0: status('calibration',maps=mi+1,total_maps=20)
    quantile=conformal_quantile(scores.values(),.1)
    result=dict(alpha=.1,map_scores=scores,quantile=quantile if math.isfinite(quantile) else None,
                cases=detail,seconds=time.perf_counter()-t,calibration_maps=20,finite_sample_order=19)
    save(dest,result); return result

def ood_cases():
    rng=random.Random(262026); out=[]
    for m in range(20):
        barriers=[]
        for j,x in enumerate([10,20,30,40]):
            gap=rng.randint(4,8)
            barriers.append([x,0,30-gap] if (j+m)%2==0 else [x,gap,30])
        for p in range(5):
            out.append(dict(id=f'ood{m:02d}-p{p:02d}',map_id=1000+m,pair_id=p,range_x=[0,51],range_y=[0,31],
                            horizontal_barriers=[],vertical_barriers=barriers,start=[rng.randint(2,8),rng.randint(2,28)],
                            goal=[rng.randint(42,48),rng.randint(2,28)]))
    return out

def evaluate(calibration,signature,stages=None):
    import torch
    torch.set_num_threads(2); models={}
    for seed in SEEDS:
        m=make_network().cuda(); m.load_state_dict(torch.load(OUT/'models'/f'seed{seed}.pt',map_location='cuda',weights_only=True)); m.eval(); models[f'TRANSPATH_{seed}']=m
    torch.manual_seed(11); untrained=make_network().cuda().eval(); models['UNTRAINED_11']=untrained
    heldout=[q for q in load_cases() if q['map_id'] in splits()['test']]
    scaleqs=[scale_case(q,f) for q in sorted(heldout,key=lambda x:(x['map_id'],x['pair_id'])) if q['pair_id']==0 and q['map_id'] in sorted(splits()['test'])[:10] for f in (2,4)]
    stagecases=dict(main=heldout,ood=ood_cases(),scaling=scaleqs)
    for stage,cases in stagecases.items():
        if stages and stage not in stages: continue
        for qi,q in enumerate(cases):
            dest=OUT/'evaluation'/stage/(q['id']+'.json')
            if dest.exists():
                assert json.loads(dest.read_text())['signature']==signature; continue
            grid=Grid(q); truth,parents,_=grid.distances(q['start']); optimum=float(truth[grid.index(q['goal'])])
            optimal_path=reconstruct(grid,parents,q['start'],q['goal']); assert math.isfinite(optimum)
            rows=[]; coverage=None; radius=None; raw_hashes=[]
            def add(name,secondary=None,prep=0.,meta=None,mode='focal',incumbent=None,calls=0,latency=0.):
                result=focal(grid,q,secondary,weight=1.1,mode=mode,incumbent=incumbent)
                assert result['valid'],(q['id'],name,result['failure'])
                assert result['length']<=optimum*(1. if mode=='astar' else 1.1)+1e-7,(q['id'],name)
                rows.append(dict(case_id=q['id'],map_id=q['map_id'],stage=stage,method=name,optimal=optimum,
                                 **result,preparation_seconds=prep,graph_seconds=grid.seconds,
                                 local_cold_seconds=grid.seconds+prep+result['search_seconds'],length_ratio=result['length']/optimum,
                                 historical_llm_seconds=latency,llm_responses_required=calls,metadata=meta or {}))
            add('ASTAR',mode='astar'); add('WEIGHTED_1_1',mode='weighted'); add('FOCAL_EUCLIDEAN')
            ffield,fmeta=fast_marching(grid,q['goal']); add('FMM_UNIFORM',ffield,prep=fmeta['field_seconds'],meta=fmeta)
            if stage!='ood':
                t=time.perf_counter(); proposals,latency,raw_hashes=raw_proposals(q); field,scale=llm_field(grid,q,proposals); prep=time.perf_counter()-t
                add('LLM_TUBE',field,prep=prep,calls=3,latency=sum(latency))
                add('FIXED_TUBE_2',np.maximum(0,field-2),prep=prep,calls=3,latency=sum(latency))
                t=time.perf_counter(); quantile=calibration['quantile']; radius=quantile*scale if quantile is not None else math.inf
                cpfield=np.maximum(0,field-radius); cpseconds=time.perf_counter()-t
                coverage=all(field[grid.index(x)]<=radius+1e-9 for x in optimal_path)
                cpmeta=dict(radius=radius if math.isfinite(radius) else None,uncertainty_scale=scale,
                            canonical_path_covered=coverage,set_fraction=float(np.mean(field[grid.free]<=radius)),
                            coverage_claim=stage=='main')
                add('CONFORMAL',cpfield,prep=prep+cpseconds,meta=cpmeta,calls=3,latency=sum(latency))
                t=time.perf_counter(); valid_proposal=normalize_proposal(q,proposals[0]); from core import validate
                accepted=validate(q,valid_proposal)[0]
                gate=point_segment_distance(grid.xy,valid_proposal) if accepted else None
                add('VGATE_STANDARD',gate,prep=time.perf_counter()-t,meta=dict(accepted=accepted),calls=1,latency=latency[0])
                t=time.perf_counter(); repaired,rmeta=chomp_repair(q,proposals[0]); secondary=point_segment_distance(grid.xy,repaired) if repaired else None
                add('CHOMP_STANDARD',secondary,prep=time.perf_counter()-t,meta=rmeta,incumbent=repaired or None,calls=1,latency=latency[0])
                t=time.perf_counter(); repaired,rmeta=chomp_repair(q,[q['start'],q['goal']]); secondary=point_segment_distance(grid.xy,repaired) if repaired else None
                add('CHOMP_STRAIGHT',secondary,prep=time.perf_counter()-t,meta=rmeta,incumbent=repaired or None)
                t=time.perf_counter(); speed=np.where(field<=2.,2.,1.); ffield,fmeta=fast_marching(grid,q['goal'],speed)
                add('FMM_LLM',ffield,prep=prep+time.perf_counter()-t,meta=fmeta,calls=3,latency=sum(latency))
            for name,model in models.items():
                # Warm-up per geometry is recorded separately; no ground-truth input.
                x_np=network_inputs(grid,q)
                warmstart=time.perf_counter()
                with torch.no_grad(): model(torch.from_numpy(x_np[None]).cuda())
                torch.cuda.synchronize(); warm_seconds=time.perf_counter()-warmstart
                torch.cuda.synchronize(); t=time.perf_counter()
                x_np=network_inputs(grid,q)
                with torch.no_grad(): prob=model(torch.from_numpy(x_np[None]).cuda()).sigmoid().cpu().numpy().ravel()
                torch.cuda.synchronize(); prep=time.perf_counter()-t
                add(name,-prob,prep=prep,meta=dict(warmup_seconds=warm_seconds,device='cuda',checkpoint_seed=name.split('_')[-1]))
            save(dest,dict(signature=signature,case=q,raw_hashes=raw_hashes,optimal_path=optimal_path,rows=rows))
            if (qi+1)%10==0 or qi==0: status('evaluation',stage=stage,done=qi+1,total=len(cases))

def verify(signature):
    counts={}; total=0; violations=[]
    expected={'main':(200,15),'ood':(100,8),'scaling':(20,15)}
    for stage,(n,methods) in expected.items():
        files=list((OUT/'evaluation'/stage).glob('*.json')); assert len(files)==n,(stage,len(files))
        names=None
        for p in files:
            d=json.loads(p.read_text()); assert d['signature']==signature
            assert len(d['rows'])==methods,(p,len(d['rows']))
            ns={r['method'] for r in d['rows']}; assert len(ns)==methods
            if names is None: names=ns
            assert ns==names
            for r in d['rows']:
                assert grid_valid(d['case'],r['path'])[0],(p,r['method'])
                assert abs(path_length(r['path'])-r['length'])<1e-8
                if r['length_ratio']>1.1+1e-8: violations.append([d['case']['id'],r['method']])
                total+=1
        counts[stage]=dict(cases=n,rows=n*methods,methods=sorted(names))
    assert not violations
    split=splits(); assert len(set(sum(split.values(),[])))==100
    result=dict(verified=True,total_rows=total,stages=counts,bound_violations=violations,map_splits_disjoint=True,
                signature=signature,verified_utc=now(),all_paths_revalidated=True)
    save(OUT/'verification.json',result); return result

def export():
    import zipfile
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.log') and p.name!='export.json']
    files += [ROOT/'reproduction'/n for n in ['candidate_core.py','candidate_run.py','candidate_report.py','test_candidates.py','core.py']]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}
    target=ROOT/'candidate-results-verified.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    result=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,files=len(files))
    save(OUT/'export.json',result); print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['all','verify','export']); p.add_argument('--gpu',default='0'); args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    OUT.mkdir(parents=True,exist_ok=True); signature=freeze()
    if args.action=='export': export(); return
    if args.action=='verify': print(json.dumps(verify(signature))); return
    try:
        import torch
        save(OUT/'environment.json',dict(started_utc=now(),host=platform.node(),python=sys.version,platform=platform.platform(),
             numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),selected_gpu=args.gpu))
        prepare_data(); train(); calibration=calibrate(); evaluate(calibration,signature); verify(signature)
        from candidate_report import report
        report(); status('complete',complete_rows=4100)
        current=json.loads((OUT/'status.json').read_text()); current['complete']=True; save(OUT/'status.json',current); export()
    except Exception as exc:
        status('failed',error=str(exc)); traceback.print_exc(); raise

if __name__=='__main__': main()
