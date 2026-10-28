"""Rebuild labels and retrain the existing small S4 network on CPU/Python 3.11."""
import math
import time
import numpy as np
import torch
from common import *
from candidate_core import Grid, network_inputs, make_network

def labels():
    dest=OUT/'models/training_data.npz';meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        assert sha(dest)==read(meta)['sha256'];return
    qs=[q for q in cases('public') if q['split'] in ('train','validation')]
    t=time.perf_counter();xs=[];ys=[];masks=[];is_val=[];ids=[];pops=0;graphs={}
    for i,q in enumerate(qs):
        if q['map_id'] not in graphs:graphs[q['map_id']]=Grid(q)
        grid=graphs[q['map_id']]
        ds,_,a=grid.distances(q['start']);dg,_,b=grid.distances(q['goal']);pops+=a+b
        optimum=ds[grid.index(q['goal'])];assert math.isfinite(optimum)
        xs.append(network_inputs(grid,q));ys.append(((ds+dg)<=optimum+.5).reshape(grid.h,grid.w));masks.append(grid.free.reshape(grid.h,grid.w))
        is_val.append(q['split']=='validation');ids.append(q['id'])
        if (i+1)%50==0:print(json.dumps(dict(phase='labels',done=i+1,total=len(qs))),flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,x=np.asarray(xs),y=np.asarray(ys,dtype=np.float32),mask=np.asarray(masks,dtype=np.float32),validation=np.asarray(is_val))
    save(meta,dict(sha256=sha(dest),seconds=time.perf_counter()-t,label_dijkstra_expansions=pops,case_ids=ids,environment=environment()))

def train():
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    labels();data=np.load(OUT/'models/training_data.npz')
    x=torch.from_numpy(data['x']);y=torch.from_numpy(data['y']);mask=torch.from_numpy(data['mask'])
    train_ids=np.flatnonzero(~data['validation']);val_ids=np.flatnonzero(data['validation'])
    positive=float((y[train_ids]*mask[train_ids]).sum());negative=float(mask[train_ids].sum())-positive
    weight=torch.tensor(min(30.,negative/max(positive,1.)))
    criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=weight)
    for seed in (11,22,33):
        checkpoint=OUT/'models'/f'seed{seed}.pt';meta=checkpoint.with_suffix('.json')
        if meta.exists():assert read(meta)['sha256']==sha(checkpoint);continue
        torch.manual_seed(seed);np.random.seed(seed);model=make_network()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=math.inf;best_epoch=0;history=[];t=time.perf_counter()
        for epoch in range(60):
            model.train();loss_sum=0.;order=np.random.permutation(train_ids)
            for i in range(0,len(order),32):
                idx=order[i:i+32];loss=(criterion(model(x[idx]),y[idx])*mask[idx]).sum()/mask[idx].sum()
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();loss_sum+=float(loss.item())*len(idx)
            model.eval();val_loss=0.
            with torch.no_grad():
                for i in range(0,len(val_ids),32):
                    idx=val_ids[i:i+32];loss=(criterion(model(x[idx]),y[idx])*mask[idx]).sum()/mask[idx].sum();val_loss+=float(loss.item())*len(idx)
            val_loss/=len(val_ids);history.append(dict(epoch=epoch+1,train_loss=loss_sum/len(order),validation_loss=val_loss))
            if val_loss<best:best=val_loss;best_epoch=epoch+1;torch.save(model.state_dict(),checkpoint)
            status=dict(phase='training',seed=seed,epoch=epoch+1,epochs=60,validation_loss=val_loss,best_epoch=best_epoch,seconds=time.perf_counter()-t,complete=False)
            save(OUT/'training_status.json',status)
            if (epoch+1)%5==0:print(json.dumps(status),flush=True)
        save(meta,dict(seed=seed,epochs=60,best_epoch=best_epoch,best_validation_loss=best,seconds=time.perf_counter()-t,
                       parameters=sum(p.numel() for p in model.parameters()),sha256=sha(checkpoint),history=history,positive_weight=weight.item(),device='cpu',threads=4))
    save(OUT/'training_status.json',dict(phase='trained',seeds=[11,22,33],complete=True,updated_at=now()))

if __name__=='__main__':train()
