"""Offline-only cached Llama generation; no external APIs or downloads."""
import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def model_manifest(root,out):
    root=Path(root); m=json.loads((root/'results/llama/model_manifest.json').read_text()); path=root/m['snapshot_relative_path']
    checked={}
    for name,info in m['files'].items():
        p=path/name
        assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
        checked[name]=info
    save(out,dict(m,verified_files=checked,absolute_snapshot_path=str(path)))
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-root',required=True);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    tasks=json.loads(Path(a.tasks).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    pending=[t for t in tasks if not(out/(t['id']+'.json')).exists()]
    if not pending:return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    set_seed(26101430);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    m=json.loads((Path(a.model_root)/'results/llama/model_manifest.json').read_text())
    path=Path(a.model_root)/m['snapshot_relative_path'];t=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,
          device_map={'':a.device},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    load_seconds=time.perf_counter()-t
    eos=[tok.eos_token_id,tok.convert_tokens_to_ids('<|eot_id|>')]
    # One warmup whose cost is recorded separately, not a scored task.
    warm=tok('Return JSON.',return_tensors='pt').to(model.device);t=time.perf_counter()
    with torch.inference_mode():model.generate(**warm,max_new_tokens=1,do_sample=False,pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize();warm_seconds=time.perf_counter()-t
    save(out/'runtime.json',dict(model_id=m['model_id'],revision=m['revision'],load_seconds=load_seconds,
         warmup_seconds=warm_seconds,gpu=torch.cuda.get_device_name(a.device),torch=torch.__version__,dtype='BF16',decoding='greedy'))
    for task in pending:
        text=tok.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=False,add_generation_prompt=True)
        t=time.perf_counter();batch=tok(text,return_tensors='pt').to(model.device)
        num=int(batch.input_ids.shape[1]);limit=task['max_new_tokens']
        if num+limit>model.config.max_position_embeddings:
            save(out/(task['id']+'.json'),dict(task_id=task['id'],prompt_sha256=sha_text(task['prompt']),valid=False,
                 error='context_limit',input_tokens=num,seconds=time.perf_counter()-t,ranking=None));continue
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**batch,max_new_tokens=limit,do_sample=False,temperature=None,top_p=None,
                          eos_token_id=eos,pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize();new=generated[0,num:].tolist();decoded=tok.decode(new,skip_special_tokens=True)
        r=dict(task_id=task['id'],prompt_sha256=sha_text(task['prompt']),prompt=task['prompt'],raw_text=decoded,
               input_tokens=num,output_tokens=len(new),output_ids=new,seconds=time.perf_counter()-t,
               generation_seconds=time.perf_counter()-start,finish_reason='eos' if new and new[-1] in eos else 'length',
               valid=False,ranking=None,weights=None)
        try:
            match=re.search(r'\{[^{}]*\}',decoded,re.S);obj=json.loads(match.group())
            if task['kind']=='ranking':
                ids=obj['ranking'];assert isinstance(ids,list) and all(type(x)==int for x in ids)
                assert sorted(ids)==list(range(task['candidate_count']));r['ranking']=ids
            else:
                w=obj['weights'];assert len(w)==4 and all(type(x) in (int,float) and math.isfinite(x) and abs(x)<=10 for x in w)
                r['weights']=list(map(float,w))
            r['valid']=r['finish_reason']=='eos'
            if not r['valid']:r['error']='truncated'
        except Exception as e:r['error']='invalid_json_or_schema'
        save(out/(task['id']+'.json'),r)
        print(json.dumps(dict(task=task['id'],valid=r['valid'],tokens=len(new),seconds=r['seconds'])),flush=True)


def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
if __name__=='__main__':main()
