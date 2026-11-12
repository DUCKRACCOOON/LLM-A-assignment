"""Fresh, bounded, resumable API collection for K1 ranking and K2 CoT/RePE.

Mirrors sm_src (SM cv1000) collect.py: one transport attempt per request, failures saved
and never auto-retried in-process; a later invocation skips existing successes (resume).
The key is read from the repo-root .env and never printed or stored.
"""
import argparse
import concurrent.futures
import threading
import urllib.error
import urllib.request
from common import *
import next_topology as topo

VARIANTS = ('k2_cot', 'k2_repe', 'k1_ranking')
LOCK = threading.Lock()

def key_value():
    for line in (REPO/'.env').read_text(encoding='utf-8').splitlines():
        if line.strip().startswith('OPENAI_API_KEY='): return line.split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('OPENAI_API_KEY missing in .env')

def body(content, max_tokens=1000):
    return dict(model=MODEL, messages=[dict(role='user', content=content)], temperature=0, max_tokens=max_tokens, store=False)

def template(name): return (RUN/'prompts'/f'{name}.txt').read_text(encoding='utf-8')

def tasks(selected=None, limit=None):
    cot, repe = template('cot'), template('repe'); out = []; skipped = []
    qs = cases()
    if selected: qs = [q for q in qs if q['id'] in selected]
    if limit: qs = qs[:limit]
    for q in qs:
        fmt = {k: q[k] for k in ('start', 'goal', 'horizontal_barriers', 'vertical_barriers')}
        out.append((q, 'k2_cot', body(cot.format(**fmt))))
        out.append((q, 'k2_repe', body(repe.format(**fmt))))
        proposal, _ = topo.routes(q, k=2)
        if len(proposal['candidates']) >= 2: out.append((q, 'k1_ranking', body(topo.prompt(q, proposal), 256)))
        else: skipped.append(dict(case_id=q['id'], candidates=len(proposal['candidates'])))
    return out, skipped

def request(payload, key):
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            data = json.load(res); seconds = time.perf_counter()-t
            return dict(response=data, latency_seconds=seconds, request_seconds=seconds,
                        attempts=[dict(attempt=1, status=200, seconds=seconds)], request_id=res.headers.get('x-request-id'))
    except urllib.error.HTTPError as exc:
        try: error = json.loads(exc.read()).get('error', {})
        except Exception: error = {}
        raise RuntimeError(f'OpenAI HTTP {exc.code}; code={error.get("code")}; type={error.get("type")}') from None
    except urllib.error.URLError:
        raise RuntimeError('OpenAI connection failed; credential details suppressed') from None

def obtain(q, variant, payload, key):
    dest = raw_path(q, variant)
    if dest.exists():
        record = read(dest); assert record['request_sha256'] == digest(payload); return 'existing'
    started = time.perf_counter()
    try: result = request(payload, key)
    except Exception as exc:
        save(dest.parent/'failures'/f'{dest.stem}-{time.time_ns()}.json', dict(case_id=q['id'], variant=variant, request=payload,
             request_sha256=digest(payload), created_at=now(), error=str(exc), seconds=time.perf_counter()-started, retries=0))
        return 'failed:'+str(exc)
    record = dict(case_id=q['id'], stage=q['stage'], variant=variant, created_at=now(), request=payload, request_sha256=digest(payload), **result)
    record['response_sha256'] = digest(record['response']); save(dest, record); return 'new'

def collect(workers, limit=None, selected=None, label='full'):
    key = key_value(); todo, skipped = tasks(selected, limit)
    save(RUN/'collection/ranking_skipped.json', dict(rule='k1_ranking requested only when topology routes(k=2) yields >=2 candidates', skipped=skipped)) if not limit and not selected else None
    started = now(); counts = {}; t0 = time.perf_counter()
    log = RUN/'collection'/f'{label}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(obtain, q, v, b, key): (q['id'], v) for q, v, b in todo}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            status = f.result(); kind = status.split(':')[0]; counts[kind] = counts.get(kind, 0)+1
            if kind == 'failed':
                with LOCK, open(log, 'a', encoding='utf-8') as fh: fh.write(json.dumps(dict(case=futures[f], status=status, at=now()))+'\n')
            if (i+1) % 50 == 0 or i+1 == len(futures):
                st = dict(phase='collect', label=label, done=i+1, total=len(futures), counts=counts, workers=workers, started_at=started,
                          elapsed_seconds=time.perf_counter()-t0, updated_at=now(), complete=False)
                save(RUN/'collection/collection_status.json', st); print(json.dumps(st), flush=True)
    return counts, len(todo), skipped

def usage():
    agg = dict(actual_responses=0, by_variant={}, prompt_tokens=0, completion_tokens=0, cached_prompt_tokens=0, summed_wait_seconds=0., finish_reasons={}, failures={})
    for v in VARIANTS:
        files = sorted((RUN/'raw'/v).glob('*.json')); agg['by_variant'][v] = len(files)
        agg['failures'][v] = len(list((RUN/'raw'/v/'failures').glob('*.json'))) if (RUN/'raw'/v/'failures').exists() else 0
        per = dict(prompt_tokens=0, completion_tokens=0, finish_reasons={})
        for p in files:
            r = read(p); u = r['response'].get('usage', {}); agg['actual_responses'] += 1
            for k in ('prompt_tokens', 'completion_tokens'): agg[k] += u.get(k, 0); per[k] += u.get(k, 0)
            agg['cached_prompt_tokens'] += (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            agg['summed_wait_seconds'] += r['latency_seconds']
            fr = r['response']['choices'][0].get('finish_reason'); agg['finish_reasons'][fr] = agg['finish_reasons'].get(fr, 0)+1
            per['finish_reasons'][fr] = per['finish_reasons'].get(fr, 0)+1
        agg[v] = per
    agg['estimated_usd'] = round(agg['prompt_tokens']*.15e-6+agg['completion_tokens']*.6e-6, 4)
    agg['pricing'] = 'gpt-4o-mini list price $0.15/M input, $0.60/M output (cached discount ignored)'
    agg['model'] = MODEL; agg['updated_at'] = now(); save(RUN/'collection/api_usage.json', agg); return agg

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['dry', 'smoke', 'collect', 'usage'])
    p.add_argument('--workers', type=int, default=16); p.add_argument('--label', default='full'); a = p.parse_args()
    if a.action == 'dry':
        todo, skipped = tasks(limit=3)
        for q, v, b in todo: print(q['id'], v, len(b['messages'][0]['content']), b['max_tokens'], digest(b)[:12])
    elif a.action == 'smoke':
        print(collect(4, limit=3, label='smoke')); print(json.dumps(usage(), indent=1))
    elif a.action == 'collect':
        counts, total, skipped = collect(a.workers, label=a.label); agg = usage()
        failed = counts.get('failed', 0)
        save(RUN/'collection/collection_status.json', dict(phase='collected' if not failed else 'collected_with_failures', label=a.label, total=total,
             counts=counts, ranking_skipped=len(skipped), complete=failed == 0, updated_at=now()))
        print(json.dumps(dict(counts=counts, total=total, ranking_skipped=len(skipped), usd=agg['estimated_usd'])))
    else: print(json.dumps(usage(), indent=1))
