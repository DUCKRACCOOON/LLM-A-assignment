"""Finish the frozen E3 replay using existing responses; network calls forbidden."""
import concurrent.futures
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / 'reproduction'))
from improve_run import Runner, MODEL, kinds, protocol_signature, now
from run import cases_for, digest, payload, save

source = root / 'results/improvements'
out = root / 'results/improvements_partial'
failure = json.loads((source / 'status.json').read_text())
assert failure['phase'] == 'failed' and 'insufficient_quota' in failure['error']
assert json.loads((source / 'protocol.json').read_text())['signature'] == protocol_signature()


def cache_only(stage, q, kind, variant, body):
    assert variant == 'first'
    p = source / stage / 'raw' / f'{q["id"]}-{kind}-{variant}.json'
    record = json.loads(p.read_text(encoding='utf-8'))
    assert record['payload_sha256'] == digest(body) == digest(record['payload'])
    assert record['response_sha256'] == digest(record['response'])
    assert record['response']['model'] == MODEL
    return record


for stage in ('main', 'scaling'):
    for q in cases_for(stage):
        for kind in kinds(stage):
            cache_only(stage, q, kind, 'first', payload(q, kind))

runner = Runner(None, 8)
runner.api = cache_only
progress = {'additional_api_requests': 0, 'started_at': now(), 'complete': False,
            'protocol_signature': protocol_signature(),
            'execution_exception': 'E3 cached replay follows interrupted E2, before remaining E2 scaling. Frozen intervention, inputs and evaluation rules are unchanged.'}
for stage in ('main', 'scaling'):
    qs = cases_for(stage)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8 if stage == 'main' else 3) as pool:
        futures = [pool.submit(runner.delayed, stage, q) for q in qs]
        for done, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            progress.update(stage=stage, completed_cases=done, total_cases=len(qs))
            save(out / 'replay_progress.json', progress)
            if done % 50 == 0 or done == len(qs):
                print(json.dumps(progress), flush=True)
progress.update(complete=True, finished_at=now())
save(out / 'replay_audit.json', progress)
print('Cached E3 replay complete. Additional API requests: 0.', flush=True)
