"""Shared paths/data for the K1-K6 re-evaluation under the SM unified protocol.

Adapted from sm_src/common.py (SM cv1000, commit 0c0a6082). Only paths changed:
inputs are the copied SM run inputs; outputs go to results/run_20261112.
"""
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SMSRC = HERE/'sm_src'
INPUTS = HERE/'inputs'
RUN = HERE/'results/run_20261112'
FOLD = int(os.environ.get('CV_FOLD', '0'))
OUT = RUN/'folds'/f'fold{FOLD}'
# Order: our folder first (sm_src also contains common/evaluate/report/memory.py), then SM sources, then repo.
for p in (str(REPO), str(SMSRC), str(HERE)):
    if p in sys.path: sys.path.remove(p)
    sys.path.insert(0, p)
from reproduction import core as gm                      # identical to SM vendor/gm (sha checked)
from experiments.K3.heuristic import LowerBound           # noqa: F401
from reproduction.proposed import proposed_search         # noqa: F401

MODEL = 'gpt-4o-mini-2024-07-18'
PROMPTS = INPUTS/'prompts'
STANDARD = (PROMPTS/'standard.txt').read_text(encoding='utf-8')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try: temp.replace(path); break
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)

MANIFEST = read(INPUTS/'fold_manifest.json')

def splits(fold=None):
    part = MANIFEST['folds'][FOLD if fold is None else fold]
    return {k: part[k+'_maps'] for k in ('train', 'validation', 'test')}

def cases(stage=None, fold=None):
    """stage: None=all public 1000 (split labels for `fold`), 'test'/'train'/'validation', or 'pilot'."""
    if stage == 'pilot': return read(INPUTS/'data/pilot.json')
    sp = splits(fold); ans = []
    for item in read(REPO/'experiments/K3/results/run_20261025/public_cases.json'):
        mid = item['map_index']; q = dict(item['query'])
        q.update(id=item['case_id'], map_id=f'public:{mid}', map_index=mid, pair_id=item['pair_index'],
                 stage='public', fold_id=FOLD if fold is None else fold, split=next(k for k, v in sp.items() if mid in v))
        ans.append(q)
    return [q for q in ans if q['split'] == stage] if stage else ans

def text_of(record): return record['response']['choices'][0]['message'].get('content') or ''

def raw_path(q, variant):
    """variant 'first' = SM shared standard response; others = our new collections (k1_ranking, k2_cot, k2_repe)."""
    if variant == 'first': return INPUTS/'raw'/q['stage']/f'{q["id"]}-first.json'
    return RUN/'raw'/variant/f'{q["id"]}.json'

def raw(q, variant='first'):
    record = read(raw_path(q, variant))
    assert record['request_sha256'] == digest(record['request'])
    assert record['response_sha256'] == digest(record['response'])
    return record

def pilot_reference(): return read(INPUTS/'data/pilot_reference.json')

def environment():
    import numpy, scipy, shapely, torch
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count(), numpy=numpy.__version__, scipy=scipy.__version__, shapely=shapely.__version__,
                torch=torch.__version__, device='cpu')
