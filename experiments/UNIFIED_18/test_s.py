"""Equivalence of methods_s.py (through UNIFIED_K harness.run) with SM's recorded S rows.

For every public case (OOF fold from fold_manifest) each S condition runs once and its
deterministic fields -- the same ``det`` as UNIFIED_K/evaluate.py: valid, error, length,
operation, successor_expansions, storage, api_calls, path_sha256 -- are compared with SM's
recorded row (repeat 0) in origin/sm .../folds/fold<k>/evaluations/<id>.json (read straight
from git, nothing written to disk). Also: independent Oracle path check and declared bound
vs UNIFIED_K/inputs/oracle.json. CPU, torch 1 thread, network disabled.

    python test_s.py [--limit N] [--folds 0,1] [--out results/s_port/equivalence.json]
"""
import argparse
import json
import math
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
UK = HERE.parent/'UNIFIED_K'
sys.path.insert(0, str(UK))
import common                                   # noqa: E402  UNIFIED_K/common.py
from common import cases, raw, text_of, digest, read, INPUTS, REPO, environment   # noqa: E402
import harness                                  # noqa: E402
if str(HERE) in sys.path: sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))
import methods_s                                # noqa: E402
from experiments.K3.verification import Oracle  # noqa: E402

SM_COMMIT = '0c0a60821b6b3284de30aa9572926175aa65e240'
SM_RUN = 'sm_part/cv1000_20261112/results/s1_s6_cv1000_20261111/runs/run_20261111_01'
FIELDS = ('valid', 'error', 'length', 'operation', 'successor_expansions', 'storage', 'api_calls')

for name, spec in methods_s.METHODS.items():
    assert name not in harness.EXTRA and name not in harness.BASE, name
    harness.EXTRA[name] = spec
CONDITIONS = list(methods_s.METHODS)

class GitBlobs:
    """Stream files from origin/sm via one `git cat-file --batch` process."""
    def __init__(self):
        self.p = subprocess.Popen(['git', '-C', str(REPO), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def get(self, path):
        self.p.stdin.write(f'{SM_COMMIT}:{path}\n'.encode()); self.p.stdin.flush()
        header = self.p.stdout.readline().split()
        assert len(header) == 3, header
        data = self.p.stdout.read(int(header[2])); self.p.stdout.read(1)
        return json.loads(data.decode('utf-8'))

def det_of(r, ph):
    d = {k: r.get(k) for k in FIELDS}; d['path_sha256'] = ph; return d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int); ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--out', default=str(HERE/'results/s_port/equivalence.json')); a = ap.parse_args()
    socket.socket.connect = socket.create_connection = lambda *x, **k: (_ for _ in ()).throw(RuntimeError('network disabled'))
    import torch; torch.set_num_threads(1)
    truth = read(INPUTS/'oracle.json'); blobs = GitBlobs(); oracles = {}
    match = Counter(); exact_len = Counter(); field_mismatch = defaultdict(Counter); examples = defaultdict(list)
    seconds = defaultdict(list); oracle_ok = Counter(); bound_ok = Counter(); n = 0; flagged = Counter()
    paper_ok = Counter(); multi = Counter(); paper_sum = defaultdict(lambda: [0, 0])
    for fold in map(int, a.folds.split(',')):
        qs = cases('test', fold=fold)
        if a.limit: qs = qs[:a.limit]
        for q in qs:
            rec = blobs.get(f'{SM_RUN}/folds/fold{fold}/evaluations/{q["id"]}.json')
            assert rec['case']['id'] == q['id'] and rec['case']['fold_id'] == fold
            sm_rows = {r['method']: r for r in rec['rows'] if r['repeat'] == 0}
            first = raw(q); assert first['response_sha256'] == rec['raw_hashes']['first']
            extra = methods_s.replies(q)                                  # untimed: S2 replies, fold models, warm-up
            for v, (_, _, h) in extra.items(): assert h == rec['raw_hashes'][v.split('_')[1]], (q['id'], v)
            assert set(extra) == ({'s2_feedback', 's2_control'} if len(rec['raw_hashes']) == 3 else set())
            ctx = dict(text=text_of(first), replies={v: t for v, (t, _, _) in extra.items()}, fold=fold, seed=f'test:{q["id"]}')
            oracle = oracles.get(q['map_id']) or oracles.setdefault(q['map_id'], Oracle(q))
            for m in CONDITIONS:
                r = harness.run(q, m, ctx); ph = digest(r['path']); seconds[m].append(r['total_seconds'])
                if r['extra'].get('artefacts_loaded_in_timed_call'): flagged[m] += 1
                # paper_operation consistency: all harness heap pops must be main-OPEN pops (=> aux_operation 0).
                e = r['extra']; heap_main = e.get('paper_operation_stale_pops', e['paper_operation'])
                paper_ok[m] += int(r['operation'] == heap_main and e['aux_operation'] == 0)
                paper_sum[m][0] += e['paper_operation']; paper_sum[m][1] += e['paper_storage']
                if 'successor_expansions_corrected' in e: multi[m] += int(e['successor_expansions_corrected'] != r['successor_expansions'])
                if r['valid']:
                    oracle.check_path(q, r); oracle_ok[m] += 1
                    w = harness.bound(m)
                    bound_ok[m] += int(not w or r['length'] <= w*truth[q['id']]['optimal']+1e-8)
                mine = det_of(r, ph); theirs = det_of(sm_rows[m], sm_rows[m]['path_sha256'])
                if mine == theirs: match[m] += 1
                else:
                    diff = [k for k in mine if mine[k] != theirs[k]]
                    if diff == ['length'] and mine['length'] is not None and math.isclose(mine['length'], theirs['length'], rel_tol=1e-12):
                        match[m] += 1; exact_len[m] += 1          # identical path, length summed in a different order
                    else:
                        for k in diff: field_mismatch[m][k] += 1
                        if len(examples[m]) < 5: examples[m].append(dict(case=q['id'], fold=fold, mine={k: mine[k] for k in diff}, sm={k: theirs[k] for k in diff}))
            n += 1
            if n % 50 == 0: print(json.dumps(dict(done=n, match={m: match[m] for m in CONDITIONS})), flush=True)
    summary = {m: dict(cases=n, det_match=match[m], length_ulp_only=exact_len[m], mismatched_fields=dict(field_mismatch[m]),
                       examples=examples[m], oracle_valid=oracle_ok[m], within_bound=bound_ok[m], bound=harness.bound(m),
                       family=methods_s.METHODS[m]['family'], role=methods_s.METHODS[m]['role'],
                       mean_ms=1000*sum(seconds[m])/len(seconds[m]), median_ms=1000*sorted(seconds[m])[len(seconds[m])//2],
                       artefacts_loaded_in_timed_call=flagged[m], heap_pops_equal_paper_main_pops=paper_ok[m],
                       mean_paper_operation=paper_sum[m][0]/n, mean_paper_storage=paper_sum[m][1]/n, s6_expansion_double_count_cases=multi[m]) for m in CONDITIONS}
    out = dict(sm_commit=SM_COMMIT, cases=n, folds=a.folds, fields=list(FIELDS)+['path_sha256'], model_load_seconds=methods_s.LOAD_SECONDS,
               environment=environment(), torch_threads=torch.get_num_threads(), summary=summary)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    for m in CONDITIONS:
        s = summary[m]; print(f"{m:18s} match {s['det_match']}/{n}  oracle {s['oracle_valid']}  bound {s['within_bound']}  {s['mean_ms']:.2f} ms  paper_ok {s['heap_pops_equal_paper_main_pops']}  {s['mismatched_fields'] or ''}")

if __name__ == '__main__':
    main()
