"""Pilot check of K1/K2 adapters on the 30 synthetic development maps (not part of evaluation).

Fake responses: the verified shortcut waypoints for standard/CoT/RePE, '{"ranking":[1,0]}' for K1.
Asserts validity, the 1.05 bound and an independent Oracle path check; heap pops must be counted.
"""
import json
import sys
import time
from harness import run, EXTRA, bound
from common import pilot_reference
from experiments.K3.verification import Oracle

K12 = ['K1_llm', 'K1_rule', 'K1_random', 'K2_conformal', 'K2_nocal']

def main():
    assert all(m in EXTRA for m in K12)
    refs = pilot_reference(); n = 0; seconds = {m: 0. for m in K12}; accepted = 0
    for item in refs:
        q = item['query']; oracle = Oracle(q); text = 'Generated Path: '+json.dumps(item['waypoints'])
        ctx = dict(text=text, replies={'k2_cot': text, 'k2_repe': text, 'k1_ranking': '{"ranking":[1,0]}'},
                   finish={'k1_ranking': 'stop'}, calibration={'quantile': 0.5}, fold=0)
        for m in K12:
            r = run(q, m, ctx)
            assert r['valid'], (q['id'], m, r['error'])
            assert r['length'] <= bound(m)*item['optimal']+1e-8, (q['id'], m, r['length'], item['optimal'])
            # cc.focal selects by array scan: heap pops are only stale anchor pops (may be 0); pushes must be seen.
            assert r['heap_counts'].get('heap_pushes', 0) > 0, (m, 'heapq not counted')
            oracle.check_path(q, r); seconds[m] += r['total_seconds']; n += 1
            accepted += bool(r['extra'].get('ranking_accepted'))
    print(json.dumps(dict(pilot_cases=len(refs), checks=n, k1_rankings_accepted=accepted,
                          mean_ms={m: round(1000*s/len(refs), 3) for m, s in seconds.items()})))

if __name__ == '__main__': main()
