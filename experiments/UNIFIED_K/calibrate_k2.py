"""Fold-level split-conformal calibration for K2 (COMMON18 benchmark.calibrate score, per fold).

Calibration set = the fold's inner train+validation maps only (80 maps / 800 cases); the 20
outer test maps are never read. Nonconformity score of a case = max over the canonical
shortest grid path (cv_candidate_core Grid.distances/reconstruct) of the 3-response corridor
field divided by the response-dispersion scale. Map-level exchangeability: one score per
map = max over its 10 cases. qhat = conformal_quantile(map maxima, alpha=0.1).

Usage: python calibrate_k2.py [fold ...]   (default 0-4)
       --dry-standard-only  : code check using the standard response three times; writes
                              *_dry.json and is never used for evaluation.
"""
import sys
import time
from collections import defaultdict
import numpy as np
from common import RUN, cases, raw, text_of, save, sha, digest, now, MODEL, read
import cv_candidate_core as cc
from methods_k12 import proposals

ALPHA = .1
VARIANTS = ('first', 'k2_cot', 'k2_repe')

def calibrate(fold, dry=False):
    began = time.perf_counter()
    qs = cases('train', fold)+cases('validation', fold)
    assert len(qs) == 800 and not any(q['split'] == 'test' for q in qs)
    scores = defaultdict(list); details = []; errors = defaultdict(int); hashes = {}; last = None
    for q in sorted(qs, key=lambda q: q['id']):
        if q['map_index'] != last: grid = cc.Grid(q); last = q['map_index']
        records = [raw(q, 'first')]*3 if dry else [raw(q, v) for v in VARIANTS]
        hashes[q['id']] = [r['response_sha256'] for r in records]
        paths, errs = proposals(q, [text_of(r) for r in records])
        for v, e in zip(VARIANTS, errs):
            if e: errors[f'{v}:{e}'] += 1
        field, scale = cc.llm_field(grid, q, paths)
        distances, parents, _ = grid.distances(q['start']); path = cc.reconstruct(grid, parents, q['start'], q['goal'])
        score = max(float(field[grid.index(p)]) for p in path)/scale
        scores[q['map_index']].append(score)
        details.append(dict(case_id=q['id'], map_index=q['map_index'], split=q['split'], score=score, scale=scale,
                            optimum=float(distances[grid.index(q['goal'])])))
    maxima = {str(k): max(v) for k, v in scores.items()}
    quantile = cc.conformal_quantile(maxima.values(), ALPHA); assert np.isfinite(quantile)
    value = dict(fold=fold, quantile=quantile, alpha=ALPHA, calibration_maps=len(maxima), cases=len(qs),
                 splits_used=['train', 'validation'], selection_uses_test=False, dry_standard_only=dry,
                 score='max over canonical shortest path of min-over-3-proposals point-segment distance / (1+mean std across proposals)',
                 map_scores=maxima, parse_errors=dict(errors), response_hashes_sha256=digest(hashes),
                 fold_manifest_sha256=sha(RUN.parent.parent/'inputs/fold_manifest.json'), model=MODEL,
                 seconds=time.perf_counter()-began, created_at=now(), details=details)
    save(RUN/'folds'/f'fold{fold}'/('k2_calibration_dry.json' if dry else 'k2_calibration.json'), value)
    print(f'fold {fold}: qhat={quantile:.6f} maps={len(maxima)} cases={len(qs)} parse_errors={dict(errors)}', flush=True)
    return value

if __name__ == '__main__':
    dry = '--dry-standard-only' in sys.argv
    folds = [int(a) for a in sys.argv[1:] if a.isdigit()] or range(5)
    for f in folds: calibrate(f, dry)
