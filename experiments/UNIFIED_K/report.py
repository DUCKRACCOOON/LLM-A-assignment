"""K1-K6 OOF aggregation and Korean report under the SM unified protocol.

Mirrors sm_src/report.py (per-case medians, paired geometric-mean ratios vs the original
LLM-A* `B_llm`, map-level bootstrap within folds, quality tails, reconstructed end-to-end
time). Method names are read from frozen.json; families/controls are derived generically.

    python report.py [--run DIR]        # default results/run_20261112 (or env UNIFIED_RUN)
Outputs (in the run dir): per_case.csv, summary.json, summary.csv, direct_comparisons.json,
worst_cases.json, analysis/api_usage.json, report.md, manifest.json.
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from common import *

BASELINES = ['A_GM', 'N_g2h', 'B_llm', 'K3_wall_euclidean', 'K3_wall_octile', 'A_OCT', 'FOCAL_105']
CONTROL_TOKENS = ('noLLM', 'noguide', 'rule', 'random', 'nocal', 'roundrobin')
NAMES = {'A_GM': 'A* (GM)', 'B_llm': '원 LLM-A*', 'N_g2h': 'g+2h', 'A_OCT': 'A*_octile', 'FOCAL_105': '무안내 FOCAL 1.05',
         'K3_wall_euclidean': 'K3 직선거리＋벽', 'K3_wall_octile': 'K3 octile＋벽'}
METRICS = ('total_seconds', 'end_to_end_seconds', 'operation', 'successor_expansions', 'storage', 'length')
BOOT = 2000; SEED = 20261112

def name(m): return NAMES.get(m, m)
def is_control(m): return any(t.lower() in m.lower() for t in CONTROL_TOKENS)

def run_dir(arg=None):
    return Path(arg or os.environ.get('UNIFIED_RUN') or RUN).resolve()

def method_info(R):
    fz = R/'frozen.json'
    if fz.exists():
        f = read(fz)['frozen']; return f['methods'], f['bounds'], f['families']
    from harness import METHODS, bound, family
    return METHODS, {m: bound(m) for m in METHODS}, {m: family(m) for m in METHODS}

def records(R):
    for p in sorted(R.glob('folds/fold*/evaluations/*.json')): yield p, read(p)

def raw_record(R, q, variant):
    p = INPUTS/'raw'/q['stage']/f'{q["id"]}-first.json' if variant == 'first' else R/'raw'/variant/f'{q["id"]}.json'
    if not p.exists(): return None
    r = read(p); assert r['request_sha256'] == digest(r['request']) and r['response_sha256'] == digest(r['response']), p
    return r

def used_variants(method, fam, row, hashes):
    """Responses a condition logically consumes (sequential use assumed)."""
    if not row['api_calls']: return []
    if fam == 'K1': return ['k1_ranking'] if 'k1_ranking' in hashes else []
    if fam == 'K2': return [v for v in ('first', 'k2_cot', 'k2_repe') if v in hashes]
    return ['first']

def per_case(R, methods, families, bounds):
    out = []
    for _, d in records(R):
        q = d['case']; group = defaultdict(list); cache = {}
        for r in d['rows']: group[r['method']].append(r)
        for m in methods:
            rr = sorted(group[m], key=lambda x: x['repeat']); r = rr[0]
            t = st.median(x['total_seconds'] for x in rr)
            used = []
            for v in used_variants(m, families[m], r, d['raw_hashes']):
                if v not in cache: cache[v] = raw_record(R, q, v)
                if cache[v] is not None:
                    assert cache[v]['response_sha256'] == d['raw_hashes'][v]; used.append(cache[v])
            api = sum(x['request_seconds'] for x in used); usage = [x['response'].get('usage', {}) for x in used]
            ex = r.get('extra') or {}
            out.append(dict(case_id=q['id'], map_id=q['map_id'], fold=q['fold_id'], method=m, family=families[m], valid=r['valid'], error=r.get('error'),
                            optimal=d['optimal'], length=r['length'], operation=r['operation'], successor_expansions=r['successor_expansions'],
                            storage=r['storage'], storage_definition=r['storage_definition'], peak_open=r.get('measured_peak_open'),
                            peak_open_definition=r.get('peak_open_definition'), total_seconds=t, api_seconds=api, end_to_end_seconds=t+api,
                            api_calls=r['api_calls'], logical_responses=len(used), prompt_tokens=sum(u.get('prompt_tokens', 0) for u in usage),
                            completion_tokens=sum(u.get('completion_tokens', 0) for u in usage), bound=bounds.get(m),
                            bound_violation=r['bound_violation'], path_sha256=r['path_sha256'],
                            repeat_seconds=json.dumps([x['total_seconds'] for x in rr]),
                            guidance_accepted=ex.get('ranking_accepted', ex.get('guidance_accepted')),
                            extra=json.dumps(ex, ensure_ascii=False, sort_keys=True, default=str)[:2000]))
    return out

def ratio(rr, bb, metric, seed=SEED):
    pairs = []; directions = Counter(improved=0, equal=0, worsened=0, excluded=0); invalid_pairs = 0
    for r in rr:
        b = bb.get(r['case_id'])
        if b is None: directions['excluded'] += 1; continue
        x = r.get(metric); y = b.get(metric)
        if not (r['valid'] and b['valid']): invalid_pairs += 1
        if x is None or y is None or (metric == 'length' and not (r['valid'] and b['valid'])): directions['excluded'] += 1; continue
        eq = math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-8 if metric == 'length' else 1e-12)
        directions['equal' if eq else 'improved' if x < y else 'worsened'] += 1
        if x > 0 and y > 0: pairs.append((r['fold'], r['map_id'], math.log(x/y)))
    if not pairs: return dict(ratio=None, n=0, maps=0, ci95=None, directions=dict(directions), invalid_pairs=invalid_pairs)
    by = defaultdict(list)
    for f, m, x in pairs: by[(f, m)].append(x)
    rng = np.random.default_rng(seed); sums = np.zeros(BOOT); counts = np.zeros(BOOT)
    for fold in sorted({f for f, _ in by}):
        vals = [v for (f, _), v in by.items() if f == fold]; ss = np.array([sum(v) for v in vals]); nn = np.array([len(v) for v in vals])
        choice = rng.integers(0, len(vals), size=(BOOT, len(vals))); sums += ss[choice].sum(1); counts += nn[choice].sum(1)
    boots = np.exp(sums/counts)
    return dict(ratio=math.exp(st.mean(x[2] for x in pairs)), n=len(pairs), maps=len(by), ci95=np.quantile(boots, [.025, .975]).tolist(),
                directions=dict(directions), invalid_pairs=invalid_pairs)

def summarize(subset, methods, domain):
    bb = {r['case_id']: r for r in subset if r['method'] == 'B_llm'}; part = {}
    for m in methods:
        rr = [r for r in subset if r['method'] == m]; valid = [r for r in rr if r['valid']]
        q = [r['length']/r['optimal'] for r in valid]
        v = dict(domain=domain, method=m, family=rr[0]['family'] if rr else None, cases=len(rr), valid_cases=len(valid),
                 success_rate=len(valid)/len(rr) if rr else None,
                 mean_planner_ms=1000*st.mean(r['total_seconds'] for r in rr), mean_end_to_end_seconds=st.mean(r['end_to_end_seconds'] for r in rr),
                 mean_pop=st.mean(r['operation'] for r in rr), mean_expansions=st.mean(r['successor_expansions'] for r in rr),
                 mean_peak_open=st.mean(r['peak_open'] for r in rr if r['peak_open'] is not None) if any(r['peak_open'] is not None for r in rr) else None,
                 mean_storage=st.mean(r['storage'] for r in rr if r['storage'] is not None) if any(r['storage'] is not None for r in rr) else None,
                 gm_cost_ratio=math.exp(st.mean(math.log(x) for x in q)) if q else None, mean_cost_ratio=st.mean(q) if q else None,
                 p95_excess_pct=100*(float(np.quantile(q, .95))-1) if q else None, max_excess_pct=100*(max(q)-1) if q else None,
                 optimal_paths=sum(abs(x-1) <= 1e-8 for x in q), quality_path_n=len(q), over_105=sum(x > 1.05+1e-8 for x in q),
                 over_110=sum(x > 1.1+1e-8 for x in q), bound_violations=sum(r['bound_violation'] for r in rr),
                 calls=sum(r['api_calls'] for r in rr), logical_responses=sum(r['logical_responses'] for r in rr),
                 prompt_tokens=sum(r['prompt_tokens'] for r in rr), completion_tokens=sum(r['completion_tokens'] for r in rr))
        for metric in METRICS: v[metric] = ratio(rr, bb, metric)
        part[m] = v
    return part

def direct_pairs(methods, families):
    pairs = []
    for m in methods:
        fam = families[m]
        if m in BASELINES and fam != 'K3': continue
        if fam == 'K3': pairs.append((m, 'N_g2h')); continue
        if is_control(m): continue
        for c in methods:
            if c != m and families[c] == fam and is_control(c): pairs.append((m, c))
        if fam in ('K2', 'K4') and 'FOCAL_105' in methods: pairs.append((m, 'FOCAL_105'))
    return pairs

def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def api_usage(R):
    api = []
    for p in list(R.glob('raw/*/*.json')):
        r = read(p); u = r['response'].get('usage', {})
        api.append(dict(variant=r.get('variant', p.parent.name), prompt_tokens=u.get('prompt_tokens', 0), completion_tokens=u.get('completion_tokens', 0),
                        cached=u.get('prompt_tokens_details', {}).get('cached_tokens', 0), seconds=r.get('request_seconds', 0),
                        finish=r['response']['choices'][0].get('finish_reason')))
    pt = sum(a['prompt_tokens'] for a in api); ct = sum(a['completion_tokens'] for a in api)
    return dict(new_responses=len(api), by_variant=dict(Counter(a['variant'] for a in api)), prompt_tokens=pt, completion_tokens=ct,
                cached_prompt_tokens=sum(a['cached'] for a in api), summed_wait_seconds=sum(a['seconds'] for a in api),
                finish_reasons=dict(Counter(a['finish'] for a in api)), failures=len(list(R.glob('raw/*/failures/*.json'))),
                estimated_usd_list_price=pt*0.15/1e6+ct*0.60/1e6,
                note='SM shared first responses (inputs/raw/public) are reused and not re-billed here')

def build(R):
    methods, bounds, families = method_info(R)
    rows = per_case(R, methods, families, bounds); assert rows, 'no evaluations'
    csv_write(R/'per_case.csv', rows)
    folds = sorted({r['fold'] for r in rows}); summary = {}; flat = []
    for domain in ['OOF']+[f'fold{f}' for f in folds]:
        subset = rows if domain == 'OOF' else [r for r in rows if r['fold'] == int(domain[4:])]
        summary[domain] = summarize(subset, methods, domain); flat += summary[domain].values()
    save(R/'summary.json', summary)
    csv_write(R/'summary.csv', [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in x.items()} for x in flat])
    direct = []; worst = []
    for m, c in direct_pairs(methods, families):
        rr = [r for r in rows if r['method'] == m]; bb = {r['case_id']: r for r in rows if r['method'] == c}
        direct.append(dict(method=m, control=c, family=families[m], **{k: ratio(rr, bb, k) for k in METRICS}))
    bl = {r['case_id']: r for r in rows if r['method'] == 'B_llm'}
    for m in methods:
        rr = [r for r in rows if r['method'] == m]
        for r in sorted(rr, key=lambda r: r['total_seconds']/bl[r['case_id']]['total_seconds'], reverse=True)[:3]:
            worst.append(dict(kind='time_vs_B_llm', method=m, case_id=r['case_id'], fold=r['fold'], value=r['total_seconds']/bl[r['case_id']]['total_seconds'],
                              record=f'folds/fold{r["fold"]}/evaluations/{r["case_id"]}.json'))
        for r in sorted([r for r in rr if r['valid']], key=lambda r: r['length']/r['optimal'], reverse=True)[:3]:
            worst.append(dict(kind='length_over_optimal', method=m, case_id=r['case_id'], fold=r['fold'], value=r['length']/r['optimal'],
                              record=f'folds/fold{r["fold"]}/evaluations/{r["case_id"]}.json'))
    save(R/'direct_comparisons.json', direct); save(R/'worst_cases.json', worst)
    usage = api_usage(R); save(R/'analysis/api_usage.json', usage)
    mem = [read(p) for p in R.glob('memory/*.json') if p.name != 'memory_status.json']
    ver = read(R/'verification.json') if (R/'verification.json').exists() else None
    agree = None
    if ver and 'aggregates' in ver:
        agree = all(math.isclose(ver['aggregates'][m]['time_ratio'], summary['OOF'][m]['total_seconds']['ratio'], rel_tol=1e-9)
                    and ver['aggregates'][m]['valid_cases'] == summary['OOF'][m]['valid_cases'] for m in methods if m in ver['aggregates'])
    write_md(R, methods, families, bounds, summary, direct, worst, usage, mem, ver, agree, rows)
    out = dict(status='completed', completed_at=now(), cases=len({r['case_id'] for r in rows}), methods=methods,
               report_script_sha256=sha(Path(__file__)), independent_aggregates_agree=agree,
               outputs={n: sha(R/n) for n in ('report.md', 'per_case.csv', 'summary.json', 'direct_comparisons.json')})
    save(R/'manifest.json', out); print(json.dumps(dict(report=str(R/'report.md'), cases=out['cases'], agree=agree)), flush=True)

def write_md(R, methods, families, bounds, summary, direct, worst, usage, mem, ver, agree, rows):
    T = summary['OOF']; n = len({r['case_id'] for r in rows})
    f2 = lambda x: '—' if x is None else f'{x:.2f}'
    pct = lambda x: '—' if x is None else f'{100*x:.2f}%'
    imp = lambda v: '—' if v is None else f'{100*(1-v):+.2f}%'
    ci = lambda d: '—' if not d['ci95'] else f"{pct(d['ratio'])} [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}]"
    bnd = lambda m: '없음' if not bounds.get(m) else ('최단' if bounds[m] == 1 else f'≤{bounds[m]:g}L*')
    kmethods = [m for m in methods if families[m] in ('K1', 'K2', 'K3', 'K4', 'K5', 'K6')]
    primary = [m for m in kmethods if not is_control(m)]
    env = (read(R/'frozen.json')['frozen'].get('environment', {}) if (R/'frozen.json').exists() else {})
    L = ['# K1~K6 공개 1,000문제 통합 규칙 재평가 결과', '',
         '2026-11-12 (Asia/Seoul), gm. SM 브랜치 `sm_part/cv1000_20261112`의 통합 평가 규칙(계획서·실행기)을 그대로 따라 규민 담당 6개 방법군을 다시 측정했다.', '',
         f"평가 문제 {n:,}개 × {len(methods)}조건 × 5회. "
         + (f"독립 검산: 시행 {ver['timed_trials']:,}회, 무효 경로 {ver['invalid_trials']}회, 선언 상한 위반 {ver['bound_violation_trials']}회." if ver else '독립 검산 결과 없음.'), '',
         '## 핵심 결과 (원 LLM-A* = 100)', '',
         '문제별 비율의 기하평균이다. 개선율은 양수가 감소다. 총응답시간은 저장된 실제 API 대기(조건이 논리적으로 쓰는 응답의 합)와 오프라인 계획기 시간의 합인 **재구성 추정**이며 온라인 반복 실측이 아니다.', '',
         '| 방법 | 가족 | 경로 보장 | 유효 경로 | 계획기 시간 감소 | 총응답시간 감소(추정) | heap pop 감소 | 확장 감소 | L/L* 기하평균 | 최대 초과율 |',
         '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in primary:
        d = T[m]
        L.append(f"| {name(m)} | {families[m]} | {bnd(m)} | {d['valid_cases']}/{d['cases']} | {imp(d['total_seconds']['ratio'])} | {imp(d['end_to_end_seconds']['ratio'])} | "
                 f"{imp(d['operation']['ratio'])} | {imp(d['successor_expansions']['ratio'])} | {'—' if d['gm_cost_ratio'] is None else '%.4f' % d['gm_cost_ratio']} | {f2(d['max_excess_pct'])}% |")
    L += ['', '경로 보장 조건이 서로 다르므로 하나의 순위로 묶지 않는다. heap pop은 엔진마다 의미가 다르다(FOCAL 배열 선택·다중 큐·stale pop 포함 여부). 실제 이웃 확장과 시간을 함께 본다.', '',
          '## 실행 조건', '',
          f"- Python {env.get('python', '?').split()[0]} ({env.get('platform', '?')}), CPU {env.get('processor', '?')}. NumPy {env.get('numpy', '?')}, SciPy {env.get('scipy', '?')}, Shapely {env.get('shapely', '?')}, PyTorch {env.get('torch', '?')}.",
          '- SM 규칙은 Python 3.11.18이지만 이 PC의 Windows 앱 제어 정책이 서명 없는 3.11.18 배포본의 DLL을 차단해 공식 서명 3.11.9를 사용했다. 패키지 버전은 SM과 같다.',
          '- SM `fold_manifest.json`(지도 단위 5-fold, 평가 20지도/200문제)과 SM `oracle.json`을 그대로 사용했다. 각 문제는 자기 지도가 보정·선택에 쓰이지 않은 외부 fold에서 한 번 평가했다.',
          '- standard 응답은 SM이 수집한 새 예시 프롬프트(공개 지도와 겹치지 않는 합성 예시 5개) 응답 1,000건을 공유했다. K1 순위 응답과 K2 CoT·RePE 응답은 같은 예시 지도 기준으로 이번에 새로 수집했다.',
          '- K2 conformal 보정은 fold마다 내부 학습+검증 80지도(800문제)만 사용했다. 평가 지도는 보정·설정 선택에 사용하지 않았다.',
          '- 문제마다 모든 조건을 5회 반복하고 반복마다 순서를 고정 난수로 섞었다. 계획기 시간은 5회 중앙값이다. 측정 중 네트워크를 차단했다. 파싱·준비·탐색·복원·필수 최종 검증을 시간에 포함했다.',
          '- 공개 자료는 과거 연구에서 이미 노출됐다. 지도 교차검증은 보정 중복을 막지만 hidden 평가가 아니다.', '',
          '## 전체 조건', '', '시간은 문제별 5회 중앙값의 산술평균이다.', '',
          '| 조건 | 가족 | 성공 | 계획기 ms | 총응답 초(추정) | 평균 pop | 평균 확장 | 평균 저장 | 논리 응답 수 | 입력/출력 토큰 |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in methods:
        d = T[m]
        L.append(f"| {name(m)} | {families[m]} | {d['valid_cases']} | {f2(d['mean_planner_ms'])} | {f2(d['mean_end_to_end_seconds'])} | {f2(d['mean_pop'])} | "
                 f"{f2(d['mean_expansions'])} | {f2(d['mean_storage'])} | {d['logical_responses']} | {d['prompt_tokens']:,}/{d['completion_tokens']:,} |")
    L += ['', '## 경로 품질', '', '| 조건 | 보장 | 최단/유효 | L/L* 기하평균 | P95 초과율 | 최대 초과율 | 5% 초과 | 10% 초과 | 상한 위반 | 길이 vs 원 LLM-A* |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in methods:
        d = T[m]
        g = '—' if d['gm_cost_ratio'] is None else '%.4f' % d['gm_cost_ratio']
        L.append(f"| {name(m)} | {bnd(m)} | {d['optimal_paths']}/{d['quality_path_n']} | {g} | "
                 f"{f2(d['p95_excess_pct'])}% | {f2(d['max_excess_pct'])}% | {d['over_105']} | {d['over_110']} | {d['bound_violations']} | {pct(d['length']['ratio'])} |")
    L += ['', '## 원 LLM-A* 대비 불확실성', '', 'fold 안에서 지도 단위 재표집 2,000회의 95% 구간. 고정된 응답·보정에 조건부인 탐색적 구간이다.', '',
          '| 조건 | 계획기 시간 비율 [95%] | 개선/동일/악화 | 총응답시간 비율 [95%] | 확장 비율 [95%] |', '| --- | ---: | ---: | ---: | ---: |']
    for m in methods:
        if m == 'B_llm': continue
        a = T[m]['total_seconds']; c = a['directions']
        L.append(f"| {name(m)} | {ci(a)} | {c['improved']}/{c['equal']}/{c['worsened']} | {ci(T[m]['end_to_end_seconds'])} | {ci(T[m]['successor_expansions'])} |")
    L += ['', '## 직접 대조 (같은 엔진의 안내 제거·규칙·무작위·보정 제거)', '',
          '| 방법 / 대조 | 계획기 시간 비율 [95%] | pop 비율 | 확장 비율 | 길이 비율 | 해석 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for d in direct:
        t = d['total_seconds']
        meaning = '—' if not t['ci95'] else '시간 감소 관측' if t['ci95'][1] < 1 else '시간 증가 관측' if t['ci95'][0] > 1 else '시간 차이 불확실'
        L.append(f"| {name(d['method'])} / {name(d['control'])} | {ci(t)} | {pct(d['operation']['ratio'])} | {pct(d['successor_expansions']['ratio'])} | {pct(d['length']['ratio'])} | {meaning} |")
    L += ['', '대조군보다 나빠지면 원 LLM-A* 대비 개선이 있더라도 LLM 안내 자체의 추가 효용으로 해석하지 않는다.', '',
          '## fold별 계획기 평균 ms', '', '| fold | '+' | '.join(name(m) for m in primary+['B_llm'])+' |', '| --- |'+' ---: |'*(len(primary)+1)]
    for dom in [k for k in summary if k != 'OOF']:
        L.append(f'| {dom[4:]} | '+' | '.join(f2(summary[dom][m]['mean_planner_ms']) for m in primary+['B_llm'])+' |')
    L += ['', '## LLM 비용', '',
          f"- 이번 신규 응답 {usage['new_responses']:,}건 {usage['by_variant']}: 입력 {usage['prompt_tokens']:,}, 출력 {usage['completion_tokens']:,}토큰(캐시 {usage['cached_prompt_tokens']:,}). "
          f"목록가 기준 추정 ${usage['estimated_usd_list_price']:.2f}(청구서 미확인). 종료 사유 {usage['finish_reasons']}, 실패 기록 {usage['failures']}건.",
          '- standard 응답 1,000건은 SM 수집분을 재사용했다. 위 표의 논리 응답 수·토큰은 각 방법을 단독 운영할 때 사용하는 응답 기준이다.', '']
    if mem:
        L += ['## 메모리 (별도 프로세스 RSS 표본)', '', '| 조건 | 평균 OPEN 최대 | 평균 상태 수 | RSS peak MiB 중앙값 | RSS 증가 MiB 중앙값 | 표본 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for m in methods:
            rs = [r for r in mem if r['method'] == m]
            if not rs: continue
            L.append(f"| {name(m)} | {f2(T[m]['mean_peak_open'])} | {f2(T[m]['mean_storage'])} | {st.median(r['sampled_peak_rss_bytes'] for r in rs)/2**20:.2f} | "
                     f"{st.median(r['sampled_increment_bytes'] for r in rs)/2**20:.2f} | {len(rs)} |")
        L += ['', f'RSS 보조 측정 {len(mem)}회: fold마다 가장 작은 평가 지도 4개의 pair00, 조건별 새 프로세스 3회. nominal 1ms 표본이라 순간 최대를 보장하지 않고 Python·라이브러리 상주분을 포함한다. 시간 측정과 별도로 실행했다. 상태 수는 엔진별 정의가 다르다.', '']
    L += ['## 악화 사례', '', '| 조건 | 종류 | 문제 | 값 |', '| --- | --- | --- | ---: |']
    for e in worst:
        if e['method'] in primary and e is next(x for x in worst if x['method'] == e['method'] and x['kind'] == e['kind']):
            L.append(f"| {name(e['method'])} | {'시간 배수(원 LLM-A* 대비)' if e['kind'] == 'time_vs_B_llm' else 'L/L*'} | [{e['case_id']}]({e['record']}) | {e['value']:.3f} |")
    L += ['', '## 검산과 산출물', '']
    if ver:
        L.append(f"- 독립 검산: 경로 Oracle 재검사, 길이 재계산, 상한 재판정, 반복 결정성, 응답 해시, 서명. SM 기록과의 하네스 교차 확인(기준·K3 조건 결정적 필드): 일치 {ver.get('sm_crosscheck', {}).get('matched', '?')}, 불일치 {ver.get('sm_crosscheck', {}).get('mismatched', '?')}.")
    L.append(f"- 보고서 집계와 독립 검산 집계 일치: {agree}.")
    L += ['- [문제별 결과](per_case.csv), [요약](summary.json), [직접 대조](direct_comparisons.json), [악화 사례](worst_cases.json), [API 사용](analysis/api_usage.json), [검산](verification.json), [고정 해시](frozen.json).',
          '- K1·K2 최초 구현은 승민(SM), 하네스·fold 분할·standard 응답은 SM 통합 실행기, K3~K6 구현은 gm. 이번 이식·재측정·보고서는 Claude Code(에이전트) 지원 작업이다.', '']
    (R/'report.md').write_text('\n'.join(L), encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run'); a = p.parse_args()
    build(run_dir(a.run))
