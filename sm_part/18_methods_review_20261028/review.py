"""Read-only reanalysis of the pinned remote COMMON18 run; no search/API calls."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,gzip,hashlib,json,math,statistics
import numpy as np

ROOT=Path(r'C:\인지개')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=ROOT/'tmp/common18_remote_20261028/experiments/COMMON18/results/run_20261028')
p.add_argument('--output',type=Path,default=ROOT/'results/18_methods_review_20261028')
args=p.parse_args();SOURCE=args.source;OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
REF='cd7d490eada5bd0af9c79a552c46694f06737f32'
SMREF='e6458c5d8a71b7b6ddad913b6f0202644a686ac3'
URL=f'https://github.com/DUCKRACCOOON/LLM-A-assignment/blob/{REF}/experiments/COMMON18'
SMURL=f'https://github.com/DUCKRACCOOON/LLM-A-assignment/blob/{SMREF}/sm_part/gm_retest_20261028'
PRIMARY=[f'{c}{i}' for c in 'KJS' for i in range(1,7)]
NAMES=dict(zip(PRIMARY,['통로 ID 선택','Conformal','벽 하한 g+2h','LLM FOCAL','MHA*+DTS','적응 안내 가중치','CHOMP 경로 수리','FMM 거리장','LLM 초기 해+정확 가지치기','ALT4','NBS octile','lazy OPEN','경유점 정제','오류 피드백 수정','정체 시 호출','소형 TransPath','정확 직사각형 압축','장애물 단계 반영']))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
artifact_hashes=read(SOURCE/'artifacts_sha256.json')
for n,h in artifact_hashes.items():assert sha(SOURCE/n)==h,n
summary=read(SOURCE/'summary.json');audit=read(SOURCE/'audit.json')
assert audit['passed'] and audit['primary']==18 and audit['failed_runs']==0
with (SOURCE/'per_case.csv').open(encoding='utf-8-sig',newline='') as f:flat=list(csv.DictReader(f))
numeric=['local_seconds','api_seconds','reconstructed_total_seconds','operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens','length','optimum','peak_open','peak_heap_entries']
by=defaultdict(dict)
for row in flat:
    for k in numeric:row[k]=float(row[k]) if row.get(k) not in (None,'') else None
    row['valid']=row['valid']=='True';row['map_index']=int(row['map_index'])
    by[row['method']][row['case_id']]=row
base=by['LLMA'];assert len(base)==200 and len(by)==45
trials=defaultdict(list);raw_count=0;quality_rows=defaultdict(list);unique_seeds={}
with gzip.open(SOURCE/'evaluations.jsonl.gz','rt',encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line);raw_count+=1
        assert math.isclose(row['local_seconds']+row['api_seconds'],row['reconstructed_total_seconds'],rel_tol=1e-12)
        trials[row['method'],row['case_id'],row['seed_index']].append(row)
for (m,cid,seed),rs in trials.items():
    assert len(rs)==3 and len({r['valid'] for r in rs})==1
    if rs[0]['valid']:quality_rows[m].append(rs[0]['length']/rs[0]['optimum'])
    unique_seeds[m,cid,seed]=rs[0]
assert raw_count==audit['runs']==30600
# Verify saved per-case aggregates against the original repeated records.
checks=0
for m,rows in by.items():
    assert set(rows)==set(base)
    for cid,row in rows.items():
        seedgroups=[rs for (mm,cc,ss),rs in trials.items() if mm==m and cc==cid]
        for metric in numeric[:11]:
            vals=[statistics.median(r[metric] for r in rs) if all(r.get(metric) is not None for r in rs) else None for rs in seedgroups]
            actual=statistics.mean(vals) if all(v is not None for v in vals) else None
            expected=row[metric]
            assert (actual is None and expected is None) or (actual is not None and expected is not None and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-9)),(m,cid,metric)
            checks+=1

def paired(rows,baseline,metric):
    logs=[];maps=defaultdict(list);directions=dict(improved=0,equal=0,worsened=0,missing=0);zero=0
    for cid,r in rows.items():
        a,b=r[metric],baseline[cid][metric]
        if a is None or b is None:directions['missing']+=1;continue
        if math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):directions['equal']+=1
        elif a<b:directions['improved']+=1
        else:directions['worsened']+=1
        if a<=0 or b<=0:zero+=1;continue
        log=math.log(a/b);logs.append(log);maps[r['map_index']].append(log)
    if not logs:return dict(ratio=None,improvement_pct=None,n=0,directions=directions,zero_pairs=zero,ci95=None)
    values=np.array([statistics.mean(v) for _,v in sorted(maps.items())]);rng=np.random.default_rng(20261028)
    boot=np.exp(values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1))
    ratio=math.exp(statistics.mean(logs))
    return dict(ratio=ratio,improvement_pct=100*(1-ratio),n=len(logs),directions=directions,zero_pairs=zero,ci95=np.quantile(boot,[.025,.975]).tolist())

memory={r['method']:r for r in read(SOURCE/'memory_summary.json')}
results={};rowscsv=[]
for m in PRIMARY+['K1_exact','K4_exact','S1_gate','LLMA']:
    rows=by[m];ratios=quality_rows[m];a=summary['absolute'][m]
    metrics={k:paired(rows,base,k) for k in ['local_seconds','reconstructed_total_seconds','operation','successor_expansions','length','storage']}
    # For strictly positive metrics, agree with original source aggregation.
    for k,v in metrics.items():
        orig=summary['comparisons']['LLMA'][m].get(k)
        if orig is not None and v['zero_pairs']==0:assert math.isclose(v['ratio'],orig['geometric_ratio'],rel_tol=1e-10)
    cost={k:sum(r[k] for r in rows.values()) for k in ['llm_calls','prompt_tokens','completion_tokens','api_seconds']}
    cost['total_tokens']=cost['prompt_tokens']+cost['completion_tokens']
    q=dict(valid=sum(r['valid'] for r in rows.values()),n=len(rows),individual_seed_paths=len(ratios),
       exact_count=sum(abs(x-1)<1e-8 for x in ratios),exact_pct=100*statistics.mean(abs(x-1)<1e-8 for x in ratios),
       geometric_ratio=math.exp(statistics.mean(math.log(x) for x in ratios)),p95_excess_pct=100*(float(np.quantile(ratios,.95))-1),
       max_excess_pct=100*(max(ratios)-1),formal_bound=a['formal_bound'])
    d=dict(method=m,name=NAMES.get(m,m),metrics=metrics,quality=q,online_cost=cost,memory=memory[m],
       arithmetic_mean_total_seconds=statistics.mean(r['reconstructed_total_seconds'] for r in rows.values()),
       arithmetic_mean_local_ms=1000*statistics.mean(r['local_seconds'] for r in rows.values()),
       mean_total_improvement_pct=100*(1-statistics.mean(r['reconstructed_total_seconds'] for r in rows.values())/summary['absolute']['LLMA']['means']['reconstructed_total_seconds']))
    results[m]=d
    if m in PRIMARY:
        item=dict(method=m,name=d['name'],n=q['n'],valid=q['valid'],exact_pct=q['exact_pct'],formal_bound=q['formal_bound'],
            stretch_gm=q['geometric_ratio'],p95_excess_pct=q['p95_excess_pct'],max_excess_pct=q['max_excess_pct'],
            mean_local_ms=d['arithmetic_mean_local_ms'],mean_total_seconds=d['arithmetic_mean_total_seconds'],mean_total_improvement_pct=d['mean_total_improvement_pct'],
            calls=cost['llm_calls'],prompt_tokens=cost['prompt_tokens'],completion_tokens=cost['completion_tokens'],sampled_rss_peak_mib=memory[m]['sampled_peak_rss_bytes']/1048576)
        for k,v in metrics.items():
            item[k+'_gm_ratio']=v['ratio'];item[k+'_improvement_pct']=v['improvement_pct'];item[k+'_improved']=v['directions']['improved'];item[k+'_equal']=v['directions']['equal'];item[k+'_worsened']=v['directions']['worsened'];item[k+'_missing']=v['directions']['missing']
        rowscsv.append(item)
save(OUT/'comparison.json',dict(source_commit=REF,sm_publication_commit=SMREF,methods=results,source_audit=audit))
with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rowscsv[0]));w.writeheader();w.writerows(rowscsv)

def pct(m,k):
    x=results[m]['metrics'][k]['improvement_pct'];return '미측정' if x is None else f'{x:+.3f}%'
def bound(m):
    b=results[m]['quality']['formal_bound'];return '최단' if b==1 else '≤1.05C*' if b==1.05 else '상한 없음'
def direction(m,k):
    d=results[m]['metrics'][k]['directions'];return f"{d['improved']}/{d['equal']}/{d['worsened']}"
def gm_ratio(m,k):return results[m]['metrics'][k]['ratio']
exact_primary=[m for m in PRIMARY if results[m]['quality']['formal_bound']==1]
llm_primary=[m for m in PRIMARY if results[m]['online_cost']['llm_calls']>0]
winners=dict(total_gm=min(PRIMARY,key=lambda m:gm_ratio(m,'reconstructed_total_seconds')),
    local_gm=min(PRIMARY,key=lambda m:gm_ratio(m,'local_seconds')),
    exact_total_gm=min(exact_primary,key=lambda m:gm_ratio(m,'reconstructed_total_seconds')),
    llm_total_gm=min(llm_primary,key=lambda m:gm_ratio(m,'reconstructed_total_seconds')),
    llm_mean_total=min(llm_primary,key=lambda m:results[m]['arithmetic_mean_total_seconds']),
    measured_heap_pop=min([m for m in PRIMARY if gm_ratio(m,'operation') is not None],key=lambda m:gm_ratio(m,'operation')),
    query_expansions=min(PRIMARY,key=lambda m:gm_ratio(m,'successor_expansions')),
    sampled_rss_min=min(PRIMARY,key=lambda m:memory[m]['sampled_peak_rss_bytes']))
save(OUT/'verification.json',dict(passed=True,analysis_script_sha256=sha(Path(__file__)),artifact_hashes_checked=len(artifact_hashes),raw_trials=raw_count,
    per_case_aggregate_checks=checks,primary_methods=len(PRIMARY),cases=200,seed_quality_unit='individual paths, time repetitions deduplicated',
    calls_zero_policy='No floor-to-one geometric ratios for calls/tokens; report exact totals and arithmetic means.',winners=winners))

lines=['# 18개 개선안 — 원 LLM-A* 기준 재검토','',
    '2026-10-28 · 원격 결과 읽기·재집계. 새 API 호출이나 탐색 재실험은 하지 않았다.','',
    f'**SM 공식 결과 교체 완료:** [{SMREF[:7]}](https://github.com/DUCKRACCOOON/LLM-A-assignment/commit/{SMREF}). 이전 SM 결과는 현재 평가 근거로 폐기하고 GM 기준·Python 3.11.18 결과를 공식 자료로 지정했다. 원자료와 과거 바이트는 보존했다.','',
    f'**18개 비교 근거:** JY 최신 커밋 [{REF[:7]}]({URL}/results/run_20261028/REPORT.md)의 COMMON18. 같은 PC·공개 평가 200문제·GPT-4o mini·3회 로컬 시간 반복으로 18개 주 방법과 대조군 포함 45조건, 30,600회를 측정한 별도 실험이다.','',
    '**두 실행을 합치지 않는다.** COMMON18은 Python 3.12.14이며 SM 공식 Python 3.11.18 실행과 다르다. COMMON18의 S4는 과거 seed11 가중치 한 개, S6는 `[0.5,0.3,0.1,0.1]`을 사용한다. 공식 SM의 3개 재학습 시드·S6 `[1.5,1.0,0.5,-1.0]`와 다르며 S1~S3 어댑터도 완전히 같지 않다. 아래 순위는 COMMON18의 고정된 18개 **변형**에 대한 결과이지, SM 공식 설정을 그대로 다른 12개와 비교한 결과가 아니다.','',
    '## 판정 방법','',
    '- 동일 문제의 원 LLM-A*(`LLMA`)를 모든 상대 지표의 분모로 사용했다.',
    '- 주 상대 지표는 `100 × (1 − 문제별 방법/LLMA 비율의 기하평균)`이다. 양수는 개선, 음수는 악화다. 문제별 개선/동일/악화도 집계했다.',
    '- 전체 시간은 수집된 API 대기＋같은 문제의 로컬 준비·탐색·복원·필수 검증 시간의 합인 **재구성 추정치**다. 서비스 전체 시간을 독립적으로 3회 반복 측정한 결과가 아니다. API 지연의 반복 변동은 bootstrap에 포함되지 않는다.',
    '- 호출 0회를 1회로 치환하는 기하평균은 비용 비교에 사용하지 않았다. 호출·토큰은 실제 논리 합계와 문제당 산술평균으로 비교했다. 원격 summary의 0→1 보정 호출 비율은 인용하지 않는다.',
    '- 정확 방법, 5% 제한 방법, 품질 상한 없는 방법을 나눠 해석한다. K5가 관측상 전부 최단인 것과 보장 배율 1.05는 별개다.',
    '- 파라미터는 w·trigger·호출 한도·랜드마크 수·안내 가중치 등 실행 전에 정한 설정이다. 아래 시간·pop·품질·성공률은 평가 지표다. 이 재검토에서 파라미터를 변경하지 않았다.','',
    '## 기준별 선택','',
    '| 기준 | 선택 | 근거와 범위 |','| --- | --- | --- |',
    f"| 재구성 전체 시간·로컬 시간 | **K3** | 전체 GM 개선 {pct('K3','reconstructed_total_seconds')}, 로컬 {pct('K3','local_seconds')}. 전체 시간은 200/0/0, 로컬은 185/0/15. 최단/5% 보장은 없음. |",
    f"| 최단성을 요구한 전체 시간 | **S6** | 정확 주 방법 J3/J4/J5/S5/S6 중 전체 GM 최소. 평균 {results['S6']['arithmetic_mean_total_seconds']*1000:.3f}ms. COMMON18의 고정 점수 변형에 한정. |",
    f"| 측정된 heap pop | **K3** | 원 LLMA 대비 {pct('K3','operation')}, 182/15/3. K2/J1/J2/S4는 공통 heap 수가 없어 18개 완전 순위는 불가. |",
    f"| 검색 중 실제 이웃 확장 | **K3** | {pct('K3','successor_expansions')}, 182/15/3. ALT/FMM/통로 생성의 전처리 작업과는 별도. |",
    '| 최단성·경로 품질 | **J3·J4·J5·S5·S6** | 정확 주 방법 모두 200/200 최단, P95/최대 초과율 0%. K5도 관측상 100% 최단이지만 일반 보장은 1.05배. |',
    '| 유효 경로 성공률 | **18개 모두 동률** | 각 200/200, 실패 0. 새로운 지도 전체에 대한 보장은 아님. |',
    '| 메모리 | **확정 불가** | RSS 표본의 수치상 최소 K4 55.00MiB, K3 55.11MiB. 10문제·각 1프로세스·1ms 표본으로 근소한 차이의 우열을 확정하지 않음. OPEN peak는 모든 방법/원 LLMA에 있지 않음. |',
    '| 온라인 LLM 비용 | **K3·J4·J5·S4·S5·S6** | 호출·입출력 토큰 모두 0. S4 학습 및 S6 과거 점수 제안 같은 오프라인 비용까지 0이라는 뜻은 아님. |',
    '| 실제 LLM 사용 방법의 호출 수 | **S3** | 94/200회, 원 LLMA의 200회 대비 53% 감소. |',
    '| 실제 LLM 사용 방법의 토큰 수 | **K1** | 문제당 입력 250.22·출력 9.86토큰, 합계 260.08토큰. 호출은 180/200회. |',
    '| LLM 안내의 추가 효용 | **확인된 종합 승자 없음** | K1은 규칙 대조와 순위·경로·확장이 모두 200/200 동일. 안내 포함 성능과 LLM의 기여를 구분해야 함. |','',
    '## 전체 18개: 원 LLMA 대비 개선율','',
    '아래 값은 모두 같은 COMMON18 실행에서 계산했다. 시간은 재구성 총시간이며 원격 보고서의 산술평균 중심 선택과 달리 사용자가 지정한 기하평균을 주 지표로 삼았다. heap 미측정은 0이 아니다.','',
    '| ID | 방법 | 총시간 GM 개선 | 로컬 시간 개선 | 기록된 heap pop 개선 | 검색 이웃 확장 개선 | 총시간 개선/동일/악화 | 품질 보장 |',
    '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
for m in PRIMARY:lines.append(f"| {m} | {NAMES[m]} | {pct(m,'reconstructed_total_seconds')} | {pct(m,'local_seconds')} | {pct(m,'operation')} | {pct(m,'successor_expansions')} | {direction(m,'reconstructed_total_seconds')} | {bound(m)} |")
lines += ['','K1의 기록 pop은 탐색기 카운터이고 통로 후보 생성의 모든 heap 작업을 통일 집계한 값은 아니다. J4 pop은 ALT 준비를 포함하지만 확장 열은 질의 검색만이다. K2/J1/J2/S4는 배열 기반 FOCAL 선택 및 일부 준비 작업과 heap pop의 대응이 없어 미측정으로 남겼다. 압축 정점과 격자 정점의 확장은 동일 비용의 연산도 아니다. 이 표를 총 CPU 연산량 순위로 해석하지 않는다.','',
    '## 경로 품질과 LLM 비용','',
    '품질 꼬리는 K5의 실제 개별 시드 경로를 사용하며 시간 반복은 중복 계산하지 않는다. 길이/원 LLMA와 개선·동일·악화 건수는 comparison.json/CSV에 함께 있다.','',
    '| ID | 성공 | 최단 일치율 | 길이/최단 GM | P95 초과 | 최대 초과 | 호출/문제 | 입력/출력 토큰/문제 | RSS 표본 MiB |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
for m in PRIMARY:
    d=results[m];q=d['quality'];c=d['online_cost']
    lines.append(f"| {m} | {q['valid']}/{q['n']} | {q['exact_pct']:.1f}% | {q['geometric_ratio']:.5f} | {max(0,q['p95_excess_pct']):.3f}% | {max(0,q['max_excess_pct']):.3f}% | {c['llm_calls']/200:.3f} | {c['prompt_tokens']/200:.2f}/{c['completion_tokens']/200:.2f} | {d['memory']['sampled_peak_rss_bytes']/1048576:.2f} |")
lines += ['','## LLM을 사용하는 방법만 고르면','',
    f"- **기하평균 총시간 개선: S3.** {pct('S3','reconstructed_total_seconds')}. 다만 평균 응답시간은 {results['S3']['arithmetic_mean_total_seconds']:.4f}초로 산술평균 개선은 {results['S3']['mean_total_improvement_pct']:.2f}%다. API를 호출하지 않은 106문제의 매우 작은 비율이 GM에 크게 반영된다.",
    f"- **산술평균 총시간: K1.** {results['K1']['arithmetic_mean_total_seconds']:.4f}초, 평균 {results['K1']['mean_total_improvement_pct']:.2f}% 감소. GM 개선 {pct('K1','reconstructed_total_seconds')}와 구분한다. K1은 5% 보장을 가지지만 규칙 선택과 같은 결과라 LLM의 판단 능력이 낫다는 증거는 없다.",
    f"- **LLM 사용＋최단성 보장 변형: K1_exact.** 평균 {results['K1_exact']['arithmetic_mean_total_seconds']:.4f}초. 별도 w=1 변형이며 18개 주 설정의 K1과 바꿔 쓰지 않는다. 규칙 exact 대조도 동일 경로·확장을 얻는다.",
    '- 매 문제 LLM을 호출하는 조건은 공통 standard 응답 대기가 지배한다. S1_gate/K4_exact의 수치상 작은 시간 차이를 실용적인 서비스 속도 승리로 주장하지 않는다.','',
    '## 같은 구조에서 안내 기여 확인','',
    '| 개입 / 대조 | 로컬 시간 GM 비율 | 총시간 GM 비율 | 검색 확장 GM 비율 |','| --- | ---: | ---: | ---: |']
for m,b in [('K1','K1_rule'),('K1_exact','K1_rule_exact'),('K2','FOCAL0'),('K4','K4_noLLM'),('K5','K5_noLLM'),('K6','K6_noLLM'),('J1','J1_straight'),('J2','J2_uniform'),('J3','J3_noLLM'),('J6','J6_full'),('S1','S1_gate'),('S4','S4_untrained')]:
    vals=[paired(by[m],by[b],k)['ratio'] for k in ['local_seconds','reconstructed_total_seconds','successor_expansions']]
    lines.append(f'| {m}/{b} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
lines += ['','J6/full은 LLM 제거가 아니라 같은 LLM을 쓴 큐 갱신 비교다. S4/untrained는 학습 효과다. S1/gate는 정제의 추가 효과다. S2는 COMMON18에서 동일 호출수의 일반 재검토 대조가 없어 구체 피드백 문구 자체의 효과를 분리하지 못한다. S6는 이번 COMMON18에서 규칙·무작위 점수 대조가 없어 LLM 제안 점수의 효용을 새롭게 입증하지 않는다.','',
    '## 재현·검증','',
    f'- 최신 원격: gm `c56eb83b7df4be06d402daba36e0a2b45afa950b`, jy `{REF}`, sm `{SMREF}`.',
    f'- 원격 파일 {len(artifact_hashes)}개 SHA-256 확인, 원시 시행 {raw_count:,}개에서 저장 집계 {checks:,}개 항목 재검산. [이 재검토 검증](verification.json).',
    f'- [원격 공통 규약]({URL}/PROTOCOL.md), [원격 구현 조정]({URL}/IMPLEMENTATION_NOTES.md), [원격 독립 감사]({URL}/results/run_20261028/audit.json).',
    f'- [SM 공식 Python 3.11 결과]({SMURL}/results/s1_s6_gm_20261028/report.md).',
    '- [18개 CSV](comparison.csv), [전체 상대 비율·구간·문제 수 JSON](comparison.json).','']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(verification=read(OUT/'verification.json'),report=str(OUT/'report.md')),ensure_ascii=False))
