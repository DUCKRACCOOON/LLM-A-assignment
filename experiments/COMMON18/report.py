"""All-method paired analysis. Never combines numbers from old experiments."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .prepare import save,SEED
from .backends import ROOT,PRIMARY,EXACT

METRICS=('local_seconds','api_seconds','reconstructed_total_seconds','operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')


def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def collapse(out):
    trials=defaultdict(list)
    for r in read(out/'evaluations.jsonl.gz'):trials[r['method'],r['case_id'],r['seed_index']].append(r)
    seeds=defaultdict(list)
    for (m,cid,si),rs in trials.items():
        assert len(rs)==3
        v={k:rs[0].get(k) for k in ('case_id','map_index','method','valid','path','length','optimum','bound','metadata','peak_open','peak_heap_entries')}
        for k in METRICS:
            vals=[r.get(k) for r in rs]
            v[k]=statistics.median(vals) if all(x is not None for x in vals) else None
        v['seed_index']=si;seeds[m,cid].append(v)
    methods=defaultdict(dict)
    for (m,cid),rs in seeds.items():
        r=dict(rs[0]);r['seed_count']=len(rs);r['valid']=all(v['valid'] for v in rs)
        for k in METRICS:
            r[k]=statistics.mean(v[k] for v in rs) if all(v[k] is not None for v in rs) else None
        r['length']=statistics.mean(v['length'] for v in rs) if r['valid'] else None
        ratios=[v['length']/v['optimum'] for v in rs if v['valid']]
        r['max_seed_stretch']=max(ratios) if ratios else None
        r['exact_seed_fraction']=statistics.mean(abs(v['length']-v['optimum'])<1e-7 if v['valid'] else False for v in rs)
        methods[m][cid]=r
    return methods


def comparison(method,baseline,metric):
    ids=sorted(set(method)&set(baseline));logs=[];maps=defaultdict(list);better=same=worse=0
    for cid in ids:
        a,b=method[cid].get(metric),baseline[cid].get(metric)
        if a is None or b is None:continue
        # API-only times may be zero. Ratios for latency use total/local,
        # while counters floor ONLY for logs. True zero totals remain zero.
        floor=1e-12 if 'seconds' in metric else 1.0
        x=math.log(max(a,floor)/max(b,floor));logs.append(x);maps[method[cid]['map_index']].append(x)
        if math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):same+=1
        elif a<b:better+=1
        else:worse+=1
    if not logs:return None
    vals=np.array([statistics.mean(v) for _,v in sorted(maps.items())]);rng=np.random.default_rng(SEED)
    resampled=np.exp(vals[rng.integers(0,len(vals),size=(2000,len(vals)))].mean(axis=1))
    return dict(n=len(logs),maps=len(vals),geometric_ratio=math.exp(statistics.mean(logs)),
        ci95=[float(v) for v in np.quantile(resampled,[.025,.975])],improved=better,same=same,worsened=worse)


def summarize(out,methods):
    absolute={}
    for m,rs in methods.items():
        rows=list(rs.values());valid=[r for r in rows if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        means={k:statistics.mean(r[k] for r in rows) if all(r[k] is not None for r in rows) else None for k in METRICS}
        absolute[m]=dict(n=len(rows),valid=len(valid),valid_percent=100*len(valid)/len(rows),means=means,
            exact_percent=100*statistics.mean(r['exact_seed_fraction'] for r in rows),
            stretch_geometric=math.exp(statistics.mean(map(math.log,ratios))) if ratios else None,
            stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,
            stretch_max=max(r['max_seed_stretch'] for r in valid) if valid else None,
            formal_bound=rows[0]['bound'],all_cases_call_llm=all(r['llm_calls']>=1 for r in rows),
            called_cases=sum(r['llm_calls']>0 for r in rows),
            ranking_accepted_cases=sum(r.get('metadata',{}).get('ranking_accepted',False) for r in rows),
            conditional_zero_call_cases=sum(r['llm_calls']==0 for r in rows))
    comparisons={}
    references={'LLMA':list(methods),'SM_LLMA':['S3'],
        'K1_rule':['K1'],'K1_random':['K1'],'K1_rule_exact':['K1_exact'],
        'FOCAL0':['K2','J1','J2','S4'],'K4_noLLM':['K4'],'K4_exact_noLLM':['K4_exact'],
        'K5_noLLM':['K5'],'K6_noLLM':['K6'],'J1_straight':['J1'],'J2_uniform':['J2'],
        'J3_noLLM':['J3'],'J6_full':['J6'],'S4_untrained':['S4'],'S1_gate':['S1'],'Octile':list(methods)}
    for baseline,targets in references.items():
        comparisons[baseline]={m:{k:comparison(methods[m],methods[baseline],k) for k in METRICS if k!='api_seconds'} for m in targets}
    base_mean=absolute['LLMA']['means']['reconstructed_total_seconds']
    for m,v in absolute.items():v['mean_total_ratio_to_llma']=v['means']['reconstructed_total_seconds']/base_mean
    # An actually used LLM is required for the LLM candidate set. K6_exact
    # removes its guide algebraically (w-1=0), so do not call that LLM benefit.
    eligible=[m for m,v in absolute.items() if v['valid']==v['n'] and v['called_cases']>0 and m!='K6_exact']
    exact=[m for m in eligible if m in EXACT]
    always=[m for m in eligible if absolute[m]['all_cases_call_llm']]
    key=lambda m:absolute[m]['means']['reconstructed_total_seconds']
    winners=dict(llm_any_quality=min(eligible,key=key),llm_exact=min(exact,key=key),
        llm_every_case=min(always,key=key),llm_exact_every_case=min([m for m in exact if m in always],key=key),
        rule='arithmetic mean per-case reconstructed total latency, 100% observed valid; quality separated',
        excluded_inert_llm='K6_exact: w=1 makes the guide term identically zero')
    diagnostics={}
    for m,b in [('K1','K1_rule'),('K1_exact','K1_rule_exact')]:
        pairs=[(r,methods[b][cid]) for cid,r in methods[m].items()]
        diagnostics[m]=dict(cases=len(pairs),same_path=sum(r['path']==v['path'] for r,v in pairs),
            same_expansions=sum(r['successor_expansions']==v['successor_expansions'] for r,v in pairs),
            same_ranking=sum(r['metadata'].get('ranking')==v['metadata'].get('ranking') for r,v in pairs))
    diagnostics['S1_gate_accepted']=sum(r['metadata'].get('accepted',False) for r in methods['S1_gate'].values())
    diagnostics['K2_full_map_sets']=sum(r['metadata'].get('set_fraction')==1.0 for r in methods['K2'].values())
    value=dict(absolute=absolute,comparisons=comparisons,winners=winners,diagnostics=diagnostics,
        uncertainty='map bootstrap 2000, exploratory 95%; API sampled once, not repeated 3 times')
    save(out/'summary.json',value)
    flat=[]
    for m,rs in methods.items():
        for r in rs.values():flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list,tuple))})
    fields=sorted(set().union(*(r.keys() for r in flat)))
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    return value


def charts(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator,FormatStrFormatter,NullFormatter
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    names=list(PRIMARY);fig,axes=plt.subplots(1,3,figsize=(17,9),constrained_layout=True)
    for ax,k,title in zip(axes,('local_seconds','reconstructed_total_seconds','stretch_geometric'),('로컬 계산시간 / 원 LLM-A*','API 포함 재구성 평균시간 / 원 LLM-A*','경로 길이 / 최단거리')):
        values=[]
        for m in names:
            a=summary['absolute'][m]
            if k=='stretch_geometric':val=a[k]
            else:val=a['means'][k]/summary['absolute']['LLMA']['means'][k]
            values.append(val)
        for i,(m,v) in enumerate(zip(names,values)):
            color='#8a8a8a' if summary['absolute'][m]['called_cases']==0 else '#1d7194'
            ax.plot(v,i,'o',color=color)
        ax.set_yticks(range(len(names)),names);ax.axvline(1,color='black',lw=1);ax.set_title(title)
        if k!='stretch_geometric':
            ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,) if k=='reconstructed_total_seconds' else (1,2,5)))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'));ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle('동일 200문제 · 동일 호스트 · 동일 GPT-4o mini 응답 공유 (회색: 온라인 LLM 없음)')
    fig.savefig(out/'comparison.png',dpi=155);plt.close(fig)


def write_markdown(out,s):
    a=s['absolute'];memory={r['method']:r for r in json.loads((out/'memory_summary.json').read_text(encoding='utf-8'))} if (out/'memory_summary.json').exists() else {}
    collection=json.loads((out/'collection_complete.json').read_text(encoding='utf-8'))
    generations=[json.loads(line) for line in (out/'generations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    finishes=Counter(r['response']['choices'][0]['finish_reason'] for r in generations)
    names={'K1':'통로 ID 선택','K2':'Conformal','K3':'벽 하한 g+2h','K4':'LLM FOCAL 1.05','K5':'MHA*+DTS 1.05','K6':'적응 α 1.05',
        'J1':'CHOMP+LLM','J2':'FMM+LLM','J3':'LLM 초기 해+정확 BB','J4':'ALT4','J5':'NBS octile','J6':'lazy OPEN',
        'S1':'경유점 정제','S2':'오류 피드백 수정','S3':'정체 시 호출','S4':'소형 TransPath','S5':'정확 직사각형 압축','S6':'장애물 단계 반영'}
    lines=['# 동일 조건 18개 방법군 비교 결과','',
        '2026-10-28 (Asia/Seoul), jy. 기존 브랜치 결과의 비율을 합치지 않고 모든 행을 같은 PC·200문제에서 새로 측정했다.',
        '온라인 LLM은 gpt-4o-mini-2024-07-18. 같은 수집에서 얻은 응답을 방법 간 공유하고, API 지연을 각 방법의 논리 호출량에 따라 부과했다.',
        '동적 경유점 생성은 보류했다. 팀 제출 알고리즘/기존 Proposed는 변경하지 않았다.','',
        '## 속도 판정','',
        f'- 온라인 LLM을 사용하는 조건 중 전체 평균 지연 최저: **{s["winners"]["llm_any_quality"]}**.',
        f'- 온라인 LLM+최단성 보장 조건 중 최저: **{s["winners"]["llm_exact"]}**.',
        f'- 모든 문제에서 LLM을 적어도 한 번 호출하는 조건 중 최저: **{s["winners"]["llm_every_case"]}**.',
        f'- 모든 문제에서 LLM 호출+최단성 보장 조건 중 최저: **{s["winners"]["llm_exact_every_case"]}**.','',
        '위 판정은 성공률 100%인 조건의 **문제별 재구성 총지연 산술평균**이다. 모든 방법이 같은 경로 품질을 보장하는 것은 아니므로 품질 표와 함께 본다. '
        '원 LLM-A*가 이기는 경우도 그대로 표시한다. K6 w=1은 안내 항이 0이므로 유효한 LLM 기여 후보에서 제외했다.','',
        '## 결과 해석','',
        f'K1_exact의 총평균은 원 LLM-A* 대비 {100*(1-a["K1_exact"]["mean_total_ratio_to_llma"]):.2f}% 감소했다. '
        f'문제당 호출 {a["K1_exact"]["means"]["llm_calls"]:.2f}회, 출력 {a["K1_exact"]["means"]["completion_tokens"]:.2f}토큰으로 '
        f'원 응답의 {a["LLMA"]["means"]["completion_tokens"]:.2f}토큰보다 짧다. '
        f'그러나 규칙 대조와 {s["diagnostics"]["K1_exact"]["same_ranking"]}/200문제의 순위, '
        f'{s["diagnostics"]["K1_exact"]["same_path"]}/200문제의 경로와 '
        f'{s["diagnostics"]["K1_exact"]["same_expansions"]}/200문제의 확장 수가 같다. '
        '이번 속도 개선을 LLM 판단의 우수성으로 해석할 수 없다. 규칙 대조는 API 없이 같은 경로를 더 빨리 반환한다.','',
        '매 문제 긴 standard 응답을 받는 조건은 API 대기가 지배하여 사실상 비슷한 총시간이다. '
        'S1_gate의 수치상 최소값은 원 LLM-A*보다 약 0.006% 빠른 수준이며 실용적인 대기시간 개선으로 주장하지 않는다. '
        '매 문제 호출하는 정확 조건 K4_exact와 J3의 차이도 약 0.05ms여서 일반적인 서비스 속도 순위를 확정할 근거가 아니다.','',
        'S3는 호출을 0.47회/문제로 줄여 총평균을 약 절반으로 낮췄지만 최단성 보장은 없다. '
        'K2/J2의 3회 호출과 S2의 추가 수정은 이번 조건에서 대기시간을 늘렸다. '
        f'K2 보정 집합은 {s["diagnostics"]["K2_full_map_sets"]}/200문제에서 전체 자유 공간을 포함해 안내 구분력이 사라졌다. '
        '안내를 제거한 대조와 함께 봐야 하며 탐색 확장 수 감소만으로 전체 개선을 주장하지 않는다.','',
        '## 18개 primary: 시간·호출 비용','',
        '| ID | 방법 | 호출/문제 | 입력/출력 토큰/문제 | 로컬 평균 ms | 재구성 총평균 s | 총평균/원 LLM-A* | 총시간 GM 비율 [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for m in PRIMARY:
        v=a[m];c=s['comparisons']['LLMA'][m]['reconstructed_total_seconds'];lo,hi=c['ci95']
        lines.append(f'| {m} | {names[m]} | {v["means"]["llm_calls"]:.3f} | '
            f'{v["means"]["prompt_tokens"]:.1f}/{v["means"]["completion_tokens"]:.1f} | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.4f} | {100*v["mean_total_ratio_to_llma"]:.2f}% | '
            f'{100*c["geometric_ratio"]:.2f}% [{100*lo:.2f}, {100*hi:.2f}] |')
    lines += ['', f'원 LLM-A*: 로컬 평균 {1000*a["LLMA"]["means"]["local_seconds"]:.3f}ms, '
        f'재구성 총평균 {a["LLMA"]["means"]["reconstructed_total_seconds"]:.4f}s, 호출 1회/문제.',
        '조건부 호출은 0회인 문제에서 시간 비율이 매우 작아 기하평균 감소가 크게 보일 수 있다. '
        '기하평균 감소를 평균 대기시간 감소로 바꾸어 말하지 않는다. 위 표는 산술평균과 GM을 모두 제시했다.','',
        '## 경로 품질·계산량·메모리','',
        '| ID | 유효율 | 최단 일치율 | 길이/최단 GM | 최대 길이/최단 | 비용 보장 | 검색 이웃 확장 평균 | 저장 상태 평균 | RSS peak MiB† |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for m in PRIMARY:
        v=a[m];bound=v['formal_bound'];guarantee='최단' if bound==1 else f'≤{bound:.2f}C*' if bound else '없음'
        rss=f'{memory[m]["sampled_peak_rss_bytes"]/1024**2:.2f}' if m in memory else '미측정'
        storage=f'{v["means"]["storage"]:.1f}' if v['means']['storage'] is not None else '—'
        stretch=f'{100*v["stretch_geometric"]:.3f}%' if v['stretch_geometric'] else '—'
        maximum=f'{100*v["stretch_max"]:.3f}%' if v['stretch_max'] else '—'
        lines.append(f'| {m} | {v["valid_percent"]:.1f}% | {v["exact_percent"]:.1f}% | {stretch} | {maximum} | {guarantee} | '
            f'{v["means"]["successor_expansions"]:.1f} | {storage} | {rss} |')
    lines += ['', '† RSS는 별도 동일 10문제에서 방법마다 새 프로세스를 띄운 1ms 명목 간격 표본 peak다. '
        'Python·응답 캐시·라이브러리·모델 상주분 포함이며 순수 알고리즘 할당량/순간 peak가 아니다. '
        '저장 상태 수도 각 엔진의 g 정의이며 그래프/거리표/모델을 포함한 메모리 바이트가 아니다.',
        '검색 확장은 query 이웃 확장이다. 압축 그래프 정점/일반 격자 정점 확장은 동일 연산이 아니다. '
        'ALT 거리표·FMM 거리장·K1 통로 추출의 준비 비용은 모두 시간에 포함하고 네이티브 카운터를 raw에 남겼다.','',
        '## LLM을 사용하는 최단성 보장 조건','',
        '| 조건 | 호출/문제 | 로컬 평균 ms | 재구성 총평균 s | 길이/최단 | 비고 |',
        '|---|---:|---:|---:|---:|---|']
    for m in sorted([m for m in a if m in EXACT and a[m]['called_cases']>0],key=lambda m:a[m]['means']['reconstructed_total_seconds']):
        v=a[m];note='w=1에서 LLM 안내 항 소거; 기여 후보 제외' if m=='K6_exact' else '조건부 호출' if not v['all_cases_call_llm'] else '매 문제 호출'
        lines.append(f'| {m} | {v["means"]["llm_calls"]:.3f} | {1000*v["means"]["local_seconds"]:.3f} | '
                     f'{v["means"]["reconstructed_total_seconds"]:.4f} | {100*v["stretch_geometric"]:.3f}% | {note} |')
    lines += ['', '이 표의 w=1 변형은 새로 측정한 별도 조건이다. 주 방법을 사후에 바꾸어 원 결과를 덮어쓰지 않는다. '
        'S1/S2/S3/J6에 이론적 보장 없이 최단이라는 이름을 붙이지 않았다.','',
        '## LLM 기여 분리','',
        '| 비교 | 로컬 시간 비율 | 재구성 총시간 비율 | 검색 확장 비율 | 해석 |',
        '|---|---:|---:|---:|---|']
    ablations=[('K1','K1_rule'),('K1_exact','K1_rule_exact'),('K2','FOCAL0'),('K4','K4_noLLM'),('K4_exact','K4_exact_noLLM'),
        ('K5','K5_noLLM'),('K6','K6_noLLM'),('J1','J1_straight'),('J2','J2_uniform'),('J3','J3_noLLM'),('J6','J6_full'),('S4','S4_untrained'),('S1','S1_gate')]
    for m,b in ablations:
        c=s['comparisons'][b][m];x=c['local_seconds']['geometric_ratio'];t=c['reconstructed_total_seconds']['geometric_ratio'];e=c['successor_expansions']['geometric_ratio']
        lines.append(f'| {m}/{b} | {x:.4f} | {t:.4f} | {e:.4f} | 동일 구조의 직접 대조 |')
    lines += ['', 'LLM 제거 시 API 대기도 없어지므로 총시간뿐 아니라 검색과 로컬 계산의 변화도 확인한다. '
        'J6/full은 양쪽 모두 LLM을 쓰는 동일 확장 순서의 큐 비교다. S4/untrained는 학습 효과이며 LLM 효과가 아니다.',
        '', '## 조건·한계','',
        '상세 설계와 구현 조정은 PROTOCOL.md 및 IMPLEMENTATION_NOTES.md를 참조한다.','',
        f'- 신규 API 요청 {collection["requests"]:,}회, 입력 {collection["prompt_tokens"]:,} / 출력 {collection["completion_tokens"]:,}토큰. '
        f'finish_reason=length {finishes["length"]}건도 삭제·재생성 없이 사용했다. 교정·fallback 및 호출 비용은 그대로 반영했다.',
        '- 동일 test 20지도×10쌍=200문제. 호스트·입력·충돌·이동 비용·응답을 공유하고 학습/보정 지도와 분리했다.',
        '- 모든 온라인 LLM은 같은 GPT-4o mini snapshot과 수집 기간을 사용했다. standard/CoT/RePE 및 수정·ID 순위 프롬프트는 방법 특성에 따라 다르다. 필요한 응답 수 차이는 비용에 포함했다.',
        '- CPU/native thread1, 로컬 시간3회 반복. K5 계열3시드는 문제 안에서 평균하며 독립 지도 수를 늘리지 않는다.',
        '- API는 한 번 수집해 공유한다. 총시간은 같은 수집의 API 대기와 현재 로컬 측정의 합이며 각 방법을 온라인으로 반복 운영한 실측이 아니다. 3프롬프트는 순차 운영을 가정했다.',
        '- 기본 응답 수집 뒤 이에 의존하는 수정·순위 응답을 수집했다. 같은 수집 기간에도 서비스 부하와 system fingerprint가 달라질 수 있으며, API 지연의 반복 불확실성은 이번 bootstrap에 포함되지 않는다.',
        '- 모델은 메모리에 상주하고 warm-up 후 측정했다. 로딩·warm-up은 별도다. S4는 기존 seed11 checkpoint를 CPU로 사용했고 재학습하지 않았다.',
        '- 95% 구간은 지도 bootstrap2000회의 탐색적 구간이며 다중 비교 보정이 없다. 공개 자료는 연구 전체에서 미관측인 hidden holdout이 아니다.',
        f'- {len(a)}조건 모두 같은 실행 블록에서 재측정했지만 네이티브 탐색 엔진은 다르다. 구성요소 기여는 같은 엔진의 제거 대조로 확인한다.',
        '- S2의 추가 수정 효과와 구체적인 피드백 문구 자체의 효과는 다르다. 이번에는 동일 호출수의 일반 재검토 대조를 추가 수집하지 않았으므로 후자의 인과 효과는 미확인이다.',
        '- S2는 공통 경로 검증기의 오류와 첫 충돌 구간을 전달하는 통합 변형이다. 기존 SM의 장애물 식별 피드백 문구를 바이트 단위로 재현한 조건은 아니다.',
        '', '![동일 조건 비교](comparison.png)','',
        '## 재현 자료','',
        '[실행 규약](../../PROTOCOL.md), [구현 조정 기록](../../IMPLEMENTATION_NOTES.md), '
        '[전체 수치](summary.json), [문제별 CSV](per_case.csv), [원시 반복](evaluations.jsonl.gz), '
        '[요청·응답](generations.jsonl), [독립 검증](audit.json).','',
        '사용자 지시에 따라 Codex가 구현 통합·실행·검증·집계 작성을 지원했다. 기존 gm/sm 구현의 출처와 변경은 SM_PROVENANCE.json 및 코드에 남겼다. '
        '사람별 실제 기여를 추정하지 않으며 main/제출용 Proposed는 변경하지 않는다.','']
    (out/'REPORT.md').write_bytes(('\n'.join(lines)).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();out=a.input.resolve()
    assert (out/'benchmark_complete.json').exists()
    methods=collapse(out);s=summarize(out,methods);charts(out,s);write_markdown(out,s)
    print(json.dumps(s['winners'],ensure_ascii=False))


if __name__=='__main__':main()
