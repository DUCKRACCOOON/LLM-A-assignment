"""OOF artifact audit and generated Markdown; no hard-coded performance claims."""
import csv
import math
import statistics as st
from collections import defaultdict,Counter
import numpy as np
from common import *
from algorithms import BASE,MODEL_METHODS
from evaluate import bounds
from experiments.K3.verification import Oracle

METHODS=BASE+MODEL_METHODS
PRIMARY=['S1','S2_feedback','S3','S4_mean','S5','S6_program']
NAMES={'A_GM':'A* (GM)','B_llm':'원 LLM-A*','N_g2h':'g+2h','S2_feedback':'S2','S4_mean':'S4 (3시드 평균)','S6_program':'S6 (새 LLM 계수)'}

def records():
    for p in sorted(RUN.glob('folds/fold*/evaluations/*.json')):yield p,read(p)

def verify():
    frozen=read(RUN/'frozen.json');assert digest(frozen['frozen'])==frozen['signature']
    assert sha(RUN/'protocol.json')==frozen['frozen']['protocol_sha256']
    assert sha(RUN/'fold_manifest.json')==frozen['frozen']['fold_manifest_sha256']
    assert sha(GM/'experiments/K3/results/run_20261025/public_cases.json')==read(RUN/'protocol.json')['source_input_sha256']
    for rel,h in frozen['frozen']['code'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in frozen['frozen']['artifacts'].items():assert sha(RUN/rel)==h,rel
    assert sha(PROMPT_FILE)==frozen['frozen']['prompt_sha256']
    groups=defaultdict(list);seen=set();trials=invalid=violations=0
    source={q['id']:q for q in cases()}
    for p,d in records():
        q=d['case'];cid=q['id'];fold=q['fold_id'];assert cid not in seen;seen.add(cid)
        assert cid in MANIFEST['folds'][fold]['test_case_ids'] and q['split']=='test'
        for k in ('start','goal','range_x','range_y','horizontal_barriers','vertical_barriers'):assert q[k]==source[cid][k]
        assert d['signature']==frozen['signature']
        assert len(d['rows'])==110 and {(r['method'],r['repeat']) for r in d['rows']}=={(m,i) for m in METHODS for i in range(5)}
        repeat_refs={}
        for r in d['rows']:
            assert digest(d['paths'][r['path_sha256']])==r['path_sha256']
            assert r['total_seconds']>0 and r['operation']>=0 and r['successor_expansions']>=0
            fixed={k:r.get(k) for k in ('valid','error','length','operation','successor_expansions','storage','path_sha256','api_calls')}
            if r['method'] in repeat_refs:assert fixed==repeat_refs[r['method']]
            repeat_refs[r['method']]=fixed
            trials+=1;invalid+=int(not r['valid']);violations+=int(r['bound_violation'])
        groups[q['map_id']].append(d)
    assert seen==set(source) and len(seen)==1000
    for i,(mid,ds) in enumerate(groups.items()):
        oracle=Oracle(ds[0]['case'])
        for d in ds:
            q=d['case'];opt=oracle.distances(q['goal'])[tuple(q['start'])];assert math.isclose(d['optimal'],opt,abs_tol=1e-8)
            checked=set()
            for r in d['rows']:
                if r['valid'] and r['path_sha256'] not in checked:
                    oracle.check_path(q,dict(r,path=d['paths'][r['path_sha256']]));checked.add(r['path_sha256'])
                w=bounds(r['method']);bad=bool(r['valid'] and w and r['length']>w*opt+1e-8)
                assert bad==r['bound_violation']
        if (i+1)%20==0:print(json.dumps(dict(phase='independent_verify',maps=i+1)),flush=True)
    raw_count=0
    for q in source.values():
        first=raw(q);assert first['request_sha256']==digest(payload(q));raw_count+=1
        if not diagnosis(q,text_of(first))['valid']:
            for variant,feedback in [('control',False),('feedback',True)]:
                r=raw(q,variant);assert r['request_sha256']==digest(correction(q,text_of(first),feedback));raw_count+=1
    actual=list(RUN.glob('raw/public/*.json'))+list(RUN.glob('folds/fold*/raw/s6_programs/*.json'))
    assert len(actual)==raw_count+60
    if (RUN/'tail_collection_results.json').exists():
        entries=[]
        for line in (RUN/'collection.log').read_text(encoding='utf-8').splitlines():
            try:entry=json.loads(line)
            except ValueError:continue
            if entry.get('phase')=='collect' and entry.get('done')==1000:entries.append(entry)
        assert entries
        accounting=read(RUN/'tail_collection_accounting.json')
        counted=entries[-1]['new_calls']+accounting['responses_before_restart']+accounting['new_calls']
        if (RUN/'middle_collection_status.json').exists():counted+=read(RUN/'middle_collection_status.json')['new_calls']
        assert counted==raw_count,('Duplicate or unaccounted collection',counted,raw_count)
    for p in actual:
        r=read(p);assert digest(r['request'])==r['request_sha256'] and digest(r['response'])==r['response_sha256']
        assert r['request']['model']==MODEL and r['response']['model']==MODEL
    for fold,part in enumerate(MANIFEST['folds']):
        dest=RUN/'folds'/f'fold{fold}';meta=read(dest/'models/training_data.json')
        expected={q['id'] for q in source.values() if q['map_index'] in part['train_maps']+part['validation_maps']}
        assert set(meta['case_ids'])==expected and not expected&set(part['test_case_ids'])
        assert sha(dest/'models/training_data.npz')==meta['sha256']
        data=np.load(dest/'models/training_data.npz');expected_val=np.array([source[cid]['map_index'] in part['validation_maps'] for cid in meta['case_ids']])
        assert np.array_equal(data['validation'],expected_val)
        train_ids=np.flatnonzero(~data['validation']);positive=float((data['y'][train_ids]*data['mask'][train_ids]).sum())
        weight=min(30.,(float(data['mask'][train_ids].sum())-positive)/max(positive,1.))
        for seed in (11,22,33):
            sm=read(dest/'models'/f'seed{seed}.json');assert sha(dest/'models'/f'seed{seed}.pt')==sm['sha256']
            assert math.isclose(sm['positive_weight'],weight,rel_tol=1e-6)
            assert sm['best_epoch']==min(sm['history'],key=lambda x:x['validation_loss'])['epoch']
        selected=read(dest/'program_selection.json');assert selected['selection_uses_test'] is False
        proposal=read(dest/'programs'/f'proposal{selected["proposal_index"]:02d}.json');assert proposal['valid'] and proposal['weights']==selected['weights']
        selection_records=[]
        for p in (dest/'programs').glob('selection*.json'):
            d=read(p);selection_records.append(d)
            for split,rs in d['splits'].items():
                assert {r['case_id'] for r in rs['rows']}=={q['id'] for q in source.values() if q['map_index'] in part[split+'_maps']}
                calculated=math.exp(st.mean(math.log(r['operation']/r['reference_operation']) for r in rs['rows']))
                assert math.isclose(calculated,rs['heap_pop_ratio'],rel_tol=1e-12)
        shortlisted=sorted(selection_records,key=lambda d:(d['splits']['train']['heap_pop_ratio'],d['index']))[:3]
        chosen=min(shortlisted,key=lambda d:(d['splits']['validation']['heap_pop_ratio'],d['index']))
        assert chosen['index']==selected['proposal_index']
    result=dict(complete=True,cases=len(seen),maps=len(groups),folds=5,methods=22,timing_repeats=5,timed_trials=trials,
                invalid_trials=invalid,bound_violation_trials=violations,raw_responses=len(actual),source_hashes_verified=True,
                split_and_training_labels_verified=True,independent_paths_verified=True,updated_at=now())
    save(RUN/'verification.json',result);return result

def per_case():
    out=[]
    for _,d in records():
        q=d['case'];group=defaultdict(list)
        for r in d['rows']:group[r['method']].append(r)
        first=raw(q);cache={'first':first}
        if not d['initial_proposal']['valid']:cache.update({v:raw(q,v) for v in ('feedback','control')})
        local=[]
        for m,rr in group.items():
            r=rr[0];t=sorted(rr,key=lambda x:x['total_seconds'])[2]
            used=[]
            if r['api_calls']:
                used=[cache['first']]
                if r['api_calls']==2:used.append(cache[m.split('_')[1]])
            api_seconds=sum(x['request_seconds'] for x in used);tokens=[x['response'].get('usage',{}) for x in used]
            v=dict(case_id=q['id'],map_id=q['map_id'],fold=q['fold_id'],method=m,valid=r['valid'],error=r.get('error'),
                   optimal=d['optimal'],length=r['length'],operation=r['operation'],successor_expansions=r['successor_expansions'],
                   storage=r['storage'],storage_definition=r['storage_definition'],peak_open=r['measured_peak_open'],peak_open_definition=r['peak_open_definition'],
                   total_seconds=t['total_seconds'],api_seconds=api_seconds,end_to_end_seconds=t['total_seconds']+api_seconds,
                   api_calls=r['api_calls'],prompt_tokens=sum(x.get('prompt_tokens',0) for x in tokens),completion_tokens=sum(x.get('completion_tokens',0) for x in tokens),
                   initial_proposal_valid=d['initial_proposal']['valid'],final_proposal_valid=r.get('final_proposal_valid'),
                   guidance_accepted=r.get('guidance_accepted'),trigger=r.get('trigger'),guidance_release_reason=r.get('guidance_release_reason'),
                   expansions_before_guidance=r.get('expansions_before_guidance'),expansions_after_guidance=r.get('expansions_after_guidance'),
                   rekeyed_entries=r.get('rekeyed_entries',0),queue_rebuilds=r.get('queue_rebuilds',0),
                   focal_scanned_entries=r.get('focal_scanned_entries'),heap_counts=r['heap_counts'],
                   preparation_seconds=t.get('preparation_seconds',t['extra'].get('graph_seconds')),inference_seconds=t['extra'].get('inference_seconds'),
                   search_seconds=t.get('search_seconds'),reconstruction_seconds=t.get('reconstruction_seconds'),
                   wall_validation_seconds=t.get('required_validation_seconds'),rounds=len(r['rounds']) if 'rounds' in r else None,
                   fallback=r.get('fallback'),bound_violation=r['bound_violation'],path_sha256=r['path_sha256'],
                   repeat_seconds=[x['total_seconds'] for x in rr],seed_count=1)
            local.append(v)
        seeds=[next(r for r in local if r['method']==f'S4_seed{s}') for s in (11,22,33)]
        mean=dict(seeds[0],method='S4_mean',valid=all(r['valid'] for r in seeds),seed_count=3,path_sha256=None,
                  bound_violation=any(r['bound_violation'] for r in seeds),repeat_seconds=None)
        for k in ('length','operation','successor_expansions','storage','peak_open','total_seconds','end_to_end_seconds',
                  'preparation_seconds','inference_seconds','search_seconds','reconstruction_seconds','wall_validation_seconds','focal_scanned_entries'):
            values=[r[k] for r in seeds if r[k] is not None];mean[k]=st.mean(values) if values else None
        out.extend(local+[mean])
    return out

def ratio(rr,bb,metric):
    pairs=[];directions=Counter(improved=0,equal=0,worsened=0,excluded=0);invalid_pairs=0
    for r in rr:
        b=bb[r['case_id']];x=r.get(metric);y=b.get(metric)
        if not(r['valid'] and b['valid']):invalid_pairs+=1
        if x is None or y is None or (metric=='length' and not(r['valid'] and b['valid'])):directions['excluded']+=1;continue
        eq=math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8 if metric=='length' else 1e-12)
        directions['equal' if eq else 'improved' if x<y else 'worsened']+=1
        if x>0 and y>0:pairs.append((r['fold'],r['map_id'],math.log(x/y)))
    if not pairs:return dict(ratio=None,n=0,ci95=None,ci_bonferroni6=None,directions=dict(directions),invalid_pairs=invalid_pairs)
    by=defaultdict(list)
    for f,m,x in pairs:by[(f,m)].append(x)
    rng=np.random.default_rng(20261111);sums=np.zeros(2000);counts=np.zeros(2000)
    for fold in sorted({f for f,m in by}):
        vals=[v for (f,m),v in by.items() if f==fold];ss=np.array([sum(v) for v in vals]);nn=np.array([len(v) for v in vals])
        choice=rng.integers(0,len(vals),size=(2000,len(vals)));sums+=ss[choice].sum(1);counts+=nn[choice].sum(1)
    boots=np.exp(sums/counts)
    return dict(ratio=math.exp(st.mean(x[2] for x in pairs)),n=len(pairs),maps=len(by),ci95=np.quantile(boots,[.025,.975]).tolist(),
                ci_bonferroni6=np.quantile(boots,[.05/12,1-.05/12]).tolist(),directions=dict(directions),invalid_pairs=invalid_pairs)

def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build():
    verified=read(RUN/'verification.json');assert verified['complete']
    assert read(RUN/'memory_status.json')['complete'] and len(list(RUN.glob('memory/*.json')))==1320
    rows=per_case();csv_write(RUN/'per_case.csv',rows)
    summary={};flat=[];methods=METHODS+['S4_mean']
    for domain in ['OOF1000']+[f'fold{i}' for i in range(5)]:
        subset=rows if domain=='OOF1000' else [r for r in rows if r['fold']==int(domain[4:])]
        bb={r['case_id']:r for r in subset if r['method']=='B_llm'};part={}
        for m in methods:
            rr=[r for r in subset if r['method']==m];valid=[r for r in rr if r['valid']]
            tail=[r for r in subset if r['method'] in MODEL_METHODS[:3]] if m=='S4_mean' else rr
            qualities=[r['length']/r['optimal'] for r in tail if r['valid']]
            value=dict(domain=domain,method=m,cases=len(rr),valid_cases=len(valid),success_rate=len(valid)/len(rr),
                mean_planner_ms=1000*st.mean(r['total_seconds'] for r in rr),mean_end_to_end_seconds=st.mean(r['end_to_end_seconds'] for r in rr),
                mean_pop=st.mean(r['operation'] for r in rr),mean_expansions=st.mean(r['successor_expansions'] for r in rr),
                mean_peak_open=st.mean(r['peak_open'] for r in rr),mean_storage=st.mean(r['storage'] for r in rr if r['storage'] is not None),
                mean_cost_ratio=st.mean(qualities) if qualities else None,p95_excess_pct=100*(float(np.quantile(qualities,.95))-1) if qualities else None,
                max_excess_pct=100*(max(qualities)-1) if qualities else None,optimal_paths=sum(abs(x-1)<=1e-8 for x in qualities),
                quality_path_n=len(qualities),over_105=sum(x>1.05+1e-8 for x in qualities),over_110=sum(x>1.1+1e-8 for x in qualities),
                bound_violations=sum(r['bound_violation'] for r in tail),calls=sum(r['api_calls'] for r in rr),
                prompt_tokens=sum(r['prompt_tokens'] for r in rr),completion_tokens=sum(r['completion_tokens'] for r in rr))
            for metric in ('total_seconds','end_to_end_seconds','operation','successor_expansions','length'):
                value[metric]=ratio(rr,bb,metric)
            part[m]=value;flat.append(value)
        summary[domain]=part
    save(RUN/'summary.json',summary);csv_write(RUN/'summary.csv',flat)
    comparisons=[('S1','V_GATE_NORMALIZED'),('S2_feedback','S2_control'),('S3','N_g2h'),('S4_mean','FOCAL_105'),
                 ('S4_mean','S4_untrained'),('S5','A_OCT'),('S6_program','A_OCT'),('S6_program','S6_rule'),('S6_program','S6_random')]
    direct=[];worst=[]
    for m,b in comparisons:
        rr=[r for r in rows if r['method']==m];bb={r['case_id']:r for r in rows if r['method']==b}
        direct.append(dict(method=m,control=b,**{k:ratio(rr,bb,k) for k in ('total_seconds','end_to_end_seconds','operation','successor_expansions','length')}))
        for r in sorted(rr,key=lambda r:r['total_seconds']/bb[r['case_id']]['total_seconds'],reverse=True)[:3]:
            worst.append(dict(method=m,control=b,case_id=r['case_id'],fold=r['fold'],time_ratio=r['total_seconds']/bb[r['case_id']]['total_seconds'],
                              record=f'folds/fold{r["fold"]}/evaluations/{r["case_id"]}.json'))
    save(RUN/'direct_comparisons.json',direct);save(RUN/'worst_cases.json',worst)
    api=[]
    for p in list(RUN.glob('raw/public/*.json'))+list(RUN.glob('folds/fold*/raw/s6_programs/*.json')):
        r=read(p);u=r['response'].get('usage',{});api.append(dict(variant=r['variant'],prompt_tokens=u.get('prompt_tokens',0),completion_tokens=u.get('completion_tokens',0),
             cached_prompt_tokens=u.get('prompt_tokens_details',{}).get('cached_tokens',0),seconds=r['request_seconds'],finish_reason=r['response']['choices'][0].get('finish_reason'),attempts=len(r['attempts'])))
    usage=dict(actual_responses=len(api),prompt_tokens=sum(r['prompt_tokens'] for r in api),completion_tokens=sum(r['completion_tokens'] for r in api),
               cached_prompt_tokens=sum(r['cached_prompt_tokens'] for r in api),summed_wait_seconds=sum(r['seconds'] for r in api),transport_attempts=sum(r['attempts'] for r in api),
               failed_requests=len(list(RUN.rglob('failures/*.json'))),by_variant=dict(Counter(r['variant'] for r in api)),finish_reasons=dict(Counter(r['finish_reason'] for r in api)))
    save(RUN/'api_usage.json',usage)
    def f(x):return '—' if x is None else f'{x:.2f}'
    def pct(x):return '—' if x is None else f'{100*x:.2f}'
    def name(m):return NAMES.get(m,m)
    def improvement(v):return '—' if v is None else f'{100*(1-v):+.2f}%'
    total=summary['OOF1000'];protocol=read(RUN/'protocol.json')
    lines=['# S1~S6 공개 1,000문제 교차검증 실험 결과','',
       '학습·준비: 2026-11-11, 새 평가 응답 수집·본 평가·완료: 2026-11-12 (Asia/Seoul).','',
       f"완료 검증: **100지도 × 10문제 × 22조건 × 5회 = {verified['timed_trials']:,}회**. 모든 문제는 자기 지도를 학습·검증에 사용하지 않은 외부 fold에서 한 번 평가했다.",'',
       f"무효 경로 시행: {verified['invalid_trials']}회. 선언한 경로 비용 상한 위반: {verified['bound_violation_trials']}회. 시드·시간 반복은 독립 문제 수에 합산하지 않았다.",'',
       '## 핵심 결과','',
       '아래 개선율은 원 LLM-A*에 대한 문제별 비율 기하평균으로 계산했다. 양수는 감소, 음수는 증가다. 총응답시간은 새 응답 수집의 실제 API 대기와 오프라인 계획기 시간을 합한 재구성 추정이며 독립 온라인 반복 실측이 아니다.','',
       '| 방법 | 유효 경로 | 계획기 시간 감소 | 총응답시간 감소(추정) | heap pop 감소 | 확장 감소 | 최대 경로 초과율 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in PRIMARY:
        d=total[m];lines.append(f"| {name(m)} | {d['valid_cases']}/1000 | {improvement(d['total_seconds']['ratio'])} | {improvement(d['end_to_end_seconds']['ratio'])} | {improvement(d['operation']['ratio'])} | {improvement(d['successor_expansions']['ratio'])} | {f(d['max_excess_pct'])}% |")
    lines+=['','성공률과 해당 방법의 선언 상한을 지킨 주 개선안 중 관측 최소값은 다음과 같다. 통계적 우승이나 다른 품질 조건까지의 종합 우위를 의미하지 않는다.','']
    eligible=[m for m in PRIMARY if total[m]['valid_cases']==1000 and total[m]['bound_violations']==0]
    for label,key in [('계획기 시간 기하평균 비율','total_seconds'),('총응답시간 추정 기하평균 비율','end_to_end_seconds'),('heap pop 기하평균 비율','operation'),('이웃 확장 기하평균 비율','successor_expansions')]:
        candidates=[m for m in eligible if total[m][key]['ratio'] is not None]
        if candidates:
            best=min(candidates,key=lambda m:total[m][key]['ratio']);lines.append(f"- {label}: **{name(best)}**, 원 LLM-A* 대비 {pct(total[best][key]['ratio'])}%.")
    lines+=['','S4는 세 시드의 문제별 비용 평균이다. S4의 경로 품질 꼬리값은 평균 경로가 아닌 실제 3,000개 시드별 경로를 사용한다. S1~S3에는 최단 또는 5% 보장이 없으므로 품질 조건이 다른 방법을 단일 순위로 묶지 않는다.','',
      '## 실행 조건과 데이터','',
      f"- Python {protocol['environment']['python']}; CPU {protocol['environment']['processor']}; CPU 추론, 계측 시 native 1스레드.",
      '- NumPy 1.26.4, SciPy 1.15.3, Shapely 2.2.0, PyTorch 2.7.1+cpu. 현재 AMD PC에서 모든 비교군을 재측정했으며 과거 GM Intel 시간과 합치지 않았다.',
      '- 5개 외부 fold마다 학습 64지도/640문제, 검증 16지도/160문제, 평가 20지도/200문제. S4 3시드 × 5fold = 15개 모델을 처음부터 학습했다.',
      '- S6는 fold마다 새 GPT 계수 12개를 생성하고 학습 pop 비율 상위 3개 중 검증 비율 최저를 선택했다. 외부 평가를 사용하지 않았다.',
      f'- LLM `{MODEL}`, temperature=0, max_tokens=1000, 줄바꿈 stop 없음. 처음 응답과 S2 수정 응답을 새로 수집했다.',
      '- 기존 GM standard 지시문·형식은 유지하되 공개 지도와 겹치던 예시를 별도 합성 지도 5개로 교체했다. 원 LLM-A*는 원 탐색 알고리즘이라는 뜻이며 원문 프롬프트 그대로의 Table 1 재현이 아니다.',
      '- A*, 원 LLM-A*, g+2h, 방법별 직접 대조군을 같은 실행에 포함했다. 지도 준비·파싱·추론·수리·모든 탐색·복원·필수 검증을 계획기 시간에 포함했다.',
      '- API 수집은 앞쪽 순차 공급 6worker와 아직 도달하지 않은 뒤쪽·중간 구간의 별도 32worker로 진행했다. 동시성에 따른 서비스 지연 차이가 있을 수 있으며 모든 방법은 같은 문제의 동일 응답·지연을 공유한다. 생성 건수 합계와 저장 응답 수를 대조해 중복 수집을 검사했다. 상태 파일 교체 오류로 병렬 작업을 한 번 재개했으며 성공 응답은 재생성하지 않았다.',
      '- 파일 IO·독립 정답 계산·학습·모델 로딩은 주 시간에서 제외하고 별도 기록했다. 실행 순서는 문제·반복별 고정 난수로 섞었고 평가 중 네트워크 연결을 차단했다.',
      '- 공개 자료는 과거 연구에서 이미 노출됐다. 지도 교차검증은 이번 학습 중복을 막지만 연구 전체의 미관측 hidden 평가를 의미하지 않는다.','',
      '## 전체 조건 비교','',
      '시간 ms/초는 문제별 5회 중앙값의 산술평균이다. 비율 기하평균과 산술평균의 비율은 서로 다르다.','',
      '| 조건 | 성공/1000 | 계획기 ms | 총응답 초(추정) | 평균 pop | 평균 확장 | 논리 LLM 호출 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in METHODS+['S4_mean']:
        d=total[m];lines.append(f"| {name(m)} | {d['valid_cases']} | {f(d['mean_planner_ms'])} | {f(d['mean_end_to_end_seconds'])} | {f(d['mean_pop'])} | {f(d['mean_expansions'])} | {d['calls']} |")
    lines+=['','`S6_historical`은 과거 공개 자료 노출이 있는 보조 조건이며 이번 독립 후보의 우승 판정에서 제외한다. S4·S5는 온라인 LLM 호출이 없고 S6의 LLM 호출은 오프라인 계수 제안이다. 이 조건들의 온라인 지연 감소만으로 LLM 안내의 효용을 주장할 수 없다.','',
      '## 경로 품질','',
      '| 방법 | 최단 경로/유효 경로 | 평균 L/L* | P95 초과율 | 최대 초과율 | 5% 초과 | 10% 초과 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in ['B_llm','A_GM','N_g2h']+PRIMARY:
        d=total[m];lines.append(f"| {name(m)} | {d['optimal_paths']}/{d['quality_path_n']} | {d['mean_cost_ratio']:.4f} | {f(d['p95_excess_pct'])}% | {f(d['max_excess_pct'])}% | {d['over_105']} | {d['over_110']} |")
    lines+=['','S4의 선언 상한은 1.05L*, S5/S6는 최단 경로다. 관측상 최단 일치와 알고리즘의 최단 보장은 구분한다. S5는 정확 그래프 압축·격자 복원, S6는 완화 문제의 최단 경로를 전체 벽으로 검증하고 필요하면 벽을 추가해 전체 그래프로 fallback한다.','',
      '## 원 LLM-A* 대비 불확실성 및 개선·동일·악화','',
      'fold 안에서 지도 단위 재표집 2,000회. 고정된 응답·학습 모델에 조건부인 탐색적 구간이며 API 재호출·재학습 변동 전체를 포함하지 않는다. 6개 주 방법에 대한 보수적 Bonferroni 구간도 JSON에 제공했다.','',
      '| 방법 | 계획기 시간 비율 [95% 구간] | 개선/동일/악화 | 총응답시간 비율 [95% 구간] |',
      '| --- | ---: | ---: | ---: |']
    for m in PRIMARY:
        a=total[m]['total_seconds'];b=total[m]['end_to_end_seconds'];c=a['directions']
        lines.append(f"| {name(m)} | {pct(a['ratio'])}% [{pct(a['ci95'][0])}, {pct(a['ci95'][1])}] | {c['improved']}/{c['equal']}/{c['worsened']} | {pct(b['ratio'])}% [{pct(b['ci95'][0])}, {pct(b['ci95'][1])}] |")
    lines+=['','양수 비율만 기하평균에 사용하며 분모 수와 0·누락은 JSON에 남겼다. 실패 시행은 전체 분모와 소모 시간에 보존했고 유효 경로끼리의 길이 비교는 별도다.','',
      '## 개선 원인을 확인하는 직접 대조','',
      '| 개입 / 대조 | 계획기 시간 비율 [95% 구간] | pop 비율 | 확장 비율 | 해석 |',
      '| --- | ---: | ---: | ---: | --- |']
    for d in direct:
        t=d['total_seconds'];lo,hi=t['ci95'];meaning='시간 감소 관측' if hi<1 else '시간 증가 관측' if lo>1 else '시간 차이 불확실'
        lines.append(f"| {name(d['method'])} / {name(d['control'])} | {pct(t['ratio'])}% [{pct(lo)}, {pct(hi)}] | {pct(d['operation']['ratio'])}% | {pct(d['successor_expansions']['ratio'])}% | {meaning} |")
    lines+=['','S1은 정규화 gate-only, S2는 동일한 추가 호출의 일반 재검토, S3는 끝까지 무안내 g+2h, S4는 같은 FOCAL, S5는 같은 백엔드 A*_octile, S6는 규칙·무작위 벽 선택과 비교했다. 원 LLM-A* 대비 개선과 LLM 안내 자체의 추가 효용을 분리해서 해석해야 한다.','',
      '## 결과 해석','',
      '- **S1:** 원 LLM-A*보다 계획기 시간이 감소했고, 같은 정규화 gate-only와 비교해도 작은 추가 감소를 보였다. 최단 경로 보장이 있는 방법은 아니며 API 호출 수는 그대로다.',
      '- **S2:** 구체 피드백은 일반 재검토보다 계획기 시간이 조금 짧았지만, 원 LLM-A* 대비 계획기·API 포함 시간은 증가했다. 개선 구조의 추가 호출 비용을 상쇄하지 못했다.',
      '- **S3:** 원 LLM-A*보다 호출과 계획기 시간을 줄였지만 무안내 g+2h보다 계획기 시간이 늘었다. 안내의 추가 탐색 이득은 확인하지 못했다.',
      '- **S4:** 5% 경로 상한은 지켰지만 무안내 FOCAL 대비 시간이 증가했다. 학습 전 모델보다의 개선과 무안내 탐색보다의 개선은 구분해야 한다.',
      '- **S5:** 모든 경로가 최단이었고 pop·확장은 줄었으나 압축·복원 등을 포함한 계획기 시간은 같은 백엔드 A*_octile보다 증가했다.',
      '- **S6:** 모든 경로가 최단이었고 A*_octile 및 원 LLM-A* 대비 시간 감소를 보였다. 그러나 규칙 점수보다 느려 LLM이 제안한 계수의 추가 효용은 입증되지 않았다.','',
      f"S3의 호출은 {total['S3']['calls']}/1000회다. 무호출 문제의 밀리초 시간과 호출 문제의 초 단위 시간이 섞여 총응답시간 비율의 기하평균 감소가 크게 보인다. 산술평균 재구성 총응답시간은 {total['B_llm']['mean_end_to_end_seconds']:.4f}초에서 {total['S3']['mean_end_to_end_seconds']:.4f}초로 **{100*(1-total['S3']['mean_end_to_end_seconds']/total['B_llm']['mean_end_to_end_seconds']):.2f}% 감소**했다. 기하평균 감소율을 평균 사용자 대기시간 감소율로 해석하지 않는다.",'',
      'S4·S5·S6의 매우 큰 API 포함 시간 감소는 온라인 LLM 호출이 없다는 조건이 주원인이다. 오프라인 학습·계수 제안·모델 로딩 비용은 아래에 별도 보고하며, 온라인 시간에 자동으로 포함하거나 임의의 사용 횟수로 상각하지 않았다.','',
      '## fold별 결과','',
      '| fold | S1 ms | S2 ms | S3 ms | S4 ms | S5 ms | S6 ms | 원 LLM-A* ms |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for fold in range(5):
        ds=summary[f'fold{fold}'];lines.append('| '+str(fold)+' | '+' | '.join(f(ds[m]['mean_planner_ms']) for m in PRIMARY+['B_llm'])+' |')
    lines+=['','## 추가 비용과 안내 진단','',
      f"- 실제 생성 응답 {usage['actual_responses']:,}건: {usage['by_variant']}. 입력 {usage['prompt_tokens']:,}토큰, 출력 {usage['completion_tokens']:,}토큰; 입력 중 캐시 토큰 {usage['cached_prompt_tokens']:,}.",
      f"- 요청별 대기시간 합계 {usage['summed_wait_seconds']:.2f}초는 병렬 수집의 실제 경과시간과 다르다. 실패 요청 기록 {usage['failed_requests']}건. 종료 사유: {usage['finish_reasons']}.",
      '- 금액은 청구서와 현재 단가를 검증하지 않아 환산하지 않았다. 실제 수집 호출과 방법별 단독 사용 시의 논리 호출을 구분한다. S6 프로그램 제안 60건은 오프라인 비용이다.','']
    initial_invalid=sum(not r['initial_proposal_valid'] for r in rows if r['method']=='B_llm')
    for m in ('S2_feedback','S2_control'):
        rr=[r for r in rows if r['method']==m and not r['initial_proposal_valid']]
        lines.append(f"- {name(m)}: 최초 무효 {initial_invalid}건 중 수정 제안 유효 {sum(bool(r['final_proposal_valid']) for r in rr)}건. 제안 유효성과 최종 탐색 성공은 다른 지표다.")
    for m in ('S1','V_GATE_NORMALIZED'):
        rr=[r for r in rows if r['method']==m];lines.append(f"- {m}: 안내 채택 {sum(bool(r['guidance_accepted']) for r in rr)}/1000건.")
    rr=[r for r in rows if r['method']=='S3'];lines.append(f"- S3: 안내 호출 {sum(r['api_calls'] for r in rr)}/1000건, OPEN 재계산 항목 {sum(r['rekeyed_entries'] for r in rr):,}개. 전후 확장·해제 사유는 per_case.csv에 기록했다.")
    lines+=['','| fold | S4 학습 초(3시드) | S6 선택 초 | 선택 계수 |','| --- | ---: | ---: | --- |']
    for fold in range(5):
        dest=RUN/'folds'/f'fold{fold}';training=sum(read(dest/'models'/f'seed{s}.json')['seconds'] for s in (11,22,33));sel=read(dest/'program_selection.json')
        lines.append(f"| {fold} | {training:.2f} | {sel['selection_seconds']:.2f} | {sel['weights']} |")
    lines+=['','## 메모리와 관리 연산','',
      '상태 저장 수는 엔진별 의미가 다르다. GM은 inf 포함 g 사전, FOCAL은 조밀 배열 할당 수, S5는 유한 g 사전, S6는 회차별 최대 저장 수다. 이를 같은 바이트로 해석하지 않는다. heap pop·실제 이웃 확장·FOCAL 배열 스캔·OPEN 재계산도 별개다.','',
      '| 방법 | 평균 OPEN 크기 최대값 | 평균 상태 수 | 표본 RSS peak MiB 중앙값 | 표본 RSS 증가 MiB 중앙값 |',
      '| --- | ---: | ---: | ---: | ---: |']
    mem=[read(p) for p in RUN.glob('memory/*.json')]
    for m in ['B_llm','A_GM']+PRIMARY:
        mm=MODEL_METHODS[:3] if m=='S4_mean' else [m];rs=[r for r in mem if r['method'] in mm]
        peak=st.median(r['sampled_peak_rss_bytes'] for r in rs)/2**20 if rs else None
        inc=st.median(r['sampled_increment_bytes'] for r in rs)/2**20 if rs else None
        d=total[m];lines.append(f"| {name(m)} | {f(d['mean_peak_open'])} | {f(d['mean_storage'])} | {f(peak)} | {f(inc)} |")
    lines+=['',f'RSS 보조 측정 {len(mem)}회. 사전에 정한 20지도×1문제×22조건×3회, 각각 새 프로세스. nominal 1ms 표본이며 순간 최대를 보장하지 않는다. Python·라이브러리·모델 상주량을 포함하며 시간 계측과 별도로 실행했다. OPEN 정의도 heap 중복 포함과 논리 OPEN을 CSV에서 구분했다.','',
      'heap wrapper 계측의 영향을 개발용 별도 10지도에서 on/off 3회로 검사했다. 경로가 바뀌지 않았는지 확인하고 방법별 시간 비율을 [계측 민감도](instrumentation_check.json)에 보존했다. 이 보조 비교로 평가 설정을 선택하거나 주 계측 시간을 보정하지 않았다. 다른 카운터 비용까지 제거한 비교는 아니다.','',
      '## 악화 사례','', '| 개입 / 대조 | 문제 | 계획기 시간 배수 |','| --- | --- | ---: |']
    for e in worst[::3]:lines.append(f"| {name(e['method'])} / {name(e['control'])} | [{e['case_id']}]({e['record']}) | {e['time_ratio']:.2f} |")
    lines+=['','## 재현과 산출물','',
      '- [계획](plan.md), [프로토콜](protocol.json), [지도 분할](fold_manifest.json), [고정 코드·입력·모델 해시](frozen.json), [독립 검산](verification.json).',
      '- [문제별 결과](per_case.csv), [전체·fold별 요약](summary.json), [직접 대조](direct_comparisons.json), [실제 API 사용](api_usage.json), [악화 사례](worst_cases.json).',
      '- 최초 S1~S6 구현 출처는 기존 SM 코드, GM 비교군은 고정 vendor 코드다. 이번 실행기·교차검증·계측·새 응답·보고서 작성은 Codex 지원 작업이다.',
      '- 이 결과는 S1~S6 공통 재실험이며 원격 18개 방법 전체를 같은 환경에서 재실행한 결과가 아니다. Table 1 재현·hidden 환경 입출력 검증·최종 제출은 별도다.','',
      '실행 명령과 단계별 로그는 run_log.json에 기록한다. 수집 재개는 같은 요청 해시의 성공 응답을 재사용하며 좋은 응답이 나올 때까지 재생성하지 않는다.','']
    (RUN/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    assert read(RUN/'memory_status.json')['complete'] and len(mem)==1320
    manifest=dict(status='completed',completed_at=now(),verification=verified,report_script_sha256=sha(HERE/'report.py'),
                  outputs={n:sha(RUN/n) for n in ('report.md','per_case.csv','summary.json','api_usage.json','verification.json')})
    save(RUN/'manifest.json',manifest);print(json.dumps(dict(completed=True,report=str(RUN/'report.md'))),flush=True)

if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'all'
    if action in ('all','verify'):verify()
    if action in ('all','report'):build()
