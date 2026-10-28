"""Independent artifact verification and auditable Korean Markdown summary."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
import numpy as np
from common import *
from algorithms import BASE, MODEL_METHODS
from experiments.K3.verification import Oracle

def artifact_rows():
    for path in sorted((OUT/'evaluations/main').glob('*/*.json')):yield path,read(path)

def verify():
    sources=read(OUT/'main_frozen.json');signature=sources['signature'];n=0;trials=0;invalid=0;bound_errors=0;groups=defaultdict(list);ids=set()
    for path,record in artifact_rows():
        q=record['case'];assert q['id'] not in ids;ids.add(q['id']);assert record['signature']==signature
        assert len(record['rows'])==5*(len(BASE)+len(MODEL_METHODS))
        assert {(r['method'],r['repeat']) for r in record['rows']}=={(m,i) for m in BASE+MODEL_METHODS for i in range(5)}
        groups[q['map_id']].append(record);n+=1
        for r in record['rows']:
            assert digest(record['paths'][r['path_sha256']])==r['path_sha256']
            assert r['total_seconds']>=0 and r['operation']>=0 and r['successor_expansions']>=0
            trials+=1;invalid+=int(not r['valid']);bound_errors+=int(r['bound_violation'])
    assert ids=={q['id'] for q in cases()},('Missing/extra cases',len(ids))
    for map_index,(mid,records) in enumerate(groups.items()):
        oracle=Oracle(records[0]['case'])
        for record in records:
            q=record['case'];opt=oracle.distances(q['goal'])[tuple(q['start'])];assert math.isclose(opt,record['optimal'],abs_tol=1e-8)
            checked=set()
            for r in record['rows']:
                if r['valid'] and r['path_sha256'] not in checked:
                    oracle.check_path(q,dict(r,path=record['paths'][r['path_sha256']]))
                    checked.add(r['path_sha256'])
                if r['valid']:
                    exact=r['method'] in ('A_GM','A_OCT','S5') or r['method'].startswith('S6_')
                    bounded=r['method']=='FOCAL_105' or r['method'] in MODEL_METHODS
                    independently_violated=(exact and abs(r['length']-opt)>1e-8+1e-10*opt) or (bounded and r['length']>1.05*opt+1e-8+1e-10*opt)
                    assert bool(independently_violated)==r['bound_violation']
        if (map_index+1)%20==0:print(json.dumps(dict(phase='verify',maps=map_index+1,total=len(groups))),flush=True)
    for rel,expected in sources['frozen']['code'].items():assert sha(ROOT/rel)==expected,('Source changed',rel)
    raw_files=list((OUT/'raw').glob('*/*.json'))
    for p in raw_files:
        d=read(p);assert digest(d['request'])==d['request_sha256'];assert digest(d['response'])==d['response_sha256']
        assert d['request']['model']==MODEL and d['response']['model']==MODEL
    expected_raw=12
    for q in cases():
        first=raw(q);assert first['case_id']==q['id'] and first['stage']==q['stage']
        assert first['request_sha256']==digest(payload(q));expected_raw+=1
        if not diagnosis(q,text_of(first))['valid']:
            for variant,feedback in [('control',False),('feedback',True)]:
                revised=raw(q,variant);assert revised['case_id']==q['id']
                assert revised['request_sha256']==digest(correction(q,text_of(first),feedback));expected_raw+=1
    assert len(raw_files)==expected_raw,(len(raw_files),expected_raw)
    result=dict(artifacts_verified=True,complete=True,cases=n,maps=len(groups),timed_trials=trials,invalid_trials=invalid,bound_violation_trials=bound_errors,
                methods=len(BASE)+len(MODEL_METHODS),timing_repeats=5,raw_responses=len(raw_files),request_payloads_verified=True,verified_at=now())
    save(OUT/'verification.json',result);print(json.dumps(result),flush=True)

def per_case():
    rows=[];examples=[]
    for _,record in artifact_rows():
        q=record['case'];group=defaultdict(list)
        for r in record['rows']:group[r['method']].append(r)
        for method,rr in group.items():
            r=rr[0];timed=sorted(rr,key=lambda x:x['total_seconds'])[len(rr)//2]
            values=dict(case_id=q['id'],map_id=q['map_id'],stage=q['stage'],split=q['split'],method=method,optimal=record['optimal'],
                valid=r['valid'],length=r['length'],operation=r['operation'],successor_expansions=r['successor_expansions'],storage=r['storage'],
                storage_definition=r['storage_definition'],total_seconds=statistics.median(x['total_seconds'] for x in rr),repeat_seconds=[x['total_seconds'] for x in rr],
                api_calls=r['api_calls'],initial_proposal_valid=record['initial_proposal']['valid'],final_proposal_valid=r.get('final_proposal_valid'),
                error=r.get('error'),bound_violation=r['bound_violation'],path_sha256=r['path_sha256'],seeds=1,valid_seeds=int(r['valid']))
            values.update(preparation_seconds=timed.get('preparation_seconds',timed['extra'].get('graph_seconds')),
                          inference_seconds=timed['extra'].get('inference_seconds'),search_seconds=timed.get('search_seconds'),
                          reconstruction_seconds=timed.get('reconstruction_seconds'),wall_validation_seconds=timed.get('required_validation_seconds'))
            values.update(rounds=len(r['rounds']) if 'rounds' in r else None,active_walls=r.get('active_walls'),
                          fallback=(any(w[1]==w[2] for w in q['horizontal_barriers']+q['vertical_barriers']) if method=='S5' else r.get('fallback')),
                          fallback_source='frozen_input_and_Rectangles_disabled_rule' if method=='S5' else 'recorded' if method.startswith('S6_') else None)
            rows.append(values)
        seeds=[next(r for r in rows[-len(group):] if r['method']==f'S4_seed{s}') for s in (11,22,33)]
        mean=dict(seeds[0]);mean.update(method='S4_mean',seeds=3,valid_seeds=sum(r['valid'] for r in seeds),valid=all(r['valid'] for r in seeds),
            path_sha256=None,repeat_seconds=None,bound_violation=any(r['bound_violation'] for r in seeds))
        for metric in ('length','operation','successor_expansions','storage','total_seconds'):mean[metric]=statistics.mean(r[metric] for r in seeds if r[metric] is not None)
        for metric in ('preparation_seconds','inference_seconds','search_seconds','reconstruction_seconds','wall_validation_seconds'):
            known=[r[metric] for r in seeds if r[metric] is not None];mean[metric]=statistics.mean(known) if known else None
        rows.append(mean)
    return rows

def ratio(rows,baseline,metric):
    pairs=[]
    for r in rows:
        b=baseline[r['case_id']];x=r.get(metric);y=b.get(metric)
        if metric=='length' and not(r['valid'] and b['valid']):continue
        if x is not None and y is not None and x>0 and y>0:pairs.append((r['map_id'],math.log(x/y)))
    if not pairs:return dict(ratio=None,n=0,ci95=[None,None])
    bymap=defaultdict(list)
    for mid,value in pairs:bymap[mid].append(value)
    sums=np.array([sum(v) for v in bymap.values()]);counts=np.array([len(v) for v in bymap.values()]);rng=np.random.default_rng(20261029)
    choices=rng.integers(0,len(sums),size=(2000,len(sums)));boots=np.exp(sums[choices].sum(axis=1)/counts[choices].sum(axis=1))
    return dict(ratio=math.exp(statistics.mean(v for _,v in pairs)),n=len(pairs),ci95=np.quantile(boots,[.025,.975]).tolist(),maps=len(bymap))

def direction_counts(rows,baseline,metric):
    counts=dict(improved=0,equal=0,worsened=0,excluded=0)
    for r in rows:
        b=baseline[r['case_id']];x=r.get(metric);y=b.get(metric)
        if x is None or y is None or (metric=='length' and not(r['valid'] and b['valid'])):counts['excluded']+=1;continue
        equal=math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8 if metric=='length' else 1e-12)
        counts['equal' if equal else 'improved' if x<y else 'worsened']+=1
    return counts

def build_report():
    checked=read(OUT/'verification.json');assert checked['complete'];rows=per_case();methods=BASE+MODEL_METHODS+['S4_mean']
    with (OUT/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    domains={'public_test':lambda r:r['stage']=='public' and r['split']=='test','transfer':lambda r:r['stage']=='transfer',
             'synthetic':lambda r:r['stage']=='synthetic','public_all':lambda r:r['stage']=='public'}
    summaries=[];details={}
    for domain,predicate in domains.items():
        subset=[r for r in rows if predicate(r)];by={m:[r for r in subset if r['method']==m] for m in methods};base={r['case_id']:r for r in by['N_g2h']}
        details[domain]={}
        for method,rr in by.items():
            valid=[r for r in rr if r['valid']];quality=[r['length']/r['optimal'] for r in valid]
            tailrows=([r for r in subset if r['method'] in ('S4_seed11','S4_seed22','S4_seed33')] if method=='S4_mean' else rr)
            tailvalid=[r for r in tailrows if r['valid']];tail=[r['length']/r['optimal'] for r in tailvalid]
            d=dict(dataset=domain,method=method,n=len(rr),valid=sum(r['valid_seeds'] for r in rr),trials=sum(r['seeds'] for r in rr),
                valid_cases=len(valid),mean_total_ms=1000*statistics.mean(r['total_seconds'] for r in rr),
                geometric_length_optimal=math.exp(statistics.mean(math.log(x) for x in quality)) if quality else None,
                max_length_optimal=max(tail) if tail else None,optimal_paths=sum(abs(x-1)<=1e-8 for x in tail),
                over_105=sum(x>1.05+1e-8 for x in tail),over_110=sum(x>1.10+1e-8 for x in tail),
                quality_path_n=len(tail),quality_tail_unit='individual_seed_path' if method=='S4_mean' else 'individual_path',
                bound_violations=sum(r['bound_violation'] for r in tailrows))
            for metric in ('preparation_seconds','inference_seconds','search_seconds','reconstruction_seconds','wall_validation_seconds'):
                known=[r[metric] for r in rr if r[metric] is not None];d['mean_'+metric]=statistics.mean(known) if known else None
            for metric in ('operation','successor_expansions','total_seconds','length'):
                item=ratio(rr,base,metric);d[metric+'_ratio']=item['ratio'];d[metric+'_n']=item['n'];d[metric+'_ci95']=item['ci95']
            details[domain][method]=d;summaries.append(d)
    save(OUT/'summary.json',details)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    api=[]
    for path in sorted((OUT/'raw').glob('*/*.json')):
        r=read(path);usage=r['response'].get('usage',{});api.append(dict(stage=r['stage'],variant=r['variant'],prompt_tokens=usage.get('prompt_tokens',0),completion_tokens=usage.get('completion_tokens',0),seconds=r['request_seconds'],transport_attempts=len(r['attempts']),finish_reason=r['response']['choices'][0].get('finish_reason')))
    api_summary=dict(actual_generated_responses=len(api),prompt_tokens=sum(r['prompt_tokens'] for r in api),completion_tokens=sum(r['completion_tokens'] for r in api),
                     transport_attempts=sum(r['transport_attempts'] for r in api),summed_request_seconds=sum(r['seconds'] for r in api),by_variant={})
    for variant in sorted(set(r['variant'] for r in api)):
        rr=[r for r in api if r['variant']==variant];api_summary['by_variant'][variant]=dict(responses=len(rr),prompt_tokens=sum(r['prompt_tokens'] for r in rr),completion_tokens=sum(r['completion_tokens'] for r in rr))
    api_summary['finish_reasons']={str(reason):sum(r['finish_reason']==reason for r in api) for reason in {r['finish_reason'] for r in api}}
    save(OUT/'api_usage.json',api_summary)
    def pct(x):return '—' if x is None else f'{100*x:.2f}'
    lines=['# S1~S6 재실험 결과 — GM 기준·Python 3.11','',
      '실행 조건·실제 수치·실패를 원시 파일에서 재집계한 결과다. 과거 GM·SM·JY의 다른 장비 결과를 합친 전체 18개 순위가 아니다.','',
      f"검증 상태: **{checked['cases']:,}문제 × {checked['methods']}조건 × 5회 = {checked['timed_trials']:,}회 계측 검증 완료**. 무효 경로 시행 {checked['invalid_trials']}회, 명시적 품질 상한 위반 {checked['bound_violation_trials']}회.",'',
      '## 조건과 해석 범위','',
      f"- Python 3.11.18, Windows, AMD Ryzen 5 9600X CPU. S4 추론도 CPU, 계측 중 PyTorch 1스레드. 패키지·정확한 환경은 [규약](protocol.json).",
      '- GM의 데이터·비교군·평가 규칙을 적용하되, 과거 GM의 Intel 장비 대신 현재 AMD 장비에서 모든 비교군을 함께 재측정했다. 과거 GM 절대 시간과 직접 비교하지 않는다.',
      '- GM 커밋 `c56eb83b7df4be06d402daba36e0a2b45afa950b`; GPT `gpt-4o-mini-2024-07-18`, 원문 standard/Few-shot, temperature 0, max_tokens 1000. 새 최초 응답을 안내 기반 조건끼리 공유했다.',
      '- 공통 비율 분모는 이번 실행의 g+2h이며 **100%보다 작으면 해당 비용이 적다**. A*·원 LLM-A*·방법별 대조군도 같은 조건에서 실행했다.',
      '- 시간은 파싱·준비·추론·탐색·필수 경로 검증을 포함한 5회 중앙값이다. 실제 heap 연산 계측의 공통 wrapper 비용도 포함한다. API·파일 IO·독립 oracle·학습·모델 로딩은 별도다.',
      '- pop은 모든 실제 heap pop, 확장은 목표 선택을 제외한 이웃 순회 횟수다. FOCAL 배열 스캔·재정렬·충돌·압축 비용은 pop과 다르며 시간·부가 카운터에 남겼다.',
      '- S4는 FOCAL w=1.05, S5/S6는 정확 탐색이다. S1~S3 및 가중 탐색과 품질 조건이 달라 단일 pop 수로 종합 순위를 정하지 않는다.',
      '- S4 평균은 3개 재학습 시드의 문제별 산술평균 이후 비율 집계다. 시드를 독립 지도로 세지 않는다. 개별 시드와 학습 전 대조군은 summary.csv에 보존했다.',
      '- 공개 평가 200문제는 학습·검증과 지도 단위로 분리했다. 별도 30·120문제도 따로 제시한다. 전체 연구에서 한 번도 보지 않은 자료라는 의미는 아니다.','']
    lines += ['| ID | 이번 구현 | 직접 대조군 |','| --- | --- | --- |',
              '| S1 | GPT 경유점의 왕복 제거·충돌 없는 단축·검증 | 동일 정규화의 유효성 검사만 수행 |',
              '| S2 | 첫 오류 정보를 제공해 한 번 수정 | 오류 정보를 제공하지 않는 한 번의 재검토 |',
              '| S3 | 고정 거리 정체 조건에서 안내를 한 번 삽입 | 즉시 안내 원 LLM-A*와 무안내 g+2h |',
              '| S4 | 소형 TransPath-inspired 경로 확률로 FOCAL 안내 | 같은 5% FOCAL의 안내 없음·학습 전 모델 |',
              '| S5 | 정확한 sparse 직사각형 공간 압축·격자 경로 복원 | 같은 백엔드 A*_octile |',
              '| S6 | 벽 부분집합의 정확 풀이와 전체 검증·단계적 추가 | 전체 벽 A*_octile·rule·random·과거 고정 가중치 |','']
    selected=['A_GM','N_g2h','B_llm','V_GATE_GM','V_GATE_NORMALIZED','K3_wall_euclidean','K3_wall_octile','S1','S2_feedback','S2_control','S3','FOCAL_105','S4_mean','S4_untrained','A_OCT','S5','S6_program','S6_rule','S6_random','S6_historical']
    for domain,title in [('public_test','공통 비교: 공개 평가 200문제'),('transfer','공통 비교: GM 별도 30문제'),('synthetic','공통 비교: GM 합성 120문제'),('public_all','보조 비교: 공개 전체 1,000문제 — 학습·설정 선택 노출 포함')]:
        lines += ['## '+title,'','길이는 유효 경로의 L/L* 기하평균이며 %로 표시한다. 비용 비율은 각 문제에서 g+2h와 대응한 비율의 기하평균이다. 시간(ms)은 문제별 5회 중앙값의 산술평균이다. 따라서 시간(ms) 열끼리 나눈 값은 시간 비율 열과 일치하지 않을 수 있다.','',
                  '| 방법 | 유효 시행/분모 | 길이/L* % | pop % | 확장 % | 시간 % | 시간 ms |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for m in selected:
            d=details[domain][m];lines.append(f"| {m} | {d['valid']}/{d['trials']} | {pct(d['geometric_length_optimal'])} | {pct(d['operation_ratio'])} | {pct(d['successor_expansions_ratio'])} | {pct(d['total_seconds_ratio'])} | {d['mean_total_ms']:.3f} |")
        lines.append('')
    lines += ['## 경로 품질 상세','',
              'S4 평균 행의 최단·상한 초과·최악 값은 시드 평균 경로가 아니라 세 시드의 실제 개별 경로로 집계했다.','',
              '| 집합 | 방법 | 최단 경로/유효 경로 | 최단+5% 초과 | 최단+10% 초과 | 최악 L/L* % |','| --- | --- | ---: | ---: | ---: | ---: |']
    for domain in ('public_test','transfer','synthetic'):
        for method in ('B_llm','N_g2h','K3_wall_euclidean','S1','S2_feedback','S3','S4_mean','S5','S6_program'):
            d=details[domain][method]
            lines.append(f"| {domain} | {method} | {d['optimal_paths']}/{d['quality_path_n']} | {d['over_105']} | {d['over_110']} | {pct(d['max_length_optimal'])} |")
    lines += ['','## 준비·추론·탐색 시간 구성 — 공개 평가 200문제','',
              '각 문제의 전체 시간이 중앙값인 실제 시행에서 비용 구성 값을 가져와 산술평균했다. 단위 ms, —는 미사용 또는 분리 계측 없음이다. 전체 시간에는 표의 항목 외 필수 검증·기타 관리도 포함한다.','',
              '| 방법 | 그래프/준비 | 추론 | 탐색 | 복원 | S6 전체 벽 검사 | 전체 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in ('FOCAL_105','S4_mean','A_OCT','S5','S6_program','S6_rule'):
        d=details['public_test'][method]
        components=[d['mean_'+m] for m in ('preparation_seconds','inference_seconds','search_seconds','reconstruction_seconds','wall_validation_seconds')]
        cells=['—' if v is None else f'{1000*v:.3f}' for v in components]
        lines.append('| '+method+' | '+' | '.join(cells)+f" | {d['mean_total_ms']:.3f} |")
    lines += ['','## 압축 적용과 재탐색','',
              'S5의 압축 비활성 건수는 고정 입력의 길이 0 장애물 존재 여부와 `Rectangles.disabled` 규칙으로 산출했다. S6의 fallback은 기록된 마지막 전체 벽 풀이 여부이며 모든 회차 비용은 이미 전체 시간/pop에 포함된다.','',
              '| 집합 | 방법 | 압축 비활성/전체 벽 fallback | 평균 풀이 회차 |','| --- | --- | ---: | ---: |']
    for domain,predicate in list(domains.items())[:3]:
        for method in ('S5','S6_program','S6_rule','S6_random'):
            rr=[r for r in rows if predicate(r) and r['method']==method];rounds=[r['rounds'] for r in rr if r['rounds'] is not None]
            avg='—' if not rounds else f'{statistics.mean(rounds):.3f}'
            lines.append(f"| {domain} | {method} | {sum(bool(r['fallback']) for r in rr)}/{len(rr)} | {avg} |")
    lines += ['','## 방법별 직접 비교','',
              '| 집합 | 개입 / 대조 | pop 비율 % | 계획기 시간 비율 % [지도 bootstrap 95% 구간] | 길이 비율 % |','| --- | --- | ---: | ---: | ---: |']
    comparisons=[('S1','V_GATE_NORMALIZED'),('S2_feedback','S2_control'),('S3','B_llm'),('S4_mean','FOCAL_105'),('S5','A_OCT'),('S6_program','A_OCT'),('S6_program','S6_rule')]
    comparison_data=[]
    for domain,predicate in list(domains.items())[:3]:
        for method,control in comparisons:
            rr=[r for r in rows if predicate(r) and r['method']==method];bb={r['case_id']:r for r in rows if predicate(r) and r['method']==control}
            p=ratio(rr,bb,'operation');t=ratio(rr,bb,'total_seconds');l=ratio(rr,bb,'length')
            lines.append(f"| {domain} | {method} / {control} | {pct(p['ratio'])} | {pct(t['ratio'])} [{pct(t['ci95'][0])}, {pct(t['ci95'][1])}] | {pct(l['ratio'])} |")
            directions={metric:direction_counts(rr,bb,metric) for metric in ('operation','successor_expansions','total_seconds','length')}
            comparison_data.append(dict(dataset=domain,method=method,control=control,pop=p,time=t,length=l,directions=directions))
    save(OUT/'direct_comparisons.json',comparison_data)
    lines += ['','구간은 지도 단위 2,000회 재표집의 탐색적 95% 구간이다. 다중 비교 보정된 전체 순위나 3개 학습 시드의 모든 불확실성을 뜻하지 않는다. 유효 길이 비교와 양수 비율의 실제 분모는 CSV/JSON을 따른다.','',
              '## S2 수정 효과와 S3 호출','', '| 집합 | 최초 제안 유효 | 일반 재검토 후 유효 | 오류 피드백 후 유효 | S3 논리 호출/문제 |','| --- | ---: | ---: | ---: | ---: |']
    for domain,predicate in domains.items():
        original=[r for r in rows if predicate(r) and r['method']=='S3'];control=[r for r in rows if predicate(r) and r['method']=='S2_control'];feedback=[r for r in rows if predicate(r) and r['method']=='S2_feedback']
        lines.append(f"| {domain} | {sum(r['initial_proposal_valid'] for r in original)}/{len(original)} | {sum(bool(r['final_proposal_valid']) for r in control)}/{len(control)} | {sum(bool(r['final_proposal_valid']) for r in feedback)}/{len(feedback)} | {sum(r['api_calls'] for r in original)}/{len(original)} |")
    lines += ['','최초 무효 응답만을 분모로 한 수정 성공률:','',
              '| 집합 | 수정 대상 | 일반 재검토 성공 | 오류 피드백 성공 |','| --- | ---: | ---: | ---: |']
    for domain,predicate in domains.items():
        cc=[r for r in rows if predicate(r) and r['method']=='S2_control' and not r['initial_proposal_valid']]
        ff=[r for r in rows if predicate(r) and r['method']=='S2_feedback' and not r['initial_proposal_valid']]
        lines.append(f"| {domain} | {len(cc)} | {sum(bool(r['final_proposal_valid']) for r in cc)}/{len(cc)} | {sum(bool(r['final_proposal_valid']) for r in ff)}/{len(ff)} |")
    lines += ['','## 실제 생성·학습 비용','',
      f"실제 응답 {api_summary['actual_generated_responses']:,}건, 입력 {api_summary['prompt_tokens']:,}토큰, 출력 {api_summary['completion_tokens']:,}토큰, 전송 시도 {api_summary['transport_attempts']:,}회. 최초/수정/점수 제안은 [usage](api_usage.json)에 구분했다. 요금 단가는 검증하지 않았으므로 청구 금액을 임의 추정하지 않았다.",'',
      f"응답 종료 사유: `{api_summary['finish_reasons']}`. 길이 제한으로 잘린 응답도 그대로 보존·평가했다.",'',
      'S3의 논리 호출은 실제 배치 시 필요한 호출 수다. 이번 수집에서는 비교군이 공통 응답을 사용했으므로 그 수를 실제 전체 API 수집 건수와 같다고 해석하지 않는다. API 지연의 합은 동시 요청을 포함하므로 작업 경과시간이 아니다.','']
    for seed in (11,22,33):
        meta=read(OUT/'models'/f'seed{seed}.json');lines.append(f"- S4 seed {seed}: {meta['epochs']} epochs, 선택 epoch {meta['best_epoch']}, 검증 손실 {meta['best_validation_loss']:.6f}, 학습 {meta['seconds']:.2f}초, {meta['parameters']:,} 파라미터.")
    label=read(OUT/'models/training_data.json');selection=read(OUT/'program_selection.json')
    lines += [f"- S4 라벨 준비 {label['seconds']:.2f}초, 학습/검증 600문제. CPU 학습 4스레드; 학습 시간 측정은 추론 벤치마크와 분리.",
              f"- S6 선택 가중치: `{selection['weights']}`. 유효 제안 {selection['valid_proposals']}개, 중복 제거 후 {selection['unique_proposals']}개. 선택은 학습·검증 집합만 사용.",
              f"- S6 선택 비용 재계측: {selection.get('selection_queries',0):,}개 후보-문제 평가, 전체 {selection.get('selection_seconds',0):.2f}초; 후보 계획기 {selection.get('candidate_planner_seconds',0):.2f}초와 대조군 {selection.get('reference_planner_seconds',0):.2f}초 포함. 최초 선택에서 빠진 대조군 시간 기록을 보완하기 위해 동일 선택을 반복했고 선택 결과 일치를 확인했다. 최초 기록도 보존했다.",'',
              '- 최초 S6 선택의 로그 생성 시각부터 선택 완료 기록까지 경과시간은 약 56.62초였다. 프로세스 시작 비용을 포함할 수 있는 보조 기록이며 위 59.31초 재계측과 별도다. [최초 경과시간 근거](selection_initial_wall_clock.json).','',
              '## 실패·악화 사례와 한계','',
              '- 모든 실패와 상한 위반은 전체 분모에 남겼다. 실패한 모델 응답을 좋은 응답으로 재생성하거나 가장 좋은 답을 고르지 않았다.',
              '- storage의 표현이 서로 다르다. GM의 inf 포함 g 사전, S4의 dense g 배열 할당량, S5의 finite g 사전, S6의 회차별 최대 g 크기를 동일한 RAM으로 해석하지 않는다. 자세한 정의는 per_case.csv에 있다.',
              '- 이번 S1~S3의 GM 파서·경유점 처리, S4의 5% 제한/CPU 재학습, S6의 GPT 점수 제안은 과거 SM 조건과 다르다. 이 보고서는 GM 기준 변경 실험이며 과거 수치의 정확한 재현이라고 주장하지 않는다.',
              '- S6의 12개 최초 응답은 모두 JSON 코드블록이라 strict-v1 파서가 거부했다. 후보 성능 평가 전에 단일 JSON 코드블록 표지만 제거하는 v2를 모든 응답에 똑같이 적용했다. v1 판정·원문·변경 이력은 보존했으며 새 응답을 생성하지 않았다. 이는 최초 strict 파서의 성공 결과가 아니다.',
              '- 정답·oracle 계산 및 모델 학습 비용은 온라인 계획기 시간에 섞지 않고 별도 기록했다. 신규 온라인 서비스 지연의 독립 반복 시험은 아니다.','']
    lines += ['- LLM 최초 응답은 문제당 한 번 생성했고 이후 고정했다. 5회 반복은 계획기 시간 반복이며, temperature=0 모델 출력의 독립 반복 불확실성을 평가한 것은 아니다.','']
    examples=[]
    for method,control in comparisons:
        rr=[r for r in rows if domains['public_test'](r) and r['method']==method];bb={r['case_id']:r for r in rows if domains['public_test'](r) and r['method']==control}
        worst=sorted(rr,key=lambda r:r['total_seconds']/bb[r['case_id']]['total_seconds'],reverse=True)[:3]
        for r in worst:examples.append(dict(method=method,control=control,case_id=r['case_id'],time_ratio=r['total_seconds']/bb[r['case_id']]['total_seconds'],path_record=f"evaluations/main/public/{r['case_id']}.json"))
    save(OUT/'worst_cases.json',examples)
    lines += ['| 방법 / 대조 | 공개 평가 중 최대 시간 악화 사례 | 배수 |','| --- | --- | ---: |']
    for e in examples[::3]:lines.append(f"| {e['method']} / {e['control']} | [{e['case_id']}]({e['path_record']}) | {e['time_ratio']:.2f} |")
    lines += ['','## 재현과 근거','',
      '- [사전 계획](plan.md), [프로토콜](protocol.json), [코드·설정 고정](main_frozen.json), [검증](verification.json).',
      '- GM 과거 기록의 4,000개 경로·길이·pop·storage와 현재 비교군을 대조해 불일치 0건을 확인했다. 시간은 이 일치 검사의 대상이 아니다. [경로 대조 근거](historical_path_verification.json).',
      '- [문제별 CSV](per_case.csv), [집계 CSV](summary.csv), [직접 비교](direct_comparisons.json), [악화 사례](worst_cases.json).',
      '- 최초 S1~S6 구현·실험: 승민. GM 비교군: 규민의 고정 커밋. 이번 공통 실행기·계측·재학습·응답 수집·보고서 작성은 Codex 지원 작업이다.','',
      '프로젝트 루트에서 실행한 명령(collect/programs generate는 실제 API 호출):','', '```powershell',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/common.py',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/preflight.py',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/collect.py collect --stage all --workers 24',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/train.py',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/programs.py generate',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/programs.py select',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/evaluate.py all',
      '.\.venv-gm311\Scripts\python.exe experiments/s1_s6_gm_20261028/report.py all','```','']
    def change(value):
        return f"{100*abs(1-value):.2f}% {'감소' if value<1 else '증가' if value>1 else '동일'}"
    overview=['## 핵심 결과 — 공개 평가 200문제','',
              '다음은 학습·설정 선택과 분리한 공개 평가 지도에서의 직접 비교다. 별도 30·120문제 결과는 아래 표에 분리했다.','']
    for method,control in comparisons[:6]:
        comp=next(d for d in comparison_data if d['dataset']=='public_test' and d['method']==method and d['control']==control)
        overview.append(f"- **{method} / {control}:** heap pop {change(comp['pop']['ratio'])}, 계획기 시간 {change(comp['time']['ratio'])}, 유효 경로 길이 {change(comp['length']['ratio'])}.")
    overview += ['','비용 감소만으로 종합 우위를 판정하지 않는다. S2의 제안 수정 성공률, S3의 논리 호출 수, 각 방법의 길이 제한과 추가 준비 비용을 함께 확인해야 한다.','',
      '### 결과 해석','',
      '- **S1:** 공개 평가에서는 시간 비율의 95% 구간이 100%를 포함해 시간 개선이 뚜렷하지 않았다. 별도 30·120문제에서는 감소가 관측되어 조건에 따른 효과로 해석한다.',
      '- **S2:** 오류 피드백의 공개 평가 수정 성공은 6/186건으로 일반 재검토의 2/186건보다 많았으나 절대 성공률은 낮았다. 계획기 시간 차이의 구간은 100%를 포함했고 수정 API 비용도 추가된다.',
      '- **S3:** 세 평가 집합 모두 원 LLM-A*보다 계획기 시간이 감소했다. 공개 평가에서 논리 호출은 94/200회로 53% 줄었지만, 계획기 시간은 g+2h의 120.93%로 더 길었다. 실제 온라인 API 지연까지 포함한 속도 개선을 검증한 결과는 아니다.',
      '- **S4:** 모든 최종 경로가 5% 상한을 지켰지만 같은 FOCAL 대비 계획기 시간이 세 집합 모두 증가했다. 이번 소형 모델·CPU 조건에서 성능 이득을 확인하지 못했다.',
      '- **S5:** 정확한 최단 경로를 유지하며 세 집합 모두 pop은 감소했지만 준비·복원 등을 포함한 시간은 증가했다. pop 감소를 실행시간 개선으로 해석할 수 없다.',
      '- **S6:** 공개 평가의 A*_octile 대비 시간 비율은 82.39%였으나 95% 구간 [56.94%, 118.41%]이 100%를 포함한다. 별도 30·120문제에서는 각각 161.27%, 171.06%로 더 느렸다. GPT 점수도 규칙 점수보다 낫다는 근거가 없어, 이번 결과로 일반적인 개선을 주장할 수 없다.','']
    lines[6:6]=overview
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    save(OUT/'manifest.json',dict(status='completed',completed_at=now(),protocol_sha256=sha(OUT/'protocol.json'),frozen_sha256=sha(OUT/'main_frozen.json'),verification=checked,
                                report_script_sha256=sha(HERE/'report.py'),
                                outputs={n:sha(OUT/n) for n in ('report.md','per_case.csv','summary.csv','api_usage.json','direct_comparisons.json')}))
    print(json.dumps(dict(report=str(OUT/'report.md'),completed=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','report','all']);a=p.parse_args()
    if a.action in ('verify','all'):verify()
    if a.action in ('report','all'):build_report()
