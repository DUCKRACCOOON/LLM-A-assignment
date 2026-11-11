"""Verify all interventions and create a Korean report with original metrics."""
import base64
import collections
import csv
import hashlib
import html
import io
import json
import math
import os
import statistics
import zipfile
from pathlib import Path
from core import ROOT, validate, path_length, PROMPTS
from run import cases_for, payload, save, digest
from improve_core import correction_payload, diagnosis

OUT=ROOT/'results/improvements'
LABEL={'A':'A*','D':'Dynamic weighted A*','N':'No GPT (g+2h)','GPT':'GPT only','B':'Baseline LLM-A*',
       'E1':'E1: waypoint refinement','E2C':'E2: retry control','E2F':'E2: error feedback','E3':'E3: conditional call',
       'GPT_E2C':'GPT retry control','GPT_E2F':'GPT error feedback'}
ORDER=['A','D','N','GPT','B','E1','E2C','E2F','E3','GPT_E2C','GPT_E2F']
HYBRIDS=['B','E1','E2C','E2F','E3']
PROMPT={'standard':'Few-shot','cot':'CoT','repe':'RePE',None:'—'}

def gm(xs): return 100*math.exp(statistics.mean(math.log(x) for x in xs)) if xs else None
def summary(rows):
    bases={r['case_id']:r for r in rows if r['method']=='A'}
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['method'],r['prompt'])].append(r)
    answer=[]
    for method in ORDER:
        for kind in [None,'standard','cot','repe']:
            rs=groups.get((method,kind))
            if not rs: continue
            record={'method':method,'prompt':kind,'n':len(rs),'valid_n':sum(r['valid'] for r in rs),
                    'valid_ratio':100*sum(r['valid'] for r in rs)/len(rs)}
            for key in ('operation','storage','length'):
                eligible=[r for r in rs if r['valid'] and r.get(key) is not None and r[key]>0 and bases[r['case_id']]['valid'] and bases[r['case_id']][key]>0]
                record[key+'_ratio']=gm([r[key]/bases[r['case_id']][key] for r in eligible])
                record[key+'_n']=len(eligible)
            parsed=[r for r in rs if r.get('length') is not None and r['length']>0]
            record['length_all_parseable_ratio']=gm([r['length']/bases[r['case_id']]['length'] for r in parsed])
            record['length_all_parseable_n']=len(parsed)
            record['failures']=dict(collections.Counter(r.get('failure') for r in rs if not r['valid']))
            record['api_calls']=sum(r.get('api_calls',0) for r in rs) if method in HYBRIDS else 0
            record['call_rate']=sum(r.get('api_calls',0)>0 for r in rs)/len(rs) if method in HYBRIDS else 0
            answer.append(record)
    return answer

def verify_stage(stage):
    from improve_run import kinds,protocol_signature,text_of
    rows=[]; counts=collections.Counter(); ids=set(); requests=[]
    for q in cases_for(stage):
        for phase in ('e1','e2','e3'):
            p=OUT/stage/phase/f'{q["id"]}.json'
            record=json.loads(p.read_text(encoding='utf-8'))
            assert record['protocol_signature']==protocol_signature(), 'Protocol changed'
            assert record['case']==q, 'Case changed'
            for r in record['rows']:
                key=(r['case_id'],r['method'],r['prompt'])
                assert key not in ids, 'Duplicate evaluation'; ids.add(key)
                good,reason=validate(q,[tuple(p) for p in r['path']])
                assert good==r['valid'], 'Validity mismatch'
                if r['path']:
                    assert math.isclose(path_length(r['path']),r['length'],rel_tol=1e-12,abs_tol=1e-9), 'Length mismatch'
                if not r['method'].startswith('GPT'):
                    assert r['operation']>=1 and r['storage']>=1, 'Invalid counters'
                rows.append(r);counts[(r['method'],r['prompt'])]+=1
        for kind in kinds(stage):
            p=OUT/stage/'raw'/f'{q["id"]}-{kind}-first.json'
            first=json.loads(p.read_text(encoding='utf-8'))
            needs=not diagnosis(q,text_of(first))['valid']
            variants=['first','E2C','E2F'] if needs else ['first']
            for variant in variants:
                r=json.loads((OUT/stage/'raw'/f'{q["id"]}-{kind}-{variant}.json').read_text(encoding='utf-8'))
                expected=payload(q,kind) if variant=='first' else correction_payload(q,kind,text_of(first),variant=='E2F')
                assert r['case_id']==q['id'] and r['prompt_kind']==kind and r['variant']==variant
                assert r['payload_sha256']==digest(expected)==digest(r['payload']), 'Payload hash mismatch'
                assert r['response_sha256']==digest(r['response']), 'Response hash mismatch'
                assert r['response']['model']=='gpt-3.5-turbo-0125', 'Wrong model'
                requests.append(r)
    count=len(cases_for(stage)); wanted={('A',None),('N',None)}
    if stage!='scaling': wanted.add(('D',None))
    for kind in kinds(stage):
        for m in ['GPT','B','E1','E2C','E2F','E3','GPT_E2C','GPT_E2F']: wanted.add((m,kind))
    assert set(counts)==wanted and all(n==count for n in counts.values()), 'Missing condition'
    assert len(list((OUT/stage/'raw').glob('*.json')))==len(requests), 'Unexpected response files'
    usage={'requests':len(requests),'prompt_tokens':sum(r['response']['usage']['prompt_tokens'] for r in requests),
           'completion_tokens':sum(r['response']['usage']['completion_tokens'] for r in requests),
           'finish_reasons':dict(collections.Counter(r['response']['choices'][0]['finish_reason'] for r in requests)),
           'system_fingerprints':dict(collections.Counter(str(r['response'].get('system_fingerprint')) for r in requests))}
    return rows,{'complete':True,'cases':count,'rows':len(rows),'conditions':len(counts),'usage':usage}

def write_csv(path,records):
    fields=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def compare(rows):
    lookup={(r['case_id'],r['method'],r['prompt']):r for r in rows}; out=[]
    for kind in ['standard','cot','repe']:
        for treatment,control in [('E1','B'),('E1','N'),('E2F','B'),('E2F','E2C'),('E3','B'),('E3','N')]:
            pairs=[(r,lookup[(r['case_id'],control,None if control=='N' else kind)]) for r in rows if r['method']==treatment and r['prompt']==kind]
            rec={'prompt':kind,'treatment':treatment,'control':control,'n':len(pairs)}
            for metric in ('operation','storage','length'):
                eligible=[(a,b) for a,b in pairs if a['valid'] and b['valid']]
                rec[metric+'_paired_ratio']=gm([a[metric]/b[metric] for a,b in eligible]);rec[metric+'_paired_n']=len(eligible)
            rec['valid_delta_pp']=100*(sum(a['valid'] for a,b in pairs)-sum(b['valid'] for a,b in pairs))/len(pairs)
            rec['total_api_calls_treatment']=sum(a.get('api_calls',0) for a,b in pairs)
            rec['total_api_calls_control']=sum(b.get('api_calls',0) for a,b in pairs)
            out.append(rec)
    return out

def scale_growth(rows):
    lookup={(r['case_id'],r['method'],r['prompt']):r for r in rows}; groups=collections.defaultdict(list)
    for r in rows:
        if r['method'].startswith('GPT'):continue
        scale=int(r['case_id'].rsplit('-s',1)[1]); baseid=r['case_id'].rsplit('-s',1)[0]+'-s01'
        base=lookup[(baseid,r['method'],r['prompt'])]
        for metric in ('operation','storage'):
            groups[(scale,r['method'],metric)].append(r[metric]/base[metric])
    return [{'scale':s,'method':m,'metric':k,'growth':statistics.mean(v),'n':len(v)} for (s,m,k),v in sorted(groups.items())]

def number(x): return '—' if x is None else f'{x:.2f}'
def table(records,columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(number(r.get(key)) if isinstance(r.get(key),float) else str(r.get(key,'—')))+'</td>' for key,_ in columns)+'</tr>' for r in records)+'</tbody></table>'

def build():
    from improve_run import PROTOCOL,protocol_signature
    out=OUT/'report';out.mkdir(parents=True,exist_ok=True)
    allrows={};checks={};summaries={}
    for stage in ('pilot','main','scaling'):
        allrows[stage],checks[stage]=verify_stage(stage)
        summaries[stage]=summary(allrows[stage]);save(OUT/stage/'rows.json',allrows[stage])
        write_csv(out/f'{stage}_table.csv',summaries[stage])
    reference=json.loads((ROOT/'improvement_reference.json').read_text(encoding='utf-8'))
    for r in allrows['main']:
        if r['method'] in ('A','D','N'):
            ref=reference[r['case_id']][r['method']]
            for metric in ('operation','storage','length','valid'):
                assert math.isclose(r[metric],ref[metric],rel_tol=1e-12,abs_tol=1e-9), 'Legacy classical parity failed'
    overlap={q['id'] for q in cases_for('main') for kind in PROMPTS if
        f'Start Point: {q["start"]}\nGoal Point: {q["goal"]}\nHorizontal Barriers: {q["horizontal_barriers"]}\nVertical Barriers: {q["vertical_barriers"]}' in PROMPTS[kind]}
    assert len(overlap)==5
    without=summary([r for r in allrows['main'] if r['case_id'] not in overlap])
    write_csv(out/'main_without_demo_overlap.csv',without)
    comparisons=compare(allrows['main']);write_csv(out/'paired_comparisons.csv',comparisons)
    growth=scale_growth(allrows['scaling']);save(out/'scaling.json',growth)
    diagnostics={}
    for kind in ['standard','cot','repe']:
        e1=[r for r in allrows['main'] if r['method']=='E1' and r['prompt']==kind]
        e3=[r for r in allrows['main'] if r['method']=='E3' and r['prompt']==kind]
        e2=[r for r in allrows['main'] if r['method']=='E2F' and r['prompt']==kind]
        diagnostics[kind]={'e1_rejected':sum(not r['refinement']['accepted'] for r in e1),
            'e1_input_targets_total':sum(r['refinement']['input_targets'] for r in e1),
            'e1_retained_targets_total':sum(r['refinement']['retained_targets'] for r in e1),
            'e2_revised':sum(r['revised'] for r in e2),'e3_calls':sum(r['api_calls'] for r in e3),
            'e3_trigger_pops':[r['trigger']['pop'] for r in e3 if r['trigger']]}
    save(out/'summary.json',{'tables':summaries,'without_demo_overlap':without,'paired':comparisons,'diagnostics':diagnostics,'checks':checks})
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    colors=['#64748b','#0f766e','#b45309','#7c3aed','#2563eb']
    for ax,metric in zip(axes,['operation','storage','length']):
        for j,method in enumerate(HYBRIDS):
            vals=[next(r[metric+'_ratio'] for r in summaries['main'] if r['method']==method and r['prompt']==k) for k in ['standard','cot','repe']]
            ax.bar([i+(j-2)*.15 for i in range(3)],vals,width=.14,label=method,color=colors[j])
        ax.set_xticks(range(3),['Few-shot','CoT','RePE']);ax.set_title(metric.capitalize()+' / A* (%)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,metric in zip(axes,['operation','storage']):
        for m in ['A','N',*HYBRIDS]:
            pts=[r for r in growth if r['method']==m and r['metric']==metric]
            ax.plot([r['scale'] for r in pts],[r['growth'] for r in pts],label=m,marker='.',linewidth=1)
        ax.set_xlabel('Scale');ax.set_ylabel(metric+' growth / scale 1');ax.set_yscale('log');ax.legend(ncol=4,fontsize=8)
    fig.savefig(out/'scaling.png',dpi=170);fig.savefig(out/'scaling.pdf');plt.close(fig)
    main_display=[dict(r,method=LABEL[r['method']],prompt=PROMPT[r['prompt']]) for r in summaries['main']]
    cols=[('method','방법'),('prompt','프롬프트'),('operation_ratio','Operation % ↓'),('storage_ratio','Storage % ↓'),('length_ratio','Length % ↓'),('valid_ratio','Valid % ↑'),('valid_n','유효 수'),('n','전체 수'),('api_calls','알고리즘 호출 수')]
    cmpcols=[('prompt','프롬프트'),('treatment','실험군'),('control','대조군'),('operation_paired_ratio','연산량 비율 %'),('storage_paired_ratio','저장량 비율 %'),('length_paired_ratio','경로 길이 비율 %'),('valid_delta_pp','유효율 차이 %p')]
    conclusions=[]
    for r in comparisons:
        conclusions.append(f"{PROMPT[r['prompt']]}: {r['treatment']} / {r['control']} — 연산량 {number(r['operation_paired_ratio'])}%, 저장량 {number(r['storage_paired_ratio'])}%, 경로 길이 {number(r['length_paired_ratio'])}%, 유효율 차이 {number(r['valid_delta_pp'])}%p.")
    images=''.join('<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'">' for name in ['comparison','scaling'])
    env=json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM-A* GPT 개선 실험 결과</title><style>body{font:16px/1.7 "Malgun Gothic","Noto Sans KR",sans-serif;color:#172033;background:#f5f7fa;margin:0}main{max-width:1250px;margin:36px auto;padding:40px;background:white;border-radius:14px}h1{font-size:30px}h2{margin-top:38px;color:#164e63}table{border-collapse:collapse;width:100%;font-size:12px;margin:20px 0}td,th{border-bottom:1px solid #dce4eb;padding:9px 7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf2f6}img{max-width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px;background:#f1f5f9;padding:16px}.note{background:#edf8f4;border-left:4px solid #0f766e;padding:16px}@media print{body{background:white}main{margin:0;padding:0}table{font-size:9px}h2{break-after:avoid}tr{break-inside:avoid}}</style><main><h1>LLM-A* GPT 개선 실험 결과 보고서</h1>'''
    document+='<p>세 개선안을 독립적으로 비교했습니다. 기존 재현의 네 평가 지표를 유지하며, 최초 GPT 응답은 새로 수집하여 조건 간 공유했습니다.</p>'
    document+='<div class="note">전체 검증 완료: 본실험 1,000문제 × 27조건 = 27,000행. 개발 270행, 확장성 1,000행. 기존 Windows 재현의 A*·Dynamic Weighted A*·g+2h 지표 3,000건과 서버 결과가 일치했습니다.</div>'
    document+='<h2>1. 변인 통제와 실행 환경</h2><p>모델 gpt-3.5-turbo-0125, temperature=0, max_tokens=1000, 원문 Few-shot·CoT·RePE, 동일 지도·경계·충돌 판정·탐색 규칙을 유지했습니다. 사용자 요청으로 실행 호스트만 Windows에서 Linux GPU 서버로 이동했습니다. 모든 대조군을 같은 서버에서 실행했습니다. GPT 추론은 OpenAI API이며 GPU 가속으로 탐색 알고리즘을 변경하지 않았습니다.</p>'
    document+='<pre>'+html.escape(json.dumps(env,ensure_ascii=False,indent=2))+'</pre>'
    document+='<h2>2. 개선안과 대조군</h2><p>E1: 중복·반복 구간 제거, 직선 단축, 정제 경로 검증 실패 시 내부 경유점 불채택. E2F: 최초 경로가 유효하지 않을 때 오류 피드백으로 한 번 수정. E2C: 같은 호출 횟수의 비구체적 재검토 대조군. E3: g+2h로 시작하고 사전 고정한 정체 조건에서 한 번 호출. N: GPT 경유점 없는 g+2h. 세 개선안을 서로 결합하지 않았습니다.</p>'
    document+='<h2>3. 본실험 결과</h2><p>연산량·저장량·길이는 동일 문제 A* 대비 비율의 기하평균입니다. 유효한 양수 측정치만 집계하며 분모는 CSV에 기록했습니다. 유효율 분모는 항상 전체 문제입니다. GPT-only의 연산량·저장량은 정의하지 않습니다.</p>'+table(main_display,cols)
    document+='<h2>4. 개선 효과의 직접 비교</h2><p>비율 100% 미만은 해당 대조군보다 감소했다는 뜻입니다. 두 조건에서 모두 유효한 문제를 짝지어 집계했습니다. E2F 대 E2C가 구체적 오류 정보의 효과를 구분하며, E1/E3 대 N은 GPT 안내의 추가 기여를 확인합니다.</p>'+table(comparisons,cmpcols)
    document+='<ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in conclusions)+'</ul>'+images
    document+='<h2>5. 출력 품질과 API 사용량</h2><pre>'+html.escape(json.dumps({'diagnostics':diagnostics,'usage':{s:v['usage'] for s,v in checks.items()}},ensure_ascii=False,indent=2))+'</pre>'
    document+='<h2>6. 해석의 범위</h2><p>E1과 E3의 성능이 N과 비슷하다면 GPT 경유점의 적극적 활용 효과로 해석하지 않습니다. E2F가 GPT 자체 유효율을 높여도 최종 탐색 성능 향상을 별도로 확인해야 합니다. API 호출 수는 알고리즘별 논리적 사용량이며, 응답 공유로 실제 실험 수집 요청 수와 다릅니다. 통신 지연·추가 검증·OPEN 재계산 비용은 논문의 heap-pop 지표에 포함되지 않습니다. 캐시와 공유 응답 때문에 본 보고서로 실제 지연시간 우위를 주장하지 않습니다.</p>'
    document+='<p>기존 예시 중복 5개를 모든 조건에서 공통 제외한 995문제 결과는 별도 CSV로 제공했습니다. 확장성은 기존 GPT 변경 실험과 동일한 시드의 10문제, 1~10배, Few-shot만 평가했습니다. 원문의 Llama 확장성 실험과 동일하지 않습니다. 이미 재현 결과를 관찰한 데이터에서 제안된 방법이므로 독립적인 미관측 데이터 일반화 증거는 아닙니다.</p>'
    document+='<h2>7. 재현성과 원자료</h2><p>요청·응답 해시, 원문 응답, 전체 경로, 프로토콜·코드 해시, 실패 이유, 모델 반환명, finish_reason, API 사용 토큰, 서버 환경 및 검증 기록을 보존했습니다. 실패 출력은 재생성하거나 좋은 결과로 교체하지 않았습니다.</p><p><a href="https://arxiv.org/html/2407.02511v2">원본 논문</a></p></main></html>'
    (out/'report.html').write_text(document,encoding='utf-8')
    verification={'complete':True,'protocol_signature':protocol_signature(),'checks':checks,
        'legacy_classical_parity_cases':3000,'demo_overlap_excluded':sorted(overlap),
        'total_rows':sum(v['rows'] for v in checks.values()),'total_requests':sum(v['usage']['requests'] for v in checks.values()),
        'files_sha256':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}}
    save(OUT/'verification.json',verification)
    archive=ROOT/'improvement-results-verified.zip'
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['run.log','runner.pid','status.json']]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(ROOT))
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
    save(OUT/'export.json',{'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)})
    print(json.dumps({'verified':True,'total_rows':verification['total_rows'],'total_requests':verification['total_requests']}),flush=True)

if __name__=='__main__': build()
