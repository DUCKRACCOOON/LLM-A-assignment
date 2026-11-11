"""Map-cluster intervals, raw row exports, verification and Korean report."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from core import ROOT
from next_search_core import audit_path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def comparison(rows,method,baseline,metric='total_seconds',level=.95):
    a={r['id']:r for r in rows if r['method']==method};b={r['id']:r for r in rows if r['method']==baseline}
    bymap=defaultdict(list);family={};failed=0
    for key,r in a.items():
        if key not in b:continue
        s=b[key]
        if r['status']!='ok' or s['status']!='ok':failed+=1;continue
        x=r[metric] if metric=='total_seconds' else r['counters'][metric]
        y=s[metric] if metric=='total_seconds' else s['counters'][metric]
        bymap[r['map_id']].append(math.log(max(x,1e-12)/max(y,1e-12)));family[r['map_id']]=r['family']
    if not bymap:return dict(method=method,baseline=baseline,metric=metric,failed_pairs=failed,ratio=None)
    vals={m:statistics.mean(v) for m,v in bymap.items()};groups=defaultdict(list)
    for m,v in vals.items():groups[family[m]].append(v)
    rng=np.random.default_rng(26101421);boots=np.zeros(10000)
    for vs in groups.values():
        z=np.asarray(vs);boots+=rng.choice(z,size=(10000,len(z)),replace=True).sum(axis=1)
    boots/=len(vals);alpha=(1-level)/2
    return dict(method=method,baseline=baseline,metric=metric,ratio=math.exp(statistics.mean(vals.values())),
        low=float(np.exp(np.quantile(boots,alpha))),high=float(np.exp(np.quantile(boots,1-alpha))),level=level,
        maps=len(vals),pairs=sum(map(len,bymap.values())),failed_pairs=failed,conditional_on_success=failed>0)


def build(out):
    out=Path(out);sel=read(out/'selection.json');frozen=read(out/'protocol_frozen.json')
    rows=[read(p) for p in sorted((out/'evaluation').glob('*/*/*.json'))]
    datasets={}
    for p in (out/'data').glob('*.json'):
        if p.stem.endswith('_manifest'):continue
        value=read(p)
        if isinstance(value,list):datasets.update({q['id']:q for q in value})
    violations=[];verified=0;stages=defaultdict(lambda:defaultdict(int))
    for r in rows:
        stages[r['stage']][r['method']]+=1
        if r['status']!='ok':continue
        try:audit_path(datasets[r['id']],r,r['optimal'] if r['optimal'] is not None else math.inf,r['bound'] or math.inf)
        except Exception as e:violations.append(dict(id=r['id'],method=r['method'],error=str(e)))
        verified+=1
    assert not violations,violations[:5]
    verification=dict(verified=True,total_rows=len(rows),successful_rows=verified,failures=len(rows)-verified,stages=dict(stages),violations=violations,
         final_generated_after_freeze=True,server_preflight=read(out/'server_preflight.json'))
    save(out/'verification.json',verification)
    comparisons=[];main=[r for r in rows if r['stage']=='final']
    for method in sorted(set(r['method'] for r in main)):
        for base in ('A_OCT','N_CLEAN'):
            for metric in ('total_seconds','expansions'):comparisons.append(dict(stage='final',**comparison(main,method,base,metric)))
    comparisons.append(dict(stage='final',**comparison(main,'R1_R2','R1',level=.9875)))
    for method in ('R1','R2'):comparisons.append(dict(stage='final',**comparison(main,method,'A_OCT',level=.9875)))
    for stage in ('r3_development','r3_validation','r3_final'):
        rs=[r for r in rows if r['stage']==stage];ms=sorted(set(r['method'] for r in rs))
        for method in ms:
            if not method.startswith('TOPO'):continue
            weight=method.split(':')[2];base=f'TOPO:rule:{weight}:0'
            for metric in ('total_seconds','expansions'):
                if base in ms:comparisons.append(dict(stage=stage,**comparison(rs,method,base,metric,.9875 if stage=='r3_final' and method=='TOPO:llm:1.02:0' and metric=='total_seconds' else .95)))
    summary=[]
    for method in sorted(set(r['method'] for r in main)):
        rs=[r for r in main if r['method']==method];ok=[r for r in rs if r['status']=='ok'];ratios=[r['length']/r['optimal'] for r in ok]
        item=dict(method=method,n=len(rs),success=len(ok),exact_fraction=sum(abs(r['length']-r['optimal'])<=1e-8+1e-10*r['optimal'] for r in ok)/len(rs),
                  mean_excess_percent=100*statistics.mean(v-1 for v in ratios),p95_excess_percent=100*float(np.quantile(ratios,.95)-1),max_length_ratio=max(ratios))
        for metric,label in [('total_seconds','time'),('expansions','expansions')]:
            item[label]=comparison(main,method,'A_OCT',metric)
        summary.append(item)
    save(out/'summary.json',dict(summary=summary,comparisons=comparisons,selection=sel))
    raw=[read(p) for p in (out/'raw').glob('*/*.json') if p.name!='runtime.json']
    assert len(raw)<=700
    budgets=dict(total_generations=len(raw),paid_api_calls=0,valid_outputs=sum(r.get('valid',False) for r in raw),
                 inference_seconds=sum(r.get('seconds',0) for r in raw),input_tokens=sum(r.get('input_tokens',0) for r in raw),output_tokens=sum(r.get('output_tokens',0) for r in raw))
    save(out/'actual_budget.json',budgets)
    with (out/'per_case.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'timings.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:
            for i,t in enumerate(r['timings']):f.write(json.dumps(dict(id=r['id'],method=r['method'],stage=r['stage'],repeat=i,**t))+'\n')
    def fmt(x):return f"{x:.3f}" if x is not None else 'N/A'
    table=['| 방법 | 문제 수 | 최단 길이 일치 | A* 대비 확장 수 | A* 대비 전체 시간 | P95 초과 길이 |','|---|---:|---:|---:|---:|---:|']
    for r in summary:table.append(f"| {r['method']} | {r['n']} | {r['exact_fraction']:.1%} | {fmt(r['expansions']['ratio'])}배 | {fmt(r['time']['ratio'])}배 | {r['p95_excess_percent']:.2f}% |")
    text=f'''# 세 가지 추가 개선 실험 결과

Sirius GPU 서버 · 계획 next-search-20261014-v1 · 최종 지도 {frozen['final_maps']}개 · 결과 {len(rows):,}행

R1 공간 압축, R2 장애물 단계적 반영, R3 위상 기반 안내를 개발·검증 순서로 수행하고 설정을 고정한 뒤 새 지도에서 평가했다. 탐색은 CPU, Llama 3 8B 추론은 GPU 0을 사용했다. 새 유료 API 호출은 0건이다.

## 주 평가

비율은 같은 문제의 대응 비율을 지도 단위로 집계한 기하평균이며 낮을수록 좋다. 전체 시간에는 해당 방법의 준비·필수 검증·복원을 포함한다. N_LEGACY의 확장 수는 중복을 포함한 원래 pop 처리 횟수이며 새 탐색기의 heap pop과 혼동하지 않는다.

{chr(10).join(table)}

## 개발과 검증에 따른 판단

선택한 R1 변형은 {sel['r1']}, R2 초기화는 {sel['r2']}, R3 후보 수는 {sel['k']}다. 실용 목표의 검증 통과 여부는 R1={sel['r1_gate']}, R2={sel['r2_gate']}, 결합={sel['combination_gate']}, R3 LLM 확장수 기준={sel['r3_gate']}다.

목표를 통과하지 못한 정확 후보도 최종 지도에서 진단 목적으로 측정했으며, 그 결과를 사전 통과한 확증 후보인 것처럼 승격하지 않는다. R3가 규칙 대비 확장 수 개선 기준을 못 넘으면 최종 400문제의 추가 LLM 생성을 중단하고 규칙·무작위 안내만 평가한다. 이 경우 LLM의 최종 미관측 지도 성능을 주장하지 않는다.

R3 개발의 LLM/RULE 비율은 확장 수 {sel['r3_development']['expansions']:.3f}, 추론 포함 시간 {sel['r3_development']['time']:.3f}다. 검증의 비율은 확장 수 {sel['r3_validation']['expansions']:.3f}, 시간 {sel['r3_validation']['time']:.3f}다. 후보 생성 비용도 모두 포함했다. 출입구 순서의 유한한 후보를 사용한 것으로, 엄밀한 homotopy 분류의 완전성을 주장하지 않는다.

## 검증과 반복 측정

독립 선분 교차 그래프와 Dijkstra로 정답을 만들고 각 지도의 간선 표본을 원래 스칼라 충돌 판정과 대조했다. 최종 저장 경로를 원래 충돌 규칙·정수 단위 이동·길이 상한으로 다시 검증했다. 오류 {len(violations)}건, 실행 실패 {len(rows)-verified}건이다. 작은 무작위 문제 2,000개와 도달 불가능 사례는 server_preflight.json에 별도 기록했다.

시간은 동일 CPU 설정에서 방법 순서를 섞은 5회 중앙값이다. 모델 응답은 문제당 한 번 생성해 w 조건에 공유하고 실제 생성 시간을 온라인 비용에 포함했다. RSS는 5ms 간격 표본 최대값이므로 순간적인 할당 최대값과 다를 수 있다. 실제 모델 생성 {len(raw)}회, 총 생성 토큰 {budgets['output_tokens']:,}개다.

## 추가 평가와 자료

scaling 폴더는 동일 원형 지도의 1·2·4·8배 확대 결과다. reuse 폴더는 목표 변경·고정 및 1·5·10개 질의를 구분한 실제 누적 비용이다. 같은 목표 조건에는 FMM과 역방향 Dijkstra도 포함했다. 반복 질의 결과의 지도 수와 확대된 조건 수를 독립 표본으로 더하지 않는다.

summary.json에는 지도 bootstrap 10,000회의 대응 신뢰구간이 있으며, 주요 네 비교의 시간 구간은 다중성을 반영한 98.75% 구간으로 구분한다. 실패가 있으면 성공 문제에 조건부인 요약임을 표시하고 전체 우위로 결론 내리지 않는다.

프로토콜과 세부 설정은 protocol_frozen.json, 원자료는 per_case.jsonl·timings.jsonl, 예산은 actual_budget.json, 검증은 verification.json에 있다. 프로그램·프롬프트·원응답과 모델 revision도 보존했다.
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(verified=True,rows=len(rows),generations=len(raw),report=str(out/'report.md'))))
