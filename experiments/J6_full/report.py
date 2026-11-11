"""Paired full/lazy OPEN-update analysis with case-weighted map bootstrap."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st

import numpy as np

from .run import METHODS,SEED,load
from .adapter import GUIDED
from experiments.COMMON18.prepare import save
from experiments.K3.run import sha

METRICS=('local_seconds','api_seconds','reconstructed_total_seconds','operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
UPDATES=('rekey_evaluations','target_changes','stale_pops','lazy_refreshes','heap_pushes','relax_attempts','peak_heap_entries','bound_shift_units')
REFERENCES=('LLMA','Octile','WallAstar','J6_full')


def rows(out):
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def paired(a,b,metric):
    logs=[];clusters=defaultdict(list);better=same=worse=0
    for cid,r in a.items():
        x,y=r.get(metric),b[cid].get(metric)
        if x is None or y is None:continue
        z=math.log(max(x,1e-12)/max(y,1e-12));logs.append(z);clusters[r['map_index']].append(z)
        if math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12):same+=1
        elif x<y:better+=1
        else:worse+=1
    if not logs:return None
    # Unequal changed/unchanged cases per map retain their case weights.
    sums=np.array([sum(v) for _,v in sorted(clusters.items())]);sizes=np.array([len(v) for _,v in sorted(clusters.items())])
    rng=np.random.default_rng(SEED);draw=rng.integers(0,len(sums),size=(2000,len(sums)))
    boot=np.exp(sums[draw].sum(axis=1)/sizes[draw].sum(axis=1))
    return dict(geometric_ratio=math.exp(st.mean(logs)),ci95=np.quantile(boot,[.025,.975]).tolist(),
        improved=better,same=same,worsened=worse,cases=len(logs),maps=len(sums))


def aggregate(records):
    by={m:{r['case_id']:r for r in records if r['method']==m} for m in METHODS};a={}
    for m,cs in by.items():
        rs=list(cs.values());valid=[r for r in rs if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        a[m]=dict(cases=len(rs),valid=len(valid),valid_percent=100*len(valid)/len(rs),bound=rs[0]['bound'],
            means={k:st.mean(r[k] for r in rs) for k in METRICS},updates={k:st.mean(r.get(k,0) for r in rs) for k in UPDATES},
            sums={k:sum(r.get(k,0) for r in rs) for k in ('operation',)+UPDATES},
            guide_parse_error_cases=sum(bool(r.get('guide_parse_error')) for r in rs),
            target_changed_cases=sum(r.get('target_changes',0)>0 for r in rs),
            exact_percent=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs),
            stretch_gm=math.exp(st.mean(map(math.log,ratios))) if ratios else None,
            stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,stretch_max=max(ratios,default=None))
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','reconstructed_total_seconds','operation','successor_expansions','length')} for m in METHODS} for b in REFERENCES}
    return dict(absolute=a,comparisons=comparisons)


def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    axes[0].barh(METHODS,[1000*a[m]['means']['local_seconds'] for m in METHODS],color=['#89939b']*5+['#237baa','#26957a'])
    axes[0].set_xlabel('평균 로컬 ms');axes[0].set_title('full/lazy 모두 동일 선택열 기록 포함')
    labels=['모든 score 평가','모든 heap pop','lazy refresh pop'];keys=['rekey_evaluations','operation','lazy_refreshes'];yy=np.arange(3)
    for m,off,color in [('J6_full',-.18,'#237baa'),('J6',.18,'#26957a')]:
        vals=[a[m]['sums'][k] for k in keys];axes[1].barh(yy+off,vals,height=.34,label=m,color=color)
    axes[1].set_yticks(yy,labels);axes[1].set_xlabel('1,000문제의 작업별 총횟수');axes[1].set_title('서로 다른 작업의 횟수 — 합산하지 않음');axes[1].legend()
    fig.suptitle('J6 · 공개 1,000문제 · 조건별 1회 · 같은 경로/유효 선택열 검증')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    g=s['groups']['all1000'];a=g['absolute'];full=a['J6_full'];lazy=a['J6'];c=g['comparisons']['J6_full']['J6']['local_seconds'];lo,hi=c['ci95']
    verdict='이번1회 측정에서 명확한 로컬 시간 이득은 확인하지 못했다.' if lo<=1<=hi else '이번1회 측정에서 로컬 시간 감소가 관측됐다.' if hi<1 else '이번1회 측정에서 로컬 시간이 증가했다.'
    reduction=100*(1-lazy['sums']['rekey_evaluations']/full['sums']['rekey_evaluations'])
    names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='g+2h',Octile='단일 clean octile A*',WallAstar='단일 정확 벽 하한 A*',J6_full='동일 엔진 전체 재계산',J6='동일 엔진 지연 갱신')
    lines=['# J6 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개100지도×10쌍=1,000문제, 7조건×1회=7,000회. 기존 JY full/lazy 구현을 변경 없이 평가했다.','',
        '## 결과 판단','',f'**{verdict}** 지연 갱신/full의 문제별 로컬 시간 GM 비율은 {c["geometric_ratio"]:.5f}, '
        f'지도 bootstrap95% 구간은 [{lo:.5f}, {hi:.5f}]다. 전체 재계산과 지연 갱신의 유효 선택열·경로·확장은1,000쌍 모두 같았다.','',
        f'총 score 평가 횟수는 {full["sums"]["rekey_evaluations"]:,}→{lazy["sums"]["rekey_evaluations"]:,}로 {reduction:.2f}% 감소했지만, '
        f'lazy refresh pop이 {lazy["sums"]["lazy_refreshes"]:,}회 추가됐다. '
        '우선순위 계산 감소와 큐 관리 증가를 실제 시간과 함께 판단한다.','',
        '두 방식 모두 같은 저장 경유점·parser·1e-9 정수 키·동률·closed 규칙을 쓴다. '
        '원 LLMA와는 stale-g 처리와 우선순위 표현도 달라 원 LLMA 대비 전체 차이를 J6 갱신 최적화 하나의 효과로 돌릴 수 없다.',
        'full/lazy 둘 다 동일 trace append를 타이머 안에서 수행했다. 같은 선택열의 계측 비용도 포함한 실행이며 미계측 production 성능이 아니다.','',
        '## 전체 1,000문제','',
        '| 조건 | 유효율 | 로컬 평균 ms | 재구성 총평균 s | 길이/최단 GM | 최단 일치율 | 모든 heap pop 평균 | 확장 평균 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];b='최단' if v['bound']==1 else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.6f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% | '
            f'{v["means"]["operation"]:.2f} | {v["means"]["successor_expansions"]:.2f} | {b} |')
    lines+=['','J6 계열은 최단성 보장이 없다. 같은 경로를 반환하는 것은 full/lazy 동등성이고 최단 경로 보장과는 다르다.','',
        '## 갱신 작업과 큐 비용','',
        '| 조건 | score 평가 합 | target 변경 합 | stale pop 합 | lazy refresh 합 | 모든 heap pop 합 | heappush 합 | 논리 저장 상태 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in GUIDED:
        v=a[m];p=v['sums'];lines.append(f'| {m} | {p["rekey_evaluations"]:,} | {p["target_changes"]:,} | {p["stale_pops"]:,} | '
            f'{p["lazy_refreshes"]:,} | {p["operation"]:,} | {p["heap_pushes"]:,} | {v["means"]["storage"]:.2f} |')
    lines+=['','score 평가는 초기 삽입·재삽입까지 포함한다. heappush는실제 해당 함수 호출만 세며 full list 재구성/heapify 항목 전체를 삽입으로 세지 않는다. '
        'full의 rebuild 비용은 시간과 score 평가에 반영한다. 서로 다른 카운터를 더해 총 CPU 연산이라고 표현하지 않는다.',
        'trace는 stale/refresh를 제외한 유효 상태 선택열이다. lazy에 추가 pop이 있어도 이 선택열과 경로는 full과 일치한다.','',
        '## target 변경 여부에 따른 기술적 부분집합','',
        '| 부분집합 | 문제 수 | full 평균 ms | lazy 평균 ms | lazy/full 로컬 GM [95% CI] |',
        '|---|---:|---:|---:|---|']
    for name in ('target_changed','target_unchanged'):
        v=s['groups'][name];p=v['comparisons']['J6_full']['J6']['local_seconds'];x,y=p['ci95'];fa=v['absolute']['J6_full'];la=v['absolute']['J6']
        lines.append(f'| {name} | {fa["cases"]} | {1000*fa["means"]["local_seconds"]:.3f} | {1000*la["means"]["local_seconds"]:.3f} | '
            f'{p["geometric_ratio"]:.5f} [{x:.5f}, {y:.5f}] |')
    lines+=['','변경 여부는 고정 응답·탐색 동작에 따른 기술적 분류다. 전체 결과와 두 부분집합을 함께 제시하며 고유 문제 수를 늘리지 않는다. '
        '변경0인 문제에서는두 방식의 작업 카운터가 같은지도 검산했다. 지도마다 부분집합 크기가 다르므로 bootstrap은 지도별 합·사례 수를 같이 재표집해 사례 가중을 유지했다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 로컬 평균 ms | 모든 pop 평균 | 최단 일치율 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m];lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | {v["means"]["operation"]:.2f} | {v["exact_percent"]:.1f}% |')
    lines+=['','이200문제는 전체의 부분집합이다. 이전 측정과 합산하지 않았다.','',
        '## 실행 범위와 제한','',
        '- 문제·조건별1회, CPU/native1스레드, seed20261111로 순서만 섞었다. 시간 반복 안정성을 확인한 결과가 아니다.',
        '- 저장 standard1,000응답을 요청·입력·SHA와 확인 후 고정했다. 잘린 응답도 제외하지 않았다. 새 API 호출0회, 실행 중 네트워크/API 연결을 차단했다.',
        '- LLMA/full/lazy는각각 같은 응답을 논리1회씩 사용한다. 과거 지연·토큰도 같은 값으로 부과한다.',
        '- 총시간은 과거 API 지연+현재 로컬의 재구성 추정이며 현재 온라인 서비스 반복 실측이 아니다.',
        '- 파싱·준비·모든 갱신/큐 관리·검색·선택열 기록·복원·필수 검증은 로컬 안이다. 파일 읽기·독립 Oracle/Dijkstra·trace 해시/대조는 밖이다.',
        '- 새 경유점을 생성하지 않고 정적 목록 안에서 현재 target를 바꾸는 기존 규칙이다. 동적 생성이나 재호출 실험을 재개하지 않았다.',
        '- 공개 자료는 기존 연구에 노출됐다. 지도 bootstrap2000회 구간은 구성의 탐색적 불확실성이며 시간/API 반복 변동을 포함하지 않는다.',
        '- 논리 storage와 heap peak를 실제 bytes로 표현하지 않으며 별도 RSS 측정은 없다. 기존 J1~J5/JY 결과의 절대 시간과 합산 순위를 만들지 않는다.',
        '', '![J6 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계](summary.json), [유효 선택열 포함 원시 시행](evaluations.jsonl.gz), '
        '[실행 기록](protocol.json), [강제 target 검사](unit_tests.json), [독립 검산](audit.json).','',
        '준연 담당 J6의 기존 JY 구현을 재평가했다. Codex가 실행 통합·검증·문서 작성을 지원했고 기존 결과는 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];rs=rows(out);test=set(load(out/'protocol.json')['common18_test_maps'])
    changed={r['case_id'] for r in rs if r['method']=='J6_full' and r['target_changes']>0}
    unchanged={r['case_id'] for r in rs if r['method']=='J6_full' and r['target_changes']==0}
    parts=dict(all1000=rs,common18_test200=[r for r in rs if r['map_index'] in test],
        target_changed=[r for r in rs if r['case_id'] in changed],target_unchanged=[r for r in rs if r['case_id'] in unchanged])
    s=dict(groups={k:aggregate(v) for k,v in parts.items()},report_source_sha256=sha(Path(__file__)),repeats=1,new_api_calls=0,
        bootstrap='sample maps with replacement, preserve per-map log sums and case counts')
    save(out/'summary.json',s);keys=('case_id','map_index','method','valid','length','optimum','bound')+METRICS+UPDATES+('effective_trace_sha256',)
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in rs)
    base={r['case_id']:r for r in rs if r['method']=='J6_full'}
    worst=sorted([r for r in rs if r['method']=='J6'],key=lambda r:r['local_seconds']/base[r['case_id']]['local_seconds'],reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r.get(k) for k in keys} for r in worst]);plot(out,s);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('LLMA','J6_full','J6')},ensure_ascii=False))


if __name__=='__main__':main()
