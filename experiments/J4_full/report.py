"""Cold ALT costs and explicitly reconstructed reuse of one map preparation."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics as st

import numpy as np

from .run import METHODS,load
from experiments.J1_full.report import paired,gm
from experiments.COMMON18.prepare import save
from experiments.K3.run import sha

METRICS=('local_seconds','preprocessing_seconds','prepared_query_seconds','api_seconds','reconstructed_total_seconds',
    'operation','query_operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
PREP=('preprocessing_pops','preprocessing_expansions','landmark_entries','graph_nodes','graph_edges')
REFERENCES=('LLMA','Octile','WallAstar','GraphOctile')
INDEXED=('GraphOctile','J4','J4_alt8')
REUSE=('Octile','WallAstar')+INDEXED


def rows(out):
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def aggregate(records):
    by={m:{r['case_id']:r for r in records if r['method']==m} for m in METHODS};absolute={}
    for m,cs in by.items():
        rs=list(cs.values());valid=[r for r in rs if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        absolute[m]=dict(cases=len(rs),valid=len(valid),valid_percent=100*len(valid)/len(rs),
            means={k:st.mean(r[k] for r in rs) for k in METRICS},bound=rs[0]['bound'],
            exact_percent=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs),
            stretch_gm=gm(ratios),stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,
            stretch_max=max(ratios,default=None),prep={k:st.mean(r.get(k,0) for r in rs) for k in PREP})
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','prepared_query_seconds','operation','query_operation','successor_expansions','length')} for m in METHODS} for b in REFERENCES}
    return dict(absolute=absolute,comparisons=comparisons)


def reuse(records):
    lookup={(r['map_index'],r['pair_index'],r['method']):r for r in records};maps=sorted({r['map_index'] for r in records});result={}
    for q in (1,5,10):
        costs={};details={}
        for m in REUSE:
            costs[m]={};prep=query=0.
            for mi in maps:
                rs=[lookup[mi,pi,m] for pi in range(q)]
                p=rs[0]['preprocessing_seconds'];t=sum(r['prepared_query_seconds'] for r in rs)
                prep+=p;query+=t;costs[m][str(mi)]=dict(map_index=mi,cost=p+t)
            details[m]=dict(maps=len(maps),queries_per_map=q,preparation_seconds=prep,query_seconds=query,
                total_seconds=prep+query,amortized_seconds_per_query=(prep+query)/(len(maps)*q))
        for m in REUSE:
            d=details[m];d['sum_ratio_to_octile']=d['total_seconds']/details['Octile']['total_seconds']
            d['paired_map_ratio_to_octile']=paired(costs[m],costs['Octile'],'cost')
        result[str(q)]=details
    return result


def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    yy=np.arange(len(METHODS));axes[0].scatter([1000*a[m]['means']['local_seconds'] for m in METHODS],yy,label='전체 로컬',color='#237baa')
    axes[0].scatter([1000*a[m]['means']['prepared_query_seconds'] for m in METHODS],yy,label='전처리 제외',marker='x',color='#d98030')
    axes[0].set_yticks(yy,METHODS);axes[0].set_xscale('log');axes[0].set_xlabel('평균 ms (로그 축)');axes[0].legend();axes[0].set_title('실제 매 질의 전처리 포함·제외')
    vals=[s['reuse']['all1000']['10'][m]['sum_ratio_to_octile'] for m in INDEXED]
    axes[1].barh(INDEXED,vals,color=['#89939b','#237baa','#26957a']);axes[1].axvline(1,color='black',lw=1)
    for i,v in enumerate(vals):axes[1].text(v,i,f' {v:.3f}',va='center')
    axes[1].set_xlim(0,max(1.1,max(vals)*1.18))
    axes[1].set_xlabel('합산 비용 / 단일 octile A* 합산 비용');axes[1].set_title('같은 지도 10질의·전처리1회 합산 추정')
    fig.suptitle('J4 · 공개 1,000문제 · 조건별 1회 · 재사용은 별도 시나리오')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    g=s['groups']['all1000'];a=g['absolute'];j=a['J4'];base=a['Octile'];cold=j['means']['local_seconds']/base['means']['local_seconds']
    names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='g+2h',Octile='단일 clean octile A*',WallAstar='단일 정확 벽 하한 A*',
        GraphOctile='전체 그래프+octile(랜드마크0)',J4='ALT4',J4_alt8='ALT8')
    verdict='이번 조건에서 전처리 포함 단일 질의의 속도 개선은 확인하지 못했다.' if cold>=1 else '이번 조건에서 전처리 포함 단일 질의의 평균 시간 감소가 관측됐다.'
    lines=['# J4 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개100지도×10쌍=1,000문제, 8조건×1회=8,000회. J4 ALT를 평가했다.',
        '기존 JY LandmarkIndex와 정확 검색을 변경 없이 사용했다. GraphOctile/ALT4/ALT8 전처리를 각 질의에서 실제로 새로 만들었다.','',
        '## 결과 판단','',
        f'**{verdict}** 아래 재사용 시나리오까지 구분해 해석한다.','',
        f'ALT4의 단일 질의 전체 로컬 평균은 {1000*j["means"]["local_seconds"]:.3f}ms이며 단일 octile A*의 **{cold:.3f}배**다. '
        f'전처리를 제외한 나머지 질의 평균은 {1000*j["means"]["prepared_query_seconds"]:.3f}ms지만 '
        f'전처리 평균 {1000*j["means"]["preprocessing_seconds"]:.3f}ms도 실제 비용이다. '
        '검색 절감과 전처리 회수 여부를 구분한다.','',
        f'같은 지도 첫10질의에 전처리를1회만 합산하는 시나리오에서 ALT4/단일 octile A* 비용합 비율은 '
        f'**{s["reuse"]["all1000"]["10"]["J4"]["sum_ratio_to_octile"]:.4f}**다. '
        '이는 아래 정의로 재구성한 비용이며 한 index를 상주시켜 연속 운영한 실측은 아니다.','',
        'J4는 온라인 LLM을 쓰지 않는 정확 하한 강화다. 원 LLMA 대비 API 대기가 사라지는 효과를 LLM 안내의 개선으로 해석하지 않는다. '
        '그와 별도로 같은 품질·같은 검색 엔진의 단일 정확 탐색 및 GraphOctile와 비교한다.','',
        '## 전체 1,000문제 — 실제 새 전처리','',
        '| 조건 | 유효율 | 전처리 평균 ms | 준비 후 평균 ms | 전체 로컬 평균 ms | 재구성 총평균 s | 최단 일치율 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];b='최단' if v['bound']==1 else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["preprocessing_seconds"]:.3f} | '
            f'{1000*v["means"]["prepared_query_seconds"]:.3f} | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.5f} | {v["exact_percent"]:.1f}% | {b} |')
    lines+=['','준비 후 시간은 전체 로컬에서 실제 전처리만 뺀 값으로, 검색·복원·wrapper 필수 검증도 포함한다. '
        '전처리 없는 방법은 전체 로컬과 같다. 정확 조건의 길이/최단 GM과 최대 초과율은 summary.json에 따로 남겼다.','',
        '## 대응 비교','',
        '| 개입 / 대조 | 전체 로컬 GM 비율 [95% CI] | 준비 후 GM 비율 | 총 heap pop GM 비율 | query 확장 GM 비율 |',
        '|---|---|---:|---:|---:|']
    for m,b in [('J4','Octile'),('J4_alt8','Octile'),('J4','WallAstar'),('J4','GraphOctile'),('J4_alt8','GraphOctile')]:
        c=g['comparisons'][b][m];t=c['local_seconds'];lo,hi=t['ci95']
        lines.append(f'| {m} / {b} | {t["geometric_ratio"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{c["prepared_query_seconds"]["geometric_ratio"]:.4f} | {c["operation"]["geometric_ratio"]:.4f} | {c["successor_expansions"]["geometric_ratio"]:.4f} |')
    lines+=['','비율은 1 미만이면 해당 비용이 감소한다. GraphOctile는 같은 전체 그래프를 만들고 거리표만 제외한다. '
        'GraphOctile와 on-demand Octile는같은 확장 순서/경로를 검산하지만 graph 준비·이웃 접근 비용은 다르다.','',
        '## 전처리와 검색 작업','',
        '| 조건 | 준비 heap pop 평균 | query pop 평균 | 준비 확장 평균 | query 확장 평균 | 거리표 항목 평균 | graph 정점/간선 평균 | query 저장 상태 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in INDEXED:
        v=a[m];p=v['prep'];lines.append(f'| {m} | {p["preprocessing_pops"]:.2f} | {v["means"]["query_operation"]:.2f} | '
            f'{p["preprocessing_expansions"]:.2f} | {v["means"]["successor_expansions"]:.2f} | {p["landmark_entries"]:.2f} | '
            f'{p["graph_nodes"]:.2f}/{p["graph_edges"]:.2f} | {v["means"]["storage"]:.2f} |')
    lines+=['','총 operation은 준비 heap pop+query pop이다. 준비 heap pop이 0인 GraphOctile도 전체 graph 생성 비용이 있다. '
        '거리표 조회·휴리스틱 계산·graph 생성 비용을 heap pop이 모두 설명하지 않으며, 위 항목 개수는 실제 메모리 bytes가 아니다.','',
        '## 전처리1회 재사용 합산 시나리오','',
        '| 같은 지도 첫 Q쌍 | GraphOctile / Octile 비용합 | ALT4 / Octile 비용합 | ALT8 / Octile 비용합 |',
        '|---|---:|---:|---:|']
    for q in ('1','5','10'):
        v=s['reuse']['all1000'][q];lines.append(f'| {q} | {v["GraphOctile"]["sum_ratio_to_octile"]:.4f} | {v["J4"]["sum_ratio_to_octile"]:.4f} | {v["J4_alt8"]["sum_ratio_to_octile"]:.4f} |')
    lines+=['','공식은 지도별 첫 질의 전처리1회+고정된 첫Q쌍의 준비 후 비용 합이다. 표는100지도의 비용 합끼리 나눈 산술 합산 비율이다. '
        '지도별 대응 기하평균과 bootstrap 구간은 summary.json에 함께 보존했다.',
        '물리적으로는 모든 질의에서 새 index를 생성했다. 같은 지도에서 index 내용이 deterministic하다는 전제와 랜드마크·크기·작업 카운터 일치를 확인해 재사용 비용을 재구성했다. '
        '상주 index의 연속 운영·캐시·파일 로딩 비용을 별도로 실측한 결과가 아니다. 10쌍을 초과하는 손익분기는 추정하지 않는다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 전체 로컬 평균 ms | 준비 후 평균 ms | 최단 일치율 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m];lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{1000*v["means"]["prepared_query_seconds"]:.3f} | {v["exact_percent"]:.1f}% |')
    lines+=['','이200문제는 전체의 부분집합이며 고유 문제 수에 더하지 않는다.','',
        '## 실행 범위와 제한','',
        '- 문제·조건별1회, CPU/native1스레드, seed20261111로 순서만 섞었다. 시간 반복 안정성은 평가하지 않는다.',
        '- 실제 새 전처리3,000회, dataset 검색8,000회다. 독립 Oracle/Dijkstra와 휴리스틱 검산은 타이머 밖이다.',
        '- 파싱·전처리·검색·복원·필수 경로 검증은 전체 로컬 안이다. 파일 읽기는 밖이다.',
        '- J4/ALT8/GraphOctile는온라인LLM0회, 원 LLMA 대조만 저장 standard 응답1회다. 새 API 호출0회, 실행 중 네트워크/API 연결을 차단했다.',
        '- 원 LLMA의 총시간은 과거 API 지연+현재 로컬의 재구성 추정이며 현재 온라인 서비스 반복 실측이 아니다.',
        '- 지도 bootstrap2000회는 구성의 탐색적 구간이다. 공개 전체는 기존 연구 노출 자료이고 시간/API 반복 변동은 구간에 포함되지 않는다.',
        '- 논리 저장 상태·거리표 항목을 전체 peak bytes로 표현하지 않는다. 별도 RSS 측정은 없다.',
        '- 기존 J1/J2/J3/JY/COMMON18 측정과 합쳐 순위를 만들지 않았다.','',
        '![J4 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계·재사용](summary.json), [원시 시행](evaluations.jsonl.gz), '
        '[실행 기록](protocol.json), [독립 검증](audit.json).','',
        '준연 담당 J4를 기존 JY 코드로 재평가했다. Codex가 실행 통합·검증·문서 작성을 지원했고 기존 결과는 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];rs=rows(out);test=set(load(out/'protocol.json')['common18_test_maps']);sub=[r for r in rs if r['map_index'] in test]
    s=dict(groups={'all1000':aggregate(rs),'common18_test200':aggregate(sub)},reuse={'all1000':reuse(rs),'common18_test200':reuse(sub)},
        report_source_sha256=sha(Path(__file__)),paired_utility_sha256=sha(Path(__file__).parents[1]/'J1_full/report.py'),repeats=1,new_api_calls=0)
    save(out/'summary.json',s);keys=('case_id','map_index','pair_index','method','valid','length','optimum','bound')+METRICS
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rs)
    base={r['case_id']:r for r in rs if r['method']=='Octile'}
    worst=sorted([r for r in rs if r['method']=='J4'],key=lambda r:r['local_seconds']/base[r['case_id']]['local_seconds'],reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r[k] for k in keys} for r in worst]);plot(out,s);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('Octile','GraphOctile','J4','J4_alt8')},ensure_ascii=False))


if __name__=='__main__':main()
