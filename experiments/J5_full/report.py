"""Read-only single-pass summaries for NBS and matched bidirectional controls."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics as st

import numpy as np

from .run import METHODS,load
from .adapter import BIDIRECTIONAL
from experiments.J1_full.report import paired,gm
from experiments.COMMON18.prepare import save
from experiments.K3.run import sha

METRICS=('local_seconds','api_seconds','reconstructed_total_seconds','operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
QUEUES=('queue_promotions','stale_pops','heap_pushes','pairs','reopenings','forward_storage','backward_storage','peak_heap_entries','relax_attempts')
REFERENCES=('LLMA','Octile','WallAstar','Dijkstra','J5_bidijkstra','J5_euclidean')


def rows(out):
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def aggregate(records):
    by={m:{r['case_id']:r for r in records if r['method']==m} for m in METHODS};a={}
    for m,cs in by.items():
        rs=list(cs.values());valid=[r for r in rs if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        a[m]=dict(cases=len(rs),valid=len(valid),valid_percent=100*len(valid)/len(rs),bound=rs[0]['bound'],
            means={k:st.mean(r[k] for r in rs) for k in METRICS},queue={k:st.mean(r.get(k,0) for r in rs) for k in QUEUES},
            exact_percent=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs),
            stretch_gm=gm(ratios),stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,stretch_max=max(ratios,default=None))
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','reconstructed_total_seconds','operation','successor_expansions','length')} for m in METHODS} for b in REFERENCES}
    return dict(absolute=a,comparisons=comparisons)


def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    axes[0].barh(METHODS,[1000*a[m]['means']['local_seconds'] for m in METHODS],color=['#89939b']*6+['#237baa','#58a0c0','#26957a'])
    axes[0].set_xlabel('평균 로컬 ms');axes[0].set_title('모든 큐 관리·검색·복원·필수 검증 포함')
    expansions=[a[m]['means']['successor_expansions'] for m in BIDIRECTIONAL]
    promotions=[a[m]['queue']['queue_promotions'] for m in BIDIRECTIONAL]
    stale=[a[m]['queue']['stale_pops'] for m in BIDIRECTIONAL]
    axes[1].barh(BIDIRECTIONAL,expansions,label='확장 pop',color='#237baa')
    axes[1].barh(BIDIRECTIONAL,promotions,left=expansions,label='승격 pop',color='#d8964a')
    axes[1].barh(BIDIRECTIONAL,stale,left=np.array(expansions)+promotions,label='stale pop',color='#89939b')
    axes[1].set_xlabel('실제 heap pop 평균');axes[1].set_title('양방향 탐색의 pop 구성');axes[1].legend()
    fig.suptitle('J5 · 공개 1,000문제 · 조건별 1회 · NBS/양방향/단방향 대조')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    g=s['groups']['all1000'];a=g['absolute'];j=a['J5'];base=a['Octile'];ratio=j['means']['local_seconds']/base['means']['local_seconds']
    verdict='이번 조건에서 NBS의 단일 정확 A* 대비 평균 시간 개선은 확인하지 못했다.' if ratio>=1 else '이번 조건에서 NBS의 단일 정확 A* 대비 평균 시간 감소가 관측됐다.'
    names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='g+2h',Octile='단일 clean octile A*',WallAstar='단일 정확 벽 하한 A*',
        Dijkstra='단방향 Dijkstra',J5='NBS octile',J5_euclidean='NBS Euclidean',J5_bidijkstra='양방향 Dijkstra')
    lines=['# J5 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개100지도×10쌍=1,000문제, 9조건×1회=9,000회. 기존 JY NBS/양방향 탐색을 변경 없이 평가했다.','',
        '## 결과 판단','',f'**{verdict}** 주 J5의 전체 로컬 평균은 {1000*j["means"]["local_seconds"]:.3f}ms로 '
        f'같은 실행의 단일 octile A* 대비 {ratio:.3f}배다. 양방향 검색의 확장과 큐 관리 비용을 모두 포함했다.','',
        f'평균 검색 확장은 NBS {j["means"]["successor_expansions"]:.2f}, 단일 octile A* {base["means"]["successor_expansions"]:.2f}이며, '
        f'NBS의 모든 heap pop은 {j["means"]["operation"]:.2f}회다. '
        '전방/후방을 합친 비용과 단방향 비용을 같은 기준으로 본다.','',
        '양방향 Dijkstra와 단방향 Dijkstra 비교는 휴리스틱 없는 상태에서 양방향 탐색의 효과를 보는 대조다. '
        '그 이득을 강한 하한을 쓰는 단일 A*와의 우위로 확대하지 않는다. NBS의 확장 관련 이론도 총 heap pop/실행시간의 일정 배수 보장으로 해석하지 않는다.','',
        'J5와 그 변형은 온라인 LLM을 쓰지 않는다. 원 LLMA 대조보다 API 대기가 사라지는 효과를 LLM 안내 자체의 개선으로 표현하지 않는다.','',
        '## 전체 1,000문제','',
        '| 조건 | 유효율 | 로컬 평균 ms | 재구성 총평균 s | 길이/최단 GM | 최단 일치율 | 모든 heap pop 평균 | 검색 확장 평균 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];b='최단' if v['bound']==1 else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.5f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% | '
            f'{v["means"]["operation"]:.2f} | {v["means"]["successor_expansions"]:.2f} | {b} |')
    lines+=['','## 대응 비교','',
        '| 개입 / 대조 | 로컬 GM 비율 [95% CI] | 모든 pop GM 비율 | 검색 확장 GM 비율 | 로컬 개선/동일/악화 |',
        '|---|---|---:|---:|---|']
    for m,b in [('J5','Octile'),('J5','WallAstar'),('J5','J5_euclidean'),('J5','J5_bidijkstra'),('J5_bidijkstra','Dijkstra'),('J5_euclidean','Octile')]:
        c=g['comparisons'][b][m];t=c['local_seconds'];lo,hi=t['ci95']
        lines.append(f'| {m} / {b} | {t["geometric_ratio"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{c["operation"]["geometric_ratio"]:.4f} | {c["successor_expansions"]["geometric_ratio"]:.4f} | {t["improved"]}/{t["same"]}/{t["worsened"]} |')
    lines+=['','비율은 1 미만일 때 해당 비용이 감소한다. 전체 평균 시간 비율과 문제별 대응 비율의 기하평균은 구분한다.','',
        '## 큐 관리와 양방향 상태','',
        '| 조건 | 선택 쌍 평균 | 확장 pop 평균 | 승격 pop 평균 | stale pop 평균 | push 평균 | 재개방 평균 | 전방/후방 g 항목 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in BIDIRECTIONAL:
        v=a[m];q=v['queue'];pairs=f'{q["pairs"]:.2f}' if m!='J5_bidijkstra' else '—'
        lines.append(f'| {m} | {pairs} | {v["means"]["successor_expansions"]:.2f} | {q["queue_promotions"]:.2f} | '
            f'{q["stale_pops"]:.2f} | {q["heap_pushes"]:.2f} | {q["reopenings"]:.2f} | {q["forward_storage"]:.2f}/{q["backward_storage"]:.2f} |')
    lines+=['','양방향 Dijkstra는 NBS 쌍 선택을 하지 않으므로 쌍 수는 해당 없음이다. '
        'NBS의 operation=expansions+promotions+stale_pops, expansions=2*pairs를 원시 시행별로 검산했다.',
        'storage는 전방/후방g 항목 합이다. 같은 정점이 양쪽에 있으면 두 번 세며 고유 정점수나 실제 peak bytes가 아니다. '
        '원 코드 heap peak는 반복 경계에서 관측한 논리 큐 항목 수다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 로컬 평균 ms | 모든 pop 평균 | 검색 확장 평균 | 최단 일치율 |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m];lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["operation"]:.2f} | {v["means"]["successor_expansions"]:.2f} | {v["exact_percent"]:.1f}% |')
    lines+=['','이200문제는 전체의 부분집합이다. 과거 결과와 합산하거나 고유 문제 수에 더하지 않는다.','',
        '## 실행 범위와 제한','',
        '- 문제·조건별1회, CPU/native1스레드, seed20261111로 순서만 섞었다. 시간 반복·중앙값·확률 시드 평균은 없다.',
        '- 기존 JY NBS/양방향 코드를 그대로 사용하고 단방향 Dijkstra는 같은 clean astar 엔진의 h=0 대조로 실행했다.',
        '- 작은 fixture에서 쌍 전수 대조를 수행하고 기존35개 무작위 소형 지도×3조건의 reference/최단성 검사를 통과했다.',
        '- 본 시간 측정에서는 audit=False다. Cartesian frontier 검산의 추가 비용을 본 알고리즘 시간에 포함하지 않았다.',
        '- 파싱·지도 준비·모든 큐 관리·쌍 선택·승격·검색·복원·필수 경로 검증은 로컬 안이고 파일 읽기·독립 Oracle/Dijkstra는 밖이다.',
        '- 원 LLMA만 저장 standard1,000응답을 재생해 논리1회/문제다. 다른 조건은 온라인LLM0회이며 새 API 호출은0회다. 실행 중 네트워크/API 연결을 차단했다.',
        '- LLMA 총시간은 과거 API 지연+현재 로컬의 재구성 추정이며 현재 서비스 반복 실측이 아니다.',
        '- 공개 전체는 기존 연구 노출 자료다. 지도 bootstrap2000회는 구성의 탐색적 구간이며 시간/API 반복 변동을 포함하지 않는다.',
        '- 논리 상태를 실제 메모리 bytes로 표현하지 않으며 별도 RSS 측정은 없다. 이전 J1~J4/JY의 시간과 합산하지 않는다.',
        '', '![J5 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계](summary.json), [원시 시행](evaluations.jsonl.gz), '
        '[실행 기록](protocol.json), [소형 검증](preflight.json), [무작위 검사](unit_tests.json), [독립 검산](audit.json).','',
        '준연 담당 J5의 기존 JY 구현을 재평가했다. Codex가 실행 통합·검증·문서 작성을 지원했고 기존 결과는 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];rs=rows(out);test=set(load(out/'protocol.json')['common18_test_maps'])
    s=dict(groups={'all1000':aggregate(rs),'common18_test200':aggregate([r for r in rs if r['map_index'] in test])},
        report_source_sha256=sha(Path(__file__)),paired_utility_sha256=sha(Path(__file__).parents[1]/'J1_full/report.py'),repeats=1,new_api_calls=0)
    save(out/'summary.json',s);keys=('case_id','map_index','method','valid','length','optimum','bound')+METRICS
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rs)
    base={r['case_id']:r for r in rs if r['method']=='Octile'}
    worst=sorted([r for r in rs if r['method']=='J5'],key=lambda r:r['local_seconds']/base[r['case_id']]['local_seconds'],reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r[k] for k in keys} for r in worst]);plot(out,s);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('Octile','WallAstar','Dijkstra','J5','J5_euclidean','J5_bidijkstra')},ensure_ascii=False))


if __name__=='__main__':main()
