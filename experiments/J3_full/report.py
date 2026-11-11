"""Analyze J3 matched initial-solution and pruning controls, from saved rows."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st

import numpy as np

from .run import METHODS,load
from experiments.J1_full.report import paired,gm
from experiments.COMMON18.prepare import save
from experiments.K3.run import sha

METRICS=('local_seconds','api_seconds','reconstructed_total_seconds','operation','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
REFERENCES=('LLMA','Octile','WallAstar','J3_no_prune','J3_noLLM','J3_noLLM_no_prune')
STAGES=('seed_seconds','seed_operation','refinement_operation','pruned')
TWO_STAGE=('J3','J3_no_prune','J3_noLLM','J3_noLLM_no_prune')


def rows(out):
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def aggregate(records):
    by={m:{r['case_id']:r for r in records if r['method']==m} for m in METHODS};a={}
    for m,cs in by.items():
        rs=list(cs.values());valid=[r for r in rs if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        stages={k:st.mean(r.get(k,0) for r in rs) for k in STAGES}
        stages.update(seed_optimal_cases=sum(abs(r.get('seed_length',math.inf)-r['optimum'])<1e-7 for r in rs),
            seed_fallback_cases=sum(r.get('seed_fallback',False) for r in rs),
            nonseed_local_seconds=st.mean(r['local_seconds']-r['seed_seconds'] for r in rs) if m in TWO_STAGE else 0)
        a[m]=dict(cases=len(rs),valid=len(valid),valid_percent=100*len(valid)/len(rs),
            means={k:st.mean(r[k] for r in rs) for k in METRICS},bound=rs[0]['bound'],
            exact_percent=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs),
            stretch_gm=gm(ratios),stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,
            stretch_max=max(ratios,default=None),stages=stages)
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','reconstructed_total_seconds','successor_expansions','operation','length')} for m in METHODS} for b in REFERENCES}
    return dict(absolute=a,comparisons=comparisons)


def plot(out,s,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    colors=['#89939b']*5+['#237baa','#95bdd3','#26957a','#9ac6bb']
    axes[0].barh(METHODS,[1000*a[m]['means']['local_seconds'] for m in METHODS],color=colors)
    axes[0].set_xlabel('평균 로컬 ms');axes[0].set_title('초기 해 생성·정련·필수 검증 포함')
    means=[a[m]['means']['operation'] for m in TWO_STAGE];seed=[a[m]['stages']['seed_operation'] for m in TWO_STAGE]
    axes[1].barh(TWO_STAGE,seed,label='초기 해 탐색 pop',color='#9ba7b0')
    axes[1].barh(TWO_STAGE,[a[m]['stages']['refinement_operation'] for m in TWO_STAGE],left=seed,label='정련 pop',color='#237baa')
    axes[1].set_xlabel('모든 heap pop 평균');axes[1].set_title('두 단계 비용을 모두 집계');axes[1].legend()
    fig.suptitle('J3 · 공개 1,000문제 · 조건별 1회 · 저장 standard 응답 재생')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    g=s['groups']['all1000'];a=g['absolute'];j=a['J3'];llma=a['LLMA'];cpu=j['means']['local_seconds']/llma['means']['local_seconds']
    total=j['means']['reconstructed_total_seconds']/llma['means']['reconstructed_total_seconds']
    names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='g+2h',Octile='단일 clean octile A*',WallAstar='단일 정확 벽 하한 A*',
        J3='LLM seed+가지치기',J3_no_prune='LLM seed+무가지치기',J3_noLLM='g+2h seed+가지치기',J3_noLLM_no_prune='g+2h seed+무가지치기')
    lines=['# J3 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개100지도×10쌍=1,000문제, 9조건×1회=9,000회. J3 초기 해 상한 가지치기를 평가했다.',
        '기존 experiments/JY/search.py를 변경 없이 사용했다. 주 J3는 COMMON18과 같은 LLM seed 변형이고 비LLM seed도 별도 비교한다.','',
        '## 결과 판단','',
        f'원 LLM-A* 대비 주 J3의 로컬 평균은 **{cpu:.3f}배**, 재구성 총평균은 **{total:.6f}배**다. '
        f'최단 일치율은 {llma["exact_percent"]:.1f}%에서 {j["exact_percent"]:.1f}%로 바뀌었다. '
        '최단성 확보와 전체 비용 절감은 별개이므로 단일 정확 탐색과 같은2단계 대조를 함께 본다.','',
        f'원 LLMA에 추가된 로컬 평균은 {1000*(j["means"]["local_seconds"]-llma["means"]["local_seconds"]):.3f}ms이며, '
        f'같은 API 지연을 포함한 총평균 변화는 {100*(total-1):+.4f}%다. '
        '이번 고정 응답에서는 지연을 거의 유지하면서 최단성을 추가하는 용도로 해석할 수 있다. '
        '이는 서비스 속도를 새로 반복 측정한 결과나 LLM 자체의 판단 능력 개선을 뜻하지 않는다.','',
        '가지치기 구성요소의 이득은 아래 대응 대조에서 확인하되, 단일 정확 탐색보다 빠른 방식으로 확대하지 않는다. '
        'LLM 없는 초기 해 대조와의 차이도 있어 좋은 상한을 확보하는 방법의 비용·품질이 중요하다.','',
        'LLM이 출력한 좌표를 바로 유효 상한으로 쓰는 방식이 아니다. 원 LLM-A*가 실제 격자 탐색으로 만든 유효 경로를 seed로 얻고, '
        '그 길이를 상한으로 다시 정확 탐색한다. seed 탐색과 파싱·검증도 모두 비용에 포함했다.','',
        '## 전체 1,000문제','',
        '| 조건 | 유효율 | 로컬 평균 ms | 재구성 총평균 s | 호출/문제 | 길이/최단 GM | 최단 일치율 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];b='최단' if v['bound']==1 else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.5f} | {v["means"]["llm_calls"]:.0f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% | {b} |')
    lines+=['','## 구성요소와 단일 탐색 대조','',
        '| 개입 / 대조 | 로컬 GM 비율 [95% CI] | 모든 pop GM 비율 | 확장 GM 비율 | 로컬 개선/동일/악화 |',
        '|---|---|---:|---:|---|']
    pairs=[('J3','J3_no_prune'),('J3_noLLM','J3_noLLM_no_prune'),('J3','J3_noLLM'),
        ('J3','Octile'),('J3','WallAstar'),('J3_noLLM','Octile'),('J3_noLLM','WallAstar')]
    for m,b in pairs:
        c=g['comparisons'][b][m];t=c['local_seconds'];lo,hi=t['ci95']
        lines.append(f'| {m} / {b} | {t["geometric_ratio"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{c["operation"]["geometric_ratio"]:.4f} | {c["successor_expansions"]["geometric_ratio"]:.4f} | {t["improved"]}/{t["same"]}/{t["worsened"]} |')
    lines+=['','비율은1미만이면 해당 비용이 감소한다. 무가지치기2단계 대조와의 차이는 가지치기 성분 효과다. '
        '처음부터 seed를 만들지 않는 단일 A*보다 유리하다는 뜻으로 확대하지 않는다.',
        'J3/J3_noLLM은같은 정확 refinement와 가지치기를 유지하고 초기 해 생성기를 교체한 대조다. '
        'LLM의 API 대기를 빼더라도 로컬 비용과 초기 해 품질의 차이를 함께 확인해야 한다.','',
        '## 초기 해와 정련 비용','',
        '| 조건 | 초기 해가 이미 최단 | fallback | seed 평균 ms | seed 외 로컬 평균 ms | seed pop 평균 | 정련 pop 평균 | 가지치기 이벤트 평균 | 검색 확장 평균 | 저장 상태 합 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in TWO_STAGE:
        v=a[m];p=v['stages'];lines.append(f'| {m} | {p["seed_optimal_cases"]}/1000 | {p["seed_fallback_cases"]} | '
            f'{1000*p["seed_seconds"]:.3f} | {1000*p["nonseed_local_seconds"]:.3f} | {p["seed_operation"]:.2f} | '
            f'{p["refinement_operation"]:.2f} | {p["pruned"]:.2f} | {v["means"]["successor_expansions"]:.2f} | {v["means"]["storage"]:.2f} |')
    lines+=['','seed 외 로컬 시간에는 정확 refinement뿐 아니라 지도·추가 검증·wrapper 비용이 포함된다. 순수 refinement 시간이라고 부르지 않는다.',
        '모든 pop은 seed와 refinement 합계다. 가지치기 이벤트는 후보 거부·상한 종료 기록이며 고유 삭제 노드 수가 아니다. '
        '저장 상태는 두 단계g 항목 합으로 실제 동시 peak bytes가 아니고, 원 코드의 heap peak는 refinement에 한정된다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 로컬 평균 ms | 길이/최단 GM | 최단 일치율 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m]
        lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% |')
    lines+=['','이200문제는 전체의 부분집합이다. 이번에는 전체와 같은 2026-10-13 standard 응답을 사용해 과거 COMMON18의 새 응답과 섞지 않았다.','',
        '## 실행 범위와 제한','',
        '- 문제·조건별1회, CPU/native1스레드, seed20261111로 순서만 섞었다. 시간 반복·중앙값·확률 시드 평균은 없다.',
        '- 저장 standard1,000응답을 요청·입력·SHA와 확인해 고정했다. 새 API 호출0회이며 실행 중 네트워크/API 연결을 차단했다.',
        '- LLMA/J3/J3_no_prune은논리1회, 다른 조건은0회다. 사용하는 같은 응답에 같은 과거 지연·토큰을 부과했다.',
        '- 총시간은 과거 API 지연+현재 로컬의 재구성 추정이다. 현재 온라인 서비스 반복 실측이 아니다.',
        '- 파싱·준비·seed·refinement·복원·필수 검증은 타이머 안, 파일 읽기·독립 Oracle/Dijkstra는 밖이다.',
        '- 질의/방법 사이 이웃 캐시는 공유하지 않는다. 같은 질의의 seed/refinement 안에서는 기존 코드처럼 재사용하며 seed 비용도 포함한다.',
        '- 공개 전체는 기존 연구에 노출된 자료다. 지도 bootstrap2000회는 구성의 탐색적 불확실성이며 시간/API 반복 변동을 포함하지 않는다.',
        '- 단순 상한 가지치기와 정확 refinement이며 ARA* 전체 구현이나 탐색 상태 전체의 증분 재사용 알고리즘이 아니다.',
        '- 이전 J1/J2/JY 결과를 이번 대조로 합치지 않는다. 별도 RSS 측정은 없으며 논리 상태를 실제 메모리 bytes로 표현하지 않는다.',
        '', '![J3 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계](summary.json), [원시 시행](evaluations.jsonl.gz), '
        '[응답](responses.jsonl.gz), [실행 기록](protocol.json), [독립 검증](audit.json).','',
        '준연 담당 J3의 기존 JY 구현을 재평가했다. Codex가 실행 통합·검증·문서 작성을 지원했고 기존 결과는 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];rs=rows(out);test=set(load(out/'protocol.json')['common18_test_maps'])
    s=dict(groups={'all1000':aggregate(rs),'common18_test200':aggregate([r for r in rs if r['map_index'] in test])},
        report_source_sha256=sha(Path(__file__)),paired_utility_sha256=sha(Path(__file__).parents[1]/'J1_full/report.py'),repeats=1,new_api_calls=0)
    save(out/'summary.json',s)
    keys=('case_id','map_index','method','valid','length','optimum','bound')+METRICS
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rs)
    base={r['case_id']:r for r in rs if r['method']=='Octile'}
    worst=sorted([r for r in rs if r['method']=='J3'],key=lambda r:r['local_seconds']/base[r['case_id']]['local_seconds'],reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r[k] for k in keys} for r in worst]);plot(out,s,rs);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('LLMA','Octile','WallAstar','J3','J3_no_prune','J3_noLLM')},ensure_ascii=False))


if __name__=='__main__':main()
