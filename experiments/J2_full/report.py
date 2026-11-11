"""Analyze J2 full-dataset records; never invoke search or a language model."""
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

METRICS=('local_seconds','api_seconds','reconstructed_total_seconds','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
FIELDS=('field_seconds','field_updates','field_heap_pops','field_accepted')


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
            stretch_max=max(ratios,default=None),field={k:st.mean(r['metadata'].get(k,0) for r in rs) for k in FIELDS})
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','reconstructed_total_seconds','successor_expansions','length')} for m in METHODS} for b in ('LLMA','FOCAL0','J2_uniform')}
    diagnostics=dict(same_path_as_uniform=sum(r['path']==by['J2_uniform'][cid]['path'] for cid,r in by['J2'].items()),
        same_expansions_as_uniform=sum(r['successor_expansions']==by['J2_uniform'][cid]['successor_expansions'] for cid,r in by['J2'].items()))
    return dict(absolute=absolute,comparisons=comparisons,diagnostics=diagnostics)


def plot(out,s,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].barh(METHODS,[1000*a[m]['means']['local_seconds'] for m in METHODS],color=['#89939b']*5+['#237baa','#26957a'])
    axes[0].set_xscale('log');axes[0].set_xlabel('평균 로컬 ms (로그 축)');axes[0].set_title('격자·속도장·거리장·검색·검증 포함')
    for m in ('LLMA','FOCAL0','J2_uniform','J2','J2_exact'):
        vals=sorted(r['length']/r['optimum'] for r in records if r['method']==m and r['valid'])
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=m)
    axes[1].set_xlabel('반환 경로 길이 / 독립 최단거리');axes[1].set_ylabel('누적 비율');axes[1].legend()
    axes[1].set_title('경로 품질 분포');fig.suptitle('J2 · 공개 1,000문제 · 조건별 1회 · 저장 응답 재생')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    g=s['groups']['all1000'];a=g['absolute'];j=a['J2'];base=a['LLMA'];direct=g['comparisons']['J2_uniform']['J2']['local_seconds']
    mean_local_ratio=j['means']['local_seconds']/base['means']['local_seconds']
    mean_total_ratio=j['means']['reconstructed_total_seconds']/base['means']['reconstructed_total_seconds']
    ci=direct['ci95'];evidence='1을 포함해 시간 감소가 뚜렷하지 않다' if ci[0]<=1<=ci[1] else '1보다 커 로컬 시간이 증가했다' if ci[0]>1 else '1보다 작아 로컬 시간이 감소했다'
    judgement='이번 설정에서 J2의 전체 지연 개선은 확인하지 못했다.' if mean_total_ratio>=1 else '이번 설정에서 J2의 재구성 총평균 지연 감소가 관측됐다.'
    names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='무안내 g+2h',FOCAL0='안내 없는 FOCAL 1.05',
        J2_uniform='균일 FMM+FOCAL 1.05',J2='LLM FMM+FOCAL 1.05',J2_exact='LLM FMM+FOCAL 1.00')
    lines=['# J2 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개 100지도×10쌍=1,000문제, 7조건×1회=7,000회. 이번 평가 대상은 J2뿐이다.',
        '기존 COMMON18의 Fast Marching 보조 거리장+FOCAL 구현을 변경 없이 재사용했다. J1의 이전 측정값을 이번 대조군으로 쓰지 않았다.','',
        '## 결과 판단','',
        f'**{judgement}** 로컬 비용과 논리 호출 비용을 포함한 결과다. 다음 비교는 이번 고정 구현과 저장 응답의 범위로 한정한다.','',
        f'원 LLM-A* 대비 J2의 로컬 평균 시간 비율은 **{mean_local_ratio:.2f}배**, 재구성 총평균 시간 비율은 **{mean_total_ratio:.4f}배**다. '
        f'경로 길이/최단 GM은 {100*base["stretch_gm"]:.3f}%에서 {100*j["stretch_gm"]:.3f}%로 바뀌었다. '
        '경로 품질과 비용을 함께 보고 추가 효용을 판단한다.','',
        f'같은 FMM의 균일 속도 대조 대비 J2 로컬 시간 GM 비율은 {direct["geometric_ratio"]:.4f}, '
        f'지도 bootstrap95% 구간은 [{ci[0]:.4f}, {ci[1]:.4f}]이며 {evidence}. '
        '이 대조가 LLM에서 유도한 속도장의 추가 기여를 보는 비교다. FOCAL0 대비는 거리장 생성 자체의 비용까지 포함한다.','',
        f'균일 FMM의 최단 일치율은 {a["J2_uniform"]["exact_percent"]:.1f}%, J2는 {j["exact_percent"]:.1f}%이며, '
        f'길이/최단 GM은 각각 {100*a["J2_uniform"]["stretch_gm"]:.3f}%와 {100*j["stretch_gm"]:.3f}%다. '
        'LLM 없는 같은 구조의 대조군과 비교해야 일반 FMM 효과와 LLM 안내의 추가 효과를 구분할 수 있다.','',
        'J2는 세 프롬프트를 사용해 원 LLMA의1회보다 호출·토큰 비용이 크다. 총시간의 차이를 순수 검색 속도 차이로 해석하지 않는다. '
        'J2_exact의 최단성은 FMM 값이 정확하기 때문이 아니라 admissible octile anchor와 w=1 탐색 조건에서 나온다.','',
        '## 전체 1,000문제','',
        '| 조건 | 유효율 | 로컬 평균 ms | 재구성 총평균 s | 호출/문제 | 길이/최단 GM | 최단 일치율 | 최대 초과율 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];b='최단' if v['bound']==1 else '≤1.05C*' if v['bound'] else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.4f} | {v["means"]["llm_calls"]:.0f} | {100*v["stretch_gm"]:.3f}% | '
            f'{v["exact_percent"]:.1f}% | {100*(v["stretch_max"]-1):.3f}% | {b} |')
    lines+=['','## J2 대응 비교','',
        '| J2 / 대조군 | 로컬 시간 GM 비율 [95% CI] | 검색 확장 GM 비율 | 로컬 개선/동일/악화 문제 |',
        '|---|---|---:|---|']
    for b in ('LLMA','FOCAL0','J2_uniform'):
        c=g['comparisons'][b]['J2'];t=c['local_seconds'];lo,hi=t['ci95']
        lines.append(f'| J2 / {b} | {t["geometric_ratio"]:.4f} [{lo:.4f}, {hi:.4f}] | '
            f'{c["successor_expansions"]["geometric_ratio"]:.4f} | {t["improved"]}/{t["same"]}/{t["worsened"]} |')
    lines+=['','비율은 1 미만일 때 해당 비용이 감소한다. 확장 감소는 전체 CPU 시간 감소와 다르다.',
        f'J2와 균일 FMM은 경로가 {g["diagnostics"]["same_path_as_uniform"]}/1000문제, '
        f'검색 확장 수가 {g["diagnostics"]["same_expansions_as_uniform"]}/1000문제에서 같다.','',
        '## 거리장 준비와 비용','',
        '| 조건 | FMM 평균 ms | 거리장 heap pop 평균 | update 평균 | accepted 평균 | 검색 확장 평균 | 저장 상태 평균 | 입력/출력 토큰/문제 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in ('J2_uniform','J2','J2_exact'):
        v=a[m];f=v['field'];lines.append(f'| {m} | {1000*f["field_seconds"]:.3f} | {f["field_heap_pops"]:.2f} | '
            f'{f["field_updates"]:.2f} | {f["field_accepted"]:.2f} | {v["means"]["successor_expansions"]:.2f} | '
            f'{v["means"]["storage"]:.2f} | {v["means"]["prompt_tokens"]:.2f}/{v["means"]["completion_tokens"]:.2f} |')
    lines+=['','FMM 시간은 전체 준비 비용의 일부다. 격자 생성·세 응답 파싱·polyline 거리장·속도장 준비도 로컬 전체 시간에 포함했다.',
        f'J2의 FMM 평균 {1000*j["field"]["field_seconds"]:.2f}ms와 전체 로컬 평균 {1000*j["means"]["local_seconds"]:.2f}ms를 구분한다. '
        '전체 로컬 비용을 FMM 계산 하나의 비용으로 돌리지 않는다.',
        '거리장은 cardinal 방향의 1차 Eikonal 해다. 정확한 8방향 최단거리나 admissible heuristic으로 주장하지 않고 FOCAL 보조 순위로만 쓴다.',
        '목표에서 cardinal로 도달하지 못한 거리장 값은 원 구현의 큰 보조 비용으로 정규화하며, anchor의 OPEN 후보는 제거하지 않는다.',
        '거리장 heap 연산·검색 확장·배열 기반 FOCAL 선택은 서로 다른 작업이므로 합산한 하나의 operation 순위를 만들지 않는다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 로컬 평균 ms | 길이/최단 GM | 최단 일치율 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m]
        lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% |')
    lines+=['','이200문제는 전체의 부분집합이다. 과거 COMMON18의 새 응답 대신 이번 전체와 같은 2026-10-13 응답을 사용하므로 과거 표와 수치가 달라질 수 있다.','',
        '## 실행 범위와 제한','',
        '- 문제·조건마다 정확히1회, 추가 반복·중앙값·시드 평균은 없다. CPU/native1스레드, 순서만 고정 seed로 섞었다.',
        '- 기존 GPT-4o mini standard/CoT/RePE 3,000응답을 요청·입력 SHA와 확인 후 고정했다. 새 API 호출0회, 실행 중 네트워크/API 연결을 차단했다.',
        '- 원 LLMA는standard1회, J2/J2_exact는세 응답을 모두 사용한다. 같은 응답의 지연·토큰은 동일하게 부과한다.',
        '- 총시간은 과거 API 지연과 현재 로컬 시간의 합이며, 세 프롬프트의 순차 호출 가정이다. 현재 서비스 실측이나 수집 당시 병렬 벽시계가 아니다.',
        '- 파일 읽기·독립 Oracle·Dijkstra는 타이머 밖이고, 지도 준비·파싱·제안/속도/거리장·검색·복원·필수 검증은 안이다.',
        '- 시간1회와 과거 API 지연의 반복 안정성은 평가하지 않는다. 지도 bootstrap2000회 구간은 문제 구성의 탐색적 불확실성만 나타낸다.',
        '- 공개 전체는 연구에 이미 노출된 자료다. 전체 결과를 새로운 hidden test 일반화로 주장하지 않는다.',
        '- 저장 상태·OPEN peak는 논리 개수다. 이번에는 별도 RSS 측정이 없으며 메모리 bytes나 총CPU 연산으로 표현하지 않는다.',
        '- J2/J2_uniform은같은 w1.05의 직접 대조다. J2_exact와uniform은 w가 달라 직접 성분 대조로 사용하지 않는다.',
        '', '![J2 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계](summary.json), [원시 시행](evaluations.jsonl.gz), '
        '[저장 응답](responses.jsonl.gz), [실행 기록](protocol.json), [독립 검증](audit.json).','',
        '최초 J2 구현·실험은 승민, 이번 전체 평가·검토는 준연 담당이다. Codex가 실행 통합·검증·문서 작성을 지원했다. 이전 결과는 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];records=rows(out);test=set(load(out/'protocol.json')['common18_test_maps'])
    s=dict(groups={'all1000':aggregate(records),'common18_test200':aggregate([r for r in records if r['map_index'] in test])},
        report_source_sha256=sha(Path(__file__)),paired_utility_sha256=sha(Path(__file__).parents[1]/'J1_full/report.py'),repeats=1,new_api_calls=0)
    save(out/'summary.json',s)
    keys=('case_id','map_index','method','valid','length','optimum','bound')+METRICS
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in records)
    base={r['case_id']:r for r in records if r['method']=='LLMA'}
    worst=sorted([r for r in records if r['method']=='J2'],key=lambda r:r['local_seconds']/base[r['case_id']]['local_seconds'],reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r[k] for k in keys} for r in worst]);plot(out,s,records);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('LLMA','J2_uniform','J2','J2_exact')},ensure_ascii=False))


if __name__=='__main__':main()
