"""Analyze the single-pass J1 study, without running search or calling APIs."""
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
from experiments.COMMON18.prepare import save
from experiments.K3.run import sha


def rows(out):
    with gzip.open(out/'evaluations.jsonl.gz','rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def gm(values):return math.exp(st.mean(math.log(v) for v in values)) if values else None


def paired(a,b,metric):
    values=[];maps=defaultdict(list);better=same=worse=0
    for cid,r in a.items():
        x,y=r.get(metric),b[cid].get(metric)
        if x is None or y is None:continue
        v=max(x,1e-12)/max(y,1e-12);values.append(v);maps[r['map_index']].append(math.log(v))
        if math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12):same+=1
        elif x<y:better+=1
        else:worse+=1
    if not values:return None
    rng=np.random.default_rng(SEED);m=np.array([st.mean(v) for _,v in sorted(maps.items())])
    boot=np.exp(m[rng.integers(0,len(m),size=(2000,len(m)))].mean(axis=1))
    return dict(geometric_ratio=gm(values),ci95=np.quantile(boot,[.025,.975]).tolist(),
        improved=better,same=same,worsened=worse,cases=len(values),maps=len(m))


def aggregate(records):
    by={m:{r['case_id']:r for r in records if r['method']==m} for m in METHODS};result={}
    metrics=('local_seconds','api_seconds','reconstructed_total_seconds','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
    for m,cs in by.items():
        rs=list(cs.values());valid=[r for r in rs if r['valid']];ratios=[r['length']/r['optimum'] for r in valid]
        result[m]=dict(cases=len(rs),valid=len(valid),valid_percent=100*len(valid)/len(rs),
            means={k:st.mean(r[k] for r in rs) for k in metrics},
            exact_percent=100*sum(abs(r['length']-r['optimum'])<1e-7 for r in valid)/len(rs),
            stretch_gm=gm(ratios),stretch_p95=float(np.quantile(ratios,.95)) if ratios else None,
            stretch_max=max(ratios,default=None),bound=rs[0]['bound'],
            repair=dict(normalized_input_valid=sum(r['metadata'].get('input_valid',False) for r in rs),
                repaired_valid=sum(r['metadata'].get('repaired',False) for r in rs),
                invalid_to_valid=sum(not r['metadata'].get('input_valid',False) and r['metadata'].get('repaired',False) for r in rs),
                valid_to_invalid=sum(r['metadata'].get('input_valid',False) and not r['metadata'].get('repaired',False) for r in rs),
                incumbent_returned=sum(r.get('incumbent_returned',False) for r in rs),
                mean_seconds=st.mean(r['metadata'].get('repair_seconds',0) for r in rs),
                mean_iterations=st.mean(r['metadata'].get('iterations',0) for r in rs)))
    comparisons={b:{m:{k:paired(by[m],by[b],k) for k in ('local_seconds','reconstructed_total_seconds','successor_expansions','length')} for m in METHODS} for b in ('LLMA','FOCAL0','J1_straight')}
    return dict(absolute=result,comparisons=comparisons)


def plot(out,s,rs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    a=s['groups']['all1000']['absolute'];fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    axes[0].barh(METHODS,[a[m]['means']['local_seconds']*1000 for m in METHODS],color=['#89939b']*5+['#237baa','#26957a'])
    axes[0].set_xscale('log');axes[0].set_xlabel('평균 로컬 ms (로그 축)');axes[0].set_title('전체 준비·수리·탐색·검증 포함')
    for m in ('LLMA','FOCAL0','J1','J1_exact'):
        vals=sorted(r['length']/r['optimum'] for r in rs if r['method']==m and r['valid'])
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=m)
    axes[1].set_xlabel('반환 경로 길이 / 독립 최단거리');axes[1].set_ylabel('누적 비율');axes[1].legend()
    axes[1].set_title('경로 품질 분포');fig.suptitle('J1 · 공개 1,000문제 · 조건별 1회 · 저장 응답 재생')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def markdown(out,s):
    a=s['groups']['all1000']['absolute'];names=dict(Astar='원 A*',LLMA='원 LLM-A*',G2H='무안내 g+2h',
        FOCAL0='안내 없는 FOCAL 1.05',J1_straight='직선 CHOMP+FOCAL 1.05',J1='LLM CHOMP+FOCAL 1.05',J1_exact='LLM CHOMP+FOCAL 1.00')
    lines=['# J1 전체 공개 데이터셋 1회 평가','',
        '2026-11-11 · jy · 공개 100지도×10쌍=1,000문제, 7조건×1회=7,000회. 이번 평가 대상은 J1뿐이다.',
        'CHOMP-inspired 경로 수리와 격자 FOCAL을 결합한 기존 COMMON18 구현을 재사용했다. 원 CHOMP 논문 전체의 재현 결과가 아니다.','',
        '## 결과 판단','',
        '**이번 조건에서 J1의 종합적인 효율 개선은 확인하지 못했다.** 원 LLM-A*보다 경로 품질은 좋아졌으나 준비·수리·탐색을 합친 시간은 증가했다. '
        f'로컬 평균은 {a["J1"]["means"]["local_seconds"]/a["LLMA"]["means"]["local_seconds"]:.2f}배, '
        f'같은 과거 API 지연을 더한 총평균은 {100*(a["J1"]["means"]["reconstructed_total_seconds"]/a["LLMA"]["means"]["reconstructed_total_seconds"]-1):.2f}% 증가했다.','',
        '같은 5% 보장 FOCAL0는 수리 없이 더 낮은 경로 초과율 GM과 로컬 평균 비용을 보였다. '+
        ('직선 초기화 대조와 J1의 로컬 시간 GM 구간은 1을 포함해 LLM 초기 경로가 시간을 줄였다는 뚜렷한 근거는 없다. '
         if s['groups']['all1000']['comparisons']['J1_straight']['J1']['local_seconds']['ci95'][0]<=1<=s['groups']['all1000']['comparisons']['J1_straight']['J1']['local_seconds']['ci95'][1]
         else '직선 초기화 대조와의 시간 차이는 아래 대응 비율과 구간을 따른다. ')+
        '이번 실행은 단일 반복이므로 작은 시간 차이는 특히 제한적으로 해석한다.','',
        f'수리 자체의 평균은 {1000*a["J1"]["repair"]["mean_seconds"]:.2f}ms이고, '
        f'전체 로컬 평균은 {1000*a["J1"]["means"]["local_seconds"]:.2f}ms다. '
        '전체 비용을 CHOMP 수리만의 비용으로 돌리지 않는다. NumPy 격자 준비·FOCAL 배열 선택 등의 비용도 포함한다. '
        'J1_exact의 최단성은 CHOMP가 아니라 admissible 하한을 사용하는 w=1 탐색·검증 조건에서 나온다.','',
        '## 전체 1,000문제','',
        '| 조건 | 유효율 | 로컬 평균 ms | 재구성 총평균 s | 길이/최단 GM | 최단 일치율 | 최대 초과율 | 보장 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for m in METHODS:
        v=a[m];bound=v['bound'];b='최단' if bound==1 else '≤1.05C*' if bound else '없음'
        lines.append(f'| {names[m]} | {v["valid_percent"]:.1f}% | {1000*v["means"]["local_seconds"]:.3f} | '
            f'{v["means"]["reconstructed_total_seconds"]:.4f} | {100*v["stretch_gm"]:.3f}% | '
            f'{v["exact_percent"]:.1f}% | {100*(v["stretch_max"]-1):.3f}% | {b} |')
    lines+=['','## J1 직접 대조','',
        '| J1 / 대조군 | 로컬 시간 GM 비율 [95% CI] | 확장 GM 비율 | 로컬 개선/동일/악화 문제 |',
        '|---|---|---:|---|']
    for b in ('LLMA','FOCAL0','J1_straight'):
        c=s['groups']['all1000']['comparisons'][b]['J1'];t=c['local_seconds'];e=c['successor_expansions'];lo,hi=t['ci95']
        lines.append(f'| J1 / {b} | {t["geometric_ratio"]:.4f} [{lo:.4f}, {hi:.4f}] | {e["geometric_ratio"]:.4f} | {t["improved"]}/{t["same"]}/{t["worsened"]} |')
    direct=s['groups']['all1000']['comparisons']['J1_straight']['J1']['local_seconds']['geometric_ratio']
    lines+=['','비율은 1 미만일 때 해당 비용이 감소한다. 같은 구조의 직선 초기화 대비가 LLM 초기 경로의 추가 기여를 보는 직접 대조다.',
        f'J1의 직접 대조 로컬 시간 GM 비율은 {direct:.4f}다. 최단성 보장 변형의 결과는 별도 행이며 주 J1을 대체하지 않는다.',
        'FOCAL0 대비는 CHOMP 준비 전체의 비용·효과를 포함한다. 원 A*/LLM-A*는 탐색 엔진이 달라 구성요소 하나만의 인과 효과로 해석하지 않는다.','',
        '## 수리 성공과 계산량','',
        '| 조건 | 정규화 초기 경로 유효 | 유효 격자 수리 성공 | 최종 incumbent 직접 반환 | 평균 수리 ms | 평균 반복 수 | 검색 확장 평균 | 저장 상태 평균 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in ('J1','J1_exact','J1_straight'):
        v=a[m];r=v['repair'];lines.append(f'| {m} | {r["normalized_input_valid"]}/1000 | {r["repaired_valid"]}/1000 | '
            f'{r["incumbent_returned"]}/1000 | {1000*r["mean_seconds"]:.3f} | {r["mean_iterations"]:.2f} | '
            f'{v["means"]["successor_expansions"]:.2f} | {v["means"]["storage"]:.2f} |')
    lines+=['','수리 실패여도 최종 FOCAL은 실행된다. 따라서 수리 성공률과 최종 경로 성공률을 구분한다.',
        f'J1의 정규화 초기 무효→유효 수리는 {a["J1"]["repair"]["invalid_to_valid"]}건이며, '
        f'초기 유효였으나 수리기가 유효 격자 경로를 보존하지 못한 경우는 {a["J1"]["repair"]["valid_to_invalid"]}건이다.',
        '수리 실패 시 원래 LLM polyline을 secondary로 계속 쓰는 구현이 아니며, 유효 수리가 없으면 기본 휴리스틱으로 진행한다.',
        '정규화 유효성은 parser 결과에 start/goal 삽입·중복 제거를 적용한 경로의 검사다. 원문 자체의 유효 응답률과 같지 않다.',
        'J1 계열은 배열 기반 FOCAL이므로 heap pop만으로 총 연산량을 나타낼 수 없다. 준비·수리 비용은 로컬 시간에 포함한다.','',
        f'LLM을 쓰는 LLMA/J1/J1_exact는 모두 문제당 논리 호출1회, 입력 {a["J1"]["means"]["prompt_tokens"]:.2f} / '
        f'출력 {a["J1"]["means"]["completion_tokens"]:.2f}토큰을 동일하게 부과했다. 새 API 호출은0회다.','',
        '## COMMON18 test200 부분집합','',
        '| 조건 | 로컬 평균 ms | 길이/최단 GM | 최단 일치율 |',
        '|---|---:|---:|---:|']
    for m in METHODS:
        v=s['groups']['common18_test200']['absolute'][m]
        lines.append(f'| {m} | {1000*v["means"]["local_seconds"]:.3f} | {100*v["stretch_gm"]:.3f}% | {v["exact_percent"]:.1f}% |')
    lines+=['','이 200문제는 전체의 부분집합이다. 과거 COMMON18의 새 응답 대신 이번 전체와 같은 2026-10-13 응답을 사용하므로 과거 표와 수치가 달라질 수 있다.','',
        '## 실행 범위와 제한','',
        '- 각 문제·조건을 정확히1회 실행했다. 추가 반복·시간 중앙값·확률 시드 평균은 없다. CPU/native1스레드, 순서만 고정 seed로 섞었다.',
        '- 기존 GPT-4o mini standard 1,000응답을 요청·입력 SHA와 대조 후 고정했다. 새 API 호출은0회이고 응답 재생 중 네트워크 연결을 차단했다.',
        '- 총시간은 과거 API 대기＋현재 로컬 측정의 재구성 추정이다. 새로운 온라인 서비스 속도나 API 지연 반복 실측이 아니다.',
        '- 파일 읽기·독립 Dijkstra는 타이머 밖이고 지도 준비·파싱·수리·탐색·복원·필수 검증은 타이머 안이다.',
        '- 지도 bootstrap2000회 CI는 문제 구성의 탐색적 불확실성이다. 시간1회와 과거 API 지연의 반복 안정성은 포함하지 않는다.',
        '- 공개 전체는 기존 학습·설정 선택과 평가에 노출된 자료를 포함한다. 전체 결과를 새 hidden test 일반화로 해석하지 않는다.',
        '- 저장 상태·OPEN peak는 논리 개수다. 별도 RSS 측정은 하지 않았으며 실제 메모리 bytes나 총 CPU 연산 수로 표현하지 않는다.',
        '- J1 1.05와 straight 1.05가 직접 대조다. J1_exact와 straight는 w가 달라 직접 기여 대조로 사용하지 않는다.',
        '', '![J1 비교](comparison.png)','',
        '[사전 규약](../../PROTOCOL.md), [문제별 수치](per_case.csv), [전체 집계](summary.json), [원시 시행](evaluations.jsonl.gz), '
        '[저장 응답](responses.jsonl.gz), [실행·소스 기록](protocol.json), [독립 검증](audit.json).','',
        '최초 CHOMP-inspired 구현·실험은 승민, 이번 전체 평가·검토는 준연 담당이다. Codex가 실행 통합·검증·문서 작성을 지원했다. 기존 실험은 보존했다.','']
    (out/'REPORT.md').write_bytes('\n'.join(lines).encode('utf-8'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.input.resolve()
    assert load(out/'complete.json')['complete'];rs=rows(out);protocol=load(out/'protocol.json');test=set(protocol['common18_test_maps'])
    s=dict(groups={'all1000':aggregate(rs),'common18_test200':aggregate([r for r in rs if r['map_index'] in test])},
        report_source_sha256=sha(Path(__file__)),repeats=1,new_api_calls=0)
    save(out/'summary.json',s)
    fields=('case_id','map_index','method','valid','length','optimum','bound','local_seconds','api_seconds',
        'reconstructed_total_seconds','successor_expansions','storage','llm_calls','prompt_tokens','completion_tokens')
    with (out/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rs)
    worst=sorted([r for r in rs if r['method']=='J1'],key=lambda r:r['local_seconds']/next(b['local_seconds'] for b in rs if b['method']=='LLMA' and b['case_id']==r['case_id']),reverse=True)[:10]
    save(out/'worst_local_cases.json',[{k:r[k] for k in fields} for r in worst]);plot(out,s,rs);markdown(out,s)
    print(json.dumps({m:s['groups']['all1000']['absolute'][m] for m in ('LLMA','J1','J1_exact')},ensure_ascii=False))


if __name__=='__main__':main()
