"""Regenerate statistics and a Korean HTML report from verified per-case files."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from core import ROOT

OUT=ROOT/'results/candidates_v1'

def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def geomean(x): return float(np.exp(np.mean(np.log(np.maximum(x,1e-12)))))

def bootstrap_map(logvalues,seed=2026):
    means=np.array([np.mean(v) for _,v in sorted(logvalues.items())]); rng=np.random.default_rng(seed)
    draws=np.exp(rng.choice(means,size=(2000,len(means)),replace=True).mean(axis=1))
    return [float(np.quantile(draws,q)) for q in (.025,.975)]

def report():
    output=OUT/'report'; output.mkdir(parents=True,exist_ok=True)
    verification=json.loads((OUT/'verification.json').read_text()); assert verification['verified']
    allrows=[]; summaries=[]; coverages={}; repair={}; paired={}
    for stage in ['main','ood','scaling']:
        rows=[r for p in sorted((OUT/'evaluation'/stage).glob('*.json')) for r in json.loads(p.read_text())['rows']]
        allrows.extend(rows); reference={r['case_id']:r for r in rows if r['method']=='FOCAL_EUCLIDEAN'}
        groups=defaultdict(list)
        for r in rows: groups[r['method']].append(r)
        for method,rr in sorted(groups.items()):
            logs=defaultdict(list); time_logs=defaultdict(list)
            for r in rr:
                b=reference[r['case_id']]
                logs[r['map_id']].append(math.log(max(r['expansions'],1)/max(b['expansions'],1)))
                time_logs[r['map_id']].append(math.log(r['local_cold_seconds']/b['local_cold_seconds']))
            ci=bootstrap_map(logs); tci=bootstrap_map(time_logs)
            sums=dict(stage=stage,method=method,n=len(rr),valid_fraction=np.mean([r['valid'] for r in rr]).item(),
                expansions_mean=float(np.mean([r['expansions'] for r in rr])),
                expansions_ratio=geomean([max(r['expansions'],1)/max(reference[r['case_id']]['expansions'],1) for r in rr]),
                expansions_ci_low=ci[0],expansions_ci_high=ci[1],
                length_ratio=geomean([r['length_ratio'] for r in rr]),max_length_ratio=max(r['length_ratio'] for r in rr),
                preparation_ms_mean=float(np.mean([r['preparation_seconds'] for r in rr])*1000),
                search_ms_mean=float(np.mean([r['search_seconds'] for r in rr])*1000),
                local_cold_ms_mean=float(np.mean([r['local_cold_seconds'] for r in rr])*1000),
                local_cold_ratio=geomean([r['local_cold_seconds']/reference[r['case_id']]['local_cold_seconds'] for r in rr]),
                local_cold_ci_low=tci[0],local_cold_ci_high=tci[1],
                historical_llm_seconds_mean=float(np.mean([r['historical_llm_seconds'] for r in rr])),
                llm_responses_required=rr[0]['llm_responses_required'])
            summaries.append(sums)
            if method=='CONFORMAL':
                mapcovered=defaultdict(list)
                for r in rr: mapcovered[r['map_id']].append(r['metadata']['canonical_path_covered'])
                coverages[stage]=dict(cases=len(rr),path_coverage=float(np.mean([r['metadata']['canonical_path_covered'] for r in rr])),
                   maps=len(mapcovered),simultaneous_map_coverage=float(np.mean([all(v) for v in mapcovered.values()])),
                   average_set_fraction=float(np.mean([r['metadata']['set_fraction'] for r in rr])),
                   average_radius=float(np.mean([r['metadata']['radius'] for r in rr])),coverage_claim=stage=='main')
            if method.startswith('CHOMP'):
                rejected=[r for r in rr if not r['metadata']['input_valid']]
                repair[stage+'_'+method]=dict(total=len(rr),accepted=sum(r['metadata']['repaired'] for r in rr),
                    rejected_inputs=len(rejected),recovered=sum(r['metadata']['repaired'] for r in rejected))
        if stage!='ood':
            for method,control in [('CONFORMAL','FIXED_TUBE_2'),('CHOMP_STANDARD','VGATE_STANDARD'),('CHOMP_STANDARD','CHOMP_STRAIGHT'),('FMM_LLM','FMM_UNIFORM')]:
                base={r['case_id']:r for r in groups[control]}; logvals=defaultdict(list)
                for r in groups[method]: logvals[r['map_id']].append(math.log(max(r['expansions'],1)/max(base[r['case_id']]['expansions'],1)))
                paired[stage+'_'+method+'_vs_'+control]=dict(expansion_ratio=float(np.exp(np.mean(sum(logvals.values(),[])))),ci=bootstrap_map(logvals))
    seeds={str(seed):json.loads((OUT/'models'/f'seed{seed}.json').read_text()) for seed in (11,22,33)}
    training_seconds=sum(v['seconds'] for v in seeds.values()); calibration=json.loads((OUT/'calibration.json').read_text())
    summary=dict(summary=summaries,conformal=coverages,repair=repair,paired=paired,
                 training_seconds=training_seconds,training=seeds,calibration_quantile=calibration['quantile'],
                 dataset_preparation=json.loads((OUT/'data_manifest.json').read_text()),verification=verification)
    save(output/'summary.json',summary)
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    fields=['stage','case_id','map_id','method','optimal','length','length_ratio','expansions','discovered','peak_open','preparation_seconds','search_seconds','graph_seconds','local_cold_seconds','historical_llm_seconds','llm_responses_required','valid']
    with (output/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in allrows)
    css='body{font-family:system-ui,sans-serif;color:#173044;background:#f4f7f9;margin:0}main{max-width:1280px;margin:auto;padding:36px}h1{font-size:30px}h2{margin-top:36px}p,li{line-height:1.7}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dde5eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8eff4}.card{background:white;border:1px solid #dce5ec;padding:20px;border-radius:12px;margin:18px 0}.note{color:#526779}a{color:#036f88}code{font-size:12px}'
    body=f'<h1>네 가지 개선 후보 · GPU 서버 실험</h1><p>2026-10-14 · 총 {verification["total_rows"]:,}개 평가 · 경로 검증 및 1.1배 길이 상한 검증 완료</p>'
    body+='<div class="card"><b>비교 기준</b><p>같은 격자 그래프·FOCAL 구현(w=1.1)에서 안내 정보만 바꿨습니다. 주 평가 20개 지도 200문제, 별도 미로 20개 지도 100문제, 확대 지도 20문제입니다. 학습 50개·검증 10개·보정 20개·평가 20개 지도는 서로 겹치지 않습니다. 프롬프트에 등장하는 예시 지도 5개는 학습에만 포함했습니다.</p><p>GPT 3.5 Turbo의 기존 검증된 응답을 사용했으며, 새 API 호출은 0회입니다. 학습한 신경망은 RTX GPU에서 실행했습니다. CHOMP와 TransPath는 각 논문의 핵심 아이디어를 이 문제에 맞게 구현한 계열 실험이며 원 논문의 정확 재현이 아닙니다.</p></div>'
    body+='<p>탐색량 비율은 같은 문제의 FOCAL_EUCLIDEAN 대비 기하평균입니다. 1보다 작으면 감소입니다. 괄호는 지도 단위 bootstrap 95% 구간입니다. 0회 확장으로 검증된 수리 경로를 바로 반환한 경우 비율 계산에만 1을 사용하며 평균 확장 수에는 실제 0을 유지했습니다.</p>'
    for stage,title in [('main','주 평가 · 미관측 원본 지도'),('ood','분포 밖 평가 · 새 미로 지도'),('scaling','확대 평가 · 2배 / 4배')]:
        body+=f'<h2>{title}</h2><table><tr><th>방법</th><th>N</th><th>유효</th><th>확장 수 평균</th><th>탐색량 비율 [95% CI]</th><th>길이 / 최적</th><th>준비 ms</th><th>탐색 ms</th><th>로컬 총 ms</th><th>로컬 시간 비율</th></tr>'
        for r in summaries:
            if r['stage']!=stage: continue
            body+=f'<tr><td>{r["method"]}</td><td>{r["n"]}</td><td>{100*r["valid_fraction"]:.0f}%</td><td>{r["expansions_mean"]:.1f}</td><td>{r["expansions_ratio"]:.3f} [{r["expansions_ci_low"]:.3f}, {r["expansions_ci_high"]:.3f}]</td><td>{r["length_ratio"]:.4f}</td><td>{r["preparation_ms_mean"]:.2f}</td><td>{r["search_ms_mean"]:.2f}</td><td>{r["local_cold_ms_mean"]:.2f}</td><td>{r["local_cold_ratio"]:.3f}</td></tr>'
        body+='</table>'
    cp=coverages['main']
    body+=f'<h2>Conformal: 포함률과 탐색 효과</h2><p>LLM이 제안한 세 경로 주변의 격자 후보 영역을 보정했습니다. 보정 점수는 지도 내 10문제의 최적 경로 전체에서 가장 멀리 벗어난 정도입니다. α=0.1, 보정 지도 20개의 19번째 순서통계량 {calibration["quantile"]:.4f}를 사용했습니다. 주 평가에서 기준 최적 경로 전체 포함률은 문제 단위 {cp["path_coverage"]:.1%}, 지도 내 10문제 동시 포함률은 {cp["simultaneous_map_coverage"]:.1%}입니다. 평균 후보 영역은 자유 칸의 {cp["average_set_fraction"]:.1%}였습니다.</p>'
    body+='<p>90% 보장은 교환가능한 지도라는 가정 아래의 주변적 보장입니다. 지도 20개에서 관측한 포함률은 불확실성이 큽니다. 이는 경로 후보 3개 중 정답을 분류한 실험이 아니라 <b>격자 후보 영역의 포함률을 보정한 변형</b>입니다. 최적 경로 하나를 고정해 보정·평가하며, 모든 최적 경로의 포함을 보장하지 않습니다. 후보 밖 칸을 제거하지 않아 포함에 실패해도 탐색의 길이 상한은 유지됩니다. 확대·미로 데이터에는 포함률 보장을 주장하지 않습니다.</p>'
    body+='<h2>수리·학습·대조군</h2><ul>'
    for name,value in repair.items():
        body+=f'<li>{name}: {value["total"]}개 중 수리 경로 승인 {value["accepted"]}개, 처음에 충돌 또는 경계 오류가 있던 {value["rejected_inputs"]}개 중 {value["recovered"]}개 복구.</li>'
    body+=f'<li>학습: 3개 시드 × 60 epoch, GPU 학습 총 {training_seconds/60:.2f}분. 모델 선택은 검증 손실만 사용했습니다. 학습 정답 생성 {summary["dataset_preparation"]["seconds"]:.1f}초와 보정 {calibration["seconds"]:.1f}초는 별도 준비 비용입니다.</li>'
    body+='<li>UNTRAINED_11은 같은 구조의 학습 전 네트워크입니다. CHOMP_STRAIGHT는 LLM 없이 직선에서 시작하는 수리 대조군입니다. FMM_UNIFORM은 LLM 없이 파동 도착 시간만 계산합니다. FIXED_TUBE_2는 보정 없이 폭 2의 후보 영역을 사용합니다.</li></ul>'
    body+='<h2>시간·해석의 한계</h2><p>로컬 총 시간에는 매 문제의 격자 그래프 생성, 보정 영역·수리·FMM 지도 생성 또는 GPU 입력 전송·추론·출력 전송, 실제 탐색을 포함합니다. 신경망은 메모리에 상주하며 warm-up 시간은 원자료에 별도 저장했습니다. GT Dijkstra는 평가용이며 온라인 시간에서 제외했습니다. 전체 지도 탐색에 가까운 FMM 준비 비용은 누락하지 않았습니다. Python FOCAL의 OPEN 스캔 구현이라 최적화된 생산용 탐색기의 성능을 대표하지 않습니다. 방법별 실행 순서가 고정된 단회 측정이므로 시간 수치는 기술적 관찰이며 정밀 벤치마크가 아닙니다.</p>'
    body+='<p>LLM_TUBE·FIXED_TUBE_2·CONFORMAL·FMM_LLM에는 과거 응답 3개, VGATE_STANDARD·CHOMP_STANDARD에는 1개가 필요합니다. API 지연은 CSV의 historical_llm_seconds에 별도 기록했으며 현재 측정한 온라인 지연으로 주장하지 않습니다. LLM 없는 방법과 비교할 때 이 지연과 비용을 추가해야 합니다. 확대 평가는 기존 응답의 좌표도 함께 확대했으므로 새 큰 지도를 GPT가 직접 해석한 성능을 측정한 것은 아닙니다.</p>'
    body+='<h2>계열별 짝 비교</h2><table><tr><th>비교</th><th>탐색량 비율</th><th>95% CI</th></tr>'
    for name,value in paired.items(): body+=f'<tr><td>{name}</td><td>{value["expansion_ratio"]:.3f}</td><td>{value["ci"][0]:.3f}–{value["ci"][1]:.3f}</td></tr>'
    body+='</table><h2>원자료와 재현</h2><p><a href="summary.csv">집계 CSV</a> · <a href="per_case.csv">문제별 CSV</a> · <a href="summary.json">전체 통계 JSON</a> · <a href="../protocol.json">고정 프로토콜</a> · <a href="../verification.json">검증 기록</a></p><p class="note">Conformal: <a href="https://arxiv.org/abs/2107.07511">Angelopoulos & Bates</a> · <a href="https://publications.ri.cmu.edu/chomp-covariant-hamiltonian-optimization-for-motion-planning">CHOMP</a> · <a href="https://math.berkeley.edu/~sethian/2006/Applications/Robotics/hold_robotics.html">Fast Marching</a> · <a href="https://ojs.aaai.org/index.php/AAAI/article/view/26465">TransPath</a></p>'
    (output/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>네 가지 개선 후보 실험</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
    print(json.dumps({'report':str(output/'report.html'),'rows':len(allrows)}),flush=True)

if __name__=='__main__': report()
