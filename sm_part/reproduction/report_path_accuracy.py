"""Create separate Markdown/HTML reports for the audited secondary analysis."""
import base64
import html
import json
import os
from pathlib import Path

from core import ROOT

OUT = ROOT / 'results/path_accuracy'
KINDS = ['standard', 'cot', 'repe']
PROMPTS = {'standard': 'Few-shot', 'cot': 'CoT', 'repe': 'RePE', None: '공통'}
LABELS = {'A': 'A*', 'D': 'D: 동적 가중 A* (재현 구현)', 'N': 'N: GPT 없는 g+2h',
          'B': 'B: 기본 LLM-A*', 'E1': 'E1: 경유점 정제', 'E2C': 'E2C: 일반 재검토',
          'E2F': 'E2F: 오류 피드백', 'E3': 'E3: 조건부 호출'}


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def figures(data):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {(r['method'], r['prompt']): r for r in data['main']}
    colors = {'B': '#64748b', 'E1': '#0f766e', 'E2C': '#b45309', 'E2F': '#7c3aed', 'E3': '#2563eb'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, metric, title in zip(axes, ['optimal_rate_percent', 'mean_excess_percent'],
                                ['Optimal-path rate (%)', 'Mean excess over optimal length (%)']):
        for j, method in enumerate(colors):
            ax.bar([i+(j-2)*.15 for i in range(3)], [lookup[method,k][metric] for k in KINDS],
                   width=.14, label=method, color=colors[method])
        ax.axhline(lookup['N',None][metric], color='#1f2937', linestyle='--', label='N (no GPT)')
        ax.set_xticks(range(3), ['Few-shot', 'CoT', 'RePE'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, 100)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylim(bottom=0)
    fig.savefig(OUT / 'main_accuracy.png', dpi=170)
    fig.savefig(OUT / 'main_accuracy.pdf')
    plt.close(fig)
    paired = {(r['treatment'], r['control'], r['prompt']): r for r in data['main_paired']}
    entries = [(t,c,k) for t,c in [('E1','B'), ('E2F','E2C'), ('E3','B')] for k in KINDS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout='constrained')
    for ax, metric, title in zip(axes, ['optimal_delta_pp', 'mean_excess_reduction_pp'],
                                ['Optimal-path rate gain (pp)', 'Mean excess reduction (pp)']):
        for i, key in enumerate(entries):
            r = paired[key]
            ax.errorbar(r[metric], i, xerr=[[r[metric]-r[metric+'_ci95_low']], [r[metric+'_ci95_high']-r[metric]]],
                        fmt='o', color=colors[key[0]], capsize=4)
        ax.axvline(0, color='#334155', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(entries)), [f'{t}/{c} - {PROMPTS[k]}' for t,c,k in entries])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Positive = improvement; exploratory 95% map-bootstrap CI')
    fig.savefig(OUT / 'paired_accuracy_ci.png', dpi=170)
    fig.savefig(OUT / 'paired_accuracy_ci.pdf')
    plt.close(fig)


class Document:
    def __init__(self):
        self.md, self.web = [], []

    def heading(self, title, level=2):
        self.md.append('#'*level+' '+title)
        self.web.append(f'<h{level}>{html.escape(title)}</h{level}>')

    def paragraph(self, text):
        self.md.append(text)
        self.web.append('<p>'+html.escape(text)+'</p>')

    def table(self, headers, rows):
        self.md.append('\n'.join('| '+' | '.join(str(x).replace('|','\\|') for x in r)+' |' for r in [headers, ['---']*len(headers), *rows]))
        self.web.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+
                        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>')

    def figure(self, filename, caption):
        path = OUT / filename
        self.md.append(f'![{caption}]({path.as_posix()})')
        encoded = base64.b64encode(path.read_bytes()).decode()
        self.web.append(f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')

    def links(self, pairs):
        self.md.append('\n'.join(f'- [{label}]({path.as_posix()})' for label,path in pairs))
        self.web.append('<ul>'+''.join(f'<li><a href="{os.path.relpath(path,OUT).replace(chr(92), chr(47))}">{html.escape(label)}</a></li>' for label,path in pairs)+'</ul>')

    def write(self):
        (OUT / 'report.md').write_text('\n\n'.join(self.md)+'\n', encoding='utf-8')
        css = '''body{font:16px/1.75 "Malgun Gothic",sans-serif;color:#182a3b;background:#f1f5f9;margin:0}main{max-width:1180px;padding:40px;margin:28px auto;background:white;border-radius:16px}h1{font-size:30px}h2{font-size:23px;color:#155e75;margin-top:36px}h3{font-size:18px}p{max-width:1080px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #dce4eb;white-space:nowrap;text-align:right}th{background:#eaf2f6}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:12px;color:#546477}a{color:#176d84}li{margin:6px 0}@media(max-width:700px){main{padding:18px;margin:0;border-radius:0}}@media print{body{background:white}main{padding:0;margin:0}h2{break-after:avoid}tr,figure{break-inside:avoid}}'''
        (OUT / 'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>최단 경로 정확도 보조 분석</title><style>'+css+'</style><main>'+''.join(self.web)+'</main></html>', encoding='utf-8')


def main():
    data, manifest = read('analysis.json'), read('analysis_manifest.json')
    assert manifest['complete'] and manifest['reference_cases'] == 1100 and manifest['verified_grid_paths'] == 18700
    figures(data)
    lookup = {(r['method'],r['prompt']): r for r in data['main']}
    pairs = {(r['treatment'],r['control'],r['prompt']): r for r in data['main_paired']}
    doc = Document()
    doc.heading('최단 경로 정확도 및 초과 길이 보조 분석 보고서', 1)
    doc.paragraph('분석일: 2026-10-14 | 기존 GPT 개선 실험의 저장 경로 재분석 | 추가 API 호출 0건')
    doc.paragraph('본실험에서는 1번 경유점 정제와 3번 조건부 호출이 기본 LLM-A*보다 최단 경로를 정확히 찾은 비율과 평균 초과 길이를 함께 개선했습니다. 1번의 개선 폭이 더 컸습니다. 다만 지도 확대 평가에서는 이 효과가 유지되지 않아, 모든 환경에서 정확도가 좋아진다고 일반화할 수 없습니다.')
    doc.table(['개선안','본실험의 핵심 결과','판단'],[
        ['1번 E1','최단 경로 일치율 +2.9~8.0%p; 평균 초과율 0.268~0.672%p 감소','본실험 정확도 개선 근거가 가장 뚜렷함'],
        ['2번 E2F','동일 호출 대조군 대비 일치율 +0.1~1.4%p; 탐색적 구간은 0 포함','본실험의 오류 피드백 고유 효과는 불확실'],
        ['3번 E3','최단 경로 일치율 +2.6~7.2%p; 평균 초과율 0.219~0.600%p 감소','본실험에서 정확도와 호출 효율 동시 개선']])
    doc.paragraph('이 분석은 결과를 확인한 뒤 추가한 사후 보조 분석입니다. 아래 95% 구간은 다중비교를 보정하지 않은 탐색적 구간이며, 사전 등록된 확증 검정이나 새로운 데이터에서의 재현 결과가 아닙니다.')
    doc.heading('1. 분석 대상과 최단 경로 정의')
    doc.paragraph('본실험은 100개 지도 × 지도당 10문제 = 1,000문제입니다. 각 개선안은 동일 문제에서 Few-shot·CoT·RePE를 평가했습니다. 확장성 평가는 10개 시작·목표 문제를 1~10배 확대한 100개 조건입니다. 이 10문제는 서로 다른 9개 지도에 속하므로, 100개의 독립 지도로 해석하면 안 됩니다. 개발용 10문제는 본 분석에서 제외했습니다.')
    doc.paragraph('최단 경로는 기존 탐색과 동일한 8방향 격자 그래프의 최단 경로입니다. 가로·세로 한 칸 비용은 1, 대각선 한 칸 비용은 √2이며, 기존 경계와 선분 충돌 규칙을 유지했습니다. 연속 공간에서 자유롭게 선분을 연결하는 경로의 최적값을 뜻하지 않습니다.')
    doc.paragraph('휴리스틱·GPT 경유점 로직이 없는 별도 Dijkstra 구현으로 본실험 1,000문제와 확장성 100조건의 최적 길이를 계산했습니다. 1,100개 모두 저장된 A* 길이와 일치했습니다. 최단 경로의 좌표열이 달라도 길이가 같으면 성공으로 인정합니다. 부동소수점 오차 허용치는 절대 10⁻⁸, 상대 10⁻¹⁰입니다.')
    doc.paragraph('대상 방법은 A*, 동적 가중 A*(D), GPT 없는 g+2h(N), 기본 LLM-A*(B), E1, E2C, E2F, E3입니다. GPT 원출력은 긴 직선 구간이나 비정수 좌표를 포함할 수 있어 격자 최적값과 공정하게 비교할 수 없으므로 제외했습니다. 기존 보고서의 GPT 출력 유효율 평가는 그대로 유지합니다.')
    doc.table(['지표','계산 및 분모'],[
        ['최단 경로 일치율','유효하고 최적 길이와 같은 경로 수 / 전체 평가 문제 수 × 100'],
        ['절대 초과 길이','반환 길이 L − 최적 길이 L*; 단위는 격자 좌표 길이이며 미터가 아님'],
        ['상대 초과율(%)','100 × (L − L*) / L*'],
        ['평균 초과 길이·초과율','유효 경로의 문제별 값을 산술평균; 0도 포함'],
        ['P95 초과율','문제별 초과율의 95백분위수; 하위 95%가 이 값 이하'],
        ['개선 폭','일치율은 실험군−대조군(%p), 초과율 감소는 대조군−실험군(%p)']])
    doc.paragraph('모든 대상 최종 경로가 유효해 본실험의 각 조건 분모는 1,000, 확장성은 100입니다. 유효율 100%는 최단 경로 일치율 100%와 다릅니다. 평균 초과율은 기존 보고서의 길이 비율 기하평균과 집계 방식이 다르므로 숫자를 직접 혼용하지 않습니다. 예를 들어 초과율 2.91%→2.24%는 0.67%p 감소입니다.')
    doc.heading('2. 본실험: 최단 경로를 정확히 찾은 비율')
    doc.table(['방법','Few-shot','CoT','RePE'],[
        [LABELS[m]]+[f"{lookup[m,k]['optimal_n']}/1000 ({lookup[m,k]['optimal_rate_percent']:.1f}%)" for k in KINDS]
        for m in ['B','E1','E2C','E2F','E3']])
    doc.paragraph('공통 대조군의 일치율은 A* 100.0%, D 93.6%, N 63.7%입니다. D는 기존 재현의 동적 가중치 구현이며 본 분석에서 변경하지 않았습니다. 정확도만 보면 이들 고전 탐색 대조군도 반드시 함께 고려해야 합니다.')
    doc.heading('3. 본실험: 최단 경로 대비 초과 길이')
    rows = []
    for m in ['A','D','N','B','E1','E2C','E2F','E3']:
        for k in ([None] if m in ['A','D','N'] else KINDS):
            r = lookup[m,k]
            rows.append([LABELS[m],PROMPTS[k],f"{r['mean_excess_length']:.4f}",f"{r['mean_excess_percent']:.4f}%",f"{r['median_excess_percent']:.2f}%",f"{r['p95_excess_percent']:.2f}%"])
    doc.table(['방법','프롬프트','평균 절대 초과 길이','평균 초과율','중앙값 초과율','P95 초과율'],rows)
    doc.paragraph('모든 본실험 조건은 절반 이상에서 최단 경로를 찾아 중앙값 초과율이 0%입니다. 따라서 평균과 P95도 함께 봐야 합니다. 일부 조건의 평균 개선이 긴 우회 경로까지 일관되게 줄였다는 뜻은 아닙니다.')
    doc.figure('main_accuracy.png','본실험 최단 경로 일치율과 평균 초과율. 점선은 GPT 없는 N 대조군.')
    doc.heading('4. 대응 비교와 탐색적 신뢰구간')
    doc.paragraph('동일 문제의 실험군·대조군을 짝지어 비교했습니다. 지도 안의 10문제를 독립으로 가정하지 않고, 100개 지도를 복원추출하는 군집 부트스트랩을 10,000회 수행했습니다(고정 시드 240702512). 표는 백분위 95% 구간이며, 모든 개선 폭은 양수일 때 개선입니다. 지도 구성에 따른 변동을 표현하며 반복 API 호출의 변동까지 추정하지는 않습니다.')
    rows = []
    for t,c in [('E1','B'),('E2F','E2C'),('E3','B')]:
        for k in KINDS:
            r = pairs[t,c,k]
            rows.append([f'{t} / {c}',PROMPTS[k],f"{r['optimal_delta_pp']:+.1f} [{r['optimal_delta_pp_ci95_low']:+.1f}, {r['optimal_delta_pp_ci95_high']:+.1f}]",f"{r['mean_excess_reduction_pp']:+.4f} [{r['mean_excess_reduction_pp_ci95_low']:+.4f}, {r['mean_excess_reduction_pp_ci95_high']:+.4f}]",f"{r['became_optimal_n']} / {r['lost_optimal_n']}"])
    doc.table(['실험군 / 대조군','프롬프트','일치율 차이 %p [95% 구간]','평균 초과율 감소 %p [95% 구간]','최적 획득 / 상실 문제 수'],rows)
    doc.paragraph('E1/B와 E3/B는 세 프롬프트 모두 두 지표의 탐색적 구간이 0보다 큽니다. 이번 데이터에서는 본실험 정확도 개선을 지지하는 결과입니다. E2F/E2C는 두 지표 모두 세 프롬프트에서 구간이 0을 포함하므로, 오류 정보 제공 자체의 효과가 명확하다고 판단하지 않습니다.')
    doc.paragraph('비교는 여러 방법·프롬프트·지표에 걸쳐 수행됐고 다중비교 보정을 하지 않았습니다. 따라서 구간의 0 포함 여부를 확정적인 유의성 판정으로 사용하지 않으며, 후속 독립 실험에서 재확인이 필요합니다.')
    doc.figure('paired_accuracy_ci.png','주요 대응 비교의 지도 단위 부트스트랩 95% 구간. 두 패널 모두 양수일 때 개선.')
    doc.heading('5. GPT 없는 대조군과 비교하면 어떤가?')
    rows = []
    for t in ['E1','E3']:
        for k in KINDS:
            r = pairs[t,'N',k]
            rows.append([t,PROMPTS[k],f"{r['optimal_delta_pp']:+.1f} [{r['optimal_delta_pp_ci95_low']:+.1f}, {r['optimal_delta_pp_ci95_high']:+.1f}]",f"{r['mean_excess_reduction_pp']:+.4f}"])
    doc.table(['개선안 / N','프롬프트','일치율 차이 %p [95% 구간]','평균 초과율 감소 %p'],rows)
    doc.paragraph('E1과 E3가 기본 LLM-A*를 개선했다는 사실과 GPT 없는 탐색보다 우수하다는 주장은 구분해야 합니다. N 대비 우위를 일관되게 보이지 않았으며, CoT에서는 일치율이 오히려 낮았습니다. E1의 결과는 부정확한 경유점의 영향을 줄인 효과와 부합하지만, 그 기여를 분리하는 추가 실험 없이 단일 원인으로 확정하지 않습니다.')
    doc.heading('6. 지도 확대 평가: 개선이 그대로 유지되는가?')
    scale = {r['method']:r for r in data['scaling']}
    doc.table(['방법','최단 경로 일치율','평균 절대 초과 길이','평균 초과율','P95 초과율'],[
        [LABELS[m],f"{scale[m]['optimal_n']}/100 ({scale[m]['optimal_rate_percent']:.1f}%)",f"{scale[m]['mean_excess_length']:.4f}",f"{scale[m]['mean_excess_percent']:.4f}%",f"{scale[m]['p95_excess_percent']:.2f}%"]
        for m in ['A','N','B','E1','E2C','E2F','E3']])
    doc.paragraph('E1은 기본 대비 최단 경로 일치율이 41%→40%로 낮아졌고 평균 초과율은 3.1233%→3.5639%로 증가했습니다. E3는 일치율이 41%로 같고 평균 초과율은 3.6679%로 증가했습니다. 본실험의 정확도 개선이 확대 조건에서도 유지된다는 증거는 없습니다.')
    doc.paragraph('E2F는 E2C 대비 일치율이 38%→40%, 평균 초과율이 2.4172%→2.0950%로 개선됐습니다. 이 부분은 추가 검증할 가치가 있습니다. 다만 10문제를 반복 확대한 작은 표본이며 같은 문제의 배율 조건들은 독립이 아닙니다. 확장성 집계에는 독립 100표본을 가정한 유의성 검정을 적용하지 않았습니다.')
    doc.paragraph('E3의 평균 절대 초과 길이는 기본보다 조금 작지만 평균 상대 초과율은 더 큽니다. 절대 길이와 문제별 최단 길이로 정규화한 상대 비율은 가중 방식이 달라 반대 방향일 수 있습니다. 배율별 세부 결과는 별도 CSV로 제공합니다.')
    doc.heading('7. 예시 중복 제외 분석과 해석 범위')
    sensitivity = {(r['treatment'],r['control'],r['prompt']):r for r in data['paired_without_demo_overlap']}
    doc.table(['비교','Few-shot 일치율 차이','CoT 일치율 차이','RePE 일치율 차이'],[
        [f'{t}/{c}']+[f"{sensitivity[t,c,k]['optimal_delta_pp']:+.2f}%p" for k in KINDS]
        for t,c in [('E1','B'),('E2F','E2C'),('E3','B')]])
    doc.paragraph('프롬프트 예시와 중복되는 5문제를 모든 조건에서 공통 제외한 995문제에서도 결론의 방향은 유지됐습니다. E1/B·E3/B의 두 지표 구간은 0보다 크고 E2F/E2C 구간은 0을 포함했습니다. 이는 예시 중복에 대한 민감도 확인이며, 새로운 미관측 데이터 검증은 아닙니다.')
    doc.paragraph('원래 실험의 모델·프롬프트·경로·탐색 구현은 그대로 사용했습니다. 원래 실험에서 있었던 Windows→Linux 호스트 변경, 크레딧 중단, E3 캐시 재생의 실행 순서 예외도 이 자료에 그대로 적용됩니다. 원래 실험의 한계가 이번 보조 분석으로 해소되는 것은 아닙니다.')
    doc.paragraph('과제의 정확도 관련 핵심 결과는 “E1과 E3가 기존 LLM-A* 대비 본실험의 최단 경로 일치율과 평균 초과 길이를 개선했다”로 제시할 수 있습니다. 동시에 “GPT 없는 대조군을 일관되게 능가하지 못했고, 지도 확대 조건에서는 정확도 개선이 유지되지 않았다”는 범위를 함께 제시해야 합니다. E2F의 확대 조건 초과 길이 개선은 후속 연구 후보입니다.')
    doc.heading('8. 검증 및 재현 자료')
    doc.table(['항목','검증 결과'],[
        ['최단 길이 독립 검증','1,100문제/조건, 저장 A*와 불일치 0건'],
        ['최종 격자 경로 검증','본실험 18,000행 + 확장성 700행 = 18,700행'],
        ['충돌 규칙 대조',f"{sum(a['collision_rule_edge_checks'] for a in manifest['audit']):,}개 간선 대조 통과; 본실험 크기는 전체, 큰 지도는 결정적 표본 대조"],
        ['부동소수점 최대 기준 차이',f"{data['maximum_astar_dijkstra_difference']:.3g}"],
        ['원자료 무결성','ZIP SHA-256, 분석 대상 rows.json 해시, 고정 코드·데이터 해시 확인'],
        ['새 API 호출','0건; 저장 경로의 보조 지표 계산만 수행']])
    doc.links([
        ('본실험 지표 CSV',OUT/'main_summary.csv'),('본실험 대응 비교 및 95% 구간 CSV',OUT/'main_paired_comparisons.csv'),
        ('본실험 문제별 결과 CSV',OUT/'main_per_case.csv'),('확장성 지표 CSV',OUT/'scaling_summary.csv'),
        ('배율별 확장성 CSV',OUT/'scaling_by_factor.csv'),('확장성 문제별 결과 CSV',OUT/'scaling_per_case.csv'),
        ('예시 중복 제외 지표 CSV',OUT/'main_without_demo_overlap.csv'),('예시 중복 제외 대응 비교 CSV',OUT/'paired_without_demo_overlap.csv'),
        ('Dijkstra 최적 길이 참조 CSV',OUT/'optimal_reference.csv'),('분석 설정·무결성 기록',OUT/'analysis_manifest.json'),
        ('전체 분석 JSON',OUT/'analysis.json'),('분석 코드',ROOT/'reproduction/analyze_path_accuracy.py'),
        ('기존 실험 보고서',ROOT/'results/improvements_final/report.md')])
    doc.paragraph('원자료 ZIP SHA-256: '+manifest['source_archive_sha256'])
    doc.paragraph('원래 실험 프로토콜 서명: '+manifest['original_protocol_signature'])
    doc.write()
    print(json.dumps({'report_md': str(OUT/'report.md'), 'report_html': str(OUT/'report.html'), 'complete': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
