"""Create an explanatory final report from verified, immutable experiment outputs."""
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/improvements'
DEST = ROOT / 'results/improvements_final'
KINDS = ['standard', 'cot', 'repe']
NAMES = {'standard': 'Few-shot', 'cot': 'CoT', 'repe': 'RePE', None: '공통'}
METHODS = {'A': 'A*', 'N': 'N: GPT 없는 g+2h', 'B': 'B: 기본 LLM-A*',
           'E1': 'E1: 경유점 정제', 'E2C': 'E2C: 일반 재검토',
           'E2F': 'E2F: 오류 피드백', 'E3': 'E3: 조건부 호출'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        '<th>' + html.escape(str(v)) + '</th>' for v in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def main():
    verified = read(SOURCE / 'verification.json')
    imported = read(SOURCE / 'local_import_verification.json')
    report = read(SOURCE / 'report/summary.json')
    protocol = read(SOURCE / 'protocol.json')
    deviation = read(SOURCE / 'execution_deviation.json')
    deployment = read(ROOT / 'results/improvements_deployment.json')
    assert verified['complete'] and imported['complete'] and verified['total_rows'] == 28270
    assert deviation['complete']
    lookup = {(r['method'], r['prompt']): r for r in report['tables']['main']}
    pairs = {(r['treatment'], r['control'], r['prompt']): r for r in report['paired']}
    DEST.mkdir(parents=True, exist_ok=True)

    body = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM-A* 개선 실험 최종 보고서</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#172334;font:16px/1.7 "Malgun Gothic",sans-serif}
main{max-width:1120px;margin:32px auto;padding:40px;background:#fff;border-radius:16px}
h1{font-size:32px;line-height:1.4}h2{font-size:22px;margin-top:36px;color:#125468}h3{font-size:18px}
.label{font-weight:700;color:#13765e}.lead{font-size:18px}.note{padding:16px 20px;background:#eef8f5;border-left:4px solid #14785f}
.caution{background:#fff8e8;border-color:#b07c15}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}
.card{background:#eef3f8;padding:16px;border-radius:10px}.card strong{display:block;font-size:24px;color:#124e66}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th{background:#eaf1f6}th,td{padding:10px;border-bottom:1px solid #dae2e9;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
img{width:100%;height:auto}a{color:#14647c}code{font-size:12px;overflow-wrap:anywhere}.small{font-size:13px;color:#506174}li{margin:7px 0}
@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}.cards{grid-template-columns:1fr}h1{font-size:26px}}
@media print{body{background:#fff}main{margin:0;padding:0}h2,h3{break-after:avoid}tr,.card{break-inside:avoid}}
</style></head><body><main><div class="label">전체 실행 · 원자료 검증 · 로컬 재검증 완료</div>
<h1>LLM-A* 개선 실험 최종 보고서</h1>
<p class="lead">경유점 정제(E1)와 조건부 호출(E3)은 기본 LLM-A*의 탐색 연산량을 줄였습니다.
오류 피드백(E2)은 같은 호출 횟수의 재검토보다 일관된 탐색 개선을 보이지 않았습니다.</p>
<div class="note caution">세 방법 모두 GPT 없는 g+2h 대조군(N)을 일관되게 능가한다는 증거는 없습니다.
따라서 “GPT 안내 자체의 우월성”보다 “잘못된 안내의 영향 감소”와 “API 호출 절감”으로 해석하는 것이 적절합니다.</div>'''
    body += f'''<div class="cards"><div class="card"><strong>{verified['total_rows']:,}행</strong>전체 평가 기록 검증</div>
<div class="card"><strong>1,000 × 3</strong>각 개선안의 본실험 문제 × 프롬프트</div>
<div class="card"><strong>{verified['total_requests']:,}건</strong>공유·캐시를 반영한 실제 응답 수집</div></div>'''
    body += '<h2>1. 실험 범위와 변인 통제</h2><p>1,000은 학습 step 수가 아니라 고유 경로 탐색 문제 수입니다. '
    body += '각 개선안은 동일한 1,000문제에 Few-shot·CoT·RePE를 적용해 3,000개의 최종 탐색 평가를 수행했습니다. '
    body += '대조군과 GPT-only 평가까지 합친 본실험은 27,000행이며, 개발 270행과 확장성 1,000행을 포함하면 28,270행입니다.</p>'
    body += '<p>기존 GPT 재현의 모델 <b>gpt-3.5-turbo-0125</b>, temperature=0, max_tokens=1000, '
    body += '지도·시작점·목표점, 세 프롬프트, 경계·충돌 판정, 탐색 규칙과 네 평가 지표를 유지했습니다. '
    body += '최초 GPT 응답을 새로 수집해 조건 간 공유하고, 세 개선안을 결합하지 않았습니다. '
    body += 'E2C와 E2F는 수정이 필요한 같은 문제에서 각각 한 번만 추가 호출하고 두 번째 답을 그대로 채택했습니다.</p>'
    body += table(['방법', '유일한 변경 또는 대조 역할'], [
        ['B', '기본 LLM-A*: 최초 GPT 경로를 원래 방식으로 사용'],
        ['E1', '반복 구간 제거 → 충돌 없는 직선 단축 → 검증 실패 시 내부 경유점 불채택'],
        ['E2F', '최초 경로가 무효일 때 기존 검증기의 첫 오류를 제공하여 한 번 수정'],
        ['E2C', 'E2F와 호출 횟수가 같은 비구체적 재검토 대조군'],
        ['E3', 'g+2h 탐색으로 시작하고 고정된 정체 조건에서 GPT를 한 번 호출'],
        ['N', 'GPT 경유점 없는 g+2h 대조군: GPT 안내의 추가 기여 확인'],
    ])
    body += '<p>연산량은 중복을 포함한 heap-pop 수, 저장량은 len(g), 길이는 최종 경로 길이입니다. '
    body += '아래 감소율은 같은 문제의 두 조건 비율을 기하평균한 뒤 100%에서 뺀 값입니다. '
    body += '본실험의 모든 하이브리드 조건은 1,000/1,000문제에서 유효한 경로를 얻었습니다.</p>'

    body += '<h2>2. 핵심 결과</h2>'
    effect_rows = []
    for method, control, label in [('E1', 'B', 'E1 / B 연산량 감소'),
                                   ('E2F', 'E2C', 'E2F / E2C 연산량 감소'),
                                   ('E3', 'B', 'E3 / B 연산량 감소')]:
        effect_rows.append([label] + [f"{100-pairs[method,control,k]['operation_paired_ratio']:+.2f}%" for k in KINDS])
    effect_rows.append(['E3 / B API 호출 감소'] + [f"{100*(1-lookup['E3',k]['api_calls']/lookup['B',k]['api_calls']):.2f}%" for k in KINDS])
    body += table(['비교 (양수 = 감소, 음수 = 증가)', 'Few-shot', 'CoT', 'RePE'], effect_rows)
    body += '<h3>E1 — 경유점 정제</h3><p>기본 LLM-A* 대비 연산량이 감소했지만 N보다 연산량이 적지는 않았습니다. '
    body += '검증 실패로 내부 경유점을 불채택한 경우는 Few-shot 924건, CoT 780건, RePE 701건입니다. '
    body += '개선의 상당 부분은 유용한 GPT 안내 생성보다 부정확한 안내의 제거와 관련된 것으로 해석할 수 있습니다.</p>'
    body += '<h3>E2 — 오류 피드백</h3><p>GPT 출력 자체의 유효율은 소폭 개선됐으나, 같은 호출 횟수의 E2C와 비교한 '
    body += '최종 탐색 연산량은 Few-shot에서만 소폭 감소하고 CoT·RePE에서는 증가했습니다. 오류 정보 제공의 일관된 탐색 개선 효과는 확인되지 않았습니다.</p>'
    body += table(['GPT 출력 유효율', 'Few-shot', 'CoT', 'RePE'], [
        [label] + [f"{lookup[m,k]['valid_ratio']:.1f}%" for k in KINDS]
        for m, label in [('GPT', '최초 출력'), ('GPT_E2C', '일반 재검토'), ('GPT_E2F', '오류 피드백')]
    ])
    body += '<h3>E3 — 조건부 호출</h3><p>각 프롬프트에서 1,000문제 중 498문제에서만 API를 호출했습니다. '
    body += '기본 LLM-A*보다 연산량과 호출 수가 감소했으나, GPT 없는 N보다 연산량은 더 많았습니다. '
    body += '추가 연구 후보로는 호출 비용 절감이 가장 명확합니다. 실제 응답 시간 우위는 이 실험으로 주장하지 않습니다.</p>'
    body += '<h2>3. 본실험 네 지표</h2><p>연산량·저장량·길이는 같은 문제의 A* 대비 비율(기하평균, %)이며 낮을수록 작습니다. '
    body += '유효율은 높을수록 좋습니다. N은 프롬프트에 무관한 공통 대조군입니다.</p>'
    rows = []
    for m in ['A', 'N', 'B', 'E1', 'E2C', 'E2F', 'E3']:
        for k in ([None] if m in ['A', 'N'] else KINDS):
            r = lookup[m, k]
            rows.append([METHODS[m], NAMES[k]] + [f"{r[t]:.2f}" for t in ['operation_ratio','storage_ratio','length_ratio','valid_ratio']] + [r['n'], r['api_calls']])
    body += table(['방법', '프롬프트', '연산량 %', '저장량 %', '길이 %', '유효율 %', '문제 수', '논리적 API 호출'], rows)
    body += '<p class="small">논리적 호출 수는 각 알고리즘을 따로 운영할 때의 사용량입니다. 실제 응답 수집은 조건 간 공유와 캐시를 사용하므로 이 표의 호출 수 합계와 다릅니다.</p>'
    for filename, label in [('comparison.png', '본실험 지표 비교'), ('scaling.png', '지도 확대에 따른 연산량·저장량 증가')]:
        encoded = base64.b64encode((SOURCE / 'report' / filename).read_bytes()).decode()
        if filename == 'scaling.png':
            body += '<h2>4. 확장성 평가</h2><p>기존 GPT 재현의 고정 시드 240702511로 선택한 10문제를 1~10배로 확대해 '
            body += 'Few-shot으로 평가했습니다. 모든 조건에서 100문제를 완료했습니다. 아래 그래프는 각 문제의 1배 대비 증가율을 '
            body += '배율별 산술평균한 값이며 세로축은 로그 눈금입니다. 원본 논문의 Llama 확장성 실험과 동일한 실험은 아닙니다.</p>'
        body += f'<img alt="{label}" src="data:image/png;base64,{encoded}">'
    scale_rows = [r for r in report['tables']['scaling'] if r['method'] in METHODS]
    body += table(['방법', '연산량 / A* %', '저장량 / A* %', '길이 / A* %', '유효율 %', '문제 수'], [
        [METHODS[r['method']]] + [f"{r[k]:.2f}" for k in ['operation_ratio','storage_ratio','length_ratio','valid_ratio']] + [r['n']]
        for r in scale_rows
    ])
    scale_lookup = {r['method']: r for r in scale_rows}
    body += '<p>확장성 평가에서 기본 LLM-A* 대비 연산량 감소율은 '
    body += f"E1 {100*(1-scale_lookup['E1']['operation_ratio']/scale_lookup['B']['operation_ratio']):.2f}%, "
    body += f"E3 {100*(1-scale_lookup['E3']['operation_ratio']/scale_lookup['B']['operation_ratio']):.2f}%입니다. "
    body += 'E3는 100문제 중 75문제에서 호출해 호출 수를 25% 줄였습니다. '
    body += '확장성 평가에서도 E1·E3의 연산량은 N보다 많았고, E2F는 E2C보다 연산량이 소폭 증가했습니다.</p>'
    body += '<h2>5. 실행 이력과 통제상의 한계</h2><ul>'
    body += '<li>사용자 요청으로 실행 호스트를 Windows PC에서 Linux GPU 서버로 옮겼습니다. 모든 대조군을 서버에서 다시 실행했고, '
    body += '기존 A*·Dynamic Weighted A*·g+2h의 3,000개 지표 기록과 일치함을 확인했습니다. 운영체제까지 완전히 동일한 환경은 아닙니다.</li>'
    body += '<li>GPT 추론은 OpenAI API에서 수행했고, 원래 Python 탐색은 CPU에서 수행했습니다. GPU 커널이나 병렬 탐색 알고리즘을 도입하지 않았습니다.</li>'
    body += '<li>2026-10-14 02:08 KST에 API 크레딧 소진으로 E2 확장성 평가가 중단됐습니다. '
    body += '중단 뒤 E3는 이미 수집한 응답과 실제 호출 시점 기록을 이용한 재생 평가로 먼저 완료했습니다. '
    body += '이는 사전 계획의 실행 순서에 대한 예외이며, 별도 감사 기록에 보존했습니다. 재생 결과는 실제 수집 당시의 경로·지표·호출 조건과 대조했습니다.</li>'
    body += '<li>충전 확인 후 2026-10-14 09:50 KST에 동일한 고정 코드·프로토콜로 재개했습니다. '
    body += '완료된 응답을 교체하지 않고 남은 E2 확장성 21문제만 채운 뒤 전체 검증을 수행했습니다. '
    body += '중단 전후 API 서비스 상태와 시간 차이까지 통제할 수는 없습니다.</li>'
    body += '<li>프롬프트 예시와 겹치는 5문제를 모든 조건에서 공통 제외한 995문제 결과를 별도 제공했습니다. '
    body += '이 제외 분석에서도 위 연산량 비교의 방향은 유지됐습니다. '
    body += '이미 관찰한 데이터에서 제안한 개선안이며, 독립 미관측 데이터 검증이나 반복 실험에 기반한 통계적 유의성 검정은 수행하지 않았습니다.</li>'
    body += '<li>본실험 응답 중 길이 제한 종료 21건과 content_filter 종료 3건도 좋은 출력으로 교체하지 않고 평가에 포함했습니다. '
    body += '모든 응답의 모델 반환명은 고정 모델과 일치했지만, 서비스가 system_fingerprint를 제공하지 않아 내부 버전 동일성까지 확인할 수는 없습니다.</li>'
    body += '<li>캐시와 응답 공유 때문에 실제 지연시간·실제 청구 금액 우위를 주장하지 않습니다. '
    body += '추가 검증, 경유점 정제, OPEN 재계산 비용은 원본 heap-pop 지표에 포함되지 않습니다.</li></ul>'
    body += '<h2>6. 원자료와 검증 기록</h2><ul>'
    links = [('../improvements/report/report.html','전체 자동 생성 보고서'),
             ('../improvements/report/main_table.csv','본실험 전체 지표 CSV'),
             ('../improvements/report/paired_comparisons.csv','문제별 대응 비교 집계 CSV'),
             ('../improvements/report/scaling_table.csv','확장성 지표 CSV'),
             ('../improvements/report/main_without_demo_overlap.csv','예시 중복 공통 제외 결과 CSV'),
             ('../improvements/verification.json','서버 전체 검증'),
             ('../improvements/local_import_verification.json','로컬 파일 해시 및 결과 재검증'),
             ('../improvements/execution_deviation.json','E3 실행 순서 예외 감사 기록'),
             ('../../improvement-results-verified.zip','서버 원자료·보고서 ZIP')]
    body += ''.join(f'<li><a href="{url}">{label}</a></li>' for url,label in links)
    body += f'</ul><p class="small">서버 원자료 ZIP SHA-256: <code>{imported["archive_sha256"]}</code><br>'
    body += f'고정 프로토콜 서명: <code>{protocol["signature"]}</code><br>'
    body += f'검증 파일 수: {imported["files_verified"]:,}개. 최종 응답 수: {verified["total_requests"]:,}건.</p>'
    body += '<p class="small">이 문서는 검증된 결과를 해석한 최종 안내 보고서입니다. 고정된 실험 코드와 서버 자동 생성 보고서는 수정하지 않았습니다.</p></main></body></html>'
    (DEST / 'report.html').write_text(body, encoding='utf-8')
    (DEST / 'provenance.json').write_text(json.dumps({
        'created_at': datetime.now(timezone.utc).isoformat(),
        'complete': True, 'verified_rows': verified['total_rows'],
        'saved_api_responses': verified['total_requests'],
        'source_archive_sha256': imported['archive_sha256'],
        'protocol_signature': protocol['signature'],
        'execution_deviation_preserved': True,
        'server_resume': deployment['server'].get('resume'),
        'frozen_sources_modified': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(DEST / 'report.html'))


if __name__ == '__main__':
    main()
