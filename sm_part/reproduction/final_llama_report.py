"""Create the final Korean comparison from locally verified artifacts only."""
import csv
import html
import json
from collections import Counter
from pathlib import Path

from core import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    base = ROOT / 'results/llama'
    check = read(base / 'local_verification.json')
    assert check['all_file_hashes_match'] and check['table1_conditions'] == 14
    summary = read(base / 'report/summary.json')
    out = base / 'final_report'
    out.mkdir(exist_ok=True)
    prompts = {'standard': 'Few-shot', 'cot': 'CoT', 'repe': 'RePE', None: '—'}
    metrics = [('operation', '연산량'), ('storage', '저장량'), ('length', '경로 길이'), ('valid', '유효 경로')]
    comparisons = []
    for row in summary['table1']:
        r = dict(row)
        for key, _ in metrics:
            value, paper = r.get(key + '_ratio'), r.get('paper_' + key + '_ratio')
            r['delta_' + key + '_points'] = None if value is None or paper is None else value - paper
        comparisons.append(r)
    fields = list(dict.fromkeys(k for r in comparisons for k in r))
    with (out / 'table1_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    limited = [r for p in (base / 'main/raw').glob('*.json')
               if (r := read(p))['result']['finish_reason'] == 'length']
    limited_by_prompt = dict(Counter(r['prompt_kind'] for r in limited))
    final = {'table1': comparisons, 'local_verification': check,
             'length_limited_by_prompt': limited_by_prompt,
             'length_limited_case_ids': [r['case_id'] for r in limited],
             'llama_demo_overlap': summary['llama_demo_overlap'],
             'scope': 'Table 1; Figure 3 and Figure 4 excluded'}
    (out / 'table1_comparison.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')

    def label(r):
        if r['method'] == 'astar':
            return 'A*'
        if r['method'] == 'dynamic_weighted':
            return 'Dynamic weighted A*'
        return r['base_model'] + (' only' if r['method'] == 'llm_only' else ' + LLM-A*')

    def table(rows):
        body = ''
        for r in rows:
            cells = ''
            for key, _ in metrics:
                value = r.get(key + '_ratio')
                if value is None:
                    cells += '<td class="muted">—</td>'
                else:
                    paper, delta = r['paper_' + key + '_ratio'], r['delta_' + key + '_points']
                    cells += f'<td><b>{value:.2f}</b><span>논문 {paper:.2f}</span><small>Δ {delta:+.2f}%p</small></td>'
            body += f'<tr><th>{html.escape(label(r))}<span>{prompts[r["prompt"]]}</span></th>{cells}<td>{r["valid_n"]:,}/{r["n"]:,}</td></tr>'
        return '<div class="scroll"><table><thead><tr><th>모델 / 방법</th>' + ''.join('<th>' + name + (' ↑' if k == 'valid' else ' ↓') + '<br>%</th>' for k, name in metrics) + '<th>유효 / 전체</th></tr></thead><tbody>' + body + '</tbody></table></div>'

    llama_rows = [r for r in comparisons if r['base_model'].startswith('Llama')]
    gpt_rows = [r for r in comparisons if not r['base_model'].startswith('Llama')]
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM-A* Table 1 최종 재현 보고서</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#162d3b;font:16px/1.7 system-ui,"Malgun Gothic",sans-serif}main{max-width:1220px;margin:auto;padding:42px 28px 70px}header{padding:34px;background:#142e3e;color:white;border-radius:16px}h1{font-size:32px;line-height:1.3;margin:10px 0 18px}h2{font-size:23px;margin-top:0}h3{font-size:18px;margin-bottom:8px}p{margin:12px 0}.tag{color:#95d9ca;font-size:13px;font-weight:700;letter-spacing:.08em}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 28px}.card,section{background:white;border:1px solid #dce5eb;border-radius:12px}.card{padding:18px}.card strong{font-size:28px;display:block}.card span,.muted{color:#526775}.card span{font-size:13px}section{padding:26px;margin:22px 0}.notice{border-left:4px solid #ba8043;padding:10px 18px;background:#fff8ed}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;min-width:820px}td,th{padding:13px 11px;border-bottom:1px solid #dee6eb;text-align:right;vertical-align:top}thead th{background:#edf3f6;font-size:12px}th:first-child{text-align:left;width:30%}td span,th span,td small{display:block;font-weight:400;color:#59707d;font-size:12px}td b{font-size:17px}code{overflow-wrap:anywhere;font-size:13px}li{margin:8px 0}a{color:#14656b}header a{color:#b8e8e0}.files a{display:inline-block;margin:6px 20px 6px 0}footer{color:#526775;font-size:13px}@media(max-width:650px){main{padding:18px 12px}header,section{padding:20px}h1{font-size:27px}.cards{grid-template-columns:repeat(2,1fr)}}@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:auto}.cards{break-inside:avoid}header{background:#142e3e!important;print-color-adjust:exact}.scroll{overflow:visible}table{min-width:0;font-size:11px}td b{font-size:13px}}</style></head><body><main>
<header><div class="tag">REPRODUCTION · VERIFIED 2026-10-14</div><h1>LLM-A* · Table 1 재현 결과</h1><p>Llama 3-8B-Instruct의 6개 조건과 기존 GPT·기준 알고리즘의 8개 조건을 통합했습니다. 공개 데이터 100개 지도 × 10개 시작·목표 쌍을 사용했습니다.</p><p>서버 실험 완료: 2026년 10월 14일 00:54 KST · 원문: <a href="https://arxiv.org/abs/2407.02511">LLM-A*, arXiv 2407.02511v2</a></p></header>
<div class="cards"><div class="card"><strong>3,000</strong><span>Llama 원문 응답 · 본 실험</span></div><div class="card"><strong>6,000</strong><span>Llama 조건별 평가 행</span></div><div class="card"><strong>14</strong><span>통합 Table 1 조건</span></div><div class="card"><strong>4,078</strong><span>압축 내 파일 해시 일치</span></div></div>
<section><h2>확인된 결과</h2><p><b>Llama-A*의 세 프롬프트 모두 1,000문제에서 유효 경로를 100% 찾았습니다.</b> 연산량은 A*의 45.83–66.13%, 경로 길이는 A*의 102.45–102.47%였습니다. 효율을 높이면서 경로 길이 증가는 작게 유지하는 논문의 주요 경향과 일치합니다.</p><p>Few-shot Llama-A*의 연산량은 논문 44.59% → 재현 45.83%(+1.24%p), CoT는 47.60% → 48.74%(+1.14%p), RePE는 64.08% → 66.13%(+2.05%p)였습니다.</p><div class="notice"><b>완전히 일치하지 않은 결과:</b> Llama-only RePE의 유효 경로 비율은 논문 16.40% → 재현 8.40%(−8.00%p)로 떨어졌습니다. Llama-only의 프롬프트 간 순위까지 재현됐다고 주장할 수 없습니다. GPT·Llama 간 효율 순위도 모든 조건에서 일치하지 않습니다.</div></section>
<section><h2>Llama · 남은 6개 조건</h2><p class="muted">굵은 값은 재현, 그 아래는 논문 수치와 차이(재현 − 논문)입니다. 연산량·저장량·경로 길이는 A*=100인 상대 비율입니다.</p>''' + table(llama_rows) + '''</section>
<section><h2>GPT와 기준 알고리즘 · 기존 8개 조건</h2><p class="muted">기존 gpt-3.5-turbo-0125 결과를 그대로 사용했습니다. 이번 Llama 작업에서 추가 GPT API 호출은 없습니다.</p>''' + table(gpt_rows) + '''</section>
<section><h2>검증과 재현 조건</h2><ul><li>본 실험 원문 응답 3,000개, 예비실험 응답 30개, 본 평가 1,000문제 × 6조건을 확인했습니다. 누락·중복 검사와 요청·응답 해시, 평가 파일의 입력 서명 및 집계 일치 검사를 통과했습니다.</li><li>1,000문제의 A* 연산량·저장량·경로 길이·유효성은 기존 GPT 실험의 기준값과 일치합니다. 14조건 통합 통계도 로컬에서 재계산해 확인했습니다.</li><li>모델: <code>meta-llama/Meta-Llama-3-8B-Instruct</code>, revision <code>8afb486c1db24fe5011ec46dfbe5b5dccdb575c2</code>. Llama 3.1을 사용하지 않았습니다.</li><li>RTX A5000 GPU 0에서 BF16, greedy, batch 1, Transformers 4.44.2, PyTorch 2.7.0a0+7c8ec84dab.nv25.3 / CUDA 12.8로 실행했습니다. 학습·미세조정은 없습니다. 원문 raw Llama 프롬프트와 pipeline의 기본 특수 토큰 처리를 유지했습니다.</li><li>정상 종료 2,994개, 길이 제한 종료 6개입니다. 제한 응답과 경로 생성 실패를 재생성하거나 제외하지 않았습니다. 길이 제한의 프롬프트별 개수: <code>''' + html.escape(str(limited_by_prompt)) + '''</code>.</li><li>Llama 본 응답 생성의 측정 시간 합은 약 5.29 GPU 시간입니다. 연산량·저장량 표는 탐색의 heap pop 수와 방문 노드 저장 수를 비교하며, LLM 추론 지연이나 실제 RAM 사용량을 포함하지 않습니다.</li></ul></section>
<section><h2>논문과 비교할 때의 한계</h2><ol><li>공개된 모델·프롬프트·데이터를 따르는 재현이며, 저자의 과거 응답과 실행 환경을 그대로 복원한 것은 아닙니다. 원문은 Transformers 버전을 고정하지 않았습니다.</li><li>생성 상한은 원본 8,000토큰을 따르되 모델 총 문맥 한도에서 입력 토큰 수를 뺀 값으로 제한했습니다. 입력은 자르지 않았고 6개 길이 제한 응답도 평가했습니다.</li><li>주 경로 길이 통계는 유효 경로만의 문제별 A* 대비 비율을 기하평균했습니다. 논문의 실패 경로 처리·집계 세부 규칙이 완전히 공개되지 않아 LLM-only 길이 수치의 직접 비교에는 제한이 있습니다. 원본 CSV에는 전체 파싱 가능 경로의 길이와 분모도 있습니다.</li><li>프롬프트 예시와 평가 데이터가 겹치는 5문제는 본 결과에 유지했습니다. 이를 공통 제외한 995문제 보조 결과도 제공합니다.</li><li>GPT는 gpt-3.5-turbo-0125 Chat Completions로 실행했습니다. 논문은 Turbo, 공개 코드 일부는 Turbo-Instruct를 사용하며 줄바꿈 stop도 조정했으므로 역사적 GPT 설정의 완전 동일 재현은 아닙니다.</li><li>Dynamic weighted A*의 누락된 세부 규칙은 <code>max(1, 2 × 0.99^pop_count)</code>와 매 확장 후 OPEN 재계산으로 재구성했습니다. 유효 경유점이 없을 때 hybrid는 시작·목표만 사용하며 <code>g+2h</code>로 동작합니다.</li><li><b>이번 완료 범위는 Table 1입니다. Llama의 Figure 3·4 실험은 포함하지 않습니다.</b></li></ol></section>
<section class="files"><h2>결과 파일</h2><a href="table1_comparison.csv">논문·재현·차이 CSV</a><a href="table1_comparison.json">전체 비교 JSON</a><a href="../report/report.html">서버 원본 보고서</a><a href="../report/llama_without_demo_overlap.csv">995문제 보조 결과</a><a href="../local_verification.json">로컬 검증 기록</a><a href="../../../llama-table1-verified.zip">검증된 원본 ZIP</a><p>ZIP SHA-256: <code>''' + check['archive_sha256'] + '''</code></p><p>키·인증 토큰·모델 가중치는 압축에 포함하지 않았습니다. Sirius 데이터 저장소의 공식 파일 다운로드로 받은 압축과 내부 파일 전체의 해시를 확인했습니다.</p></section><footer>실패 응답을 포함한 고정 결과입니다. 보고서 생성은 기존 파일만 읽으며 GPT API 호출이나 모델 재생성을 수행하지 않습니다.</footer></main></body></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({'report': str(out / 'report.html'), 'conditions': len(comparisons),
                      'limited_by_prompt': limited_by_prompt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
