"""Generate a complete Table 1 comparison only after all Llama rows verify."""
import collections
import csv
import html
import math
import statistics

from core import ROOT, load_cases
from llama_run import OUT, PROMPTS, KINDS, read, save, verify

PAPER = {
    ('llm_only', 'standard'): [None, None, 111.05, 12.6],
    ('llm_only', 'cot'): [None, None, 114.89, 12.0],
    ('llm_only', 'repe'): [None, None, 138.32, 16.4],
    ('llm_astar', 'standard'): [44.59, 64.02, 102.47, 100.0],
    ('llm_astar', 'cot'): [47.60, 66.27, 102.46, 100.0],
    ('llm_astar', 'repe'): [64.08, 80.19, 102.54, 100.0],
}
PROMPT_LABEL = {'standard': 'Few-shot', 'cot': 'CoT', 'repe': 'RePE', None: ''}


def gm(values):
    return 100 * math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def summarize(rows, baselines, exclude=()):
    excluded = set(exclude)
    base = {r['case_id']: r for r in baselines}
    groups = collections.defaultdict(list)
    for r in rows:
        if r['case_id'] not in excluded:
            groups[r['method'], r['prompt']].append(r)
    result = []
    for key, paper in PAPER.items():
        values = groups[key]
        if not values:
            continue
        r = {'base_model': 'Llama 3-8B-Instruct', 'method': key[0], 'prompt': key[1],
             'n': len(values), 'valid_n': sum(v['valid'] for v in values)}
        for i, metric in enumerate(('operation', 'storage', 'length')):
            eligible = [v for v in values if v['valid'] and v.get(metric) is not None and v[metric] > 0
                        and base[v['case_id']]['valid'] and base[v['case_id']][metric] > 0]
            r[metric + '_ratio'] = gm([v[metric] / base[v['case_id']][metric] for v in eligible])
            r[metric + '_n'] = len(eligible)
            r['paper_' + metric + '_ratio'] = paper[i]
        parsed = [v for v in values if v.get('length') is not None and v['length'] > 0]
        r['length_all_parseable_ratio'] = gm([v['length'] / base[v['case_id']]['length'] for v in parsed])
        r['length_all_parseable_n'] = len(parsed)
        r['valid_ratio'] = 100 * r['valid_n'] / r['n']
        r['paper_valid_ratio'] = paper[3]
        r['failures'] = dict(collections.Counter(v.get('failure') for v in values if not v['valid']))
        r['parse_failures'] = dict(collections.Counter(v.get('parse_error') for v in values if v.get('parse_error')))
        if key[0] == 'llm_astar':
            r['only_start_goal_targets_n'] = sum(len(v.get('targets', [])) == 2 for v in values)
        result.append(r)
    return result


def check_baselines(actual, expected):
    reference = {r['case_id']: r for r in expected}
    if set(reference) != {r['case_id'] for r in actual}:
        raise RuntimeError('GPT/Llama baseline case IDs differ.')
    for row in actual:
        ref = reference[row['case_id']]
        for metric in ('operation', 'storage', 'length', 'valid'):
            if not math.isclose(row[metric], ref[metric], rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError(f'GPT/Llama baseline mismatch: {row["case_id"]}/{metric}')


def csv_write(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    verification = verify('main')
    rows, baselines = read(OUT / 'main/rows.json'), read(OUT / 'main/baselines.json')
    reference = read(ROOT / 'reproduction/llama_reference.json')
    check_baselines(baselines, reference['baselines'])
    overlap = {q['id'] for q in load_cases() for prompt in PROMPTS.values() if
               f'Start Point: {q["start"]}\nGoal Point: {q["goal"]}\nHorizontal Barriers: {q["horizontal_barriers"]}\nVertical Barriers: {q["vertical_barriers"]}' in prompt}
    out = OUT / 'report'
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, baselines)
    all_rows = reference['gpt_table1'] + summary
    save(out / 'summary.json', {'llama': summary, 'table1': all_rows,
                               'llama_demo_overlap': sorted(overlap), 'verification': verification})
    csv_write(out / 'table1.csv', all_rows)
    csv_write(out / 'llama_without_demo_overlap.csv', summarize(rows, baselines, overlap))
    def number(v):
        return '—' if v is None else f'{v:.2f}'
    table = ''
    for r in all_rows:
        label = ('A*' if r['method'] == 'astar' else 'Dynamic weighted A*' if r['method'] == 'dynamic_weighted'
                 else r['base_model'] + (' only' if r['method'] == 'llm_only' else ' + LLM-A*'))
        cells = ''.join(f'<td>{number(r.get(m + "_ratio"))}<small>논문 {number(r.get("paper_" + m + "_ratio"))}</small></td>'
                        for m in ('operation', 'storage', 'length', 'valid'))
        table += f'<tr><th>{html.escape(label)}<small>{PROMPT_LABEL[r["prompt"]]}</small></th>{cells}<td>{r["valid_n"]}/{r["n"]}</td></tr>'
    manifest = read(OUT / 'main/manifest.json')
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>LLM-A* Table 1 재현</title>
    <style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#172b3a}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border-bottom:1px solid #dce4e9;text-align:right}}th:first-child{{text-align:left}}small{{display:block;color:#607481;font-size:12px}}code{{overflow-wrap:anywhere}}aside{{background:#eff5f8;padding:18px}}</style>
    <h1>LLM-A*: Table 1 재현 결과</h1><p>Llama 본실험 1,000문제 × 6조건 완료. 기존 GPT·기본 알고리즘 결과와 합쳐 14조건을 비교합니다.</p>
    <aside>모델: Meta-Llama-3-8B-Instruct, BF16, greedy, batch 1. 학습 없음.<br>모델 revision: <code>{manifest['identity']['model_revision']}</code><br>실행 식별자: <code>{manifest['run_id']}</code></aside>
    <table><thead><tr><th>조건</th><th>연산량 %</th><th>저장량 %</th><th>경로 길이 %</th><th>유효 경로 %</th><th>유효/전체</th></tr></thead><tbody>{table}</tbody></table>
    <p>연산량·저장량·경로 길이는 문제별 A* 대비 비율의 기하평균입니다. 주 길이 통계는 유효 경로만 포함하며, CSV에는 전체 파싱 가능 경로의 길이와 각 분모도 기록했습니다. 실패 응답은 재생성하거나 제외하지 않았습니다.</p>
    <p>저자의 raw Llama 프롬프트 및 특수 토큰 처리 방식을 유지했습니다. 생성 상한 8,000토큰은 모델의 총 문맥 한도에서 입력 길이를 뺀 값으로 제한했습니다. 이는 원본 코드에 추가한 보호 장치이며, 종료 사유는 verification.json에 기록됩니다. 원본의 정확한 라이브러리 버전은 공개되지 않아 동일한 역사적 실행 환경이라고 주장하지 않습니다.</p>
    <p>GPT 결과는 기존 gpt-3.5-turbo-0125 실험입니다. Dynamic weighted A*의 세부 감쇠 구현과 실패 경로 길이 집계에는 기존 보고서에 명시한 재구성·해석 한계가 있습니다. 이번 범위는 Table 1이며 Figure 3·4의 Llama 실험은 포함하지 않습니다.</p></html>'''
    (out / 'report.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
