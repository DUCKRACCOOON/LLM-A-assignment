"""Make tables and figures from stored K3 measurements, without API calls.

Example: python -m experiments.K3.report --input experiments/K3/results/run_20261025
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .heuristic import METHODS, PRIMARY, run_once

BASE = 'g2h_euclidean'
LABELS = {'astar_euclidean': '기존 A*', BASE: '기존 g+2h', 'g2h_octile': 'octile만 적용',
          'g2h_wall_euclidean': '직선거리 + 벽 하한', PRIMARY: 'K3: octile + 벽 하한'}
COLORS = ['#6b7280', '#3b82b5', '#27836a', '#dd7722']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def load_runs(path):
    first, times = {}, {}
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        key = (row['case_id'], row['method'])
        times.setdefault(key, []).append(row['total_seconds'])
        if row['repeat'] == 0:
            first.setdefault(row['case_id'], {})[row['method']] = row
    for cid, rows in first.items():
        for method, row in rows.items():
            row['median_seconds'] = statistics.median(times[(cid, method)])
    return first


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    output = args.input/'analysis'
    output.mkdir(exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    if 'Malgun Gothic' in fonts:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'NanumGothic' in fonts:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    summaries = {group: read(args.input/group/'summary.json') for group in ('pilot', 'public', 'synthetic')}
    groups = {group: load_runs(args.input/group/'evaluations.jsonl') for group in ('public', 'synthetic')}
    lines = ['# 저장된 원시 결과에서 생성한 수치표', '',
             f'기준 실행: `{args.input.name}`. 비율은 문제별 비율의 기하평균 ×100이다. 작을수록 좋다. 시간은 문제별 3회 중앙값을 사용했다.', '']
    for group, summary in summaries.items():
        lines += [f'## {group}', '', '아래 비율의 기준은 기존 g+2h=100이다.', '',
                  '| 방법 | pop | g 사전 크기 | 경로 길이 | 총시간 | 유효/전체 | 최단경로 수 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            comparison = summary['comparisons'][BASE][method]
            absolute = summary['absolute'][method]
            ratios = ' | '.join(f'{comparison[k]["geometric_ratio"]*100:.3f}%' for k in ('operation', 'storage', 'length', 'total_seconds'))
            lines.append(f'| {LABELS[method]} | {ratios} | {absolute["valid"]}/{absolute["cases"]} | {absolute["optimal_paths"]} |')
        lines += ['', '주 방법 K3의 지도 단위 bootstrap 95% 구간 및 개선/동일/악화 건수:', '',
                  '| 지표 | 비율 95% 구간 | 개선 | 동일 | 악화 |', '| --- | ---: | ---: | ---: | ---: |']
        for metric in ('operation', 'storage', 'length', 'total_seconds'):
            d = summary['comparisons'][BASE][PRIMARY][metric]
            lines.append(f'| {metric} | {d["ci95"][0]*100:.3f}–{d["ci95"][1]*100:.3f}% | {d["better"]} | {d["equal"]} | {d["worse"]} |')
        lines += ['', '절대 값은 아래의 **산술평균**이다. 위 비율표와 집계 방식이 다르다.', '',
                  '| 방법 | pop | 길이 | 총시간(ms) | 벽 검사 수 |', '| --- | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            means = summary['absolute'][method]['means']
            lines.append(f'| {LABELS[method]} | {means["operation"]:.3f} | {means["length"]:.4f} | {means["total_seconds"]*1000:.4f} | {means["wall_checks"]:.1f} |')
        lines.append('')
    (output/'TABLES.md').write_text('\n'.join(lines).rstrip()+'\n', encoding='utf-8')

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), layout='constrained')
    display = list(METHODS[1:])
    for row_index, group in enumerate(('public', 'synthetic')):
        for col, (metric, name) in enumerate((('operation', '탐색량 (pop)'), ('length', '경로 길이'), ('total_seconds', '실행시간'))):
            ax = axes[row_index, col]
            for i, method in enumerate(display):
                d = summaries[group]['comparisons'][BASE][method][metric]
                value, lo, hi = d['geometric_ratio']*100, d['ci95'][0]*100, d['ci95'][1]*100
                ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', color=COLORS[i], markersize=7, capsize=4)
                ax.annotate(f'{value:.2f}%', (value, i), xytext=(0, -17), textcoords='offset points', ha='center', fontsize=9, color=COLORS[i])
            ax.set_yticks(range(4), [LABELS[m] for m in display] if col == 0 else [])
            ax.set_ylim(3.7, -.5)
            ax.set_xlim((35, 110) if metric == 'operation' else ((97.8, 102.2) if metric == 'length' else (50, 130)))
            ax.axvline(100, color='#a0a0a0', ls='--', lw=1)
            ax.grid(axis='x', alpha=.2)
            group_label = '공개 1,000문제' if group == 'public' else '새 지도 120문제'
            ax.set_title(f'{group_label} · {name}', pad=12)
            ax.set_xlabel('기존 g+2h 대비 (%)')
    fig.suptitle('K3 비교: 동일 문제의 비율 기하평균과 지도 bootstrap 95% 구간\n작을수록 좋음 · LLM 호출 0회 · 축 범위는 지표마다 다름', fontsize=14)
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)

    selected = [
        ('public', '탐색량 비율이 가장 크게 감소한 공개 사례',
         min(groups['public'], key=lambda cid: groups['public'][cid][PRIMARY]['operation']/groups['public'][cid][BASE]['operation'])),
        ('public', '경로 길이 비율이 가장 크게 증가한 공개 사례',
         max(groups['public'], key=lambda cid: groups['public'][cid][PRIMARY]['length']/groups['public'][cid][BASE]['length'])),
        ('synthetic', '탐색량 비율이 가장 크게 증가한 새 지도 사례',
         max(groups['synthetic'], key=lambda cid: groups['synthetic'][cid][PRIMARY]['operation']/groups['synthetic'][cid][BASE]['operation'])),
    ]
    queries = {group: {c['case_id']: c['query'] for c in read(args.input/name)} for group, name in (
        ('public', 'public_cases.json'), ('synthetic', 'synthetic_cases.json'))}
    fig, axes = plt.subplots(3, 2, figsize=(13, 12.8), layout='constrained')
    examples = []
    for i, (group, reason, cid) in enumerate(selected):
        query = queries[group][cid]
        example = dict(group=group, case_id=cid, selection_rule=reason, methods={})
        for j, method in enumerate((BASE, PRIMARY)):
            stored = groups[group][cid][method]
            # These six trace replays are illustrative only, excluded from the
            # measured experiment. Assert unchanged result/path before drawing.
            replay = run_once(query, method, trace=True)
            for key in ('operation', 'storage', 'length', 'path'):
                assert json.dumps(replay[key]) == json.dumps(stored[key]), (cid, method, key)
            ax = axes[i, j]
            popped = [e['node'] for e in replay['trace'] if e['event'] == 'pop']
            ax.scatter(*zip(*popped), s=9, c='#abc0cf', alpha=.45, marker='s', linewidths=0, label='탐색한 위치')
            for y, a, b in query['horizontal_barriers']:
                ax.plot([a, b], [y, y], color='#222222', lw=2.5)
            for x, a, b in query['vertical_barriers']:
                ax.plot([x, x], [a, b], color='#222222', lw=2.5)
            color = COLORS[0] if j == 0 else COLORS[-1]
            ax.plot(*zip(*stored['path']), color=color, lw=2, label='반환 경로')
            ax.scatter(*query['start'], c='#27836a', s=50, marker='o', zorder=5, label='시작')
            ax.scatter(*query['goal'], c='#b93840', s=70, marker='*', zorder=5, label='목표')
            ax.set_xlim(query['range_x'][0], query['range_x'][1]-1)
            ax.set_ylim(query['range_y'][0], query['range_y'][1]-1)
            ax.set_aspect('equal'); ax.grid(alpha=.12)
            ax.set_title(f'{cid} · {LABELS[method]}\npop {stored["operation"]}, 길이 {stored["length"]:.3f}', fontsize=12)
            if i == 0 and j == 0:
                ax.legend(loc='upper left', fontsize=8, ncol=2)
            example['methods'][method] = {k: stored[k] for k in ('operation', 'unique_pops', 'duplicate_pops', 'length', 'optimum')}
        examples.append(example)
    fig.suptitle('좋아진 사례와 나빠진 사례를 함께 확인\n왼쪽: 기존 g+2h / 오른쪽: K3 · 회색 칸은 탐색한 위치', fontsize=15)
    fig.savefig(output/'case_paths.png', dpi=180)
    plt.close(fig)
    save(output/'case_examples.json', examples)

    diagnostics = {}
    for group, all_rows in groups.items():
        diagnostics[group] = {}
        for method in (PRIMARY, 'g2h_wall_euclidean'):
            pairs = [(rows[method], rows[BASE]) for rows in all_rows.values()]
            same = [(r, b) for r, b in pairs if r['path'] == b['path'] and r['operation'] == b['operation'] and r['storage'] == b['storage']]
            diagnostics[group][method] = dict(
                unique_pop_ratio=gm(r['unique_pops']/b['unique_pops'] for r, b in pairs),
                mean_unique_pops=statistics.mean(r['unique_pops'] for r, _ in pairs),
                baseline_mean_unique_pops=statistics.mean(b['unique_pops'] for _, b in pairs),
                mean_wall_count=statistics.mean(r['wall_count'] for r, _ in pairs),
                mean_heuristic_calls=statistics.mean(r['heuristic_calls'] for r, _ in pairs),
                total_wall_checks=sum(r['wall_checks'] for r, _ in pairs),
                same_path_operation_storage=dict(cases=len(same), timing_geometric_ratio=gm(r['median_seconds']/b['median_seconds'] for r, b in same)))
    save(output/'diagnostics.json', diagnostics)
    inputs = [args.input/'manifest.json', *[args.input/g/f for g in ('public', 'synthetic', 'pilot') for f in ('evaluations.jsonl', 'summary.json')]]
    save(output/'manifest.json', dict(purpose='Post-experiment descriptive tables and illustrative traces; no tuning',
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(args.input)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        illustrative_trace_replays_per_render=6, new_api_calls=0))
    print(json.dumps(dict(output=str(output), selected_cases=[e['case_id'] for e in examples], diagnostics=diagnostics), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
