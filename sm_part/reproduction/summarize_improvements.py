"""Summarize verified results without altering the frozen experiment outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'results/improvements'
    check = json.loads((out / 'local_import_verification.json').read_text(encoding='utf-8'))
    if not check.get('complete'):
        raise RuntimeError('Local artifact verification must complete first')
    report = json.loads((out / 'report/summary.json').read_text(encoding='utf-8'))
    table = {(r['method'], r['prompt']): r for r in report['tables']['main']}
    paired = {(r['treatment'], r['control'], r['prompt']): r for r in report['paired']}
    highlights = []
    for kind in ('standard', 'cot', 'repe'):
        record = {'prompt': kind}
        for treatment, control in [('E1', 'B'), ('E1', 'N'), ('E2F', 'B'),
                                   ('E2F', 'E2C'), ('E3', 'B'), ('E3', 'N')]:
            pair = paired[treatment, control, kind]
            record[f'{treatment}_vs_{control}'] = {
                'operation_reduction_percent': 100 - pair['operation_paired_ratio'],
                'storage_reduction_percent': 100 - pair['storage_paired_ratio'],
                'length_change_percent': pair['length_paired_ratio'] - 100,
                'valid_delta_pp': pair['valid_delta_pp'],
                'paired_n': pair['operation_paired_n'],
            }
        record['raw_path_valid_percent'] = {
            method: table[method, kind]['valid_ratio']
            for method in ('GPT', 'GPT_E2C', 'GPT_E2F')
        }
        record['e3_api_calls'] = table['E3', kind]['api_calls']
        record['e3_call_reduction_percent'] = (
            100 * (1 - table['E3', kind]['api_calls'] / table['B', kind]['api_calls'])
        )
        record['hybrid_valid_percent'] = {
            method: table[method, kind]['valid_ratio']
            for method in ('B', 'E1', 'E2C', 'E2F', 'E3')
        }
        highlights.append(record)
    result = {'source': 'results/improvements/report/summary.json',
              'archive_sha256': check['archive_sha256'], 'main': highlights,
              'diagnostics': report['diagnostics'], 'checks': report['checks']}
    path = ROOT / 'results/improvements_review.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
