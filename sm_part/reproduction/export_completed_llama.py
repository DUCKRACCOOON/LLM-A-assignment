"""Verify and package a completed run; never load a model or regenerate output."""
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from core import ROOT
from llama_run import OUT, KINDS, cases_for, digest, read, verify
from llama_report import check_baselines, summarize


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if read(OUT / 'status.json')['phase'] != 'complete':
        raise RuntimeError('The original runner has not finished; export refused.')
    verification = verify('main')
    rows, baselines = read(OUT / 'main/rows.json'), read(OUT / 'main/baselines.json')
    reference = read(ROOT / 'reproduction/llama_reference.json')
    check_baselines(baselines, reference['baselines'])
    stage = read(OUT / 'main/manifest.json')
    rebuilt_rows, rebuilt_baselines = [], []
    expected_evaluations = {q['id'] + '.json' for q in cases_for('main')}
    assert {p.name for p in (OUT / 'main/evaluation').glob('*.json')} == expected_evaluations
    for q in cases_for('main'):
        records = {k: read(OUT / 'main/raw' / (q['id'] + '-' + k + '.json')) for k in KINDS}
        saved = read(OUT / 'main/evaluation' / (q['id'] + '.json'))
        expected = digest({'case': q, 'raw': records, 'core': stage['identity']['protocol']['core_sha256']})
        assert saved['signature'] == expected and saved['case'] == q, q['id']
        rebuilt_rows.extend(saved['rows'])
        rebuilt_baselines.append(saved['baseline'])
    assert rebuilt_rows == rows and rebuilt_baselines == baselines
    report = read(OUT / 'report/summary.json')
    assert report['table1'][:8] == reference['gpt_table1']
    assert report['table1'][8:] == report['llama'] == summarize(rows, baselines)
    assert len(report['table1']) == 14

    sources = [
        'requirements-llama.txt', 'bundle_manifest.json',
        'reproduction/core.py', 'reproduction/llama_run.py',
        'reproduction/llama_report.py', 'reproduction/llama_reference.json',
        'reproduction/test_llama.py', 'reproduction/server_smoke.py',
        'reproduction/server_setup.sh', 'reproduction/server_start.sh',
        'vendor/llm-astar/dataset/environment_50_30.json',
        'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py',
        'vendor/llm-astar/LICENSE', 'reproduction/export_completed_llama.py',
    ]
    names = set(sources)
    for name in ['preflight.json', 'server-requirements-lock.txt',
                 'setup-isolated.log', 'infrastructure_smoke.json', 'smoke.log',
                 'model_manifest.json', 'status.json', 'run.log', 'runner.pid']:
        names.add('results/llama/' + name)
    for phase in ['pilot', 'main']:
        for name in ['manifest.json', 'token_audit.json', 'verification.json',
                     'rows.json', 'baselines.json']:
            names.add('results/llama/' + phase + '/' + name)
        for directory in ['raw', 'evaluation']:
            names.update(p.relative_to(ROOT).as_posix()
                         for p in (OUT / phase / directory).glob('*.json'))
    names.update('results/llama/report/' + name for name in
                 ['report.html', 'table1.csv', 'summary.json', 'llama_without_demo_overlap.csv'])
    check = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'main': verification, 'raw_count': 3000, 'evaluation_files': 1000,
        'evaluation_rows': 6000, 'table1_conditions': 14,
        'response_hashes_and_unique_expected_keys': True,
        'evaluation_signatures_and_aggregate_equality': True,
        'gpt_reference_and_astar_baselines_match': True,
        'model_id': stage['identity']['protocol']['model_id'],
        'model_revision': stage['identity']['model_revision'],
        'scope': 'Table 1 only; Figure 3 and Figure 4 excluded',
    }
    (OUT / 'export_verification.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    names.add('results/llama/export_verification.json')
    archive = OUT / 'llama-table1-verified.zip'
    if archive.exists():
        raise FileExistsError('Existing export preserved: ' + str(archive))
    files = {}
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(names):
            path = ROOT / name
            assert path.resolve().is_relative_to(ROOT.resolve())
            data = path.read_bytes()
            if re.search(rb'\bhf_[A-Za-z0-9]{24,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}', data):
                raise RuntimeError('Potential credential found; export refused: ' + name)
            files[name] = {'bytes': len(data), 'sha256': sha(data)}
            z.writestr(name, data)
        z.writestr('archive_manifest.json', json.dumps(files, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, entry in files.items():
            assert sha(z.read(name)) == entry['sha256'], name
    receipt = {'archive': archive.relative_to(ROOT).as_posix(),
               'bytes': archive.stat().st_size, 'files': len(files),
               'sha256': sha(archive.read_bytes()), 'verification': check}
    (OUT / 'export_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
