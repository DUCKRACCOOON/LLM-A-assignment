"""Validate the expected archive and copy verified result artifacts locally."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from core import ROOT
from llama_run import verify
from llama_report import check_baselines, summarize


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    archive_hash = sha(args.archive.read_bytes())
    assert archive_hash == args.sha256.lower(), 'Archive SHA-256 mismatch'
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive members'
        manifest = json.loads(z.read('archive_manifest.json'))
        assert set(names) == set(manifest) | {'archive_manifest.json'}
        for name, entry in manifest.items():
            p = PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            assert ':' not in name and not name.endswith(('.safetensors', '.bin', '.pt'))
            data = z.read(name)
            assert sha(data) == entry['sha256'] and len(data) == entry['bytes'], name
        for name in ['reproduction/core.py', 'reproduction/llama_run.py',
                     'reproduction/llama_report.py', 'reproduction/llama_reference.json',
                     'vendor/llm-astar/dataset/environment_50_30.json',
                     'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py']:
            assert sha((ROOT / name).read_bytes()) == manifest[name]['sha256'], name
        result_names = [name for name in manifest if name.startswith('results/llama/')]
        for name in result_names:
            path = ROOT / name
            assert path.resolve().is_relative_to((ROOT / 'results/llama').resolve())
            if path.exists():
                assert sha(path.read_bytes()) == manifest[name]['sha256'], 'Existing file differs: ' + name
        for name in result_names:
            path = ROOT / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(name))
    out = ROOT / 'results/llama'
    main_check = verify('main')
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    rows, baselines = read(out / 'main/rows.json'), read(out / 'main/baselines.json')
    reference = read(ROOT / 'reproduction/llama_reference.json')
    check_baselines(baselines, reference['baselines'])
    report = read(out / 'report/summary.json')
    assert report['table1'] == reference['gpt_table1'] + summarize(rows, baselines)
    assert len(report['table1']) == 14
    receipt = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(args.archive.resolve()), 'archive_sha256': archive_hash,
        'archive_members': len(manifest), 'result_files_imported': len(result_names),
        'all_file_hashes_match': True, 'local_protocol_sources_match': True,
        'main': main_check, 'gpt_reference_and_astar_baselines_match': True,
        'table1_conditions': len(report['table1']),
    }
    (out / 'local_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
