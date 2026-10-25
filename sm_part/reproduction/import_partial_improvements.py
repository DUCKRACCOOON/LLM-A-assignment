"""Safely import a verified partial export, separate from complete-run outputs."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive', type=Path)
    ap.add_argument('--sha256', required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    actual = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    assert actual == args.sha256, 'Archive checksum mismatch'
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        manifest = json.loads(z.read('archive_manifest.json'))
        assert set(names) == set(manifest) | {'archive_manifest.json'}
        for name, checksum in manifest.items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert tuple(rel.parts[:2]) == ('results', 'improvements_partial')
            data = z.read(name)
            assert hashlib.sha256(data).hexdigest() == checksum, name
            target = root.joinpath(*rel.parts)
            assert not target.exists() or target.read_bytes() == data, 'Existing file differs: ' + name
        for name in manifest:
            target = root.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(z.read(name))
    out = root / 'results/improvements_partial'
    audit = json.loads((out / 'audit.json').read_text(encoding='utf-8'))
    assert audit['experiment_complete'] is False and audit['verified_saved_artifacts'] is True
    assert audit['checks']['main']['verified_rows'] == 27000
    assert audit['e3_replay']['complete'] and audit['e3_replay']['additional_api_requests'] == 0
    verification = {'experiment_complete': False, 'import_verified': True,
                    'archive_sha256': actual, 'files_verified': len(manifest),
                    'verified_rows': audit['total_verified_rows'],
                    'api_responses': audit['total_saved_api_responses']}
    (out / 'local_import_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
