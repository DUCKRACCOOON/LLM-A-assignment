#!/usr/bin/env python3
"""Verify collected files and every S1-S6 source representation, without extraction."""

from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True

from restore_results import load_manifest, records, unique_path, verify_file, verify_sources


ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        manifest = load_manifest(ROOT, "collection_manifest.json")
        files, seen = records(manifest, "files"), set()
        for record in files:
            unique_path(record.get("path"), seen, "collection path")
            verify_file(ROOT, record)
        result = verify_sources(ROOT)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    # Extra files are permitted: --extract may restore archive members here.
    print(f"Verified {len(files)} collected files, {result['archives']} archives, "
          f"and {result['source_files']} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
