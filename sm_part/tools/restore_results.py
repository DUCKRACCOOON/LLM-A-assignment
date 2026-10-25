#!/usr/bin/env python3
"""Verify S1-S6 source bytes; optionally restore archived files without overwrite."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024


def relative_path(value):
    """Accept only portable, unambiguous relative POSIX paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    for part in parts:
        if (part in ("", ".", "..") or part.endswith((".", " "))
                or re.search(r'[<>:"|?*\x00-\x1f]', part)
                or part.split(".")[0].upper() in reserved):
            raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def disk_path(root, value):
    relative_path(value)
    current = root
    for part in value.split("/"):
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"Symlink/junction is not allowed: {current}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes collection: {value}")
    return current


def load_manifest(root, name):
    with disk_path(root, name).open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return data


def records(manifest, name):
    values = manifest.get(name)
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError(f"{name}: expected a list of objects")
    return values


def expected_bytes(record, label):
    size, digest = record.get("bytes"), record.get("sha256")
    if type(size) is not int or size < 0:
        raise ValueError(f"{label}: invalid bytes value")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError(f"{label}: invalid SHA-256")
    return size, digest.lower()


def digest_stream(stream, output=None):
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
        digest.update(chunk)
        size += len(chunk)
        if output is not None:
            output.write(chunk)
    return size, digest.hexdigest()


def check_digest(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Integrity mismatch: {label} (expected {expected}, found {actual})")


def verify_file(root, record, path_key="path"):
    name = relative_path(record.get(path_key))
    expected = expected_bytes(record, name)
    with disk_path(root, name).open("rb") as stream:
        check_digest(digest_stream(stream), expected, name)


def unique_path(value, seen, label):
    name = relative_path(value)
    key = name.casefold()
    if key in seen:
        raise ValueError(f"Duplicate {label}: {name}")
    seen.add(key)
    return name


def inspect_archive(archive, label):
    seen = set()
    for entry in archive.infolist():
        if entry.orig_filename != entry.filename:
            raise ValueError(f"Ambiguous ZIP entry name in {label}: {entry.orig_filename!r}")
        name = entry.filename[:-1] if entry.is_dir() else entry.filename
        unique_path(name, seen, f"ZIP entry in {label}")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError(f"Symlink ZIP entry is not allowed: {label}!{name}")


def verify_sources(root=ROOT, extract=False):
    manifest = load_manifest(root, "s1_s6_manifest.json")
    archive_records = records(manifest, "archives")
    source_records = records(manifest, "source_files")
    archives, archive_names, source_names, restore = {}, set(), set(), []
    with ExitStack() as stack:
        for record in archive_records:
            name = unique_path(record.get("path"), archive_names, "archive path")
            stream = stack.enter_context(disk_path(root, name).open("rb"))
            check_digest(digest_stream(stream), expected_bytes(record, name), name)
            stream.seek(0)
            archive = stack.enter_context(zipfile.ZipFile(stream))
            inspect_archive(archive, name)
            archives[name] = archive

        for record in source_records:
            source = unique_path(record.get("source_path"), source_names, "source_path")
            expected = expected_bytes(record, source)
            storage = record.get("storage")
            if not isinstance(storage, dict):
                raise ValueError(f"{source}: missing storage object")
            name = relative_path(storage.get("path"))
            if "member" in storage:
                member = relative_path(storage["member"])
                if name not in archives:
                    raise ValueError(f"{source}: undeclared archive {name}")
                archive = archives[name]
                info = archive.getinfo(member)
                if info.is_dir():
                    raise ValueError(f"{source}: ZIP member is a directory")
                with archive.open(info) as stream:
                    check_digest(digest_stream(stream), expected, f"{name}!{member}")
                restore.append((source, expected, archive, info))
            else:
                with disk_path(root, name).open("rb") as stream:
                    check_digest(digest_stream(stream), expected, name)

        # Validate every destination before creating any files.
        pending, skipped = [], 0
        if extract:
            for source, expected, archive, info in restore:
                target = disk_path(root, source)
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"Restore destination is not a file: {source}")
                    with target.open("rb") as stream:
                        actual = digest_stream(stream)
                    if actual != expected:
                        raise ValueError(f"Refusing to overwrite different file: {source}")
                    skipped += 1
                else:
                    for parent in target.parents:
                        if parent == root:
                            break
                        if parent.exists() and not parent.is_dir():
                            raise ValueError(f"Restore parent is not a directory: {parent}")
                    pending.append((source, expected, archive, info))

            for source, expected, archive, info in pending:
                target = disk_path(root, source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target = disk_path(root, source)
                # Exclusive creation also protects files appearing after preflight.
                output = target.open("xb")
                try:
                    with output, archive.open(info) as stream:
                        actual = digest_stream(stream, output)
                    check_digest(actual, expected, source)
                except BaseException:
                    output.close()
                    target.unlink()
                    raise

    return {"archives": len(archives), "source_files": len(source_records),
            "restored": len(pending), "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true",
                        help="After verification, restore archived files under sm_part; never overwrite.")
    args = parser.parse_args()
    try:
        result = verify_sources(extract=args.extract)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Verified {result['archives']} archives and {result['source_files']} source files.")
    if args.extract:
        print(f"Restored {result['restored']} files; skipped {result['skipped']} identical files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
