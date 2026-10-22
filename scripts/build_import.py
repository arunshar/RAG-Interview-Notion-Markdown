#!/usr/bin/env python3
"""Build or verify the portable Notion Markdown import. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from validate_math import scan_math

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'exports' / 'RAG-Interview-Notion-Markdown.zip'
SUMS = ROOT / 'exports' / 'SHA256SUMS'
PREFIX = 'RAG-Interview-Notion-Markdown/'
STAMP = (1980, 1, 1, 0, 0, 0)
GUIDE = '''# Import into Notion

[Book index](README.md)

1. Download the dedicated RAG-Interview-Notion-Markdown.zip archive.
2. In the destination Notion workspace, open Settings → Import → Text & Markdown.
3. Select the ZIP to import the folder structure and Markdown pages.
4. Open the landing index, then Start here or Reading routes.
5. Check cross-page links, tables, equations, and diagrams after import.

This archive contains all 60 book pages, a landing index, this guide, and the
source-rights notice. It is a point-in-time copy, not a live synchronization.
Comments, permissions, page history, icons, and database configuration are not
included. Anchor links and tool-specific formatting may not import cleanly;
math and Mermaid rendering may need manual adjustment in Notion.
GitHub's protected math fences are converted to native display-equation
delimiters in this import, with their TeX contents unchanged.

See Notion's current import documentation:
https://www.notion.so/help/import-data-into-notion

Source attribution and third-party rights remain unchanged. See NOTICE.md.
'''


def notion_markdown(text: str) -> str:
    """Use Notion-native display blocks without changing their TeX contents."""
    spans, errors = scan_math(text)
    if errors:
        raise ValueError("Cannot package malformed math delimiters: " + str(errors))
    for span in reversed(spans):
        if span.kind == 'display' and span.syntax == 'fence':
            native = '$$\n' + span.tex(text) + '\n$$'
            text = text[:span.start] + native + text[span.end:]
    return text


def expected_entries() -> dict[str, bytes]:
    manifest = json.loads((ROOT / 'EXPORT-MANIFEST.json').read_text())
    landing = (ROOT / 'docs' / 'IMPORT_INDEX.md').read_text()
    landing = landing.replace('](../book/', '](book/')
    landing = landing.replace('](IMPORTING.md)', '](IMPORTING.md)')
    entries = {
        PREFIX + 'README.md': landing.encode(),
        PREFIX + 'IMPORTING.md': GUIDE.encode(),
        PREFIX + 'NOTICE.md': (ROOT / 'NOTICE.md').read_bytes(),
    }
    for item in manifest['files']:
        text = (ROOT / item['path']).read_text(encoding='utf-8')
        entries[PREFIX + item['path']] = notion_markdown(text).encode('utf-8')
    return dict(sorted(entries.items()))


def build_bytes(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    return stream.getvalue()


def verify_archive() -> list[str]:
    errors = []
    if not ARCHIVE.exists() or not SUMS.exists():
        return ['Import ZIP or SHA256SUMS is missing; run scripts/build_import.py.']
    expected = expected_entries()
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            if archive.testzip() is not None:
                errors.append('Import ZIP failed its CRC check.')
            if archive.namelist() != list(expected):
                errors.append('Import ZIP inventory or entry order differs from the book.')
            else:
                for name, data in expected.items():
                    if archive.read(name) != data:
                        errors.append('Import ZIP is stale: ' + name)
                    info = archive.getinfo(name)
                    if info.date_time != STAMP or info.external_attr >> 16 != 0o100644:
                        errors.append('Import ZIP metadata is not normalized: ' + name)
    except (zipfile.BadZipFile, OSError) as exc:
        errors.append('Cannot read import ZIP: ' + str(exc))
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    line = digest + '  exports/' + ARCHIVE.name + '\n'
    if SUMS.read_text() != line:
        errors.append('SHA256SUMS does not match the committed import ZIP.')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without changing files.')
    args = parser.parse_args()
    if args.check:
        errors = verify_archive()
        if errors:
            print('\n'.join('ERROR: ' + e for e in errors), file=sys.stderr)
            return 1
        print('PASS: import ZIP inventory, content, metadata, CRC, and SHA-256.')
        return 0
    ARCHIVE.parent.mkdir(exist_ok=True)
    entries = expected_entries()
    data = build_bytes(entries)
    ARCHIVE.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    SUMS.write_text(digest + '  exports/' + ARCHIVE.name + '\n')
    print(f'Built {ARCHIVE.relative_to(ROOT)}: {len(entries)} files, {len(data):,} bytes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
