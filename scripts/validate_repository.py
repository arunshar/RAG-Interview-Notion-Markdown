#!/usr/bin/env python3
"""Validate book inventory, checksums, local links, and the import archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

from build_import import verify_archive

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'EXPORT-MANIFEST.json'
STRUCTURAL_TAGS = re.compile(r'<(?:mention-page|page|callout|table|tr|td)\b')
LINKS = re.compile(r'\[(?:\\.|[^\]\\\n])*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)')


def visible_markdown(text: str) -> str:
    """Omit fenced literals and math so examples are not mistaken for real links."""
    result = []
    fence = None
    math = False
    for line in text.splitlines(keepends=True):
        if fence:
            if re.match(r'^ {0,3}' + re.escape(fence[0]) +
                        '{' + str(fence[1]) + r',}\s*$', line):
                fence = None
            continue
        match = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if match:
            fence = (match[1][0], len(match[1]))
            continue
        if line.strip() == '$$':
            math = not math
            continue
        if not math:
            result.append(line)
    text = ''.join(result)
    return re.sub(r'(`+).*?\1', '', text, flags=re.S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-manifest', action='store_true',
                        help='Record intentional book edits; rebuild the import afterward.')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    records = manifest['files']
    errors = []
    tracked = {record['path'] for record in records}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'book').rglob('*.md')}
    if len(records) != 60 or manifest['book_pages'] != 60 or tracked != actual:
        errors.append('Book inventory must contain exactly the 60 manifest pages.')
    if args.refresh_manifest:
        if errors:
            print('\n'.join('ERROR: ' + e for e in errors), file=sys.stderr)
            return 1
        for record in records:
            data = (ROOT / record['path']).read_bytes()
            record['bytes'] = len(data)
            record['sha256'] = hashlib.sha256(data).hexdigest()
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print('Manifest refreshed. Rebuild the import archive, then run full validation.')
        return 0
    for record in records:
        file = ROOT / record['path']
        if not file.exists():
            errors.append('Missing book page: ' + record['path'])
            continue
        data = file.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            errors.append('Book checksum mismatch: ' + record['path'])
        if not re.fullmatch(r'[a-f0-9]{64}', record['source_markdown_sha256']):
            errors.append('Invalid original-source digest: ' + record['path'])
    markdown = sorted(p for p in ROOT.rglob('*.md') if '.git' not in p.parts)
    links_checked = 0
    for file in markdown:
        text = visible_markdown(file.read_text())
        relative = file.relative_to(ROOT).as_posix()
        if STRUCTURAL_TAGS.search(text):
            errors.append('Unconverted Notion structural tag: ' + relative)
        for match in LINKS.finditer(text):
            url = match[1] or match[2]
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            # Mathematical expressions without file-like targets are not file links.
            if '/' not in parsed.path and not re.search(r'\.[A-Za-z0-9]{1,12}$', parsed.path):
                continue
            target = (file.parent / unquote(parsed.path)).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                errors.append(f'Link escapes repository: {relative} -> {url}')
            elif not target.exists():
                errors.append(f'Missing local target: {relative} -> {url}')
            links_checked += 1
    errors.extend(verify_archive())
    if errors:
        print('\n'.join('ERROR: ' + e for e in errors), file=sys.stderr)
        return 1
    print(f'PASS: 60 book pages; {len(markdown)} Markdown files; '
          f'{links_checked} local file links; all book and archive checksums.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
