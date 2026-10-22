# Export notes

[Repository home](../README.md) · [Content map](CONTENT_MAP.md)

## Inventory

| Item | Count |
| --- | ---: |
| Chapters | 41 |
| Front-matter pages | 9 |
| Appendices | 7 |
| Navigation pages | 3 |
| Total book pages | 60 |

The original root page was a 60-link index. Its complete inventory is represented in the repository content map and the import archive’s landing index. Each child page was retrieved in full; the export responses reported no truncation or unknown blocks.

## Formatting conversions

- 732 Notion page mentions were converted to relative Markdown links.
- 462 other internal page URLs were remapped, along with the root index’s 60 page links.
- 353 tables, containing 10,871 rows and 31,667 cells, were converted to Markdown tables. Layout-only Notion attributes were omitted.
- 47 callouts were converted to blockquotes, retaining their text and icons.
- All 200 fenced code and diagram blocks were checked against the source export and retained byte-for-byte.
- The page text was not rewritten. Existing prose may still refer to the Notion edition or original source page labels.

## Math compatibility

- Replaced 451 uses of GitHub-blocked `\operatorname` with equivalent supported upright operator notation, retaining side-subscript behavior.
- Separated all 1,040 book display equations from surrounding paragraphs so GitHub recognizes them as display math.
- Used named vertical-bar delimiters in affected table expressions, distinguishing original absolute-value bars from norm notation before Markdown table escaping.
- Preserved all 28,968 inline book expressions, all 1,040 display expressions, and all 200 original fenced code and diagram blocks. The mathematical content and prose were not rewritten.
- Rebuilt the Notion import archive from the same corrected book files.

See [math compatibility examples](MATH_COMPATIBILITY.md) for inline, table, aligned-equation, and matrix smoke tests.

## Validation scope

`EXPORT-MANIFEST.json` records each page’s path, group, byte size, and SHA-256 digest. Its `source_markdown_sha256` values identify the original retrieved Markdown, before formatting and link conversion.

The repository checks validate the 60-file book inventory, current file digests, local file targets, remaining Notion-specific structural tags, the dedicated import archive, math delimiters, display-block separation, balanced braces and environments, known incompatible operators, and unwrapped TeX outside intentional code. Math regression tests run in CI.

GitHub previews are also checked during math-rendering releases for actual rendered expressions, rendering errors, and raw TeX in prose or tables. The dependency-free local checks are not a full TeX engine and cannot guarantee every future GitHub renderer change. These checks do not verify the scientific correctness of every claim, third-party link availability, or rendering in every other Markdown application.

## Rebuild the import

```sh
python3 scripts/build_import.py
python3 scripts/validate_repository.py
python3 scripts/build_import.py --check
```

The archive is reproducible: entries are sorted, timestamps are fixed, and file modes are normalized. It contains a landing index, the 60 book files, and an import guide. The generated ZIP and digest are tracked because the portable Notion import is a primary deliverable of this repository.

## Updating a book page

After an intentional, source-grounded correction:

```sh
python3 scripts/validate_repository.py --refresh-manifest
python3 scripts/build_import.py
python3 scripts/validate_repository.py
```

Refreshing the manifest records the edited repository bytes; it does not re-read Notion or alter the original-source digests. Explain source changes in the pull request and retain appropriate attribution.
