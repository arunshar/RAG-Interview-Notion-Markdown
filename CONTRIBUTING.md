# Contributing

Thanks for helping keep this collection accurate, navigable, and useful.

## Good contributions

- Fix broken internal links, transcription mistakes, and formatting defects.
- Clarify an explanation without changing the source claim silently.
- Add precise supporting references or identify a source discrepancy.
- Improve navigation and import instructions while preserving all book pages.

## Before opening a pull request

1. Identify the affected page and explain the issue or proposed improvement.
2. Keep the change focused. Avoid unrelated rewrites or mass reformatting.
3. Preserve author attribution, quotations, mathematical conditions, and source references.
4. If book files change, refresh their digests and rebuild the import archive:

```sh
python3 scripts/validate_repository.py --refresh-manifest
python3 scripts/build_import.py
python3 scripts/validate_repository.py
python3 scripts/build_import.py --check
```

5. Include the updated manifest, ZIP, and checksum when their contents change.
6. Follow the [math compatibility guide](docs/MATH_COMPATIBILITY.md), run `python3 -m unittest discover -s tests`, and inspect affected GitHub previews. Static validation is not a substitute for checking the actual renderer.

Python 3.11 or later is sufficient; there are no third-party runtime dependencies. Do not include private workspace exports, API keys, access tokens, or personal account metadata.

## Reporting an issue

Use the content-correction template. Include the file path, a concise description, and a supporting source where relevant. A rendering difference should name the renderer and include a minimal example.

## Rights

Do not add material you do not have permission to contribute. Existing third-party source rights are retained; publication here does not grant a new license to that material. See [NOTICE.md](NOTICE.md).
