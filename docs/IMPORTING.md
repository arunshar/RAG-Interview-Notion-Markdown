# Import the complete collection into Notion

[Repository home](../README.md) · [Complete content map](CONTENT_MAP.md)

## Recommended download

Use [RAG-Interview-Notion-Markdown.zip](../exports/RAG-Interview-Notion-Markdown.zip?raw=true), not the source-code ZIP. The dedicated import archive contains the book and its navigation without repository-maintenance files.

## Import steps

1. Download the dedicated import ZIP to your computer.
2. Open the destination Notion workspace.
3. Go to **Settings → Import → Text & Markdown**.
4. Select the ZIP. Notion supports importing a Markdown folder as a ZIP to preserve its folder structure.
5. After import, open the landing index, then **Start here** or **Reading routes**.
6. Spot-check cross-page links, a table, an equation, and a diagram before sharing your imported copy.

Notion’s [official import guide](https://www.notion.so/help/import-data-into-notion) documents Text & Markdown imports, ZIP folders, supported blocks, and current limits. Product labels and plan limits may change.

## What to expect

- Each of the 60 book pages is an individual Markdown file.
- The part folders and relative cross-page links are included in the archive.
- The archive also contains a landing index and a copy of this guide.
- Headings, lists, standard Markdown tables, and fenced code are retained in the exported files.
- Anchor links and tool-specific Markdown extensions may not import cleanly. Notion math and Mermaid rendering are not guaranteed to match GitHub after import; verify those blocks manually.
- A Markdown export is not a clone of Notion comments, permissions, page history, page icons, or database configuration. Those are not included here.

## Verify the download

The expected digest is listed in [SHA256SUMS](../exports/SHA256SUMS).

```sh
# Run inside the repository root on Linux:
sha256sum -c exports/SHA256SUMS

# To calculate a standalone download’s digest on macOS:
shasum -a 256 RAG-Interview-Notion-Markdown.zip
```

## Troubleshooting

| Issue | Next step |
| --- | --- |
| Maintenance files appear as imported pages | Use the dedicated import ZIP rather than GitHub’s source-code ZIP. |
| A cross-page link is not remapped | Navigate through the content index and update the destination link in the imported copy. The Markdown target remains available in the ZIP. |
| Equations or diagrams appear as text | Recreate the corresponding native Notion block if needed; the source notation is preserved. |
| Import stops or is incomplete | Check the current plan limits and retry smaller part folders. Keep the original ZIP unchanged for comparison. |

This repository is a point-in-time export. Importing it does not enable synchronization with the original workspace.
