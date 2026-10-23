# Changelog

## October 2026 review corrections

- Ported the verified corrections from the October 2026 review of the companion edition `arunshar/rag-interview-chapters`. Each correction keeps the original statement and adds a labeled "Correction:" sentence or a "(corrected from ...)" table cell with its primary source. The pages now carry 64 correction sentences and 60 corrected cells.
- Checked every correction against the reworded text of this edition. A correction was ported only where this edition made the same claim. A separate verifier searched each page for missed corrections.
- Corrected wrong author lists in the annotated reading list against arXiv and venue records.
- Replaced the author's personal email address with his public website as the contact route.
- Added an unofficial-edition notice to the README, the start page, and NOTICE.md.
- Refreshed book checksums and rebuilt the import ZIP.

## Math rendering compatibility

- Fixed unsupported operator macros throughout the complete book without changing equation meanings.
- Protected display equations against paragraph, list, link, and emphasis parsing; normalized comparison signs to avoid HTML interpretation and fixed table-bar delimiter ambiguity.
- Added a live-rendered math compatibility page and dependency-free regression checks in CI.
- Rebuilt the complete import ZIP with native Notion display equations and refreshed book and archive checksums.

## Initial publication

- Published the complete 60-page RAG interview study collection.
- Organized 41 chapters into the source book’s 12 parts, with separate front matter and appendices.
- Converted internal Notion page links, tables, and callouts for portable Markdown reading.
- Added a complete content map, Notion import guide, and source-rights notice.
- Added a reproducible import ZIP, file checksums, and automated repository validation.
