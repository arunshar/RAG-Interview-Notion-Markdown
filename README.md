# RAG Interview

A complete, structured study collection for retrieval-augmented generation interviews and system design.

**41 chapters · 9 front-matter pages · 7 appendices · 3 navigation pages**

[Start reading](book/00-START-HERE.md) · [Choose a reading route](book/01-READING-ROUTES.md) · [Full contents](docs/CONTENT_MAP.md) · [Download the Notion import](exports/RAG-Interview-Notion-Markdown.zip?raw=true)

## What is inside

The complete 60-page Notion study collection, organized into the book’s 12 parts. The export retains the page text, equations, code examples, diagrams, and references; internal page links work locally and on GitHub.

- **Learn the system:** foundations, generators, representations, indexing, retrieval, ranking, and control flow.
- **Design and evaluate:** training, context assembly, evaluation, credibility, robustness, and production trade-offs.
- **Practice and review:** end-to-end design drills, a question bank, formulas, design checklists, and a glossary.

This is a documentation collection, not an executable RAG framework or a live Notion synchronization.

## Get started

1. Open [Start here](book/00-START-HERE.md) for the book overview.
2. Pick a route in [Reading routes](book/01-READING-ROUTES.md) or browse the [complete page map](docs/CONTENT_MAP.md).
3. Use the [question bank](book/appendices/A2_question_bank.md) and [design checklists](book/appendices/A3_design_checklists.md) to rehearse.
4. To use the collection in Notion, download the [ready-to-import ZIP](exports/RAG-Interview-Notion-Markdown.zip?raw=true) and follow the [import guide](docs/IMPORTING.md).

## Curriculum

| Part | Chapters | Topic |
| --- | --- | --- |
| [I](book/part-01-landscape/) | 1–3 | The RAG Interview Landscape |
| [II](book/part-02-generator/) | 4–8 | The Generator Side |
| [III](book/part-03-prompting/) | 9–11 | Prompting and Context Construction |
| [IV](book/part-04-representations/) | 12–14 | Representing What You Retrieve |
| [V](book/part-05-indexing/) | 15–17 | Indexing and Vector Search |
| [VI](book/part-06-retrieval-ranking/) | 18–23 | Retrieval and Ranking |
| [VII](book/part-07-query-control/) | 24–26 | Query Understanding and Control Flow |
| [VIII](book/part-08-training/) | 27–29 | Training the RAG System |
| [IX](book/part-09-context-assembly/) | 30–31 | Generation and Context Assembly |
| [X](book/part-10-evaluation/) | 32–34 | Evaluation |
| [XI](book/part-11-trust-robustness/) | 35–36 | Trust, Credibility, and Adversarial Robustness |
| [XII](book/part-12-systems-design/) | 37–41 | Scaling, Advanced Variants, and Design Drills |

[Front matter](book/front-matter/) · [All appendices](book/appendices/) · [Original-edition source index](book/98-SOURCE-INDEX.md)

## Repository layout

```text
book/          The complete collection, grouped into 12 parts and appendices
docs/         Content map, import instructions, export notes, and math examples
exports/      Ready-to-import ZIP and SHA-256 checksum
scripts/      Dependency-free validation and reproducible packaging
tests/        Math-syntax regression tests
.github/      Repository checks and contribution templates
```

## Download or clone

- **For Notion:** [download the import ZIP](exports/RAG-Interview-Notion-Markdown.zip?raw=true).
- **For Markdown:** read online, use GitHub’s **Code → Download ZIP**, or clone the repository:

```sh
git clone https://github.com/arunshar/RAG-Interview-Notion-Markdown.git
cd RAG-Interview-Notion-Markdown
```

The import ZIP contains the 60 book pages, a landing-page index, and a short import guide. It excludes CI configuration and other repository-maintenance files.

## Quality checks

```sh
python3 -m unittest discover -s tests
python3 scripts/validate_repository.py
python3 scripts/build_import.py --check
```

Checks cover the complete page inventory, file checksums, relative file links, source-conversion markers, math delimiters and compatibility rules, and the import archive. Fenced code and Mermaid blocks were preserved byte-for-byte during export. No third-party Python packages are required. See [math compatibility](docs/MATH_COMPATIBILITY.md) for rendered examples and [export notes](docs/EXPORT_NOTES.md) for scope and limitations.

## Contributing and source rights

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before making corrections. Preserve citations and distinguish a source statement from a new interpretation.

Source attribution and third-party rights remain unchanged. This repository does not claim authorship of the original book or grant a new license to third-party material. See [NOTICE.md](NOTICE.md), the [title page](book/front-matter/00_01_title_page.md), and the [acknowledgments](book/front-matter/00_08_acknowledgments.md).
