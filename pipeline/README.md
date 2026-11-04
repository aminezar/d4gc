# Data pipeline

Build SQLite document bundles for local retrieval. Use the included fictional sample or supply a document collection.

## Commands

```sh
uv sync --frozen
uv run python -m pipeline.demo                            # sample, keyword-only
uv run python -m pipeline.demo --semantic                 # sample with Qwen vectors
uv run python -m pipeline.build                          # source collection in data/
uv run python -m pipeline.build --sample                 # 10 documents from data/
uv run python -m pipeline.build --country 'DR Congo' --out bundles/cod.sqlite
uv run python -m pipeline.export 'DR Congo' --full bundles/demo.sqlite --out bundles/demo-cod.sqlite
```

`pipeline.demo` builds the included sample. `pipeline.build --sample` selects ten documents from `data/`. For a source collection, provide `data/original/` with HTML/PDF files, `data/parsed/` with Markdown files using the same document IDs, and `data/documentation.csv` with `id`, `genre` and `language` columns.

## Stages

| Stage | Module | Behavior |
|---|---|---|
| Load | `load.py` | Metadata CSV rows and parsed source text; retain missing-file records |
| Language | `langid.py` | Detect clear language signals, otherwise keep CSV label |
| Clean | `clean.py` | Normalize ligatures/line wrapping, remove repeated headers and site chrome |
| Title | `clean.py`, `titles.py` | HTML title extraction; verified PDF annotations or a heuristic |
| Deduplicate | `hygiene.py` | Remove navigation pages and exact/near duplicates |
| Chunk | `chunk.py` | Paragraph/sentence-aware passages, usually 200–300 words; a short tail may be merged |
| Core tags | `core_tags.py` | Rule-based country and year tags |
| Crisis tags | `flex_tags.py` | Optional annotations whose evidence is checked against source text |
| Embed | `embed.py` | Pinned Qwen3 checkpoint, float32, 256 dimensions, normalized; disk cache |
| Write/export | `bundle.py` | SQLite bundle creation and country export without re-embedding |

The cleaner recognizes navigation labels and boilerplate using normalized text and exact-match hashes.

## Optional annotations

Crisis tags and PDF title annotations are read from:

```text
data/annotations/flex_tags.json
data/annotations/titles.json
```

When absent, crisis tags are empty and PDF titles use the heuristic. To regenerate through the Claude CLI:

```sh
uv run python -m pipeline.flex_tags generate --model opus
uv run python -m pipeline.titles generate --model sonnet
```

If using an Anthropic API key instead, install the optional SDK with `uv sync --extra tagging` and provide a currently supported full model ID through `--model` or `ANTHROPIC_MODEL`. CLI aliases are not SDK model IDs. Configure credentials in the environment; never commit them. Generation sends source text to the provider/CLI, so it requires appropriate data authorization. A normal build performs no LLM API calls.

## Bundle contract

```text
meta       (key, value)
docs       (id, title, genre, language, year, countries, source_file)
tags       (doc_id, tag, evidence)
chunks     (chunk_id, doc_id, position, text, vector)
chunks_fts FTS5(text, title), rowid = chunk_id
removed    (id, reason)
```

- `docs.countries` uses canonical names joined by `'; '`. Empty means global; use `split_countries()`.
- Missing-file records have `source_file=NULL` and no chunks. They appear in the quality report but not search.
- Passage vectors are 256 little/native-endian float32 values, normalized for dot-product cosine similarity. Passage embeddings use `title + '\n\n' + text`; question embeddings use the shared instruction prompt.
- Metadata includes model, pinned revision, dimensions, thresholds, creation/expiry dates, counts and an embedding-runtime signature. Country exports preserve model provenance, tuned thresholds, demo notices and source expiry.
- The embedding cache includes model revision, dimensions, maximum sequence length, dtype, query prompt and relevant library versions in its key. An old cache is left intact but is not reused under a changed signature.
- A new bundle with a mismatched embedding signature is rejected at search/startup. Bundles created before this field was introduced cannot provide that check.
- The SQLite file contains passages, not original PDFs/HTML or model weights. Country export does not re-embed passages and must use a destination different from its source.

Tune thresholds only after building and evaluating the intended collection. `eval/run.py` evaluates stored thresholds by default; `--tune` compares a new configuration, and `--write` explicitly persists it. See [evaluation notes](../eval/RESULTS.md).
