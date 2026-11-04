# Local application

FastAPI serves two interfaces with plain HTML/CSS/JavaScript and local assets. No CDN, map tile or remote font is needed.

```sh
uv run python -m pipeline.demo
uv run python app/server.py --bundle bundles/demo.sqlite
```

A bundle without `meta.country` defaults to Brussels mode and serves both `/brussels` and `/field`. A country bundle defaults to field mode. `--mode full|field` overrides this. Bundle names do not determine the mode.

```sh
uv run python -m pipeline.export 'DR Congo' --full bundles/demo.sqlite --out bundles/demo-cod.sqlite
uv run python app/server.py --bundle bundles/demo-cod.sqlite --port 8001
```

For real or semantic demo bundles, cache the model first, then use `HF_HUB_OFFLINE=1`. The default original demo is explicitly labelled keyword-only and does not load Qwen.

## Behavior

- `/api/search`: validated question (1–2,000 characters), hybrid retrieval for Qwen bundles, keyword fallback for original/development bundles. Results contain the best returned passage per document and source metadata. Similarity labels are not correctness probabilities.
- `/api/meta`: bundle metadata, mode, file size and whether question logging is enabled.
- `/api/catalog`, `/api/readiness`, `/api/crises`, `/api/quality`: Brussels filters and collection summaries.
- `/api/bundle/{country}`: selectable country and global documents.
- `POST /api/bundle/{country}/build`: validate document IDs against that country's candidates and return a downloadable SQLite file. Each request has a separate temporary output, cleaned after the response. It never overwrites the loaded bundle.
- `/api/field/docs`, `/api/doc/{id}`, `/doc/{id}`: browse passages stored inside the bundle.
- `/api/doc/{id}/file`: serve a source file only within `data/original/` or `data/parsed/`. Path traversal and symlink escape are rejected. Source HTML is sandboxed. Original files are separate from exported text bundles; the passage reader works without them.
- `/api/gaps`: most recent 1,000 valid unanswered-question log entries, grouped by query. `--no-gap-log` disables writing and reading the log.

The “Download bundle” link is offered after building. Save the file, transfer it to the field laptop and run `uv run python app/server.py --bundle /path/to/file.sqlite` there. Documents keep the source collection's expiry date; repackaging does not refresh old guidance.

## Query privacy and connectivity

By default, refusals append to `gaps.jsonl` in the repository root. Both interfaces share it when running in the same checkout. Separate laptops must transfer this file manually; automatic synchronization is not implemented. Questions may be sensitive. See [SECURITY.md](../SECURITY.md).

The network indicator reflects the browser's network status, not a guarantee of server reachability or synchronization. A local server can continue working with Wi-Fi off after dependencies and weights are cached.

The default bind address is `127.0.0.1`. `--host` exists for intentional deployments, but this server has no authentication or multi-user isolation. Do not expose it as a public service without additional controls.
