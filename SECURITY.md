# Security

The application is intended for trusted, local use. Keep the server on its default `127.0.0.1` address: it has no authentication or multi-user isolation.

Open bundles only from trusted sources. Source-file access is restricted to `data/original/` and `data/parsed/`, and source HTML is sandboxed.

Unanswered questions are stored as plaintext in `gaps.jsonl`. Use `--no-gap-log` to disable logging, and avoid entering sensitive information. Logs do not synchronize automatically.

Report vulnerabilities through GitHub's security reporting feature when available. Do not put credentials or sensitive documents in public issues.
