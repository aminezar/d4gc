# Contributing

Install Python 3.11+ and uv, then run `uv sync --frozen` and `uv run pytest -q`. Build the sample with `uv run python -m pipeline.demo`.

For bug reports, include your OS, Python and uv versions, the command you ran, expected behavior and error output. Use the sample documents where possible and remove credentials or sensitive information from logs.

Keep changes focused, preserve offline behavior and add regression tests for functional bugs. Update the lockfile when changing dependencies.
