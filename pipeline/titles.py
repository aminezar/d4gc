"""PDF titles, read once by a cloud LLM and frozen to ``data/annotations/titles.json``.

PDF text starts with journal headers, licence notices and author lists, so the line-based
heuristic in ``clean.extract_title`` often picks the wrong line ("MEETING REPORT",
"Available online 4 August 2021"). The LLM only *locates* the title: ``load_titles`` keeps
a title when ``flex_tags.verify`` finds it verbatim in the document, so nothing is invented.
Unverified or missing titles fall back to the heuristic. HTML documents use their <title> tag.

    uv run python -m pipeline.titles generate [--model sonnet] [--workers 8] [--ids a,b]
    uv run python -m pipeline.titles stats
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline import flex_tags

ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = ROOT / "data" / "parsed"
ORIGINAL_DIR = ROOT / "data" / "original"
OUT_PATH = ROOT / "data" / "annotations" / "titles.json"
MAX_WORDS = 600

SYSTEM_PROMPT = """You find the title of a document from Médecins Sans Frontières (MSF). The text is the start of a PDF converted to text, so it may begin with journal headers, page numbers, licence notices, running heads or author lists.

Return the document's own title: the title of the article, report, guide or letter itself.
- COPY it exactly as it appears in the text, character for character, in its original language. Do not translate, paraphrase, shorten or fix it. If the title is broken over several lines, join the lines with single spaces.
- A subtitle that directly follows the title may be included after a colon if it appears in the text that way.
- Not a title: journal names, article types ("MEETING REPORT", "Research article", "Open Access"), dates ("Available online ..."), section names ("Abstract", "Background"), author names, affiliations or licence text.
- If the document has no visible title (for example a letter or a page from the middle of a report), return null.
- Output ONLY JSON, no prose, no markdown fences: {"title": "<exact title>"} or {"title": null}"""


def _parse(raw: str) -> str | None:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    title = json.loads(raw[start : end + 1]).get("title")
    return " ".join(str(title).split()) if title else None


def find_title(text: str, model: str, backend: str, retries: int = 3, timeout: int = 120) -> tuple[str | None, str]:
    doc = flex_tags.prepare(text, MAX_WORDS)
    call = flex_tags._call_sdk if backend == "sdk" else flex_tags._call_cli
    last: Exception | None = None
    for attempt in range(retries):
        try:
            raw, used = call(doc, model, timeout, SYSTEM_PROMPT)
            return _parse(raw), used
        except Exception as e:  # noqa: BLE001 - retry anything, record the last error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def _load_out(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for k in ("meta", "titles", "failures"):
            data.setdefault(k, {})
        return data
    return {"meta": {}, "titles": {}, "failures": {}}


def _save_out(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["titles"] = dict(sorted(data["titles"].items()))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_titles(texts: dict[str, str], path: Path = OUT_PATH) -> dict[str, str]:
    """Frozen titles that occur verbatim in their document (``texts``: id -> parsed text)."""
    if not path.exists():
        return {}
    titles = json.loads(path.read_text(encoding="utf-8")).get("titles", {})
    return {i: t for i, t in titles.items() if t and i in texts and flex_tags.verify(t, texts[i])}


def generate(model: str, workers: int, ids: list[str] | None, out: Path, retries: int) -> None:
    backend = "sdk" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
    data = _load_out(out)
    pdfs = sorted(p.stem for p in ORIGINAL_DIR.glob("*.pdf") if (PARSED_DIR / f"{p.stem}.md").exists())
    todo = [i for i in (ids or pdfs) if i not in data["titles"]]
    data["meta"].update({"backend": backend, "model_requested": model})
    print(f"backend={backend} model={model} todo={len(todo)} done={len(data['titles'])}", file=sys.stderr)

    lock = threading.Lock()
    t0 = time.time()
    done = 0

    def work(doc_id: str):
        text = (PARSED_DIR / f"{doc_id}.md").read_text(encoding="utf-8", errors="replace")
        return find_title(text, model, backend, retries=retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            with lock:
                done += 1
                try:
                    title, used = fut.result()
                    data["titles"][doc_id] = title
                    data["failures"].pop(doc_id, None)
                    data["meta"]["model_used"] = used
                    msg = title or "(none)"
                except Exception as e:  # noqa: BLE001
                    data["failures"][doc_id] = str(e)[:500]
                    msg = f"FAILED {str(e)[:120]}"
                _save_out(data, out)
                print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {doc_id}: {msg[:100]}", file=sys.stderr, flush=True)
    data["meta"]["last_run_seconds"] = round(time.time() - t0, 1)
    _save_out(data, out)


def stats(out: Path) -> None:
    data = _load_out(out)
    texts = {i: (PARSED_DIR / f"{i}.md").read_text(encoding="utf-8", errors="replace") for i in data["titles"]}
    verified = load_titles(texts, out)
    given = [i for i, t in data["titles"].items() if t]
    print(f"docs: {len(data['titles'])}; with a title: {len(given)}; verified verbatim: {len(verified)};"
          f" failures: {len(data['failures'])}")
    for i in given:
        if i not in verified:
            print(f"  UNVERIFIED {i}: {data['titles'][i][:120]}")
    print("meta:", json.dumps(data["meta"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m pipeline.titles")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="find PDF titles with the cloud LLM (resumable)")
    g.add_argument("--model", default="sonnet", help="model alias or id (default: sonnet)")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--ids", default=None, help="comma-separated doc ids")
    g.add_argument("--retries", type=int, default=3)
    g.add_argument("--out", type=Path, default=OUT_PATH)
    s = sub.add_parser("stats", help="verification report on the frozen JSON")
    s.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        generate(args.model, args.workers, args.ids.split(",") if args.ids else None, args.out, args.retries)
    else:
        stats(args.out)


if __name__ == "__main__":
    main()
